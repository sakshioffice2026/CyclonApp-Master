using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Http.Json;
using CyclonApp.Database;
using CyclonApp.Model.DTOs;
using CyclonApp.Repositories.Contracts;
using Microsoft.Extensions.Configuration;

namespace CyclonApp.Repositories.Repositories
{
    public class CyclonePredictionRepository : ICyclonePrediction
    {
        private readonly IHttpClientFactory _httpClientFactory;
        private readonly ICyclonCalculation _calculationRepository;
        private readonly string _baseUrl;

        // Trusted-range thresholds — matches the known limits of the Lapple
        // correlation your existing calculation engine already relies on.
        private const double MinTrustedParticleSizeMicron = 5.0;
        private const double MaxPhysicsResidualPercent = 8.0;

        public CyclonePredictionRepository(
            IHttpClientFactory httpClientFactory,
            ICyclonCalculation calculationRepository,
            IConfiguration configuration)
        {
            _httpClientFactory = httpClientFactory;
            _calculationRepository = calculationRepository;
            _baseUrl = configuration["CyclonePredictionService:BaseUrl"]
                       ?? "http://localhost:8000";
        }

        public async Task<CyclonePredictionDto> PredictAsync(DesignRevision input, CyclonTypeRatios ratios)
        {
            // ── 1. Call the external prediction service ─────────────────────────
            var client = _httpClientFactory.CreateClient("CyclonePrediction");
            client.BaseAddress = new Uri(_baseUrl);

            var request = new PredictionRequest
            {
                FlowRateCFM = (double)input.FlowRateCFM,
                InletLineSizeIn = (double)input.InletLineSizeIn,
                OperatingTempC = (double)input.OperatingTempC,
                OperatingPressKPa = (double)input.OperatingPressKPa,
                GasType = input.GasType,
                ParticleSizeMicron = (double)input.ParticleSizeMicron,
                ParticleDensityKgm3 = (double)input.ParticleDensityKgm3,
                EffectiveTurns = (double)input.EffectiveTurns,
                InletHeightRatio = ratios.InletHeightRatio,
                InletWidthRatio = ratios.InletWidthRatio,
                OutletDiamRatio = ratios.OutletDiamRatio
            };

            var response = await client.PostAsJsonAsync("/predict", request);
            response.EnsureSuccessStatusCode();

            var result = await response.Content.ReadFromJsonAsync<PredictionResponse>()
                         ?? throw new Exception("Prediction service returned an empty response.");

            // ── 2. Cross-check against the existing Lapple calculation ──────────
            //     This is the "physics rule" enforcement — the prediction is
            //     never trusted purely on the external service's word.
            var lappleResult = _calculationRepository.Calculate(input, ratios);

            double efficiencyResidualPct = Math.Abs(result.PredictedEfficiency - lappleResult.Efficiency);

            // ── 3. Decide trusted-range flag and build a human-readable note ────
            bool withinParticleRange = (double)input.ParticleSizeMicron >= MinTrustedParticleSizeMicron;
            bool withinResidualTolerance = efficiencyResidualPct <= MaxPhysicsResidualPercent;
            bool isWithinTrustedRange = withinParticleRange && withinResidualTolerance;

            string? notes = null;
            if (!withinParticleRange)
                notes = $"Particle size ({input.ParticleSizeMicron} micron) is below the " +
                         $"{MinTrustedParticleSizeMicron}-micron range the underlying correlation was built on.";
            else if (!withinResidualTolerance)
                notes = $"Prediction differs from the standard calculation by {efficiencyResidualPct:F1}%, " +
                         "beyond the normal tolerance — treat as indicative only.";

            return new CyclonePredictionDto
            {
                Efficiency = Math.Round(result.PredictedEfficiency, 2),
                PressureDropPa = Math.Round(result.PredictedPressureDropPa, 2),
                PhysicsResidual = Math.Round(efficiencyResidualPct, 3),
                IsWithinTrustedRange = isWithinTrustedRange,
                Notes = notes
            };
        }

        // ── Async field-solving job — see ICyclonePrediction for contract notes ──

        public async Task<string> StartFieldPredictionAsync(DesignRevision input, CyclonDimensions dimensions)
        {
            var client = _httpClientFactory.CreateClient("CyclonePrediction");
            client.BaseAddress = new Uri(_baseUrl);

            var request = new PredictFieldStartRequest
            {
                BarrelDiameterMm = dimensions.BarrelDiameterMm,
                BarrelHeightMm = dimensions.BarrelHeightMm,
                ConeHeightMm = dimensions.ConeHeightMm,
                ExhaustDiaMm = dimensions.ExhaustDiaMm,
                ExhaustLengthMm = dimensions.ExhaustLengthMm,
                BottomOutletMm = dimensions.BottomOutletMm,
                InletHeightMm = dimensions.InletHeightMm,
                InletWidthMm = dimensions.InletWidthMm,
                FlowRateCFM = (double)input.FlowRateCFM,
                OperatingTempC = (double)input.OperatingTempC,
                OperatingPressKPa = (double)input.OperatingPressKPa,
                GasType = input.GasType
            };

            var response = await client.PostAsJsonAsync("/predict_field/start", request);

            if (response.StatusCode == HttpStatusCode.TooManyRequests)
            {
                var detail = await TryReadErrorDetailAsync(response);
                throw new FieldPredictionCapacityExceededException(
                    detail ?? "Too many field-prediction jobs running. Try again shortly.");
            }

            response.EnsureSuccessStatusCode();

            var result = await response.Content.ReadFromJsonAsync<PredictFieldStartResponse>()
                         ?? throw new Exception("Field prediction service returned an empty start response.");

            return result.JobId;
        }

        public async Task<FieldPredictionStatusDto?> GetFieldPredictionStatusAsync(string jobId)
        {
            var client = _httpClientFactory.CreateClient("CyclonePrediction");
            client.BaseAddress = new Uri(_baseUrl);

            var response = await client.GetAsync($"/predict_field/status/{Uri.EscapeDataString(jobId)}");

            if (response.StatusCode == HttpStatusCode.NotFound)
            {
                // Normal "no such job" outcome — never existed, or expired
                // past FIELD_JOB_TTL_SECONDS and was swept. Not an error.
                return null;
            }

            response.EnsureSuccessStatusCode();

            var wire = await response.Content.ReadFromJsonAsync<PredictFieldStatusResponse>()
                       ?? throw new Exception("Field prediction service returned an empty status response.");

            return new FieldPredictionStatusDto
            {
                JobId = wire.JobId,
                Status = wire.Status,
                ErrorMessage = wire.ErrorMessage,
                Result = wire.Result is null ? null : new FieldResultDto
                {
                    RMeters = wire.Result.RMeters,
                    ZMeters = wire.Result.ZMeters,
                    VRMs = wire.Result.VRMs,
                    VThetaMs = wire.Result.VThetaMs,
                    VZMs = wire.Result.VZMs,
                    PressurePa = wire.Result.PressurePa,
                    RhoKgm3 = wire.Result.RhoKgm3,
                    NuM2s = wire.Result.NuM2s,
                    VInletMs = wire.Result.VInletMs
                },
                // Unix seconds (float, matches Python's time.time()) -> UTC DateTime.
                CreatedAtUtc = wire.CreatedAtUnix.HasValue
                    ? DateTimeOffset.FromUnixTimeMilliseconds((long)(wire.CreatedAtUnix.Value * 1000)).UtcDateTime
                    : null,
                CompletedAtUtc = wire.CompletedAtUnix.HasValue
                    ? DateTimeOffset.FromUnixTimeMilliseconds((long)(wire.CompletedAtUnix.Value * 1000)).UtcDateTime
                    : null
            };
        }

        // FastAPI's HTTPException body is {"detail": "..."} — best-effort
        // read for a human-readable message; a malformed/empty body must
        // never blow up the 429/error path itself.
        private static async Task<string?> TryReadErrorDetailAsync(HttpResponseMessage response)
        {
            try
            {
                var body = await response.Content.ReadFromJsonAsync<Dictionary<string, object>>();
                return body != null && body.TryGetValue("detail", out var detail) ? detail?.ToString() : null;
            }
            catch
            {
                return null;
            }
        }

        // ── Wire-format classes for the external service call ───────────────────
        private class PredictionRequest
        {
            public double FlowRateCFM { get; set; }
            public double InletLineSizeIn { get; set; }
            public double OperatingTempC { get; set; }
            public double OperatingPressKPa { get; set; }
            public string GasType { get; set; } = "Air";
            public double ParticleSizeMicron { get; set; }
            public double ParticleDensityKgm3 { get; set; }
            public double EffectiveTurns { get; set; }
            public double InletHeightRatio { get; set; }
            public double InletWidthRatio { get; set; }
            public double OutletDiamRatio { get; set; }
        }

        private class PredictionResponse
        {
            public double PredictedEfficiency { get; set; }
            public double PredictedPressureDropPa { get; set; }
        }

        private class PredictFieldStartRequest
        {
            public double BarrelDiameterMm { get; set; }
            public double BarrelHeightMm { get; set; }
            public double ConeHeightMm { get; set; }
            public double ExhaustDiaMm { get; set; }
            public double ExhaustLengthMm { get; set; }
            public double BottomOutletMm { get; set; }
            public double InletHeightMm { get; set; }
            public double InletWidthMm { get; set; }
            public double FlowRateCFM { get; set; }
            public double OperatingTempC { get; set; } = 25.0;
            public double OperatingPressKPa { get; set; } = 101.325;
            public string GasType { get; set; } = "Air";
        }

        private class PredictFieldStartResponse
        {
            public string JobId { get; set; } = string.Empty;
            public string Status { get; set; } = string.Empty;
        }

        private class PredictFieldStatusResponse
        {
            public string JobId { get; set; } = string.Empty;
            public string Status { get; set; } = string.Empty;
            public string? ErrorMessage { get; set; }
            public FieldResultWireDto? Result { get; set; }
            public double? CreatedAtUnix { get; set; }
            public double? CompletedAtUnix { get; set; }
        }

        private class FieldResultWireDto
        {
            public List<double> RMeters { get; set; } = new();
            public List<double> ZMeters { get; set; } = new();
            public List<double> VRMs { get; set; } = new();
            public List<double> VThetaMs { get; set; } = new();
            public List<double> VZMs { get; set; } = new();
            public List<double> PressurePa { get; set; } = new();
            public double RhoKgm3 { get; set; }
            public double NuM2s { get; set; }
            public double VInletMs { get; set; }
        }
    }
}