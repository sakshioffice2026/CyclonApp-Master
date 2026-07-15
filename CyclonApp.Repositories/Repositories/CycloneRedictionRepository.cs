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
    }
}