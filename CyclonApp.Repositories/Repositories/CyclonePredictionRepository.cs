using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using CyclonApp.Database;
using CyclonApp.Model.DTOs;
using CyclonApp.Repositories.Contracts;
using Microsoft.Extensions.Configuration;

namespace CyclonApp.Repositories.Repositories
{
    public class CyclonePredictionRepository : ICyclonePrediction
    {
        private readonly IHttpClientFactory _httpClientFactory;
        private readonly string _baseUrl;

        // jobId -> real Lapple-model efficiency (%), set by StartFieldPredictionAsync
        // when the caller has one. Deliberately a `static` field rather than an
        // instance field: this repository is registered AddScoped (see
        // Program.cs), so a new instance is created per HTTP request — a static
        // dictionary is what lets a value stashed on the "start job" request
        // still be readable on the later "poll status" / "download report"
        // requests. Mirrors the same in-memory-with-TTL pattern the Python
        // service already uses for job storage (see app.py), so no extra
        // persistence layer is introduced here. Capped and opportunistically
        // trimmed so a long-running process doesn't grow this unbounded if
        // jobs are started far more often than their results are ever read.
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, double> _knownEfficiencyByJobId = new();
        private const int MaxCachedEfficiencyEntries = 2000;

        // jobId -> the deterministic Shepherd-Lapple pressure drop (Pa) for
        // THIS design's own geometry, computed by CyclonCalculationRepository
        // (CyclonOutputDto.PressureDropPa). Same static/ConcurrentDictionary/
        // TTL-by-trim pattern as _knownEfficiencyByJobId above, and for the
        // same reason: this repository is AddScoped, so a value stashed when
        // the job starts needs a home that survives to the later "poll
        // status" request on a different instance. Lets the pressure-drop
        // insight compare the field-solve's result against the actual
        // design's calculated baseline instead of one fixed Pa threshold
        // shared by every cyclone type (Lapple/HE/GP all have different
        // "normal" pressure drops by design) — see
        // EngineeringInsightRepository.EvaluatePressureDrop.
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, double> _knownPressureDropByJobId = new();
        private const int MaxCachedPressureDropEntries = 2000;

        // PostAsJsonAsync, when called with no explicit JsonSerializerOptions,
        // defaults to JsonSerializerDefaults.Web — which camelCases property
        // names ("BarrelDiameterMm" -> "barrelDiameterMm"). The Python service's
        // Pydantic models declare PascalCase aliases (e.g. alias="BarrelDiameterMm")
        // with populate_by_name=True, so they accept that exact alias OR the
        // snake_case field name — but NOT camelCase. Sending camelCase made every
        // field look "missing" (422). PropertyNamingPolicy = null keeps outgoing
        // JSON keys exactly as the C# properties are written, matching the aliases.
        private static readonly JsonSerializerOptions OutgoingJsonOptions = new()
        {
            PropertyNamingPolicy = null
        };

        public CyclonePredictionRepository(
            IHttpClientFactory httpClientFactory,
            IConfiguration configuration)
        {
            _httpClientFactory = httpClientFactory;
            _baseUrl = configuration["CyclonePredictionService:BaseUrl"]
                       ?? "http://localhost:8000";
        }

        // ── Async field-solving job — see ICyclonePrediction for contract notes ──
        //
        // NOTE: the old synchronous PredictAsync() (scalar CyclonePINN
        // correction model, POST /predict) has been removed. The Python
        // service retired that endpoint and everything it depended on —
        // see app.py's module docstring — so this call site would only
        // ever 422/404. Field-solving (/predict_field/*) below is the only
        // prediction contract this service still serves.

        public async Task<string> StartFieldPredictionAsync(DesignRevision input, CyclonDimensions dimensions, double? knownEfficiencyPercent = null, double? knownPressureDropPa = null)
        {
            var client = _httpClientFactory.CreateClient("CyclonePrediction");
            client.BaseAddress = new Uri(_baseUrl);

            var request = new PredictFieldStartRequest
            {
                // app.py now resolves a per-type checkpoint via
                // FIELD_MODEL_CHECKPOINT_PATHS_BY_TYPE keyed on this field
                // (_get_inference_state), falling back to LAPPLE (or the
                // first configured type) with only a server-side console
                // warning if it's missing/blank — so an omitted value here
                // does not fail loudly, it silently evaluates the wrong
                // cyclone type's field. Must be populated for GP/Swift HE/
                // any non-default type to actually get their own model.
                CycloneTypeCode = input.CycloneDesign?.CycloneType?.Code ?? "LAPPLE",
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
                // The Python model requires a non-null string. The C# property
                // defaults to "Air" for objects created in code, but that
                // default never applies to a row already sitting in the DB
                // with GasType = NULL — sending that through as JSON null was
                // failing Pydantic validation (422) before this guard.
                GasType = string.IsNullOrWhiteSpace(input.GasType) ? "Air" : input.GasType
            };

            var response = await client.PostAsJsonAsync("/predict_field/start", request, OutgoingJsonOptions);

            if (response.StatusCode == HttpStatusCode.TooManyRequests)
            {
                var detail = await TryReadErrorDetailAsync(response);
                throw new FieldPredictionCapacityExceededException(
                    detail ?? "Too many field-prediction jobs running. Try again shortly.");
            }

            if (!response.IsSuccessStatusCode)
            {
                // Previously this went straight to EnsureSuccessStatusCode(),
                // which throws before the response body (FastAPI's {"detail": ...})
                // is ever read — every non-429 failure showed up in logs as a
                // bare "422 Unprocessable Content" with no indication of which
                // field was invalid. Surface it.
                var detail = await TryReadErrorDetailAsync(response);
                throw new Exception(
                    $"Field prediction service returned {(int)response.StatusCode} " +
                    $"{response.StatusCode}: {detail ?? "(no error detail in response body)"}");
            }

            var result = await response.Content.ReadFromJsonAsync<PredictFieldStartResponse>()
                         ?? throw new Exception("Field prediction service returned an empty start response.");

            if (knownEfficiencyPercent.HasValue)
            {
                if (_knownEfficiencyByJobId.Count >= MaxCachedEfficiencyEntries)
                {
                    // Best-effort trim, not a strict LRU — good enough to bound
                    // memory without adding a background sweep for what's
                    // already a short-lived, TTL-bounded set of job ids.
                    foreach (var staleKey in _knownEfficiencyByJobId.Keys.Take(MaxCachedEfficiencyEntries / 4))
                    {
                        _knownEfficiencyByJobId.TryRemove(staleKey, out _);
                    }
                }

                _knownEfficiencyByJobId[result.JobId] = knownEfficiencyPercent.Value;
            }

            if (knownPressureDropPa.HasValue)
            {
                if (_knownPressureDropByJobId.Count >= MaxCachedPressureDropEntries)
                {
                    foreach (var staleKey in _knownPressureDropByJobId.Keys.Take(MaxCachedPressureDropEntries / 4))
                    {
                        _knownPressureDropByJobId.TryRemove(staleKey, out _);
                    }
                }

                _knownPressureDropByJobId[result.JobId] = knownPressureDropPa.Value;
            }

            return result.JobId;
        }

        public double? GetKnownEfficiencyPercent(string jobId)
        {
            return _knownEfficiencyByJobId.TryGetValue(jobId, out var value) ? value : null;
        }

        public double? GetKnownPressureDropPa(string jobId)
        {
            return _knownPressureDropByJobId.TryGetValue(jobId, out var value) ? value : null;
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
                    VInletMs = wire.Result.VInletMs,
                    MassConservationStatus = wire.Result.MassConservationStatus,
                    MassFlowSpread = wire.Result.MassFlowSpread,
                    FinalLoss = wire.Result.FinalLoss,
                    // wire.Result.PngUrl is relative ("/renders/<jobId>/cfd_result.png")
                    // — resolve it against the same _baseUrl this repository
                    // already uses to reach the Python service, so the
                    // browser can load it directly as an <img src>.
                    PngUrl = string.IsNullOrEmpty(wire.Result.PngUrl)
                        ? null
                        : new Uri(new Uri(_baseUrl), wire.Result.PngUrl).ToString()
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
        private class PredictFieldStartRequest
        {
            public string CycloneTypeCode { get; set; } = "LAPPLE";
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

            // Optional — nullable so a response missing these (older
            // service version, or a run that didn't compute them) still
            // deserializes cleanly instead of throwing.
            public string? MassConservationStatus { get; set; }
            public double? MassFlowSpread { get; set; }
            public double? FinalLoss { get; set; }

            // Relative path on the Python service, e.g.
            // "/renders/<jobId>/cfd_result.png". Resolved to an absolute
            // URL (against _baseUrl) before being handed to FieldResultDto
            // below — the browser loading this <img src> has no reason to
            // know the Python service's base address otherwise.
            public string? PngUrl { get; set; }
        }
    }
}