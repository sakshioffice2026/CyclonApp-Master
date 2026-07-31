using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace CyclonApp.Model.DTOs
{

    /// <summary>
    /// Result of polling GET /predict_field/status/{jobId} on the Python
    /// field-solving service. Mirrors app.py's PredictFieldStatusResponse.
    /// A null return from ICyclonePrediction.GetFieldPredictionStatusAsync
    /// means the service returned 404 — the job never existed, or it
    /// finished and was TTL-swept out of the service's job store; both are
    /// a normal "nothing to show" outcome, not an error condition.
    ///
    /// JSON property names are pinned explicitly rather than left to
    /// ASP.NET's default CamelCase naming policy: that policy's handling
    /// of consecutive capitals is not the simple "lowercase the first
    /// letter" most people assume (e.g. "VRMs" -> "vrMs", "VZMs" ->
    /// "vzMs" — not "vRMs"/"vZMs"), which is exactly the failure mode for
    /// abbreviation-heavy names like these. Pinning avoids depending on
    /// that ambiguity for a JSON API contract the frontend JS reads.
    /// </summary>
    public class FieldPredictionStatusDto
    {
        [JsonPropertyName("jobId")]
        public string JobId { get; set; } = string.Empty;

        /// <summary>"running" | "completed" | "failed"</summary>
        [JsonPropertyName("status")]
        public string Status { get; set; } = string.Empty;

        [JsonPropertyName("errorMessage")]
        public string? ErrorMessage { get; set; }

        [JsonPropertyName("result")]
        public FieldResultDto? Result { get; set; }

        [JsonPropertyName("createdAtUtc")]
        public DateTime? CreatedAtUtc { get; set; }

        [JsonPropertyName("completedAtUtc")]
        public DateTime? CompletedAtUtc { get; set; }

        /// <summary>AI Engineering Insights health report for this job's
        /// result, computed by IEngineeringInsight.GenerateReport once the
        /// job is "completed". Null while running/failed, or if insight
        /// generation itself threw (a bad insight must never break the
        /// underlying field-solve result the client already has).</summary>
        [JsonPropertyName("insights")]
        public CycloneHealthReportDto? Insights { get; set; }
    }

    /// <summary>
    /// Full velocity/pressure field grid for a completed field-solving job.
    /// Parallel lists — index i across all lists is the same (r, z) point.
    /// Mirrors app.py's FieldResultDto. See FieldPredictionStatusDto remarks
    /// on why JSON names are pinned explicitly here.
    /// </summary>
    public class FieldResultDto
    {
        [JsonPropertyName("rMeters")]
        public List<double> RMeters { get; set; } = new();

        [JsonPropertyName("zMeters")]
        public List<double> ZMeters { get; set; } = new();

        [JsonPropertyName("vRMs")]
        public List<double> VRMs { get; set; } = new();

        [JsonPropertyName("vThetaMs")]
        public List<double> VThetaMs { get; set; } = new();

        [JsonPropertyName("vZMs")]
        public List<double> VZMs { get; set; } = new();

        [JsonPropertyName("pressurePa")]
        public List<double> PressurePa { get; set; } = new();

        [JsonPropertyName("rhoKgm3")]
        public double RhoKgm3 { get; set; }

        [JsonPropertyName("nuM2s")]
        public double NuM2s { get; set; }

        [JsonPropertyName("vInletMs")]
        public double VInletMs { get; set; }

        /// <summary>Inlet-ring-average minus vortex-finder-bore-average
        /// static pressure at the z=0 plane, computed Python-side (see
        /// compute_pressure_drop in sanity_check.py). This is the
        /// field-solve's estimate of the same quantity
        /// CyclonCalculationRepository's Shepherd-Lapple dP_Pa describes,
        /// and should be preferred over any max/min-based approximation
        /// derived from PressurePa here -- that instead captures the
        /// radial (swirl) pressure spread, a different physical quantity.
        /// Null means the Python side couldn't isolate usable inlet/outlet
        /// points (e.g. too coarse a grid) -- treat as "not computed", not
        /// as a zero pressure drop.</summary>
        [JsonPropertyName("pressureDropPa")]
        public double? PressureDropPa { get; set; }

        // ── Mass-conservation diagnostics ────────────────────────────────
        // Nullable: older/unmodified Python service responses (or a
        // failed/partial solve) may omit these entirely. Treating them as
        // optional means System.Text.Json simply leaves them null instead
        // of throwing, so existing grid-array deserialization (RMeters,
        // VRMs, VInletMs, etc.) is unaffected either way.

        /// <summary>e.g. "ok" | "warning" | "failed" — solver's own verdict
        /// on how well mass was conserved across the field.</summary>
        [JsonPropertyName("massConservationStatus")]
        public string? MassConservationStatus { get; set; }

        /// <summary>Spread/variance of mass flow across the solved field
        /// (units as defined by the Python service, typically kg/s or a
        /// dimensionless ratio — treat as opaque unless documented).</summary>
        [JsonPropertyName("massFlowSpread")]
        public double? MassFlowSpread { get; set; }

        /// <summary>Final training/solver loss value at convergence.</summary>
        [JsonPropertyName("finalLoss")]
        public double? FinalLoss { get; set; }

        /// <summary>Relative URL on the Python service (e.g.
        /// "/renders/&lt;jobId&gt;/cfd_result.png") for the matplotlib
        /// CFD-style contour PNG produced by render_field.py. Null means
        /// rendering hasn't completed yet, or failed — the failure is
        /// logged service-side but never fails the underlying field-solve
        /// job, so absence here should read as "image unavailable", not
        /// as an error for the whole result. CyclonePredictionRepository
        /// resolves this into an absolute URL (PngUrl is relative to the
        /// Python service's own base address, not this app's).</summary>
        [JsonPropertyName("pngUrl")]
        public string? PngUrl { get; set; }

        /// <summary>Short exception message when PngUrl is null because
        /// rendering failed (e.g. "ValueError: ..."). The full traceback
        /// is persisted service-side at /renders/&lt;jobId&gt;/render_error.log
        /// for deeper debugging; this is just enough for the UI to show a
        /// concrete reason instead of a generic failure message.</summary>
        [JsonPropertyName("renderError")]
        public string? RenderError { get; set; }
    }


}