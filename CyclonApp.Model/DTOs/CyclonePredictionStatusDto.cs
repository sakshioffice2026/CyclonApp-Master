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
    }


}
