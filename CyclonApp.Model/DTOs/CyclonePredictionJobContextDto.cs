namespace CyclonApp.Model.DTOs
{
    /// <summary>
    /// Everything about a field-prediction job that isn't part of the field
    /// solve's own output but is still needed later (when polling status or
    /// generating the PDF) to build an EngineeringInsightRequestDto — the
    /// cyclone type the job was started for, and the standard Lapple-model
    /// calculation for that revision, if one existed at job-start time.
    /// Replaces the old efficiency-only cache (GetKnownEfficiencyPercent)
    /// with the full standard-calculation object, since the insight engine
    /// now also needs cut size and other fields off it, not just Efficiency.
    /// </summary>
    public class CyclonePredictionJobContextDto
    {
        public string CycloneTypeCode { get; set; } = string.Empty;
        public CyclonOutputDto? StandardCalculation { get; set; }
    }
}