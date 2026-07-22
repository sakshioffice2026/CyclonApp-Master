using CyclonApp.Model.DTOs;

namespace CyclonApp.Repositories.Contracts
{
    public interface IEngineeringInsight
    {
        /// <summary>
        /// <paramref name="knownEfficiencyPercent"/>: the real Lapple-model
        /// collection efficiency for this revision (CyclonOutputDto.Efficiency),
        /// if the caller has one. When supplied, the "Separation Efficiency"
        /// risk indicator reports this real figure instead of the field-solve's
        /// swirl-based placeholder estimate — see EstimateSeparationEfficiency's
        /// remarks for why that placeholder is not a substitute for it.
        ///
        /// <paramref name="knownPressureDropPa"/>: the deterministic
        /// Shepherd-Lapple pressure drop (Pa) for THIS design's own geometry
        /// (CyclonOutputDto.PressureDropPa), if the caller has one. When
        /// supplied, the "Pressure Drop" insight judges the field-solve's
        /// result against this design's own calculated baseline instead of
        /// one fixed Pa threshold shared by every cyclone type — a GP
        /// cyclone's normal operating ΔP is supposed to be lower than an
        /// HE cyclone's, so a single absolute threshold either over-flags
        /// GP-appropriate performance or under-flags HE's. Falls back to
        /// the fixed absolute threshold when not supplied (e.g. the standard
        /// calculation for this revision hasn't been run yet) — see
        /// EvaluatePressureDrop's remarks.
        /// </summary>
        CycloneHealthReportDto GenerateReport(FieldResultDto result, double? knownEfficiencyPercent = null, double? knownPressureDropPa = null);

        /// <summary>Renders a standalone, styled HTML document for a
        /// already-generated report — same "HTML served as the PDF
        /// download, user does File > Print > Save as PDF" pattern
        /// ExportRepository.GeneratePdfAsync/BuildPdfHtml already uses
        /// for the main design report (no real PDF library is wired up
        /// server-side yet; see that class's comments).</summary>
        string BuildReportHtml(CycloneHealthReportDto report, string? tagNumber, int revisionNumber, string? projectName);
    }
}