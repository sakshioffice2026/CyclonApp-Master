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
        /// </summary>
        CycloneHealthReportDto GenerateReport(FieldResultDto result, double? knownEfficiencyPercent = null);

        /// <summary>Renders a standalone, styled HTML document for a
        /// already-generated report — same "HTML served as the PDF
        /// download, user does File > Print > Save as PDF" pattern
        /// ExportRepository.GeneratePdfAsync/BuildPdfHtml already uses
        /// for the main design report (no real PDF library is wired up
        /// server-side yet; see that class's comments).</summary>
        string BuildReportHtml(CycloneHealthReportDto report, string? tagNumber, int revisionNumber, string? projectName);
    }
}