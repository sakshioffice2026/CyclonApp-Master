using CyclonApp.Model.DTOs;

namespace CyclonApp.Repositories.Contracts
{
    public interface IEngineeringInsight
    {
        /// <summary>
        /// Generates the AI Engineering Insight health report for a
        /// completed field-solve. <paramref name="request"/> carries the
        /// field-solve result, the cyclone type (for type-aware
        /// thresholds), and the standard calculation output (for cut
        /// size, Reynolds number, and efficiency) when available.
        /// </summary>
        CycloneHealthReportDto GenerateReport(EngineeringInsightRequestDto request);

        /// <summary>Renders a standalone, styled HTML document for an
        /// already-generated report — same "HTML served as the PDF
        /// download" pattern ExportRepository uses for the main design
        /// report.</summary>
        string BuildReportHtml(CycloneHealthReportDto report, string? tagNumber, int revisionNumber, string? projectName);
    }
}