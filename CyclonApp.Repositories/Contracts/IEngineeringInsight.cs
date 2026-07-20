using CyclonApp.Model.DTOs;

namespace CyclonApp.Repositories.Contracts
{
    public interface IEngineeringInsight
    {
        CycloneHealthReportDto GenerateReport(FieldResultDto result);

        /// <summary>Renders a standalone, styled HTML document for a
        /// already-generated report — same "HTML served as the PDF
        /// download, user does File > Print > Save as PDF" pattern
        /// ExportRepository.GeneratePdfAsync/BuildPdfHtml already uses
        /// for the main design report (no real PDF library is wired up
        /// server-side yet; see that class's comments).</summary>
        string BuildReportHtml(CycloneHealthReportDto report, string? tagNumber, int revisionNumber, string? projectName);
    }
}