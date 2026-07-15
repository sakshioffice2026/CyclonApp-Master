using CyclonApp.Database;
using CyclonApp.Model.DTOs;

namespace CyclonApp.Repositories.Contracts
{
    public interface IExport
    {
        Task<(DesignRevision Revision, CyclonOutputDto? Output)> GetRevisionForExportAsync(int revisionId);
        Task<List<ExportLog>> GetExportLogsAsync(int? tenantId = null, int take = 200);
        Task<List<ExportLog>> GetExportLogsByDesignAsync(int designId);
        Task LogExportAsync(int revisionId, int tenantId, int? exportedByUserId, ExportType type);
        Task<byte[]> GeneratePdfAsync(DesignRevision rev, CyclonOutputDto? output);
        Task<byte[]> GenerateExcelAsync(DesignRevision rev, CyclonOutputDto? output);
    }
}
