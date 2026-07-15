using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CyclonApp.Database;
using CyclonApp.Model;
using CyclonApp.Repositories.Contracts;


namespace CyclonApp.Repositories.Repositories
{
    public class AuditService : IAudit
    {
        public Task LogExportAsync(int revisionId, ExportType exportType, string? userId)
            => Task.CompletedTask;   // Full impl in Step 8

        Task IAudit.LogExportAsync(int revisionId, ExportType exportType, string? userId)
        {
            throw new NotImplementedException();
        }
    }
}
