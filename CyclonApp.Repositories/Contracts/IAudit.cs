using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CyclonApp.Database;
using CyclonApp.Model;

namespace CyclonApp.Repositories.Contracts
{

    // ── Audit ─────────────────────────────────────────────────────────────────────

    public interface IAudit
    {
        Task LogExportAsync(int revisionId, ExportType exportType, string? userId);
      
    }

}
