using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CyclonApp.Database;

namespace CyclonApp.Model.ViewModel
{
    public class DesignExportViewModel
    {
        public int Id { get; set; }
        public string TagNumber { get; set; } = "—";
        public string CycloneType { get; set; } = "—";
        public int CurrentRevision { get; set; }
        public List<RevisionRowViewModel> Revisions { get; set; } = new();

        public List<ExportLog> ExportLogs { get; set; } = new();

        public enum ExportType
        {
            PDF,
            Excel
        }
    }
}
