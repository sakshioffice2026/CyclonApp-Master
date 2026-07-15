using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CyclonApp.Database;
namespace CyclonApp.Model
{
    public class ExportLogModel
    {
        public int Id { get; set; }

        public int TenantId { get; set; }
        public TenantModel Tenant { get; set; } = null!;

        public int DesignRevisionId { get; set; }
        public DesignRevisionModel DesignRevision { get; set; } = null!;

        public ExportType ExportType { get; set; }

        public string? ExportedByUserId { get; set; }
        public AppUserModel? ExportedBy { get; set; }

        public DateTime ExportedAt { get; set; } = DateTime.UtcNow;
    }

    //public enum ExportType
    //{
    //    PDF,
    //    Excel
    //}

}
