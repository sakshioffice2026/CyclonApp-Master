using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CyclonApp.Database;

namespace CyclonApp.Model
{
    public class CyclonDesignModel
    
        {
        public int Id { get; set; }
        public int CycloneDesignId { get; set; }
        // Relations
        public int ProjectId { get; set; }
        public Project Project { get; set; } = null!;

        public int TenantId { get; set; }
        public TenantModel Tenant { get; set; } = null!;

        public int CyclonTypeId { get; set; }
        public CyclonTypeModel CyclonType { get; set; } = null!;

        [MaxLength(50)]
        public string? TagNumber { get; set; }    // Equipment tag e.g. "CYC-101"

        [MaxLength(200)]
        public string? Name { get; set; }

        public int CurrentRevision { get; set; } = 1;

        [MaxLength(1000)]
        public string? Notes { get; set; }

        // Audit
        public string? CreatedByUserId { get; set; }
        public AppUserModel? CreatedBy { get; set; }

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public DateTime? UpdatedAt { get; set; }

        // Navigation
        public ICollection<DesignRevisionModel> Revisions { get; set; } = new List<DesignRevisionModel>();
    }
}

