using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CyclonApp.Models
{
    public class CycloneDesign
    {
        public int Id { get; set; }

        // Relations
        public int ProjectId { get; set; }
        public Project Project { get; set; } = null!;

        public int TenantId { get; set; }
        public Tenant Tenant { get; set; } = null!;

        public int CycloneTypeId { get; set; }
        public CycloneType CycloneType { get; set; } = null!;

        [MaxLength(50)]
        public string? TagNumber { get; set; }    // Equipment tag e.g. "CYC-101"

        [MaxLength(200)]
        public string? Name { get; set; }

        public int CurrentRevision { get; set; } = 1;

        [MaxLength(1000)]
        public string? Notes { get; set; }

        // Audit
        public string? CreatedByUserId { get; set; }
        public AppUser? CreatedBy { get; set; }

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public DateTime? UpdatedAt { get; set; }

        // Navigation
        public ICollection<DesignRevision> Revisions { get; set; } = new List<DesignRevision>();
    }
}
