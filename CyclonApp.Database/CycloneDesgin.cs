using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace CyclonApp.Database
{
    public class CycloneDesign
    {
        public int Id { get; set; }

        public string? Notes { get; set; } 
        public int CycloneDesignId {  get; set; }
        public int ProjectId { get; set; }
        public Project Project { get; set; } = null!;

        public int TenantId { get; set; }
        public Tenant Tenant { get; set; } = null!;

        public int CycloneTypeId { get; set; }
        public CycloneType CycloneType { get; set; } = null!;

        [MaxLength(50)]
        public string? TagNumber { get; set; }

        [MaxLength(200)]
        public string? Name { get; set; }

        public int CurrentRevision { get; set; }

        // explicit FK property
        public int CreatedByUserId { get; set; }

        // explicit navigation bound to FK
        [ForeignKey(nameof(CreatedByUserId))]
        public AppUser CreatedBy { get; set; } = null!;

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        public DateTime? UpdatedAt { get; set; }

        public ICollection<DesignRevision> Revisions { get; set; } = new List<DesignRevision>();
    }
}
