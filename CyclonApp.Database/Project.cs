using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace CyclonApp.Database
{
    public class Project
    {
        public int Id { get; set; }

        public int TenantId { get; set; }
        public Tenant Tenant { get; set; } = null!;

        [MaxLength(50)]
        public string? ProjectNumber { get; set; }

        [Required, MaxLength(200)]
        public string Name { get; set; } = string.Empty;

        [MaxLength(200)]
        public string? ClientName { get; set; }

        [MaxLength(100)]
        public string? Location { get; set; }

        [MaxLength(1000)]
        public string? Description { get; set; }

        public ProjectStatus Status { get; set; } = ProjectStatus.Draft;

        // Explicit FK property
        public int CreatedByUserId { get; set; }

        // Explicitly bind navigation to FK
        [ForeignKey("CreatedByUserId")]
        public AppUser CreatedBy { get; set; } = null!;

        public int? LastModifiedByUserId { get; set; }

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public DateTime? UpdatedAt { get; set; }
        public DateTime? ArchivedAt { get; set; }

        public ICollection<CycloneDesign> Designs { get; set; } = new List<CycloneDesign>();
    }

    public enum ProjectStatus { Draft, Active, Completed, Archived }
}
