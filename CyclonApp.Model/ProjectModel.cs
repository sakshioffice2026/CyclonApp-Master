using System.ComponentModel.DataAnnotations;
using CyclonApp.Database;

namespace CyclonApp.Model
{
    public class ProjectModel
    {
        public int Id { get; set; }

        // Multi-tenancy
        public int TenantId { get; set; }

        public TenantModel Tenant { get; set; } = null!;

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

        // Keep ONLY ONE Status property
        public ProjectStatus Status { get; set; }

        // Audit
        public string CreatedByUserId { get; set; } = string.Empty;

        public AppUserModel CreatedBy { get; set; } = null!;

        public string? LastModifiedByUserId { get; set; }

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        public DateTime? UpdatedAt { get; set; }

        public DateTime? ArchivedAt { get; set; }

        // Navigation
        public ICollection<CyclonDesignModel> Designs { get; set; }
            = new List<CyclonDesignModel>();

     
    }
}