using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CyclonApp.Models.ViewModel
{
    public class ProjectModel
    {
        public int Id { get; set; }

        // Multi-tenancy
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

        // Audit
        public string CreatedByUserId { get; set; } = string.Empty;
        public AppUser CreatedBy { get; set; } = null!;

        public string? LastModifiedByUserId { get; set; }

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public DateTime? UpdatedAt { get; set; }
        public DateTime? ArchivedAt { get; set; }

        // Navigation
        public ICollection<CycloneDesign> Designs { get; set; } = new List<CycloneDesign>();
    }

    public enum ProjectStatus
    {
        Draft,
        Active,
        Completed,
        Archived
    }
}
