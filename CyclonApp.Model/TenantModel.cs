using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CyclonApp.Database;

namespace CyclonApp.Model
{
    public class TenantModel
    {
        public int Id { get; set; }

        [Required, MaxLength(150)]
        public string Name { get; set; } = string.Empty;

        [Required, MaxLength(80)]
        public string Slug { get; set; } = string.Empty;   // URL-safe identifier e.g. "acme-engineers"

        [MaxLength(300)]
        public string? LogoUrl { get; set; }

        [MaxLength(200)]
        public string? ContactEmail { get; set; }

        [MaxLength(100)]
        public string? Phone { get; set; }

        [MaxLength(500)]
        public string? Address { get; set; }

        public bool IsActive { get; set; } = true;

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        public DateTime? UpdatedAt { get; set; }

        // Navigation
        public ICollection<AppUserModel> Users { get; set; } = new List<AppUserModel>();
        public ICollection<Project> Projects { get; set; } = new List<Project>();
        public ICollection<CyclonDesignModel> CycloneDesign { get; set; } = new List<CyclonDesignModel>();
    }

}
