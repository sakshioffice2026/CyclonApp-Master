using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNet.Identity.EntityFramework;

namespace CyclonApp.Models
{
    public class AppUser : IdentityUser
    {
        // Multi-tenancy
        public int TenantId { get; set; }
        public Tenant Tenant { get; set; } = null!;

        [MaxLength(100)]
        public string? FirstName { get; set; }

        [MaxLength(100)]
        public string? LastName { get; set; }

        [MaxLength(150)]
        public string? Designation { get; set; }   // e.g. "Senior Engineer"

        public bool IsActive { get; set; } = true;

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        public DateTime? LastLoginAt { get; set; }

        // Computed
        public string FullName => $"{FirstName} {LastName}".Trim();

        public string DisplayName => !string.IsNullOrWhiteSpace(FullName)
            ? FullName
            : (UserName ?? Email ?? "Unknown");
    }
}
