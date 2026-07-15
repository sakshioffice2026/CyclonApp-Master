using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CyclonApp.Model
{


    public class AppUserModel
    {
        // Multi-tenancy
        public int Id { get; set; }

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

        //  UserRole FK
        public int UserRoleId { get; set; }
        public UserRoleModel UserRole { get; set; } = null!;

        public string DisplayName =>
    !string.IsNullOrWhiteSpace(FullName) ? FullName : $"User {Id}";
    }
}

