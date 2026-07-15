using System.ComponentModel.DataAnnotations.Schema;
using CyclonApp.Database;

public class AppUser
{
    public int Id { get; set; }
    public string FirstName { get; set; } = string.Empty;
    public string LastName { get; set; } = string.Empty;
    public string? Designation { get; set; }
    public string Email { get; set; } = string.Empty;
    public string Password { get; set; } = string.Empty;
    [NotMapped]
    public string Role { get; set; } = string.Empty;
    public bool IsActive { get; set; } = true;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime? LastLoginAt { get; set; }
    public int TenantId { get; set; }
    public Tenant Tenant { get; set; } = null!;
    public int UserRoleId { get; set; }
    public UserRole UserRole { get; set; } = null!;

    // used in ProjectController
    public string DisplayName =>
        $"{FirstName} {LastName}".Trim().Length > 0
        ? $"{FirstName} {LastName}".Trim()
        : Email;

}