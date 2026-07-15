using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CyclonApp.Models.ViewModel
{
    public class LoginViewModel
    {
        [Required, EmailAddress, Display(Name = "Email Address")]
        public string Email { get; set; } = string.Empty;

        [Required, DataType(DataType.Password)]
        public string Password { get; set; } = string.Empty;

        [Display(Name = "Remember me")]
        public bool RememberMe { get; set; }

        public string? ReturnUrl { get; set; }
    }

    // ── REGISTER (Engineer / Client Admin self-register) ──────────────────────────

    public class RegisterViewModel
    {
        [Required, Display(Name = "First Name"), MaxLength(100)]
        public string FirstName { get; set; } = string.Empty;

        [Required, Display(Name = "Last Name"), MaxLength(100)]
        public string LastName { get; set; } = string.Empty;

        [MaxLength(150), Display(Name = "Designation / Job Title")]
        public string? Designation { get; set; }

        [Required, EmailAddress, Display(Name = "Email Address")]
        public string Email { get; set; } = string.Empty;

        [Required, DataType(DataType.Password), MinLength(8),
         Display(Name = "Password")]
        public string Password { get; set; } = string.Empty;

        [Required, DataType(DataType.Password),
         Compare("Password", ErrorMessage = "Passwords do not match."),
         Display(Name = "Confirm Password")]
        public string ConfirmPassword { get; set; } = string.Empty;

        // Tenant selection (only used by SuperAdmin when creating users)
        public int? TenantId { get; set; }
    }

    // ── FORGOT PASSWORD ───────────────────────────────────────────────────────────

    public class ForgotPasswordViewModel
    {
        [Required, EmailAddress, Display(Name = "Email Address")]
        public string Email { get; set; } = string.Empty;
    }

    // ── RESET PASSWORD ────────────────────────────────────────────────────────────

    public class ResetPasswordViewModel
    {
        [Required]
        public string UserId { get; set; } = string.Empty;

        [Required]
        public string Token { get; set; } = string.Empty;

        [Required, DataType(DataType.Password), MinLength(8),
         Display(Name = "New Password")]
        public string Password { get; set; } = string.Empty;

        [Required, DataType(DataType.Password),
         Compare("Password", ErrorMessage = "Passwords do not match."),
         Display(Name = "Confirm New Password")]
        public string ConfirmPassword { get; set; } = string.Empty;
    }

    // ── CHANGE PASSWORD ───────────────────────────────────────────────────────────

    public class ChangePasswordViewModel
    {
        [Required, DataType(DataType.Password), Display(Name = "Current Password")]
        public string CurrentPassword { get; set; } = string.Empty;

        [Required, DataType(DataType.Password), MinLength(8),
         Display(Name = "New Password")]
        public string NewPassword { get; set; } = string.Empty;

        [Required, DataType(DataType.Password),
         Compare("NewPassword", ErrorMessage = "Passwords do not match."),
         Display(Name = "Confirm New Password")]
        public string ConfirmNewPassword { get; set; } = string.Empty;
    }

    // ── USER PROFILE ──────────────────────────────────────────────────────────────

    public class ProfileViewModel
    {
        [Required, Display(Name = "First Name"), MaxLength(100)]
        public string FirstName { get; set; } = string.Empty;

        [Required, Display(Name = "Last Name"), MaxLength(100)]
        public string LastName { get; set; } = string.Empty;

        [MaxLength(150), Display(Name = "Designation")]
        public string? Designation { get; set; }

        [EmailAddress, Display(Name = "Email")]
        public string? Email { get; set; }

        public string? TenantName { get; set; }
        public string? Role { get; set; }
    }

}
