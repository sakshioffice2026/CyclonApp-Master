using System.Security.Claims;
using CyclonApp.Model.ViewModel;
using CyclonApp.Repositories.Contracts;
using CyclonApp.Repositories.Repositories;
using CyclonApp.Utilities;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace Web.Controllers
{
    public class AccountController : Controller
    {
        private readonly IAccount _accountRepository;
        public readonly IUnitOfWork _uow;
        private readonly ILogger<AccountController> _logger;
        private readonly ExceptionHandlerRepository _exceptionHandlerRepository;

        public AccountController(
            IAccount accountRepository,
            IUnitOfWork uow,
            ILogger<AccountController> logger,
            ExceptionHandlerRepository exceptionHandlerRepository)
        {
            _uow = uow;
            _accountRepository = accountRepository;
            _logger = logger;
            _exceptionHandlerRepository = exceptionHandlerRepository;
        }
        // ── LOGIN ─────────────────────────────────────────────────────────────
        [HttpGet, AllowAnonymous]
        public IActionResult Login(string? returnUrl = null)
        {
            try
            {
                if (User.Identity?.IsAuthenticated == true)
                    return RedirectToAction("Index", "Dashboard");

                return View(new LoginViewModel { ReturnUrl = returnUrl });
            }
            catch (Exception ex)
            {

                _uow.exceptionHandlerRepository.SaveException("AccountController",
                    "Login_GET",
                    ex.ToString());

                return View();
            }
        }
        [HttpPost, AllowAnonymous, ValidateAntiForgeryToken]
        public async Task<IActionResult> Login(LoginViewModel model)
        {
            try
            {
                if (!ModelState.IsValid)
                    return View(model);

                string encryptedPassword = Encryp_Decrypt.Encryptdata(model.Password);

                var user = await _accountRepository
                    .GetUserByCredentialsAsync(model.Email, encryptedPassword);

                if (user == null)
                {
                    TempData["Error"] = "Invalid Email or Password";
                    return View(model);
                }

                if (!user.IsActive)
                {
                    TempData["Error"] = "User is inactive";
                    return View(model);
                }

                await _accountRepository.UpdateLastLoginAsync(user);

                var claims = new List<Claim>
        {
            new Claim(ClaimTypes.NameIdentifier, user.Id.ToString()),
            new Claim(ClaimTypes.Name, user.Email),
            new Claim(ClaimTypes.Role, user.UserRole.RoleName),
            new Claim("TenantId", user.TenantId.ToString())
        };

                var principal = new ClaimsPrincipal(
                    new ClaimsIdentity(claims, CookieAuthenticationDefaults.AuthenticationScheme));

                await HttpContext.SignInAsync(
                    CookieAuthenticationDefaults.AuthenticationScheme,
                    principal,
                    new AuthenticationProperties { IsPersistent = model.RememberMe });

                return RedirectToAction("Index", "Dashboard");
            }
            catch (Exception ex)
            {

                _uow.exceptionHandlerRepository.SaveException(
                            "AccountController",
                    "Login",
                    ex.ToString());

                TempData["Error"] = "An error occurred while logging in.";
                return View(model);
            }
        }
        // ── LOGOUT ────────────────────────────────────────────────────────────
        [HttpPost, ValidateAntiForgeryToken]
        public async Task<IActionResult> Logout()
        {
            try
            {
                await HttpContext.SignOutAsync(
                    CookieAuthenticationDefaults.AuthenticationScheme);

                return RedirectToAction("Login");
            }
            catch (Exception ex)
            {

                _uow.exceptionHandlerRepository.SaveException(
                            "AccountController",
                    "Logout",
                    ex.ToString());

                return RedirectToAction("Login");
            }
        }
        // ── REGISTER ──────────────────────────────────────────────────────────
        [HttpGet]
        public async Task<IActionResult> Register()
        {
            try
            {
                await LoadRegisterViewBagsAsync();
                return View(new RegisterViewModel());
            }
            catch (Exception ex)
            {

                _uow.exceptionHandlerRepository.SaveException(
                    "AccountController",
                    "Register_GET",
                    ex.ToString());

                return RedirectToAction("Login");
            }
        }

        [HttpPost, ValidateAntiForgeryToken]
        public async Task<IActionResult> Register(RegisterViewModel model)
        {
            try
            {
                await LoadRegisterViewBagsAsync();

                if (!ModelState.IsValid)
                    return View(model);

                if (await _accountRepository.EmailExistsAsync(model.Email))
                {
                    ModelState.AddModelError("Email", "Email already exists.");
                    return View(model);
                }

                var userRole = await _accountRepository.GetRoleByIdAsync(model.UserRoleId);

                if (userRole is null)
                {
                    ModelState.AddModelError("UserRoleId", "Invalid role selected.");
                    return View(model);
                }

                int tenantId;

                if (User.Identity != null &&
                    User.Identity.IsAuthenticated &&
                    User.IsInRole("SuperAdmin"))
                {
                    if (!model.TenantId.HasValue)
                    {
                        ModelState.AddModelError(
                            "TenantId",
                            "Please select a tenant.");

                        return View(model);
                    }

                    tenantId = model.TenantId.Value;
                }
                else if (model.TenantId.HasValue)
                {
                    tenantId = model.TenantId.Value;
                }
                else
                {
                    tenantId = 1;
                }

                var user = new AppUser
                {
                    FirstName = model.FirstName,
                    LastName = model.LastName,
                    Designation = model.Designation,
                    Email = model.Email,
                    Password = Encryp_Decrypt.Encryptdata(model.Password),
                    Role = userRole.RoleName,
                    UserRoleId = userRole.Id,
                    TenantId = tenantId,
                    IsActive = true,
                    CreatedAt = DateTime.UtcNow
                };

                await _accountRepository.CreateUserAsync(user);

                TempData["Success"] = "Account created! Please login.";

                return RedirectToAction("Login");
            }
            catch (Exception ex)
            {

                _uow.exceptionHandlerRepository.SaveException(
                            "AccountController",
                    "Register_POST",
                    ex.ToString());

                TempData["Error"] = "Registration failed.";

                return View(model);
            }
        }
        // ── CHANGE PASSWORD ───────────────────────────────────────────────────

        [HttpGet]
        public IActionResult ChangePassword()
        {
            try
            {
                var email = User.Identity?.IsAuthenticated == true
                    ? User.FindFirstValue(ClaimTypes.Name) ?? string.Empty
                    : string.Empty;

                return View(new ChangePasswordViewModel
                {
                    Email = email
                });
            }
            catch (Exception ex)
            {
                _uow.exceptionHandlerRepository.SaveException(
                    "AccountController",
                    "ChangePassword_GET",
                    ex.ToString());

                return View();
            }
        }

        [HttpPost, ValidateAntiForgeryToken]
        public async Task<IActionResult> ChangePassword(ChangePasswordViewModel model)
        {
            try
            {
                if (!ModelState.IsValid)
                    return View(model);

                if (string.IsNullOrEmpty(model.Email))
                {
                    TempData["Error"] = "Session expired. Please login first.";
                    return RedirectToAction("Login");
                }

                var user = await _accountRepository.GetUserByEmailAsync(model.Email);
                if (user is null)
                {
                    TempData["Error"] = "User not found.";
                    return RedirectToAction("Login");
                }

                await _accountRepository.UpdatePasswordAsync(user, Encryp_Decrypt.Encryptdata(model.NewPassword));

                TempData["Success"] = "Password changed successfully!";
                return User.Identity?.IsAuthenticated == true
                    ? RedirectToAction("Index", "Dashboard")
                    : RedirectToAction("Login");
            }
            catch (Exception ex)
            {
                _uow.exceptionHandlerRepository.SaveException(
                    "AccountController",
                    "ChangePassword_POST",
                    ex.ToString());

                TempData["Error"] = "Password update failed.";
                return View(model);
            }
        }


        // ── PROFILE ───────────────────────────────────────────────────────────

        [HttpGet, Authorize]
        public async Task<IActionResult> Profile()
        {
            try
            {
                var userId = int.Parse(
                    User.FindFirstValue(ClaimTypes.NameIdentifier)!);

                var user = await _accountRepository.GetUserByIdAsync(userId);

                if (user == null)
                    return RedirectToAction("Login");

                return View(new ProfileViewModel
                {
                    FirstName = user.FirstName ?? "",
                    LastName = user.LastName ?? "",
                    Designation = user.Designation,
                    Email = user.Email,
                    TenantName = user.Tenant?.Name,
                    Role = user.Role
                });
            }
            catch (Exception ex)
            {
                _uow.exceptionHandlerRepository.SaveException(
                    "AccountController",
                    "Profile_GET",
                    ex.ToString());

                return RedirectToAction("Login");
            }
        }
        [HttpPost, Authorize, ValidateAntiForgeryToken]
        public async Task<IActionResult> Profile(ProfileViewModel model)
        {
            try
            {
                if (!ModelState.IsValid)
                    return View(model);

                var userId = int.Parse(
                    User.FindFirstValue(ClaimTypes.NameIdentifier)!);

                var user = await _accountRepository.GetUserByIdAsync(userId);

                if (user == null)
                    return RedirectToAction("Login");

                user.FirstName = model.FirstName;
                user.LastName = model.LastName;
                user.Designation = model.Designation;

                await _accountRepository.UpdateProfileAsync(user);

                TempData["Success"] = "Profile updated successfully.";

                return RedirectToAction("Profile");
            }
            catch (Exception ex)
            {
                _uow.exceptionHandlerRepository.SaveException(
                    "AccountController",
                    "Profile_POST",
                    ex.ToString());

                TempData["Error"] = "Profile update failed.";

                return View(model);
            }
        }

        // ── ACCESS DENIED ─────────────────────────────────────────────────────

        [HttpGet, AllowAnonymous]
        public IActionResult AccessDenied() => View();

        // ── HELPERS ───────────────────────────────────────────────────────────

        private async Task LoadRegisterViewBagsAsync()
        {
            try
            {
                if (User.IsInRole("SuperAdmin"))
                    ViewBag.Tenants = await _accountRepository.GetActiveTenantsAsync();

                var roles = await _accountRepository.GetActiveRolesAsync();

                if (User.IsInRole("ClientAdmin"))
                    roles = roles.Where(r => r.RoleName != "SuperAdmin").ToList();

                ViewBag.Roles = roles;
            }
            catch (Exception ex)
            {
                _uow.exceptionHandlerRepository.SaveException("AccountController","LoadRegisterViewBagsAsync",ex.ToString());

                throw;
            }
        }
         

    }
}
