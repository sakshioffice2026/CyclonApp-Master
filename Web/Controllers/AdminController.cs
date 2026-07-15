using System.Security.Claims;
using CyclonApp.Model.ViewModel;
using CyclonApp.Repositories.Contracts;
using CyclonApp.Repositories.Repositories;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace CyclonApp.Controllers;

[Authorize(Roles = "SuperAdmin,ClientAdmin")]
public class AdminController : Controller
{
    private readonly IAdminRepository _adminRepository;
    private readonly IAccount _accountRepository;
    public readonly IUnitOfWork _uow;

    public AdminController(
        IAdminRepository adminRepository,
        IAccount accountRepository,
        IUnitOfWork uow)
    {
        _adminRepository = adminRepository;
        _accountRepository = accountRepository;
        _uow = uow;
    }

    // GET /Admin/Index — overview panel
    public async Task<IActionResult> Index()
    {
        try
        {
            await SetTenantNameAsync();

            var isSuperAdmin = User.IsInRole("SuperAdmin");
            var tenantId = GetCurrentTenantId();

            var users = isSuperAdmin
                ? await _adminRepository.GetAllUsersAsync()
                : await _adminRepository.GetUsersByTenantAsync(tenantId);

            var allTenants = await _adminRepository.GetAllTenantsAsync();
            var tenants = isSuperAdmin ? allTenants : allTenants.Take(0).ToList();

            var vm = new AdminPanelViewModel
            {
                TotalUsers = users.Count(),
                ActiveUsers = users.Count(u => u.IsActive),
                TotalTenants = tenants.Count(),
                TotalProjects = 0,

                UsersByRole = users
                    .GroupBy(u => u.Role ?? "Unknown")
                    .ToDictionary(g => g.Key, g => g.Count()),

                TenantSummaries = tenants.Select(t => new TenantSummaryRow
                {
                    Name = t.Name,
                    UserCount = users.Count(u => u.TenantId == t.Id),
                    ProjectCount = t.Projects != null ? t.Projects.Count() : 0,
                    IsActive = t.IsActive
                }).ToList(),

                RecentUsers = users
                    .OrderByDescending(u => u.LastLoginAt ?? u.CreatedAt)
                    .Take(6)
                    .ToList()
            };

            return View(vm);
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "AdminController",
                "Index",
                ex.ToString());

            return RedirectToAction("Index", "Dashboard");
        }
    }

    // GET /Admin/Users
    public async Task<IActionResult> Users()
    {
        try
        {
            await SetTenantNameAsync();

            var users = User.IsInRole("SuperAdmin")
                ? await _adminRepository.GetAllUsersAsync()
                : await _adminRepository.GetUsersByTenantAsync(GetCurrentTenantId());

            return View(users);
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "AdminController",
                "Users",
                ex.ToString());

            return RedirectToAction("Index", "Dashboard");
        }
    }

    // GET /Admin/Tenants (SuperAdmin only)
    [Authorize(Roles = "SuperAdmin")]
    public async Task<IActionResult> Tenants()
    {
        try
        {
            await SetTenantNameAsync();
            return View(await _adminRepository.GetAllTenantsAsync());
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "AdminController",
                "Tenants",
                ex.ToString());

            return RedirectToAction("Index", "Dashboard");
        }
    }

    // ── HELPERS ───────────────────────────────────────────────────────────────

    private int GetCurrentTenantId()
    {
        var val = User.FindFirstValue("TenantId");
        return int.TryParse(val, out var id) ? id : 0;
    }

    private async Task SetTenantNameAsync()
    {
        try
        {
            var userIdClaim = User.FindFirstValue(ClaimTypes.NameIdentifier);
            if (!int.TryParse(userIdClaim, out var userId)) return;
            var user = await _accountRepository.GetUserByIdAsync(userId);
            ViewBag.TenantName = user?.Tenant?.Name ?? "—";
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "AdminController",
                "SetTenantNameAsync",
                ex.ToString());

            throw;
        }
    }
}
