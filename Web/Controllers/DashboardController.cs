using CyclonApp.Model.DTOs;
using CyclonApp.Repositories.Contracts;
using CyclonApp.Repositories.Repositories;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Security.Claims;

namespace Web.Controllers
{
    [Authorize]
    public class DashboardController : Controller
    {
        private readonly IDashboardRepository _dashboardRepository;
        private readonly IAccount _accountRepository;
        public readonly IUnitOfWork _uow;

        public DashboardController(
            IDashboardRepository dashboardRepository,
            IAccount accountRepository,
            IUnitOfWork uow)
        {
            _dashboardRepository = dashboardRepository;
            _accountRepository = accountRepository;
            _uow = uow;
        }

        public async Task<IActionResult> Index()
        {
            try
            {
                var userId = int.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier)!);
                var tenantId = int.Parse(User.FindFirstValue("TenantId")!);
                var role = User.FindFirstValue(ClaimTypes.Role) ?? "";

                var user = await _accountRepository.GetUserByIdAsync(userId);
                ViewBag.TenantName = user?.Tenant?.Name ?? "—";
                ViewBag.UserName = user?.DisplayName ?? "—";
                ViewBag.Role = role;

                var dto = role switch
                {
                    "SuperAdmin" => await _dashboardRepository.GetSuperAdminDashboardAsync(),
                    "ClientAdmin" => await _dashboardRepository.GetClientAdminDashboardAsync(tenantId),
                    "Engineer" => await _dashboardRepository.GetEngineerDashboardAsync(tenantId, userId),
                    _ => await _dashboardRepository.GetViewerDashboardAsync(tenantId)
                };

                return View(dto);
            }
            catch (Exception ex)
            {
                _uow.exceptionHandlerRepository.SaveException(
                    "DashboardController",
                    "Index",
                    ex.ToString());

                TempData["Error"] = "An error occurred while loading the dashboard.";
                return View(new DashboardDto());   // was: return View();
            }
        }
    }
}
