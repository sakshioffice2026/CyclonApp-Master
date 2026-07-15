using System.Security.Claims;
using CyclonApp.Database;
using CyclonApp.Model.ViewModel;
using CyclonApp.Repositories;
using CyclonApp.Repositories.Contracts;
using CyclonApp.Repositories.Repositories;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace CyclonApp.Controllers;

[Authorize]
public class ProjectController : Controller
{
    private readonly ProjectRepository _projectRepository;
    private readonly IAccount _accountRepository;
    private readonly ITenant _tenantRepository;
    private readonly ILogger<ProjectController> _logger;
    public readonly IUnitOfWork _uow;

    public ProjectController(
        ProjectRepository projectRepository,
        IAccount accountRepository,
        ITenant tenantRepository,
        ILogger<ProjectController> logger,
        IUnitOfWork uow)
    {
        _projectRepository = projectRepository;
        _accountRepository = accountRepository;
        _tenantRepository = tenantRepository;
        _logger = logger;
        _uow = uow;
    }

    // ── INDEX ─────────────────────────────────────────────────────────────────

    [HttpGet]
    public async Task<IActionResult> Index(string? status, string? search)
    {
        try
        {
            SetBreadcrumb(("Projects", null));

            var vm = await _projectRepository.GetProjectIndexDataAsync(status, search);

            await SetTenantNameAsync();
            return View(vm);
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "ProjectController",
                "Index",
                ex.ToString());

            TempData["Error"] = "An error occurred while loading projects.";
            return View();
        }
    }

    // ── DETAIL ────────────────────────────────────────────────────────────────

    [HttpGet]
    public async Task<IActionResult> Detail(int id)
    {
        try
        {
            var vm = await _projectRepository.GetProjectDetailAsync(id);
            if (vm == null) return NotFound();

            SetBreadcrumb(
                ("Projects", Url.Action("Index")),
                (vm.Name, null));

            await SetTenantNameAsync();
            return View(vm);
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "ProjectController",
                "Detail",
                ex.ToString());

            TempData["Error"] = "An error occurred while loading the project.";
            return RedirectToAction("Index");
        }
    }

    // ── CREATE (GET) ──────────────────────────────────────────────────────────

    [HttpGet]
    [Authorize(Roles = "SuperAdmin,ClientAdmin,Engineer")]
    public async Task<IActionResult> Create()
    {
        try
        {
            SetBreadcrumb(
                ("Projects", Url.Action("Index")),
                ("New Project", null));

            await SetTenantNameAsync();

            var projectNumber = await _projectRepository.GenerateProjectNumberAsync();

            return View(new ProjectFormViewModel
            {
                ProjectNumber = projectNumber,
                Status = ProjectStatus.Draft
            });
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "ProjectController",
                "Create_GET",
                ex.ToString());

            return RedirectToAction("Index");
        }
    }

    // ── CREATE (POST) ─────────────────────────────────────────────────────────

    [HttpPost]
    [Authorize(Roles = "SuperAdmin,ClientAdmin,Engineer")]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Create(ProjectFormViewModel vm)
    {
        try
        {
            if (!ModelState.IsValid)
            {
                await SetTenantNameAsync();
                return View(vm);
            }

            var isDuplicate = await _projectRepository.IsProjectNumberExistsAsync(vm.ProjectNumber);
            if (isDuplicate)
            {
                ModelState.AddModelError("ProjectNumber",
                    "This project number already exists. Please use a unique number.");
                await SetTenantNameAsync();
                return View(vm);
            }

            var currentUser = await GetCurrentUserAsync();
            if (currentUser == null) return RedirectToAction("Login", "Account");

            var projectId = await _projectRepository.CreateProjectAsync(
                vm,
                _tenantRepository.CurrentTenantId,
                currentUser.Id);

            _logger.LogInformation("Project {ProjectNumber} created by {User}.",
                vm.ProjectNumber, currentUser.Email);

            TempData["Success"] = $"Project \"{vm.Name}\" created successfully.";
            return RedirectToAction("Detail", new { id = projectId });
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "ProjectController",
                "Create_POST",
                ex.ToString());

            TempData["Error"] = "An error occurred while creating the project.";
            await SetTenantNameAsync();
            return View(vm);
        }
    }

    // ── EDIT (GET) ────────────────────────────────────────────────────────────

    [HttpGet]
    [Authorize(Roles = "SuperAdmin,ClientAdmin,Engineer")]
    public async Task<IActionResult> Edit(int id)
    {
        try
        {
            var (canEdit, message) = await _projectRepository.CanEditProjectAsync(id);
            if (!canEdit)
            {
                TempData["Error"] = message;
                return RedirectToAction("Detail", new { id });
            }

            var vm = await _projectRepository.GetProjectForEditAsync(id);
            if (vm == null) return NotFound();

            SetBreadcrumb(
                ("Projects", Url.Action("Index")),
                (vm.Name, Url.Action("Detail", new { id })),
                ("Edit", null));

            await SetTenantNameAsync();
            return View(vm);
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "ProjectController",
                "Edit_GET",
                ex.ToString());

            TempData["Error"] = "An error occurred while loading the project for editing.";
            return RedirectToAction("Detail", new { id });
        }
    }

    // ── EDIT (POST) ───────────────────────────────────────────────────────────

    [HttpPost]
    [Authorize(Roles = "SuperAdmin,ClientAdmin,Engineer")]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Edit(int id, ProjectFormViewModel vm)
    {
        try
        {
            if (!ModelState.IsValid)
            {
                await SetTenantNameAsync();
                return View(vm);
            }

            var project = await _projectRepository.GetProjectByIdAsync(id);
            if (project == null) return NotFound();

            var isDuplicate = await _projectRepository.IsProjectNumberExistsAsync(vm.ProjectNumber, id);
            if (isDuplicate)
            {
                ModelState.AddModelError("ProjectNumber",
                    "This project number already exists. Please use a unique number.");
                await SetTenantNameAsync();
                return View(vm);
            }

            var currentUser = await GetCurrentUserAsync();
            if (currentUser == null) return RedirectToAction("Login", "Account");

            var success = await _projectRepository.UpdateProjectAsync(id, vm, currentUser.Id);
            if (!success) return NotFound();

            _logger.LogInformation("Project {Id} updated by {User}.", id, currentUser.Email);

            TempData["Success"] = $"Project \"{vm.Name}\" updated successfully.";
            return RedirectToAction("Detail", new { id });
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "ProjectController",
                "Edit_POST",
                ex.ToString());

            TempData["Error"] = "An error occurred while updating the project.";
            await SetTenantNameAsync();
            return View(vm);
        }
    }

    // ── CHANGE STATUS ─────────────────────────────────────────────────────────

    [HttpPost]
    [Authorize(Roles = "SuperAdmin,ClientAdmin,Engineer")]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> ChangeStatus(int id, string newStatus)
    {
        try
        {
            var project = await _projectRepository.GetProjectByIdAsync(id);
            if (project == null) return NotFound();

            if (!Enum.TryParse<ProjectStatus>(newStatus, out var status))
            {
                TempData["Error"] = "Invalid status.";
                return RedirectToAction("Detail", new { id });
            }

            var currentUser = await GetCurrentUserAsync();
            if (currentUser == null) return RedirectToAction("Login", "Account");

            var success = await _projectRepository.ChangeStatusAsync(id, status, currentUser.Id);
            if (!success) return NotFound();

            _logger.LogInformation("Project {Id} status changed to {Status} by {User}.",
                id, newStatus, currentUser.Email);

            TempData["Success"] = $"Project status changed to {newStatus}.";
            return RedirectToAction("Detail", new { id });
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "ProjectController",
                "ChangeStatus",
                ex.ToString());

            TempData["Error"] = "An error occurred while changing the project status.";
            return RedirectToAction("Detail", new { id });
        }
    }

    // ── DELETE ────────────────────────────────────────────────────────────────

    [HttpPost]
    [Authorize(Roles = "SuperAdmin,ClientAdmin")]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Delete(int id)
    {
        try
        {
            var (success, message) = await _projectRepository.DeleteProjectAsync(id);

            if (!success)
            {
                TempData["Error"] = message;
                return RedirectToAction("Detail", new { id });
            }

            _logger.LogInformation("Project '{Name}' deleted.", message);
            TempData["Success"] = $"Project \"{message}\" deleted.";
            return RedirectToAction("Index");
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "ProjectController",
                "Delete",
                ex.ToString());

            TempData["Error"] = "An error occurred while deleting the project.";
            return RedirectToAction("Detail", new { id });
        }
    }

    // ── HELPERS ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Sets breadcrumb navigation in ViewBag
    /// </summary>
    private void SetBreadcrumb(params (string Label, string? Url)[] crumbs)
    {
        ViewBag.Breadcrumbs = crumbs.Select(c => (c.Label, c.Url)).ToList();
    }

    /// <summary>
    /// Retrieves current logged-in user from claims
    /// </summary>
    private async Task<AppUser?> GetCurrentUserAsync()
    {
        var userIdClaim = User.FindFirstValue(ClaimTypes.NameIdentifier);
        if (!int.TryParse(userIdClaim, out var userId)) return null;
        return await _accountRepository.GetUserByIdAsync(userId);
    }

    /// <summary>
    /// Sets tenant name in ViewBag from current user
    /// </summary>
    private async Task SetTenantNameAsync()
    {
        try
        {
            var user = await GetCurrentUserAsync();
            if (user?.Tenant != null)
                ViewBag.TenantName = user.Tenant.Name;
            else
                ViewBag.TenantName = "-";
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "ProjectController",
                "SetTenantNameAsync",
                ex.ToString());

            throw;
        }
    }
}
