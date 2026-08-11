using System.Linq;
using System.Security.Claims;
using CyclonApp.Database;
using CyclonApp.Model.ViewModel;
using CyclonApp.Repositories.Contracts;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace CyclonApp.Controllers;

// Dedicated controller for 3D CAD deliverables (STEP/DXF/OBJ/PDF), kept
// separate from ExportController (which handles the PDF report / Excel
// sheet exports) since this talks to a different backing capability
// (FreeCAD via the Python service) with its own failure modes (subprocess
// timeouts, missing FreeCAD install) and its own richer result shape
// (5 files instead of 1).
[Authorize]
public class CadController : Controller
{
    private readonly ICadGeneration _cadRepository;
    private readonly IExport _exportRepository;
    private readonly ApplicationDbContext _db;
    private readonly ILogger<CadController> _logger;
    public readonly IUnitOfWork _uow;

    public CadController(
        ICadGeneration cadRepository,
        IExport exportRepository,
        ApplicationDbContext db,
        ILogger<CadController> logger,
        IUnitOfWork uow)
    {
        _cadRepository = cadRepository;
        _exportRepository = exportRepository;
        _db = db;
        _logger = logger;
        _uow = uow;
    }

    // ── INDEX — landing page: Project -> Design -> Revision cascading
    //    pickers + file-type checkboxes. No revisionId needed to land here. ──
    [HttpGet]
    public async Task<IActionResult> Index()
    {
        try
        {
            var tenantId = GetCurrentTenantId();
            var projects = await _db.Projects
                .Where(p => p.TenantId == tenantId)
                .OrderBy(p => p.Name)
                .Select(p => new { p.Id, p.Name, p.ProjectNumber })
                .ToListAsync();

            ViewBag.Projects = projects;
            ViewBag.Breadcrumbs = new (string, string?)[]
            {
                ("Projects", Url.Action("Index", "Project")),
                ("3D / CAD", null)
            };

            return View(new CadViewModel());
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException("CadController", "Index", ex.ToString());
            TempData["Error"] = "An error occurred while loading the CAD page.";
            return RedirectToAction("Index", "Project");
        }
    }

    // ── AJAX: designs for a chosen project ──────────────────────────────────
    [HttpGet]
    public async Task<IActionResult> GetDesigns(int projectId)
    {
        var tenantId = GetCurrentTenantId();
        var designs = await _db.CycloneDesign
            .Where(d => d.ProjectId == projectId && d.TenantId == tenantId)
            .OrderByDescending(d => d.UpdatedAt)
            .Select(d => new { d.Id, d.TagNumber, d.Name })
            .ToListAsync();

        return Json(designs);
    }

    // ── AJAX: revisions for a chosen design (only ones with calculated
    //    results can generate CAD — same HasResults rule as ExportController) ──
    [HttpGet]
    public async Task<IActionResult> GetRevisions(int designId)
    {
        var revisions = await _db.DesignRevisions
            .Where(r => r.CycloneDesignId == designId)
            .OrderByDescending(r => r.RevisionNumber)
            .Select(r => new
            {
                r.Id,
                r.RevisionNumber,
                r.RevisionNote,
                HasResults = r.CalculatedAt != null
            })
            .ToListAsync();

        return Json(revisions);
    }

    // ── GENERATE — calls the Python service once (it always produces all 5
    //    files in one FreeCAD run — there's no partial-generation mode), then
    //    the view shows/downloads only the file types the user checked. ──
    [HttpPost]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Generate(int revisionId, List<string> fileTypes)
    {
        var vm = await BuildViewModelAsync(revisionId);
        vm.SelectedFileTypes = fileTypes ?? new List<string>();
        SetBreadcrumb(revisionId, vm);

        try
        {
            vm.Result = await _cadRepository.GenerateCadAsync(revisionId);

            await _exportRepository.LogExportAsync(
                revisionId, GetCurrentTenantId(), GetCurrentUserId(), ExportType.CadBundle);

            _logger.LogInformation("CAD generated for RevisionId={RevisionId}", revisionId);
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException("CadController", "Generate", ex.ToString());
            vm.ErrorMessage = "CAD generation failed. " + ex.Message;
        }

        return View("Result", vm);
    }

    // ── HISTORY — past CAD generations across the tenant, mirrors
    //    ExportController.History() ──────────────────────────────────────
    [HttpGet]
    public async Task<IActionResult> History()
    {
        try
        {
            var logs = await _exportRepository.GetExportLogsAsync(GetCurrentTenantId(), take: 200);
            var cadLogs = logs.Where(l =>
                l.ExportType == ExportType.CadBundle ||
                l.ExportType == ExportType.STEP ||
                l.ExportType == ExportType.DXF ||
                l.ExportType == ExportType.OBJ).ToList();

            ViewBag.Breadcrumbs = new (string, string?)[]
            {
                ("Projects", Url.Action("Index", "Project")),
                ("CAD History", null)
            };

            return View(cadLogs);
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException("CadController", "History", ex.ToString());
            TempData["Error"] = "An error occurred while loading CAD history.";
            return RedirectToAction("Index", "Project");
        }
    }

    // ── HELPERS ───────────────────────────────────────────────────────────

    private async Task<CadViewModel> BuildViewModelAsync(int revisionId)
    {
        var (rev, _) = await _exportRepository.GetRevisionForExportAsync(revisionId);
        return new CadViewModel
        {
            RevisionId = rev.Id,
            DesignId = rev.CycloneDesignId,
            TagNumber = rev.CycloneDesign?.TagNumber,
            CycloneType = rev.CycloneDesign?.CycloneType?.Name,
            RevisionNumber = rev.RevisionNumber
        };
    }

    private void SetBreadcrumb(int revisionId, CadViewModel vm)
    {
        ViewBag.Breadcrumbs = new (string, string?)[]
        {
            ("Projects", Url.Action("Index", "Project")),
            ("Export / Reports", Url.Action("Index", "Export", new { designId = vm.DesignId })),
            ($"CAD — {vm.TagNumber} Rev {vm.RevisionNumber}", null)
        };
    }

    private int GetCurrentUserId()
    {
        var claim = User.FindFirstValue(ClaimTypes.NameIdentifier);
        return int.TryParse(claim, out var id) ? id : 0;
    }

    private int GetCurrentTenantId()
    {
        var claim = User.FindFirstValue("TenantId");
        return int.TryParse(claim, out var id) ? id : 0;
    }
}