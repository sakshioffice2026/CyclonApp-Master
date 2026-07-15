using System.Security.Claims;
using CyclonApp.Database;
using CyclonApp.Model.ViewModel;
using CyclonApp.Repositories.Contracts;
using CyclonApp.Repositories.Repositories;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace CyclonApp.Controllers;

[Authorize]
public class ExportController : Controller
{
    private readonly IExport _exportRepository;
    private readonly IDesignRepository _designRepository;
    private readonly IAccount _accountRepository;
    private readonly ILogger<ExportController> _logger;
    public readonly IUnitOfWork _uow;

    public ExportController(
        IExport exportRepository,
        IDesignRepository designRepository,
        IAccount accountRepository,
        ILogger<ExportController> logger,
        IUnitOfWork uow)
    {
        _exportRepository = exportRepository;
        _designRepository = designRepository;
        _accountRepository = accountRepository;
        _logger = logger;
        _uow = uow;
    }

    // ── INDEX ─────────────────────────────────────────────────────────────────

    [HttpGet]
    public async Task<IActionResult> Index(int? designId)
    {
        try
        {
            await SetTenantNameAsync();
            SetBreadcrumb(("Projects", Url.Action("Index", "Project")), ("Export", null));

            // No designId — show landing page with recent export logs
            if (designId == null)
            {
                var logs = await _exportRepository.GetExportLogsAsync(GetCurrentTenantId());
                return View(new DesignExportViewModel { ExportLogs = logs });
            }

            // designId provided — show revision picker for this design
            var design = await _designRepository.GetDesignWithDetailsAsync(designId.Value);
            if (design == null) return NotFound();

            var vm = new DesignExportViewModel
            {
                Id = design.Id,
                TagNumber = design.TagNumber,
                CycloneType = design.CycloneType?.Name,
                CurrentRevision = design.CurrentRevision,
                Revisions = design.Revisions
                    .OrderByDescending(r => r.RevisionNumber)
                    .Select(r => new RevisionRowViewModel
                    {
                        Id = r.Id,
                        RevisionNumber = r.RevisionNumber,
                        RevisionNote = r.RevisionNote,
                        FlowRateCFM = r.FlowRateCFM,
                        ParticleSizeMicron = r.ParticleSizeMicron,
                        ParticleDensityKgm3 = r.ParticleDensityKgm3,
                        HasResults = r.CalculatedAt != null,
                        CreatedAt = r.CreatedAt,
                        IsLatest = r.RevisionNumber == design.CurrentRevision
                    }).ToList(),
                ExportLogs = await _exportRepository.GetExportLogsByDesignAsync(designId.Value)
            };

            return View(vm);
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "ExportController",
                "Index",
                ex.ToString());

            TempData["Error"] = "An error occurred while loading the export page.";
            return RedirectToAction("Index", "Project");
        }
    }

    // ── PDF DOWNLOAD ──────────────────────────────────────────────────────────

    [HttpGet]
    public async Task<IActionResult> Pdf(int revisionId)
    {
        try
        {
            var (rev, output) = await _exportRepository.GetRevisionForExportAsync(revisionId);
            var bytes = await _exportRepository.GeneratePdfAsync(rev, output);

            await _exportRepository.LogExportAsync(
                revisionId, GetCurrentTenantId(), GetCurrentUserId(), ExportType.PDF);

            // Returns HTML in browser — user can File > Print > Save as PDF
            // Replace with DinkToPdf File() call once libwkhtmltox is installed on server
            return Content(
                System.Text.Encoding.UTF8.GetString(bytes),
                "text/html",
                System.Text.Encoding.UTF8);
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "ExportController",
                "Pdf",
                ex.ToString());

            TempData["Error"] = "An error occurred while generating the PDF.";
            return RedirectToAction("Index");
        }
    }

    // ── EXCEL DOWNLOAD ────────────────────────────────────────────────────────

    [HttpGet]
    public async Task<IActionResult> Excel(int revisionId)
    {
        try
        {
            var (rev, output) = await _exportRepository.GetRevisionForExportAsync(revisionId);
            var bytes = await _exportRepository.GenerateExcelAsync(rev, output);

            await _exportRepository.LogExportAsync(
                revisionId, GetCurrentTenantId(), GetCurrentUserId(), ExportType.Excel);

            var fileName = $"CycloneDesign_{rev.CycloneDesign.TagNumber ?? "Report"}" +
                           $"_Rev{rev.RevisionNumber}_{DateTime.UtcNow:yyyyMMdd}.xlsx";

            return File(bytes,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                fileName);
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "ExportController",
                "Excel",
                ex.ToString());

            TempData["Error"] = "An error occurred while generating the Excel file.";
            return RedirectToAction("Index");
        }
    }

    // ── HISTORY ───────────────────────────────────────────────────────────────

    [HttpGet]
    public async Task<IActionResult> History()
    {
        try
        {
            await SetTenantNameAsync();
            SetBreadcrumb(
                ("Export / Reports", Url.Action("Index")),
                ("History", null));

            var logs = await _exportRepository.GetExportLogsAsync(GetCurrentTenantId(), take: 200);
            return View(new DesignExportViewModel { ExportLogs = logs });
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "ExportController",
                "History",
                ex.ToString());

            TempData["Error"] = "An error occurred while loading the export history.";
            return RedirectToAction("Index");
        }
    }

    // ── HELPERS ───────────────────────────────────────────────────────────────

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

    private async Task SetTenantNameAsync()
    {
        try
        {
            var user = await _accountRepository.GetUserByIdAsync(GetCurrentUserId());
            ViewBag.TenantName = user?.Tenant?.Name ?? "—";
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "ExportController",
                "SetTenantNameAsync",
                ex.ToString());

            throw;
        }
    }

    private void SetBreadcrumb(params (string Label, string? Url)[] crumbs)
        => ViewBag.Breadcrumbs = crumbs.Select(c => (c.Label, c.Url)).ToList();
}
