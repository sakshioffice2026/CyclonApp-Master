using System.Security.Claims;
using System.Text.Json;
using System.Text.Json.Serialization;
using CyclonApp.Database;
using CyclonApp.Model.DTOs;
using CyclonApp.Model.ViewModel;
using CyclonApp.Repositories.Contracts;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace CyclonApp.Controllers;

[Authorize]
public class DesignController : Controller
{
    private readonly IDesignRepository _designRepository;
    private readonly IAccount _accountRepository;
    private readonly ITenant _tenantRepository;
    private readonly ICyclonCalculation _calculationRepository;
    private readonly ICyclonePrediction _predictionRepository;
    private readonly ILogger<DesignController> _logger;
    private readonly IEngineeringInsight _engineeringInsight;
    public readonly IUnitOfWork _uow;


    private static readonly JsonSerializerOptions _jsonOpts =
        new() { PropertyNameCaseInsensitive = true };

    public DesignController(
    IDesignRepository designRepository,
    IAccount accountRepository,
    ITenant tenantRepository,
    ICyclonCalculation calculationRepository,
    ICyclonePrediction predictionRepository,
    IEngineeringInsight engineeringInsight,
    ILogger<DesignController> logger,
    IUnitOfWork uow)
    {
        _designRepository = designRepository;
        _accountRepository = accountRepository;
        _tenantRepository = tenantRepository;
        _calculationRepository = calculationRepository;
        _predictionRepository = predictionRepository;
        _engineeringInsight = engineeringInsight;
        _logger = logger;
        _uow = uow;
    }

    // ── INDEX ─────────────────────────────────────────────────────────────────

    [HttpGet]
    public async Task<IActionResult> Index()
    {
        try
        {
            await SetTenantNameAsync();
            SetBreadcrumb(("Designs", null));
            return View(await _designRepository.GetAllDesignsAsync());
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "DesignController",
                "Index",
                ex.ToString());

            return RedirectToAction("Index", "Dashboard");
        }
    }

    // ── CREATE ────────────────────────────────────────────────────────────────

    [HttpGet]
    [Authorize(Roles = "SuperAdmin,ClientAdmin,Engineer")]
    public async Task<IActionResult> Create(int projectId)
    {
        try
        {
            await SetTenantNameAsync();
            SetBreadcrumb(
                ("Projects", Url.Action("Index", "Project")),
                ("New Design", null));

            var types = await _designRepository.GetActiveCycloneTypesAsync();

            return View(new DesignCreateViewModel
            {
                ProjectId = projectId,
                CycloneTypes = types.Select(MapToTypeOption).ToList()
            });
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "DesignController",
                "Create_GET",
                ex.ToString());

            return RedirectToAction("Index", "Project");
        }
    }

    [HttpPost]
    [Authorize(Roles = "SuperAdmin,ClientAdmin,Engineer")]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Create(DesignCreateViewModel vm)
    {
        try
        {
            if (!ModelState.IsValid)
            {
                await RehydrateCreateVm(vm);
                await SetTenantNameAsync();
                return View(vm);
            }

            var currentUser = await GetCurrentUserAsync();
            if (currentUser == null) return RedirectToAction("Login", "Account");

            var cycloneType = await _designRepository.GetCycloneTypeByIdAsync(vm.CycloneTypeId);
            if (cycloneType == null)
            {
                ModelState.AddModelError(nameof(vm.CycloneTypeId), $"Invalid Cyclone Type selected.");
                await RehydrateCreateVm(vm);
                await SetTenantNameAsync();
                return View(vm);
            }

            var design = new CycloneDesign
            {
                ProjectId = vm.ProjectId,
                TenantId = _tenantRepository.CurrentTenantId,
                CycloneTypeId = cycloneType.Id,
                TagNumber = vm.TagNumber,
                Name = vm.Name,
                Notes = vm.Notes,
                CurrentRevision = 1,
                CreatedByUserId = currentUser.Id,
                CreatedAt = DateTime.UtcNow
            };

            await _designRepository.CreateDesignAsync(design);

            _logger.LogInformation("Design {TagNumber} created. DesignId={DesignId}", design.TagNumber, design.Id);
            TempData["Success"] = $"Design '{design.TagNumber}' created successfully.";
            return RedirectToAction("Calculate", new { id = design.Id });
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "DesignController",
                "Create_POST",
                ex.ToString());

            TempData["Error"] = "An error occurred while creating the design.";
            await RehydrateCreateVm(vm);
            await SetTenantNameAsync();
            return View(vm);
        }
    }

    // ── CALCULATE GET ─────────────────────────────────────────────────────────

    [HttpGet]
    [Authorize(Roles = "SuperAdmin,ClientAdmin,Engineer")]
    public async Task<IActionResult> Calculate(int id, int? fromRevision)
    {
        try
        {
            var design = await _designRepository.GetDesignWithDetailsAsync(id);
            if (design == null) return NotFound();

            await SetTenantNameAsync();
            SetBreadcrumb(
                ("Projects", Url.Action("Index", "Project")),
                (design.Project.Name, Url.Action("Detail", "Project", new { id = design.ProjectId })),
                (design.TagNumber ?? "Design", Url.Action("Detail", new { id })),
                ("Calculate", null));

            var latestRev = fromRevision.HasValue
                ? design.Revisions.FirstOrDefault(r => r.Id == fromRevision.Value)
                : design.Revisions.FirstOrDefault();

            var vm = new DesignCalculateViewModel
            {
                DesignId = design.Id,
                ProjectId = design.ProjectId,
                ProjectName = design.Project.Name,
                TagNumber = design.TagNumber,
                DesignName = design.Name,
                CycloneTypeId = design.CycloneTypeId,
                CycloneTypeName = design.CycloneType.Name,
            };

            if (latestRev != null)
            {
                vm.RevisionNumber = design.CurrentRevision + 1;
                vm.FlowRateCFM = latestRev.FlowRateCFM;
                vm.InletLineSizeIn = latestRev.InletLineSizeIn;
                vm.GasType = latestRev.GasType;
                vm.OperatingTempC = latestRev.OperatingTempC;
                vm.OperatingPressKPa = latestRev.OperatingPressKPa;
                vm.ParticleSizeMicron = latestRev.ParticleSizeMicron;
                vm.ParticleSizeD10 = latestRev.ParticleSizeD10;
                vm.ParticleSizeD50 = latestRev.ParticleSizeD50;
                vm.ParticleSizeD90 = latestRev.ParticleSizeD90;
                vm.ParticleDensityKgm3 = latestRev.ParticleDensityKgm3;
                vm.BulkDensityKgm3 = latestRev.BulkDensityKgm3;
                vm.ShapeFactor = latestRev.ShapeFactor;
                vm.EffectiveTurns = latestRev.EffectiveTurns;
                vm.GasViscosityKgms = latestRev.GasViscosityKgms;
                vm.ViscosityAutoCalc = latestRev.ViscosityAutoCalc;
                vm.NumberOfCyclones = latestRev.NumberOfCyclones;
                vm.SafetyFactor = latestRev.SafetyFactor;
                vm.InletShape = latestRev.InletShape;
            }
            else
            {
                vm.RevisionNumber = 1;
                vm.EffectiveTurns = design.CycloneType.DefaultEffectiveTurns;
            }

            return View(vm);
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "DesignController",
                "Calculate_GET",
                ex.ToString());

            return RedirectToAction("Index");
        }
    }

    // ── CALCULATE POST ────────────────────────────────────────────────────────

    [HttpPost]
    [Authorize(Roles = "SuperAdmin,ClientAdmin,Engineer")]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Calculate(DesignCalculateViewModel vm)
    {
        try
        {
            if (!ModelState.IsValid)
            {
                await SetTenantNameAsync();
                return View(vm);
            }

            var cycloneType = await _designRepository.GetCycloneTypeByIdAsync(vm.CycloneTypeId);
            if (cycloneType == null)
            {
                ModelState.AddModelError("", "Invalid cyclone type.");
                await SetTenantNameAsync();
                return View(vm);
            }

            var ratios = _calculationRepository.ParseRatios(cycloneType.DimensionRatiosJson);
            if (ratios == null)
            {
                ModelState.AddModelError("", "Cyclone type configuration error. Contact admin.");
                await SetTenantNameAsync();
                return View(vm);
            }

            var currentUser = await GetCurrentUserAsync();
            if (currentUser == null) return RedirectToAction("Login", "Account");

            var revision = new DesignRevision
            {
                CycloneDesignId = vm.DesignId,
                RevisionNumber = vm.RevisionNumber,
                RevisionNote = vm.RevisionNote,
                FlowRateCFM = vm.FlowRateCFM,
                InletLineSizeIn = vm.InletLineSizeIn,
                GasType = vm.GasType,
                OperatingTempC = vm.OperatingTempC,
                OperatingPressKPa = vm.OperatingPressKPa,
                ParticleSizeMicron = vm.ParticleSizeMicron,
                ParticleSizeD10 = vm.ParticleSizeD10,
                ParticleSizeD50 = vm.ParticleSizeD50,
                ParticleSizeD90 = vm.ParticleSizeD90,
                ParticleDensityKgm3 = vm.ParticleDensityKgm3,
                BulkDensityKgm3 = vm.BulkDensityKgm3,
                ShapeFactor = vm.ShapeFactor,
                EffectiveTurns = vm.EffectiveTurns,
                GasViscosityKgms = vm.GasViscosityKgms,
                ViscosityAutoCalc = vm.ViscosityAutoCalc,
                NumberOfCyclones = vm.NumberOfCyclones,
                SafetyFactor = vm.SafetyFactor,
                InletShape = vm.InletShape,
                CreatedByUserId = currentUser.Id,
                CreatedAt = DateTime.UtcNow,
            };

            var result = _calculationRepository.Calculate(revision, ratios);

            revision.FlowRateM3hr = (decimal)result.FlowRateM3hr;
            revision.AvgVelocityMs = (decimal)result.InletVelocityMs;
            if (revision.ViscosityAutoCalc)
                revision.GasViscosityKgms = (decimal)result.GasViscosityKgms;

            var jsonOptions = new JsonSerializerOptions
            {
                NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
            };

            revision.DimensionsJson = JsonSerializer.Serialize(result.Dimensions, jsonOptions);
            revision.EfficiencyJson = JsonSerializer.Serialize(result, jsonOptions);
            revision.CalculatedAt = DateTime.UtcNow;

            var design = await _designRepository.GetDesignByIdAsync(vm.DesignId);
            if (design == null) return NotFound();

            await _designRepository.UpdateDesignRevisionAsync(design, revision);

            _logger.LogInformation(
                "Design {DesignId} Rev {Rev} calculated. Eff={Eff:F1}%, ΔP={DP:F1} Pa.",
                vm.DesignId, vm.RevisionNumber, result.Efficiency, result.PressureDropPa);

            TempData["Success"] = $"Calculation complete — Revision {vm.RevisionNumber} saved.";
            return RedirectToAction("Results", new { id = revision.Id });
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "DesignController",
                "Calculate_POST",
                ex.ToString());

            TempData["Error"] = "An error occurred while performing the calculation.";
            await SetTenantNameAsync();
            return View(vm);
        }
    }

    // ── RESULTS ───────────────────────────────────────────────────────────────

    [HttpGet]
    public async Task<IActionResult> Results(int id)
    {
        try
        {
            var revision = await _designRepository.GetRevisionWithDetailsAsync(id);
            if (revision == null) return NotFound();

            await SetTenantNameAsync();
            SetBreadcrumb(
                ("Projects", Url.Action("Index", "Project")),
                (revision.CycloneDesign.Project.Name,
                 Url.Action("Detail", "Project", new { id = revision.CycloneDesign.ProjectId })),
                (revision.CycloneDesign.TagNumber ?? "Design",
                 Url.Action("Detail", new { id = revision.CycloneDesignId })),
                ($"Rev {revision.RevisionNumber} Results", null));

            return View(BuildResultsVm(revision));
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "DesignController",
                "Results",
                ex.ToString());

            return RedirectToAction("Index");
        }
    }

    // ── DETAIL ────────────────────────────────────────────────────────────────
    //
    // NOTE: PredictWithModel (POST, called PredictAsync -> /predict) has been
    // removed. The Python service retired the scalar CyclonePINN correction
    // model that endpoint depended on — see app.py's module docstring — so
    // this action would only ever surface "prediction service unavailable".
    // Field-solving (StartFieldPrediction / FieldPredictionStatus below) is
    // the only live prediction flow now. Previously-saved predictions
    // (revision.PredictionJson) are still read and displayed in Results —
    // see BuildResultsVm — since that's just showing historical data.

    [HttpGet]
    public async Task<IActionResult> Detail(int id)
    {
        try
        {
            var design = await _designRepository.GetDesignWithDetailsAsync(id);
            if (design == null) return NotFound();

            await SetTenantNameAsync();
            SetBreadcrumb(
                ("Projects", Url.Action("Index", "Project")),
                (design.Project.Name, Url.Action("Detail", "Project", new { id = design.ProjectId })),
                (design.TagNumber ?? "Design", null));

            return View(MapToRevisionList(design));
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "DesignController",
                "Detail",
                ex.ToString());

            return RedirectToAction("Index");
        }
    }

    // ── REVISIONS ─────────────────────────────────────────────────────────────

    [HttpGet]
    public async Task<IActionResult> Revisions(int id)
    {
        try
        {
            var design = await _designRepository.GetDesignWithDetailsAsync(id);
            if (design == null) return NotFound();

            await SetTenantNameAsync();
            SetBreadcrumb(
                ("Projects", Url.Action("Index", "Project")),
                (design.Project.Name, Url.Action("Detail", "Project", new { id = design.ProjectId })),
                (design.TagNumber ?? "Design", Url.Action("Detail", new { id })),
                ("Revisions", null));

            return View(MapToRevisionList(design));
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "DesignController",
                "Revisions",
                ex.ToString());

            return RedirectToAction("Index");
        }
    }

    // ── COMPARE ───────────────────────────────────────────────────────────────

    [HttpGet]
    public async Task<IActionResult> Compare(int designId, int? revAId, int? revBId)
    {
        try
        {
            var design = await _designRepository.GetDesignWithDetailsAsync(designId);
            if (design == null) return NotFound();

            await SetTenantNameAsync();
            SetBreadcrumb(
                ("Projects", Url.Action("Index", "Project")),
                (design.Project.Name, Url.Action("Detail", "Project", new { id = design.ProjectId })),
                (design.TagNumber ?? "Design", Url.Action("Detail", new { id = designId })),
                ("Compare", null));

            DesignResultsViewModel? revA = null, revB = null;

            if (revAId.HasValue)
            {
                var r = await _designRepository.GetRevisionWithDetailsAsync(revAId.Value);
                if (r != null) revA = BuildResultsVm(r);
            }
            if (revBId.HasValue)
            {
                var r = await _designRepository.GetRevisionWithDetailsAsync(revBId.Value);
                if (r != null) revB = BuildResultsVm(r);
            }

            return View(new CompareViewModel
            {
                DesignId = designId,
                TagNumber = design.TagNumber,
                RevA = revA,
                RevB = revB,
                AllRevisions = MapToRevisionList(design).Revisions,
                RevAId = revAId ?? 0,
                RevBId = revBId ?? 0,
            });
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "DesignController",
                "Compare",
                ex.ToString());

            return RedirectToAction("Index");
        }
    }

    // ── HELPERS ───────────────────────────────────────────────────────────────

    private async Task<AppUser?> GetCurrentUserAsync()
    {
        var userIdClaim = User.FindFirstValue(ClaimTypes.NameIdentifier);
        if (!int.TryParse(userIdClaim, out var userId)) return null;
        return await _accountRepository.GetUserByIdAsync(userId);
    }

    private async Task SetTenantNameAsync()
    {
        try
        {
            var user = await GetCurrentUserAsync();
            ViewBag.TenantName = user?.Tenant?.Name ?? "—";
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "DesignController",
                "SetTenantNameAsync",
                ex.ToString());

            throw;
        }
    }

    private void SetBreadcrumb(params (string Label, string? Url)[] crumbs)
        => ViewBag.Breadcrumbs = crumbs.Select(c => (c.Label, c.Url)).ToList();

    private async Task RehydrateCreateVm(DesignCreateViewModel vm)
    {
        var types = await _designRepository.GetActiveCycloneTypesAsync();
        vm.CycloneTypes = types.Select(MapToTypeOption).ToList();
    }

    private static CycloneTypeOptionViewModel MapToTypeOption(CycloneType t) => new()
    {
        Id = t.Id,
        Code = t.Code,
        Name = t.Name,
        Description = t.Description,
        ApplicationNote = t.ApplicationNote,
        DefaultEffectiveTurns = (double)t.DefaultEffectiveTurns,
    };

    private RevisionListViewModel MapToRevisionList(CycloneDesign design)
    {
        var latestRevNum = design.Revisions.Any()
            ? design.Revisions.Max(r => r.RevisionNumber) : 0;

        return new RevisionListViewModel
        {
            DesignId = design.Id,
            ProjectId = design.ProjectId,
            ProjectName = design.Project.Name,
            TagNumber = design.TagNumber,
            DesignName = design.Name,
            CycloneType = design.CycloneType.Name,
            Revisions = design.Revisions.Select(r =>
            {
                CyclonOutputDto? out_ = null;
                if (!string.IsNullOrEmpty(r.EfficiencyJson))
                    try { out_ = JsonSerializer.Deserialize<CyclonOutputDto>(r.EfficiencyJson, _jsonOpts); } catch { }

                return new RevisionRowViewModel
                {
                    Id = r.Id,
                    RevisionNumber = r.RevisionNumber,
                    RevisionNote = r.RevisionNote,
                    FlowRateCFM = r.FlowRateCFM,
                    ParticleSizeMicron = r.ParticleSizeMicron,
                    ParticleDensityKgm3 = r.ParticleDensityKgm3,
                    HasResults = r.CalculatedAt != null,
                    Efficiency = out_?.Efficiency,
                    CutDiameter = out_?.CutDiameterMicron,
                    PressureDropPa = out_?.PressureDropPa,
                    CreatedBy = "—",
                    CreatedAt = r.CreatedAt,
                    IsLatest = r.RevisionNumber == latestRevNum,
                };
            }).ToList()
        };
    }

    public IActionResult TestException()
    {
        try
        {
            throw new Exception("Test from DesignController");
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "DesignController",
                "TestException",
                ex.ToString());

            return Content("Exception Logged");
        }
    }

    // ── FIELD PREDICTION (physics-guided field solve, async job) ────────────
    // JSON endpoints, not redirects: a field solve takes real minutes, so the
    // client starts a job and polls status via AJAX rather than the
    // request/redirect flow the old PredictWithModel action used.

    [HttpPost]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> StartFieldPrediction(int id)
    {
        try
        {
            var revision = await _designRepository.GetRevisionWithDetailsAsync(id);
            if (revision == null) return NotFound();

            var output = string.IsNullOrEmpty(revision.EfficiencyJson)
                ? null
                : JsonSerializer.Deserialize<CyclonOutputDto>(revision.EfficiencyJson, _jsonOpts);
            var dims = output?.Dimensions;
            if (dims == null)
            {
                return BadRequest(new { error = "Run the standard calculation for this revision first — no geometry available yet." });
            }

            var jobId = await _predictionRepository.StartFieldPredictionAsync(revision, dims);

            _logger.LogInformation("Field prediction job {JobId} started for Revision {RevId}.", jobId, id);

            return Ok(new { jobId, status = "running" });
        }
        catch (FieldPredictionCapacityExceededException ex)
        {
            return StatusCode(StatusCodes.Status429TooManyRequests, new { error = ex.Message });
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "DesignController",
                "StartFieldPrediction",
                ex.ToString());

            return StatusCode(StatusCodes.Status503ServiceUnavailable,
                new { error = "The field prediction service is unavailable. Please try again shortly." });
        }
    }

    [HttpGet]
    public async Task<IActionResult> FieldPredictionStatus(string jobId)
    {
        try
        {
            var status = await _predictionRepository.GetFieldPredictionStatusAsync(jobId);
            if (status == null)
            {
                return NotFound(new { error = "Job not found — it may have expired. Start a new field prediction." });
            }

            return Ok(status);
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException(
                "DesignController",
                "FieldPredictionStatus",
                ex.ToString());

            return StatusCode(StatusCodes.Status503ServiceUnavailable,
                new { error = "The field prediction service is unavailable. Please try again shortly." });
        }
    }

    private DesignResultsViewModel BuildResultsVm(DesignRevision revision)
    {
        var output = string.IsNullOrEmpty(revision.EfficiencyJson)
            ? null
            : JsonSerializer.Deserialize<CyclonOutputDto>(revision.EfficiencyJson, _jsonOpts);

        var dims = output?.Dimensions ?? new CyclonDimensions();

        var prediction = string.IsNullOrEmpty(revision.PredictionJson)
            ? null
            : JsonSerializer.Deserialize<CyclonePredictionDto>(revision.PredictionJson, _jsonOpts);

        return new DesignResultsViewModel
        {
            RevisionId = revision.Id,
            DesignId = revision.CycloneDesignId,
            ProjectId = revision.CycloneDesign.ProjectId,
            ProjectName = revision.CycloneDesign.Project.Name,
            TagNumber = revision.CycloneDesign.TagNumber,
            DesignName = revision.CycloneDesign.Name,
            CycloneType = revision.CycloneDesign.CycloneType.Name,
            CycloneCode = revision.CycloneDesign.CycloneType.Code,
            RevisionNumber = revision.RevisionNumber,
            RevisionNote = revision.RevisionNote,
            CalculatedAt = revision.CalculatedAt ?? revision.CreatedAt,
            FlowRateCFM = revision.FlowRateCFM,
            InletLineSizeIn = revision.InletLineSizeIn,
            GasType = revision.GasType,
            OperatingTempC = revision.OperatingTempC,
            OperatingPressKPa = revision.OperatingPressKPa,
            ParticleSizeMicron = revision.ParticleSizeMicron,
            ParticleDensityKgm3 = revision.ParticleDensityKgm3,
            EffectiveTurns = revision.EffectiveTurns,
            NumberOfCyclones = revision.NumberOfCyclones,
            FlowRateM3hr = output?.FlowRateM3hr ?? 0,
            InletVelocityMs = output?.InletVelocityMs ?? 0,
            GasViscosityKgms = output?.GasViscosityKgms ?? 0,
            GasDensityKgm3 = output?.GasDensityKgm3 ?? 0,
            CutDiameterMicron = output?.CutDiameterMicron ?? 0,
            Efficiency = output?.Efficiency ?? 0,
            PressureDropPa = output?.PressureDropPa ?? 0,
            PressureDropMmWc = output?.PressureDropMmWc ?? 0,
            PressureDropInWc = output?.PressureDropInWc ?? 0,
            BarrelDiameterIn = dims.BarrelDiameterIn,
            BarrelDiameterMm = dims.BarrelDiameterMm,
            BarrelDiameterM = dims.BarrelDiameterM,
            InletHeightIn = dims.InletHeightIn,
            InletHeightMm = dims.InletHeightMm,
            InletWidthIn = dims.InletWidthIn,
            InletWidthMm = dims.InletWidthMm,
            BarrelHeightIn = dims.BarrelHeightIn,
            BarrelHeightMm = dims.BarrelHeightMm,
            ConeHeightIn = dims.ConeHeightIn,
            ConeHeightMm = dims.ConeHeightMm,
            ExhaustDiaIn = dims.ExhaustDiaIn,
            ExhaustDiaMm = dims.ExhaustDiaMm,
            ExhaustLengthIn = dims.ExhaustLengthIn,
            ExhaustLengthMm = dims.ExhaustLengthMm,
            BottomOutletIn = dims.BottomOutletIn,
            BottomOutletMm = dims.BottomOutletMm,
            TotalHeightIn = dims.TotalHeightIn,
            TotalHeightMm = dims.TotalHeightMm,
            GradeEfficiencyCurveJson = output != null
                ? JsonSerializer.Serialize(output.GradeEfficiencyCurve) : "[]",
            DimensionsJson = revision.DimensionsJson ?? "{}",
            HasPrediction = prediction != null,
            PredictionEfficiency = prediction?.Efficiency ?? 0,
            PredictionPressureDropPa = prediction?.PressureDropPa ?? 0,
            PredictionPhysicsResidual = prediction?.PhysicsResidual ?? 0,
            PredictionIsWithinTrustedRange = prediction?.IsWithinTrustedRange ?? false,
            PredictionNotes = prediction?.Notes,
            PredictedAt = revision.PredictedAt,
        };
    }
}