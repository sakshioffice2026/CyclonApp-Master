using System.Security.Claims;
using System.Text.Json;
using System.Text.RegularExpressions;
using CyclonApp.Agent;
using CyclonApp.Database;
using CyclonApp.Repositories.Contracts;
using EngineeringAI.Core.Agent;
using EngineeringAI.Core.State;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace CyclonApp.Controllers;

[Authorize(Roles = "SuperAdmin,ClientAdmin,Engineer")]
public sealed class AskAgentController : Controller
{
    private const int MaxMessageLength = 2000;
    private static readonly TimeSpan RequestTimeout = TimeSpan.FromSeconds(180);
    private static readonly Regex TabIdPattern = new("[^A-Za-z0-9_-]", RegexOptions.Compiled);

    private readonly IIntentRouter _router;
    private readonly AgentOrchestrator<CycloneDraftState> _orchestrator;
    private readonly DesignFlowStore<CycloneDraftState> _store;
    private readonly IDesignRepository _designRepository;
    private readonly ICyclonCalculation _calculation;
    private readonly ApplicationDbContext _db;
    private readonly IUnitOfWork _uow;
    private readonly ILogger<AskAgentController> _logger;

    public AskAgentController(
        IIntentRouter router,
        AgentOrchestrator<CycloneDraftState> orchestrator,
        DesignFlowStore<CycloneDraftState> store,
        IDesignRepository designRepository,
        ICyclonCalculation calculation,
        ApplicationDbContext db,
        IUnitOfWork uow,
        ILogger<AskAgentController> logger)
    {
        _router = router;
        _orchestrator = orchestrator;
        _store = store;
        _designRepository = designRepository;
        _calculation = calculation;
        _db = db;
        _uow = uow;
        _logger = logger;
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Message([FromBody] AgentMessageRequest request, CancellationToken cancellationToken)
    {
        if (request is null || string.IsNullOrWhiteSpace(request.Message))
        {
            return BadRequest(new { error = "Message is required." });
        }

        if (request.Message.Length > MaxMessageLength)
        {
            return BadRequest(new { error = $"Message must be {MaxMessageLength} characters or fewer." });
        }

        var sessionId = BuildSessionId(request.TabId);

        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(RequestTimeout);

        try
        {
            var decision = await _router.RouteAsync(sessionId, request.Message, timeout.Token);

            if (decision.Intent == AgentIntent.Knowledge)
            {
                return Ok(Plain(
                    decision,
                    sessionId,
                    "Knowledge Q&A is not available yet. For now I can help you size and design a cyclone — tell me your flow rate, particle size and density."));
            }

            if (decision.Intent == AgentIntent.Unclear)
            {
                return Ok(Plain(
                    decision,
                    sessionId,
                    "I can help you design a cyclone. Describe your requirement, for example: \"Size a Stairmand cyclone for 10,000 m³/h at 150 °C\"."));
            }

            var reply = await _orchestrator.HandleAsync(sessionId, request.Message, timeout.Token);

            return Ok(new AgentMessageResponse(
                Intent: decision.Intent.ToString(),
                Source: decision.Source,
                Message: reply.Message,
                DesignComplete: reply.DesignComplete,
                MissingFields: reply.MissingFields,
                Draft: Snapshot(sessionId),
                Result: ToElement(reply.CalculationJson),
                Checks: ToElement(reply.ChecksJson),
                CanSave: reply.DesignComplete && HasNoError(reply.CalculationJson)));
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            return StatusCode(StatusCodes.Status503ServiceUnavailable,
                new { error = "The assistant is busy. Please try again in a moment." });
        }
        catch (OperationCanceledException)
        {
            return StatusCode(StatusCodes.Status499ClientClosedRequest);
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException("AskAgentController", "Message", ex.ToString());
            return StatusCode(StatusCodes.Status500InternalServerError,
                new { error = "The assistant could not process your message." });
        }
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    public IActionResult Reset([FromBody] AgentResetRequest request)
    {
        _store.Remove(BuildSessionId(request?.TabId));
        return Ok(new { reset = true });
    }

    [HttpGet]
    public async Task<IActionResult> Projects(CancellationToken cancellationToken)
    {
        var tenantId = GetTenantId();
        if (tenantId == 0)
        {
            return Forbid();
        }

        var projects = await _db.Projects
            .AsNoTracking()
            .Where(p => p.TenantId == tenantId && p.Status != ProjectStatus.Archived)
            .OrderByDescending(p => p.CreatedAt)
            .Select(p => new { p.Id, p.ProjectNumber, p.Name })
            .ToListAsync(cancellationToken);

        return Ok(projects);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Save([FromBody] AgentSaveRequest request, CancellationToken cancellationToken)
    {
        if (request is null ||
            request.ProjectId <= 0 ||
            string.IsNullOrWhiteSpace(request.TagNumber) ||
            request.TagNumber.Length > 50)
        {
            return BadRequest(new { error = "Project and a tag number (max 50 characters) are required." });
        }

        var tenantId = GetTenantId();
        var userId = GetUserId();
        if (tenantId == 0 || userId == 0)
        {
            return Forbid();
        }

        var sessionId = BuildSessionId(request.TabId);

        try
        {
            var project = await _db.Projects
                .AsNoTracking()
                .FirstOrDefaultAsync(p => p.Id == request.ProjectId && p.TenantId == tenantId, cancellationToken);

            if (project is null)
            {
                return NotFound(new { error = "Project not found." });
            }

            if (!_store.TryGet(sessionId, out var draft) || draft is null)
            {
                return BadRequest(new { error = "There is no active design draft." });
            }

            if (draft.GetMissingMandatoryFields().Count > 0)
            {
                return BadRequest(new { error = "The draft is incomplete." });
            }

            var types = await _designRepository.GetActiveCycloneTypesAsync();
            var type = CycloneAgentDomain.MatchType(types, draft.CycloneTypeCode);
            if (type is null)
            {
                return BadRequest(new { error = "Cyclone type is not valid." });
            }

            var ratios = _calculation.ParseRatios(type.DimensionRatiosJson);
            if (ratios is null)
            {
                return BadRequest(new { error = "Cyclone type configuration error. Contact admin." });
            }

            var revision = CycloneAgentDomain.BuildRevision(draft, type);
            var result = _calculation.Calculate(revision, ratios);
            CycloneAgentDomain.ApplyResult(revision, result);

            var design = new CycloneDesign
            {
                ProjectId = project.Id,
                TenantId = tenantId,
                CycloneTypeId = type.Id,
                TagNumber = request.TagNumber.Trim(),
                Name = string.IsNullOrWhiteSpace(request.Name) ? null : request.Name.Trim(),
                Notes = "Created with Cyclone Assistant",
                CurrentRevision = 1,
                CreatedByUserId = userId,
                CreatedAt = DateTime.UtcNow
            };

            await _designRepository.CreateDesignAsync(design);

            revision.CycloneDesignId = design.Id;
            revision.RevisionNumber = 1;
            revision.RevisionNote = "Created with Cyclone Assistant";
            revision.CreatedByUserId = userId;

            await _designRepository.UpdateDesignRevisionAsync(design, revision);

            _store.Remove(sessionId);

            _logger.LogInformation(
                "Agent saved design {TagNumber}. DesignId={DesignId} RevisionId={RevisionId}",
                design.TagNumber,
                design.Id,
                revision.Id);

            return Ok(new
            {
                designId = design.Id,
                revisionId = revision.Id,
                resultsUrl = Url.Action("Results", "Design", new { id = revision.Id })
            });
        }
        catch (Exception ex)
        {
            _uow.exceptionHandlerRepository.SaveException("AskAgentController", "Save", ex.ToString());
            return StatusCode(StatusCodes.Status500InternalServerError,
                new { error = "The design could not be saved." });
        }
    }

    private AgentMessageResponse Plain(IntentDecision decision, string sessionId, string message)
    {
        return new AgentMessageResponse(
            Intent: decision.Intent.ToString(),
            Source: decision.Source,
            Message: message,
            DesignComplete: false,
            MissingFields: Array.Empty<string>(),
            Draft: Snapshot(sessionId),
            Result: null,
            Checks: null,
            CanSave: false);
    }

    private IReadOnlyList<DraftFieldView> Snapshot(string sessionId)
    {
        return _store.TryGet(sessionId, out var draft) && draft is not null
            ? draft.Snapshot()
            : new CycloneDraftState().Snapshot();
    }

    private string BuildSessionId(string? tabId)
    {
        var safeTab = TabIdPattern.Replace(tabId ?? "default", string.Empty);
        if (safeTab.Length == 0)
        {
            safeTab = "default";
        }

        if (safeTab.Length > 36)
        {
            safeTab = safeTab[..36];
        }

        return $"{GetTenantId()}:{GetUserId()}:{safeTab}";
    }

    private int GetTenantId() =>
        int.TryParse(User.FindFirstValue("TenantId"), out var id) ? id : 0;

    private int GetUserId() =>
        int.TryParse(User.FindFirstValue(ClaimTypes.NameIdentifier), out var id) ? id : 0;

    private static bool HasNoError(string? calculationJson)
    {
        var element = ToElement(calculationJson);
        return element is { ValueKind: JsonValueKind.Object } e && !e.TryGetProperty("error", out _);
    }

    private static JsonElement? ToElement(string? json)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            return null;
        }

        try
        {
            using var doc = JsonDocument.Parse(json);
            return doc.RootElement.Clone();
        }
        catch (JsonException)
        {
            return null;
        }
    }
}

public sealed record AgentMessageRequest(string? Message, string? TabId);

public sealed record AgentResetRequest(string? TabId);

public sealed record AgentSaveRequest(int ProjectId, string? TagNumber, string? Name, string? TabId);

public sealed record AgentMessageResponse(
    string Intent,
    string Source,
    string Message,
    bool DesignComplete,
    IReadOnlyList<string> MissingFields,
    IReadOnlyList<DraftFieldView> Draft,
    JsonElement? Result,
    JsonElement? Checks,
    bool CanSave);
