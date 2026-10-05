using EngineeringAI.Core.Security;
using EngineeringAI.Core.State;
using Microsoft.SemanticKernel.ChatCompletion;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace CyclonApp.Agent;

public sealed class IntentRouter : IIntentRouter
{
    private const double MinConfidence = 0.6;

    private const string ClassifierPrompt = """
        You classify one user message for an engineering assistant that designs cyclone separators.

        Labels:
        DESIGN: the user wants to size, design, specify, change, optimize or recalculate a cyclone or a dust collection / gas cleaning unit, or provides process, particle or equipment parameters.
        KNOWLEDGE: the user asks a question about existing designs, projects, results, history, or general cyclone knowledge.
        UNCLEAR: greeting, unrelated, or too ambiguous to tell.

        Respond with one JSON object only, no other text:
        {"intent":"DESIGN","confidence":0.0}
        The intent must be DESIGN, KNOWLEDGE or UNCLEAR. The confidence is a number from 0 to 1.
        """;

    private static readonly Regex UnitPattern = new(
        @"(?<![A-Za-z])\d[\d,]*(\.\d+)?\s*(m3/h(r)?|m³/h(r)?|cfm|°?\s?c|celsius|µm|μm|um|microns?|kpa|pa|kg/m3|kg/m³|mm|inch(es)?|"")(?![A-Za-z])",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);

    private static readonly Regex TypePattern = new(
        @"\b(lapple|stairmand|swift)\b",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);

    private readonly IChatCompletionService _chat;
    private readonly DesignFlowStore<CycloneDraftState> _store;
    private readonly ILogger<IntentRouter> _logger;

    public IntentRouter(
        IChatCompletionService chat,
        DesignFlowStore<CycloneDraftState> store,
        ILogger<IntentRouter> logger)
    {
        _chat = chat;
        _store = store;
        _logger = logger;
    }

    public async Task<IntentDecision> RouteAsync(
        string sessionId,
        string message,
        CancellationToken cancellationToken = default)
    {
        var decision = await DecideAsync(sessionId, message, cancellationToken).ConfigureAwait(false);

        _logger.LogInformation(
            "Intent routed. Session={SessionId} Intent={Intent} Source={Source} Confidence={Confidence:0.00}",
            sessionId,
            decision.Intent,
            decision.Source,
            decision.Confidence);

        return decision;
    }

    private async Task<IntentDecision> DecideAsync(
        string sessionId,
        string message,
        CancellationToken cancellationToken)
    {
        var clean = PromptSanitizer.Sanitize(message);

        if (string.IsNullOrWhiteSpace(clean))
        {
            return new IntentDecision(AgentIntent.Unclear, "empty", 0);
        }

        if (_store.TryGet(sessionId, out var draft) && draft is not null && draft.HasAnyValue)
        {
            return new IntentDecision(AgentIntent.Design, "state", 1.0);
        }

        if (UnitPattern.IsMatch(clean) || TypePattern.IsMatch(clean))
        {
            return new IntentDecision(AgentIntent.Design, "rule", 0.9);
        }

        try
        {
            var history = new ChatHistory(ClassifierPrompt);
            history.AddUserMessage(clean);

            var response = await _chat
                .GetChatMessageContentAsync(history, cancellationToken: cancellationToken)
                .ConfigureAwait(false);

            return ParseClassifier(response.Content);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Intent classifier failed for session {SessionId}", sessionId);
            return new IntentDecision(AgentIntent.Unclear, "error", 0);
        }
    }

    private static IntentDecision ParseClassifier(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return new IntentDecision(AgentIntent.Unclear, "llm-empty", 0);
        }

        var start = text.IndexOf('{');
        var end = text.LastIndexOf('}');

        if (start < 0 || end <= start)
        {
            return new IntentDecision(AgentIntent.Unclear, "llm-unparsed", 0);
        }

        try
        {
            using var doc = JsonDocument.Parse(text[start..(end + 1)]);
            var root = doc.RootElement;

            var label = root.TryGetProperty("intent", out var i) ? i.GetString()?.Trim().ToUpperInvariant() : null;
            var confidence = root.TryGetProperty("confidence", out var c) && c.TryGetDouble(out var d) ? d : 0.5;

            var intent = label switch
            {
                "DESIGN" => AgentIntent.Design,
                "KNOWLEDGE" => AgentIntent.Knowledge,
                _ => AgentIntent.Unclear
            };

            if (intent != AgentIntent.Unclear && confidence < MinConfidence)
            {
                return new IntentDecision(AgentIntent.Unclear, "llm-low", confidence);
            }

            return new IntentDecision(intent, "llm", confidence);
        }
        catch (JsonException)
        {
            return new IntentDecision(AgentIntent.Unclear, "llm-unparsed", 0);
        }
    }
}
