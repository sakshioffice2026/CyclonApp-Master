namespace CyclonApp.Agent;

public enum AgentIntent
{
    Design,
    Knowledge,
    Unclear
}

public sealed record IntentDecision(AgentIntent Intent, string Source, double Confidence);

public interface IIntentRouter
{
    Task<IntentDecision> RouteAsync(
        string sessionId,
        string message,
        CancellationToken cancellationToken = default);
}
