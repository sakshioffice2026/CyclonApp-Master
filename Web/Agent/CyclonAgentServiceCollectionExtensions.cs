using EngineeringAI.Core.Agent;
using EngineeringAI.Core.Llm;
using EngineeringAI.Core.State;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.SemanticKernel.ChatCompletion;

namespace CyclonApp.Agent;

public static class CyclonAgentServiceCollectionExtensions
{
    private const string LlamaSection = "EngineeringAI:Llama";

    public static IServiceCollection AddCyclonAgent(
        this IServiceCollection services,
        IConfiguration configuration)
    {
        services.AddOptions<LlamaOptions>()
            .Configure(options => configuration.GetSection(LlamaSection).Bind(options));

        services.TryAddSingleton<LlamaModelProvider>();
        services.TryAddSingleton<IChatCompletionService, LlamaChatCompletionService>();
        services.AddHostedService<LlamaWarmupService>();

        services.TryAddSingleton(new DesignFlowStore<CycloneDraftState>(TimeSpan.FromHours(2)));
        services.TryAddSingleton(sp => CycloneAgentDomain.Create(sp.GetRequiredService<IServiceScopeFactory>()));
        services.TryAddSingleton<AgentOrchestrator<CycloneDraftState>>();

        services.TryAddSingleton<IIntentRouter, IntentRouter>();

        return services;
    }
}
