using System.Threading.Tasks;
using CyclonApp.Database;
using CyclonApp.Model.DTOs;

namespace CyclonApp.Repositories.Contracts
{
    public interface ICyclonePrediction
    {
        Task<string> StartFieldPredictionAsync(DesignRevision input, CyclonDimensions dimensions, double? knownEfficiencyPercent = null);
        Task<FieldPredictionStatusDto?> GetFieldPredictionStatusAsync(string jobId);
        double? GetKnownEfficiencyPercent(string jobId);
    }
}
