using CyclonApp.Database;
using CyclonApp.Model.DTOs;

namespace CyclonApp.Repositories.Contracts
{
    public interface ICyclonePrediction
    {
        Task<CyclonePredictionDto> PredictAsync(DesignRevision input, CyclonTypeRatios ratios);
    }
}