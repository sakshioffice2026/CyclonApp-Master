using System.Threading.Tasks;
using CyclonApp.Database;
using CyclonApp.Model.DTOs;

namespace CyclonApp.Repositories.Contracts
{
    public interface ICyclonePrediction
    {
        /// <summary>
        /// <paramref name="standardCalculation"/>: the standard Lapple-model
        /// output for this revision (CyclonOutputDto), if the caller has one
        /// yet. Replaces the old knownEfficiencyPercent double — the insight
        /// engine now needs cut size and other fields off this object, not
        /// just Efficiency, so the whole thing is cached per job instead of
        /// just one number pulled out of it.
        /// </summary>
        Task<string> StartFieldPredictionAsync(DesignRevision input, CyclonDimensions dimensions, CyclonOutputDto? standardCalculation = null);

        Task<FieldPredictionStatusDto?> GetFieldPredictionStatusAsync(string jobId);

        /// <summary>Cyclone type code and standard-calculation output cached
        /// at job-start time, if the job is still within the cache's
        /// capacity/lifetime. Null if the job id is unknown or was trimmed.
        /// Replaces GetKnownEfficiencyPercent.</summary>
        CyclonePredictionJobContextDto? GetJobContext(string jobId);
    }
}