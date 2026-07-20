using CyclonApp.Database;
using CyclonApp.Model.DTOs;

namespace CyclonApp.Repositories.Contracts
{
    public interface ICyclonePrediction
    {
        /// <summary>
        /// Starts an async field-solving job on the Python service
        /// (POST /predict_field/start) and returns the job id to poll.
        /// Throws FieldPredictionCapacityExceededException if the service
        /// is at MAX_CONCURRENT_FIELD_JOBS (HTTP 429) — callers should
        /// catch this specifically and ask the user to retry shortly.
        ///
        /// <paramref name="knownEfficiencyPercent"/>: the real Lapple-model
        /// collection efficiency (CyclonOutputDto.Efficiency from
        /// ICyclonCalculation.Calculate / revision.EfficiencyJson) for this
        /// revision's geometry and particle size, if the caller has it.
        /// Cached in-memory against the returned jobId so the Engineering
        /// Insight report can use the real efficiency figure instead of
        /// the field-solve's swirl-based placeholder estimate. Optional —
        /// pass null if the standard calculation hasn't been run yet.
        /// </summary>
        Task<string> StartFieldPredictionAsync(DesignRevision input, CyclonDimensions dimensions, double? knownEfficiencyPercent = null);

        /// <summary>
        /// Polls GET /predict_field/status/{jobId}. Returns null if the
        /// service responds 404 — the job never existed, or it finished
        /// and was TTL-swept out of the service's job store. This is a
        /// normal outcome for the caller to handle (e.g. "job expired,
        /// start a new one"), not an exceptional one.
        /// </summary>
        Task<FieldPredictionStatusDto?> GetFieldPredictionStatusAsync(string jobId);

        /// <summary>
        /// Looks up the real Lapple-model efficiency percentage cached for
        /// this jobId by StartFieldPredictionAsync, if any was supplied
        /// when the job was started. Null means "no known figure" — the
        /// caller should fall back to the field-solve's own estimate.
        /// </summary>
        double? GetKnownEfficiencyPercent(string jobId);
    }
}