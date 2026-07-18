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
        /// </summary>
        Task<string> StartFieldPredictionAsync(DesignRevision input, CyclonDimensions dimensions);

        /// <summary>
        /// Polls GET /predict_field/status/{jobId}. Returns null if the
        /// service responds 404 — the job never existed, or it finished
        /// and was TTL-swept out of the service's job store. This is a
        /// normal outcome for the caller to handle (e.g. "job expired,
        /// start a new one"), not an exceptional one.
        /// </summary>
        Task<FieldPredictionStatusDto?> GetFieldPredictionStatusAsync(string jobId);
    }
}