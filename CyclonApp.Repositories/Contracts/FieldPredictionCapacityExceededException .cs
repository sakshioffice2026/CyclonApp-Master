using System;

namespace CyclonApp.Repositories.Contracts
{
    /// <summary>
    /// Thrown by ICyclonePrediction.StartFieldPredictionAsync when the
    /// Python field-solving service returns 429 — MAX_CONCURRENT_FIELD_JOBS
    /// is already running. This is an expected, recoverable condition (the
    /// caller should ask the user to retry shortly), not a service failure,
    /// so it gets its own exception type rather than being treated the same
    /// as a transport error or a 5xx.
    /// </summary>
    public class FieldPredictionCapacityExceededException : Exception
    {
        public FieldPredictionCapacityExceededException(string message) : base(message)
        {
        }
    }
}