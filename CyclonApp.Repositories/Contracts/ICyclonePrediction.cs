using System.Threading.Tasks;
using CyclonApp.Database;
using CyclonApp.Model.DTOs;

namespace CyclonApp.Repositories.Contracts
{
    public interface ICyclonePrediction
    {
        Task<string> StartFieldPredictionAsync(DesignRevision input, CyclonDimensions dimensions, double? knownEfficiencyPercent = null, double? knownPressureDropPa = null);
        Task<FieldPredictionStatusDto?> GetFieldPredictionStatusAsync(string jobId);
        double? GetKnownEfficiencyPercent(string jobId);

        /// <summary>
        /// The deterministic Shepherd-Lapple pressure drop (Pa) computed for
        /// this design's actual geometry by CyclonCalculationRepository
        /// (CyclonOutputDto.PressureDropPa), if the caller supplied one when
        /// starting the job. Used by EngineeringInsightRepository to judge
        /// the field-solve's pressure drop against THIS design's own
        /// calculated baseline instead of one fixed Pa threshold shared by
        /// every cyclone type — see EvaluatePressureDrop's remarks.
        /// </summary>
        double? GetKnownPressureDropPa(string jobId);
    }
}