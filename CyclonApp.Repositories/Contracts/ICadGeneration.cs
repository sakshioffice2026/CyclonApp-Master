using System.Threading.Tasks;
using CyclonApp.Model.DTOs;

namespace CyclonApp.Repositories.Contracts
{
    /// <summary>
    /// Generates CAD deliverables (STEP solid, DXF drawings, OBJ mesh, PDF)
    /// for a design revision via the Python CyclonePredictionService's
    /// POST /generate_cad endpoint (which itself shells out to FreeCAD).
    /// Kept separate from ICyclonePrediction: CAD generation and field-solve
    /// prediction are different concerns hitting the same Python host.
    /// </summary>
    public interface ICadGeneration
    {
        Task<CadGenerationResultDto> GenerateCadAsync(int revisionId);
    }
}
