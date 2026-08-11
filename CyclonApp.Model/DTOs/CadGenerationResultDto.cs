namespace CyclonApp.Model.DTOs
{
    /// <summary>
    /// Result of a CAD generation call to the Python CyclonePredictionService's
    /// POST /generate_cad endpoint. All URLs are absolute (resolved against the
    /// service's base address) so views/controllers can use them directly as
    /// download links or <a>/<model-viewer> sources without knowing the
    /// Python service's address.
    /// </summary>
    public class CadGenerationResultDto
    {
        public string? StepUrl { get; set; }
        public string? DxfUrl { get; set; }
        public string? PdfUrl { get; set; }
        public string? ObjUrl { get; set; }
        public string? AllPartsDxfUrl { get; set; }
    }
}
