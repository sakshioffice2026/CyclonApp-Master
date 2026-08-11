using CyclonApp.Model.DTOs;

namespace CyclonApp.Model.ViewModel
{
    public class CadViewModel
    {
        public int RevisionId { get; set; }
        public int DesignId { get; set; }
        public string? TagNumber { get; set; }
        public string? CycloneType { get; set; }
        public int RevisionNumber { get; set; }

        // Populated only after a successful Generate call; null on first
        // page load ("Generate CAD" button not yet pressed).
        public CadGenerationResultDto? Result { get; set; }

        public string? ErrorMessage { get; set; }
    }
}
