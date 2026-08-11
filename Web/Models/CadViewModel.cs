using System.Collections.Generic;
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

        // File types the user checked before generating: any of
        // "step", "dxf", "pdf", "obj", "allparts" — these are the only 5
        // file types the Python /generate_cad endpoint returns (see
        // GenerateCadResponse in app.py). Purely a display/download filter —
        // the underlying FreeCAD run always produces all 5 in one go.
        public List<string> SelectedFileTypes { get; set; } = new();

        // Populated only after a successful Generate call; null on first
        // page load ("Generate CAD" button not yet pressed).
        public CadGenerationResultDto? Result { get; set; }

        public string? ErrorMessage { get; set; }
    }
}
