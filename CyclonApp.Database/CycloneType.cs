using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CyclonApp.Database
{
    public class CycloneType
    {
        public int Id { get; set; }

        [Required, MaxLength(20)]
        public string Code { get; set; } = string.Empty;   // e.g. "STAIRMAND", "LAPPLE"

        [Required, MaxLength(100)]
        public string Name { get; set; } = string.Empty;   // Display name

        [MaxLength(500)]
        public string? Description { get; set; }

        // JSON blob: { "InletHeightRatio": 0.5, "InletWidthRatio": 0.25, ... }
        public string DimensionRatiosJson { get; set; } = "{}";

        // Recommended number of effective turns for this type
        public decimal DefaultEffectiveTurns { get; set; } = 6;

        // Typical application note
        [MaxLength(300)]
        public string? ApplicationNote { get; set; }

        public bool IsActive { get; set; } = true;

        public int SortOrder { get; set; } = 0;

        // Navigation
        public ICollection<CycloneDesign> Designs { get; set; } = new List<CycloneDesign>();
    }

}
