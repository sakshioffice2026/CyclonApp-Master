
using System.ComponentModel.DataAnnotations;
using CyclonApp.Database;
namespace CyclonApp.Model.ViewModel
{
    
    // ── CREATE / EDIT PROJECT ─────────────────────────────────────────────────────

    public class ProjectFormViewModel
    {
        public int Id { get; set; }

        [Required, MaxLength(50), Display(Name = "Project Number")]
        public string? ProjectNumber { get; set; }

        [Required, MaxLength(200), Display(Name = "Project Name")]
        public string Name { get; set; } = string.Empty;

        [MaxLength(200), Display(Name = "Client / Company Name")]
        public string? ClientName { get; set; }

        [MaxLength(100), Display(Name = "Site / Location")]
        public string? Location { get; set; }

        [MaxLength(1000), Display(Name = "Description")]
        public string? Description { get; set; }

        public ProjectStatus Status { get; set; } = ProjectStatus.Draft;

        public bool IsEdit => Id > 0;
    }
    //Index
    public class ProjectIndexViewModel
    {
        public List<ProjectListItemViewModel> Projects { get; set; } = new();
        public int TotalProjects { get; set; }
        public int ActiveProjects { get; set; }
        public int DraftProjects { get; set; }
        public int CompletedProjects { get; set; }
        public int TotalDesigns { get; set; }
        public string? FilterStatus { get; set; }
        public string? SearchTerm { get; set; }
    }

    // ── LIST ROW ──────────────────────────────────────────────────────────────────

    public class ProjectListItemViewModel
    {
        public int Id { get; set; }
        public string ProjectNumber { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string? ClientName { get; set; }
        public string? Location { get; set; }
        public ProjectStatus Status { get; set; }
        public int DesignCount { get; set; }
        public string CreatedBy { get; set; } = string.Empty;
        public DateTime CreatedAt { get; set; }
        public DateTime? UpdatedAt { get; set; }
    }

 

    // ── DETAIL ────────────────────────────────────────────────────────────────────

    public class ProjectDetailViewModel
    {
        public int Id { get; set; }
        public string ProjectNumber { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string? ClientName { get; set; }
        public string? Location { get; set; }
        public string? Description { get; set; }
        public ProjectStatus Status { get; set; }
        public string CreatedBy { get; set; } = string.Empty;
        public DateTime CreatedAt { get; set; }
        public DateTime? UpdatedAt { get; set; }
        public List<DesignSummaryViewModel> Designs { get; set; } = new();
    }

    // ── DESIGN SUMMARY (used inside Project Detail) ───────────────────────────────

    public class DesignSummaryViewModel
    {
        public int Id { get; set; }

        // Display tag number or fallback like "DES-001"
        public string? TagNumber { get; set; }

        // Human-readable name of the design
        public string? Name { get; set; }

        // Cyclone type name (from CycloneType entity)
        public string CycloneType { get; set; } = "—";

        // Current revision number
        public int CurrentRevision { get; set; }

        // When the design was created
        public DateTime CreatedAt { get; set; }

        // Whether any revision has results
        public bool HasResults { get; set; }
    }


}
