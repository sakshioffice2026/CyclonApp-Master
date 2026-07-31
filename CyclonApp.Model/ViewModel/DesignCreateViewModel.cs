using System.ComponentModel.DataAnnotations;
using CyclonApp.Database;


namespace CyclonApp.Model.ViewModel;

// ── CREATE DESIGN ─────────────────────────────────────────────────────────────

public class DesignCreateViewModel
{
    public int ProjectId { get; set; }
    public string? ProjectName { get; set; }

    [Required, MaxLength(50), Display(Name = "Tag Number")]
    public string? TagNumber { get; set; }

    [MaxLength(200), Display(Name = "Design Name / Description")]
    public string? Name { get; set; }

    [Required, Display(Name = "Cyclone Type")]
    public int CycloneTypeId { get; set; }

    [MaxLength(1000), Display(Name = "Notes")]
    public string? Notes { get; set; }

    // Physics-guided prediction (optional — populated only if generated)
    public bool HasPrediction { get; set; }
    public double PredictionEfficiency { get; set; }
    public double PredictionPressureDropPa { get; set; }
    public double PredictionPhysicsResidual { get; set; }
    public bool PredictionIsWithinTrustedRange { get; set; }
    public string? PredictionNotes { get; set; }
    public DateTime? PredictedAt { get; set; }

    // Available types for dropdown
    public List<CycloneTypeOptionViewModel> CycloneTypes { get; set; } = new();
}

public class CycloneTypeOptionViewModel
{
    public int Id { get; set; }
    public string Code { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string? Description { get; set; }
    public string? ApplicationNote { get; set; }
    public double DefaultEffectiveTurns { get; set; }
}

// ── CALCULATE (Input Form) ────────────────────────────────────────────────────

public class DesignCalculateViewModel
{
    public int DesignId { get; set; }
    public int ProjectId { get; set; }
    public string? ProjectName { get; set; }
    public string? TagNumber { get; set; }
    public string? DesignName { get; set; }
    public int CycloneTypeId { get; set; }
    public string? CycloneTypeName { get; set; }
    public int RevisionNumber { get; set; } = 1;

    [MaxLength(300), Display(Name = "Revision Note")]
    public string? RevisionNote { get; set; }

    // ── SECTION A: Process Parameters ──────────────────────────────────────

    [Required, Range(0.01, 1000000), Display(Name = "Gas Flow Rate (CFM)")]
    public decimal FlowRateCFM { get; set; } = 800;

    [Required, Range(0.5, 100), Display(Name = "Inlet Line Size (inches Ø)")]
    public decimal InletLineSizeIn { get; set; } = 5;

    [Display(Name = "Gas Type")]
    public string GasType { get; set; } = "Air";

    [Range(-50, 1000), Display(Name = "Operating Temperature (°C)")]
    public decimal OperatingTempC { get; set; } = 25;

    [Range(1, 10000), Display(Name = "Operating Pressure (kPa)")]
    public decimal OperatingPressKPa { get; set; } = 101.325m;

    // ── SECTION B: Particle / Solids ───────────────────────────────────────

    [Required, Range(0.01, 100000), Display(Name = "Average Particle Size (µm)")]
    public decimal ParticleSizeMicron { get; set; } = 600;

    [Range(0.01, 100000), Display(Name = "D10 (µm)")]
    public decimal ParticleSizeD10 { get; set; } = 100;

    [Range(0.01, 100000), Display(Name = "D50 (µm)")]
    public decimal ParticleSizeD50 { get; set; } = 300;

    [Range(0.01, 100000), Display(Name = "D90 (µm)")]
    public decimal ParticleSizeD90 { get; set; } = 800;

    [Required, Range(1, 25000), Display(Name = "Particle / Solids Density (kg/m³)")]
    public decimal ParticleDensityKgm3 { get; set; } = 660;

    [Range(1, 25000), Display(Name = "Bulk Density (kg/m³)")]
    public decimal BulkDensityKgm3 { get; set; } = 400;

    [Range(0.1, 1.0), Display(Name = "Shape Factor (1.0 = sphere)")]
    public decimal ShapeFactor { get; set; } = 1.0m;

    // ── SECTION C: Design Parameters ──────────────────────────────────────

    [Required, Range(1, 20), Display(Name = "Number of Effective Turns (Nt)")]
    public decimal EffectiveTurns { get; set; } = 6;

    [Display(Name = "Gas Viscosity (kg/m·s)")]
    public decimal GasViscosityKgms { get; set; } = 0;

    [Display(Name = "Auto-Calculate Viscosity")]
    public bool ViscosityAutoCalc { get; set; } = true;

    [Range(1, 100), Display(Name = "Number of Cyclones in Parallel")]
    public int NumberOfCyclones { get; set; } = 1;

    [Range(0.5, 3.0), Display(Name = "Safety Factor")]
    public decimal SafetyFactor { get; set; } = 1.0m;

    [Display(Name = "Inlet Shape")]
    public CyclonApp.Database.InletShape InletShape { get; set; }
}

// ── RESULTS ───────────────────────────────────────────────────────────────────

public class DesignResultsViewModel
{
    public int RevisionId { get; set; }
    public int DesignId { get; set; }
    public int ProjectId { get; set; }
    public string ProjectName { get; set; } = string.Empty;
    public string? TagNumber { get; set; }
    public string? DesignName { get; set; }
    public string CycloneType { get; set; } = string.Empty;
    public string CycloneCode { get; set; } = string.Empty;
    public int RevisionNumber { get; set; }
    public string? RevisionNote { get; set; }
    public DateTime CalculatedAt { get; set; }

    // Inputs (summary)
    public decimal FlowRateCFM { get; set; }
    public decimal InletLineSizeIn { get; set; }
    public string GasType { get; set; } = string.Empty;
    public decimal OperatingTempC { get; set; }
    public decimal OperatingPressKPa { get; set; }
    public decimal ParticleSizeMicron { get; set; }
    public decimal ParticleDensityKgm3 { get; set; }
    public decimal EffectiveTurns { get; set; }
    public int NumberOfCyclones { get; set; }

    // Calculated Outputs
    public double FlowRateM3hr { get; set; }
    public double InletVelocityMs { get; set; }
    public double GasViscosityKgms { get; set; }
    public double GasDensityKgm3 { get; set; }
    public double CutDiameterMicron { get; set; }
    public double Efficiency { get; set; }
    public double PressureDropPa { get; set; }
    public double PressureDropMmWc { get; set; }
    public double PressureDropInWc { get; set; }

    // Dimensions
    public double BarrelDiameterIn { get; set; }
    public double BarrelDiameterMm { get; set; }
    public double BarrelDiameterM { get; set; }
    public double InletHeightIn { get; set; }
    public double InletHeightMm { get; set; }
    public double InletWidthIn { get; set; }
    public double InletWidthMm { get; set; }
    public double BarrelHeightIn { get; set; }
    public double BarrelHeightMm { get; set; }
    public double ConeHeightIn { get; set; }
    public double ConeHeightMm { get; set; }
    public double ExhaustDiaIn { get; set; }
    public double ExhaustDiaMm { get; set; }
    public double ExhaustLengthIn { get; set; }
    public double ExhaustLengthMm { get; set; }
    public double BottomOutletIn { get; set; }
    public double BottomOutletMm { get; set; }
    public double TotalHeightIn { get; set; }
    public double TotalHeightMm { get; set; }

    // Grade efficiency curve (JSON for Chart.js)
    public string GradeEfficiencyCurveJson { get; set; } = "[]";

    // Dimensions JSON (for Three.js)
    public string DimensionsJson { get; set; } = "{}";

    // Physics-guided prediction (optional — populated only if generated)
    public bool HasPrediction { get; set; }
    public double PredictionEfficiency { get; set; }
    public double PredictionPressureDropPa { get; set; }
    public double PredictionPhysicsResidual { get; set; }
    public bool PredictionIsWithinTrustedRange { get; set; }
    public string? PredictionNotes { get; set; }
    public DateTime? PredictedAt { get; set; }

    // CFD Visualization (rendered PNG) — persisted per-revision so a saved
    // image is shown again on later visits. See DesignRevision.CfdImageUrl.
    public string? CfdImageUrl { get; set; }
    public DateTime? CfdImageGeneratedAt { get; set; }
}

// ── REVISION LIST ─────────────────────────────────────────────────────────────

public class RevisionListViewModel
{
    public int DesignId { get; set; }
    public int ProjectId { get; set; }
    public string ProjectName { get; set; } = string.Empty;
    public string? TagNumber { get; set; }
    public string? DesignName { get; set; }
    public string CycloneType { get; set; } = string.Empty;
    public List<RevisionRowViewModel> Revisions { get; set; } = new();
}

public class RevisionRowViewModel
{
    public int Id { get; set; }
    public int RevisionNumber { get; set; }
    public string? RevisionNote { get; set; }
    public decimal FlowRateCFM { get; set; }
    public decimal ParticleSizeMicron { get; set; }
    public decimal ParticleDensityKgm3 { get; set; }
    public bool HasResults { get; set; }
    public double? Efficiency { get; set; }
    public double? CutDiameter { get; set; }
    public double? PressureDropPa { get; set; }
    public string? CreatedBy { get; set; }
    public DateTime CreatedAt { get; set; }
    public bool IsLatest { get; set; }
}

// ── COMPARE REVISIONS ─────────────────────────────────────────────────────────

public class CompareViewModel
{
    public int DesignId { get; set; }
    public string? TagNumber { get; set; }
    public DesignResultsViewModel? RevA { get; set; }
    public DesignResultsViewModel? RevB { get; set; }
    public List<RevisionRowViewModel> AllRevisions { get; set; } = new();
    public int RevAId { get; set; }
    public int RevBId { get; set; }
}