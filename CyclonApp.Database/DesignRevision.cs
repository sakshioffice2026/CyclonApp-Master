using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using CyclonApp.Database;

namespace CyclonApp.Database;

public class DesignRevision
{
    public int Id { get; set; }

    public int CycloneDesignId { get; set; }
    public CycloneDesign CycloneDesign { get; set; } = null!;

    public int RevisionNumber { get; set; } = 1;

    [MaxLength(300)]
    public string? RevisionNote { get; set; }

    // ── PROCESS INPUTS ────────────────────────────────────────────────────────

    [Column(TypeName = "decimal(12,4)")]
    public decimal FlowRateCFM { get; set; }

    [Column(TypeName = "decimal(12,4)")]
    public decimal FlowRateM3hr { get; set; }       // computed from CFM

    [Column(TypeName = "decimal(8,4)")]
    public decimal InletLineSizeIn { get; set; }

    [Column(TypeName = "decimal(10,4)")]
    public decimal AvgVelocityMs { get; set; }       // computed

    [Column(TypeName = "decimal(8,2)")]
    public decimal OperatingTempC { get; set; } = 25;

    [Column(TypeName = "decimal(10,4)")]
    public decimal OperatingPressKPa { get; set; } = 101.325m;

    [MaxLength(80)]
    public string GasType { get; set; } = "Air";    // Air | N2 | FlueGas | Custom

    // ── SOLIDS / PARTICLE INPUTS ──────────────────────────────────────────────

    [Column(TypeName = "decimal(10,4)")]
    public decimal ParticleSizeMicron { get; set; }  // avg / D50

    [Column(TypeName = "decimal(10,4)")]
    public decimal ParticleSizeD10 { get; set; }

    [Column(TypeName = "decimal(10,4)")]
    public decimal ParticleSizeD50 { get; set; }

    [Column(TypeName = "decimal(10,4)")]
    public decimal ParticleSizeD90 { get; set; }

    [Column(TypeName = "decimal(10,4)")]
    public decimal ParticleDensityKgm3 { get; set; }

    [Column(TypeName = "decimal(10,4)")]
    public decimal BulkDensityKgm3 { get; set; }

    [Column(TypeName = "decimal(6,4)")]
    public decimal ShapeFactor { get; set; } = 1.0m;   // 1.0 = perfect sphere

    // ── DESIGN PARAMETERS ─────────────────────────────────────────────────────

    [Column(TypeName = "decimal(6,2)")]
    public decimal EffectiveTurns { get; set; } = 6;

    [Column(TypeName = "decimal(14,10)")]
    public decimal GasViscosityKgms { get; set; }     // 0 = auto (Sutherland)

    public bool ViscosityAutoCalc { get; set; } = true;

    public int NumberOfCyclones { get; set; } = 1;    // parallel arrangement

    [Column(TypeName = "decimal(5,3)")]
    public decimal SafetyFactor { get; set; } = 1.0m;

    public InletShape InletShape { get; set; } = InletShape.Rectangular;

    // ── CALCULATED OUTPUTS (stored as JSON) ────────────────────────────────────

    public string? DimensionsJson { get; set; }   // CycloneDimensions object

    public string? EfficiencyJson { get; set; }   // CycloneOutputDto object

    // ── AUDIT ─────────────────────────────────────────────────────────────────

    public DateTime? CalculatedAt { get; set; }

    public int CreatedByUserId { get; set; }
    public AppUser? CreatedBy { get; set; }


    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;


    // ── CALCULATED OUTPUTS (stored as JSON) ────────────────────────────────────

  

    public string? PredictionJson { get; set; }    // CyclonePredictionDto object — physics-guided
                                                   // prediction, nullable until run

    public DateTime? PredictedAt { get; set; }      // when the prediction was last generated


    // Navigation
    public ICollection<ExportLog> ExportLogs { get; set; } = new List<ExportLog>();
}

public enum InletShape
{
    Rectangular,
    Circular
}
