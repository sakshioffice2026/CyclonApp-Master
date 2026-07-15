using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using CyclonApp.Database;

public class DesignRevisionModel
{
    public int Id { get; set; }

    public int CycloneDesignId { get; set; }
    public CycloneDesign CycloneDesign { get; set; } = null!;

    public int? CreatedByUserId { get; set; }
    [ForeignKey(nameof(CreatedByUserId))]
    public AppUser? CreatedBy { get; set; }

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public int RevisionNumber { get; set; } = 1;

    [MaxLength(300)]
    public string? RevisionNote { get; set; }

    // ── Process Inputs ──
    [Column(TypeName = "decimal(12,4)")]
    public decimal FlowRateCFM { get; set; }

    [Column(TypeName = "decimal(12,4)")]
    public decimal FlowRateM3hr { get; set; }

    [Column(TypeName = "decimal(8,4)")]
    public decimal InletLineSizeIn { get; set; }

    [Column(TypeName = "decimal(10,4)")]
    public decimal AvgVelocityMs { get; set; }

    [Column(TypeName = "decimal(8,2)")]
    public decimal OperatingTempC { get; set; } = 25;

    [Column(TypeName = "decimal(10,4)")]
    public decimal OperatingPressKPa { get; set; } = 101.325m;

    [MaxLength(80)]
    public string GasType { get; set; } = "Air";

    // ── Particle Inputs ──
    [Column(TypeName = "decimal(10,4)")]
    public decimal ParticleSizeMicron { get; set; }

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
    public decimal ShapeFactor { get; set; } = 1.0m;

    // ── Design Parameters ──
    [Column(TypeName = "decimal(6,2)")]
    public decimal EffectiveTurns { get; set; } = 6;

    [Column(TypeName = "decimal(14,10)")]
    public decimal GasViscosityKgms { get; set; }

    public bool ViscosityAutoCalc { get; set; } = true;

    public int NumberOfCyclones { get; set; } = 1;

    [Column(TypeName = "decimal(5,3)")]
    public decimal SafetyFactor { get; set; } = 1.0m;

    [MaxLength(20)]
    public string InletShape { get; set; } = "Rectangular";

    // ── Calculated Outputs ──
    public string? DimensionsJson { get; set; }
    public string? EfficiencyJson { get; set; }
    public DateTime? CalculatedAt { get; set; }

    public ICollection<ExportLog> ExportLogs { get; set; } = new List<ExportLog>();
}
