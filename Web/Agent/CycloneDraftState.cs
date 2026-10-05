using System.Globalization;
using EngineeringAI.Core.Abstractions;

namespace CyclonApp.Agent;

public sealed record DraftFieldView(string Key, string Label, string? Value, bool Mandatory);

public sealed class CycloneDraftState : IDraftState
{
    private const decimal M3hrPerCfm = 1.699011m;

    public string SessionId { get; set; } = string.Empty;

    public DateTime LastUpdatedUtc { get; set; } = DateTime.UtcNow;

    public string? CycloneTypeCode { get; set; }

    public decimal? FlowRateCFM { get; set; }

    public decimal? InletLineSizeIn { get; set; }

    public string? GasType { get; set; }

    public decimal? OperatingTempC { get; set; }

    public decimal? OperatingPressKPa { get; set; }

    public decimal? ParticleSizeMicron { get; set; }

    public decimal? ParticleDensityKgm3 { get; set; }

    public decimal? BulkDensityKgm3 { get; set; }

    public decimal? EffectiveTurns { get; set; }

    public int? NumberOfCyclones { get; set; }

    public decimal? SafetyFactor { get; set; }

    public decimal? ShapeFactor { get; set; }

    public bool HasAnyValue =>
        CycloneTypeCode is not null ||
        FlowRateCFM is not null ||
        InletLineSizeIn is not null ||
        GasType is not null ||
        OperatingTempC is not null ||
        OperatingPressKPa is not null ||
        ParticleSizeMicron is not null ||
        ParticleDensityKgm3 is not null ||
        BulkDensityKgm3 is not null ||
        EffectiveTurns is not null ||
        NumberOfCyclones is not null ||
        SafetyFactor is not null ||
        ShapeFactor is not null;

    public IReadOnlyList<string> GetMissingMandatoryFields()
    {
        var missing = new List<string>();

        if (string.IsNullOrWhiteSpace(CycloneTypeCode))
        {
            missing.Add("cyclone type");
        }

        if (FlowRateCFM is null)
        {
            missing.Add("gas flow rate (m³/h or CFM)");
        }

        if (InletLineSizeIn is null)
        {
            missing.Add("inlet line size (inches or mm)");
        }

        if (ParticleSizeMicron is null)
        {
            missing.Add("average particle size (µm)");
        }

        if (ParticleDensityKgm3 is null)
        {
            missing.Add("particle density (kg/m³)");
        }

        return missing;
    }

    public IReadOnlyList<DraftFieldView> Snapshot()
    {
        return new[]
        {
            new DraftFieldView("cycloneType", "Cyclone type", CycloneTypeCode, true),
            new DraftFieldView("flow", "Gas flow",
                FlowRateCFM is null ? null : $"{Fmt(FlowRateCFM.Value * M3hrPerCfm)} m³/h ({Fmt(FlowRateCFM.Value)} CFM)", true),
            new DraftFieldView("inletLine", "Inlet line size",
                InletLineSizeIn is null ? null : $"{Fmt(InletLineSizeIn.Value)} in ({Fmt(InletLineSizeIn.Value * 25.4m)} mm)", true),
            new DraftFieldView("particleSize", "Particle size",
                ParticleSizeMicron is null ? null : $"{Fmt(ParticleSizeMicron.Value)} µm", true),
            new DraftFieldView("particleDensity", "Particle density",
                ParticleDensityKgm3 is null ? null : $"{Fmt(ParticleDensityKgm3.Value)} kg/m³", true),
            new DraftFieldView("gasType", "Gas", GasType, false),
            new DraftFieldView("temp", "Temperature",
                OperatingTempC is null ? null : $"{Fmt(OperatingTempC.Value)} °C", false),
            new DraftFieldView("pressure", "Pressure",
                OperatingPressKPa is null ? null : $"{Fmt(OperatingPressKPa.Value)} kPa", false),
            new DraftFieldView("bulkDensity", "Bulk density",
                BulkDensityKgm3 is null ? null : $"{Fmt(BulkDensityKgm3.Value)} kg/m³", false),
            new DraftFieldView("turns", "Effective turns",
                EffectiveTurns is null ? null : Fmt(EffectiveTurns.Value), false),
            new DraftFieldView("count", "Cyclones in parallel",
                NumberOfCyclones is null ? null : NumberOfCyclones.Value.ToString(CultureInfo.InvariantCulture), false),
            new DraftFieldView("safety", "Safety factor",
                SafetyFactor is null ? null : Fmt(SafetyFactor.Value), false)
        };
    }

    private static string Fmt(decimal value) =>
        value.ToString("0.##", CultureInfo.InvariantCulture);
}
