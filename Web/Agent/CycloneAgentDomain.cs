using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using CyclonApp.Database;
using CyclonApp.Model.DTOs;
using CyclonApp.Repositories.Contracts;
using EngineeringAI.Core.Agent;
using Microsoft.Extensions.DependencyInjection;

namespace CyclonApp.Agent;

public sealed record AgentCheck(string Name, string Parameter, string Status, string Detail);

public static class CycloneAgentDomain
{
    public const decimal M3hrToCfm = 0.588578m;

    private static readonly JsonSerializerOptions ReplyJson = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
    };

    private static readonly JsonSerializerOptions StoreJson = new()
    {
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
    };

    private const string FieldSchemaJson = """
        {
          "cycloneType": "string, one of LAPPLE | STAIRMAND | STAIRMAND_GP | SWIFT_HE",
          "flowRateM3hr": "number, gas flow rate in m3/h",
          "flowRateCFM": "number, gas flow rate in CFM",
          "inletLineSizeIn": "number, inlet line diameter in inches",
          "inletLineSizeMm": "number, inlet line diameter in millimetres",
          "gasType": "string, one of Air | N2 | FlueGas",
          "operatingTempC": "number, operating temperature in degrees Celsius",
          "operatingPressKPa": "number, operating pressure in kPa",
          "particleSizeMicron": "number, average particle size in micrometres",
          "particleDensityKgm3": "number, particle density in kg/m3",
          "bulkDensityKgm3": "number, bulk density in kg/m3",
          "effectiveTurns": "number, effective gas turns, 1 to 20",
          "numberOfCyclones": "integer, number of cyclones in parallel",
          "safetyFactor": "number, 0.5 to 3.0",
          "shapeFactor": "number, 0.1 to 1.0"
        }
        """;

    public static AgentDomainDefinition<CycloneDraftState> Create(IServiceScopeFactory scopeFactory)
    {
        return new AgentDomainDefinition<CycloneDraftState>
        {
            DomainDescription = "cyclone separator sizing and performance design (gas-solid separation, dust collection)",
            FieldSchemaJson = FieldSchemaJson,
            ApplyExtracted = Apply,
            ComputeAsync = (state, ct) => ComputeAsync(scopeFactory, state, ct)
        };
    }

    public static CycloneType? MatchType(IEnumerable<CycloneType> types, string? code)
    {
        if (string.IsNullOrWhiteSpace(code))
        {
            return null;
        }

        var list = types.ToList();
        var key = NormalizeType(code);

        return list.FirstOrDefault(t => NormalizeType(t.Code) == key)
            ?? list.FirstOrDefault(t => NormalizeType(t.Name) == key)
            ?? list.FirstOrDefault(t => NormalizeType(t.Code).StartsWith(key, StringComparison.Ordinal));
    }

    public static DesignRevision BuildRevision(CycloneDraftState s, CycloneType type)
    {
        var d50 = s.ParticleSizeMicron ?? 0m;
        var density = s.ParticleDensityKgm3 ?? 0m;

        return new DesignRevision
        {
            RevisionNumber = 1,
            FlowRateCFM = s.FlowRateCFM ?? 0m,
            InletLineSizeIn = s.InletLineSizeIn ?? 0m,
            GasType = s.GasType ?? "Air",
            OperatingTempC = s.OperatingTempC ?? 25m,
            OperatingPressKPa = s.OperatingPressKPa ?? 101.325m,
            ParticleSizeMicron = d50,
            ParticleSizeD10 = Math.Round(d50 * 0.3m, 4),
            ParticleSizeD50 = d50,
            ParticleSizeD90 = Math.Round(d50 * 2.5m, 4),
            ParticleDensityKgm3 = density,
            BulkDensityKgm3 = s.BulkDensityKgm3 ?? Math.Round(density * 0.6m, 4),
            ShapeFactor = s.ShapeFactor ?? 1.0m,
            EffectiveTurns = s.EffectiveTurns ?? type.DefaultEffectiveTurns,
            ViscosityAutoCalc = true,
            NumberOfCyclones = s.NumberOfCyclones ?? 1,
            SafetyFactor = s.SafetyFactor ?? 1.0m,
            InletShape = InletShape.Rectangular,
            CreatedAt = DateTime.UtcNow
        };
    }

    public static void ApplyResult(DesignRevision revision, CyclonOutputDto result)
    {
        revision.FlowRateM3hr = (decimal)result.FlowRateM3hr;
        revision.AvgVelocityMs = (decimal)result.InletVelocityMs;

        if (revision.ViscosityAutoCalc)
        {
            revision.GasViscosityKgms = (decimal)result.GasViscosityKgms;
        }

        revision.DimensionsJson = JsonSerializer.Serialize(result.Dimensions, StoreJson);
        revision.EfficiencyJson = JsonSerializer.Serialize(result, StoreJson);
        revision.CalculatedAt = DateTime.UtcNow;
    }

    private static void Apply(CycloneDraftState s, JsonElement root)
    {
        if (TryText(root, "cycloneType", out var type))
        {
            s.CycloneTypeCode = NormalizeType(type);
        }

        if (TryNum(root, "flowRateCFM", out var cfm) && InRange(cfm, 0.01m, 1_000_000m))
        {
            s.FlowRateCFM = cfm;
        }

        if (TryNum(root, "flowRateM3hr", out var m3) && InRange(m3 * M3hrToCfm, 0.01m, 1_000_000m))
        {
            s.FlowRateCFM = Math.Round(m3 * M3hrToCfm, 4);
        }

        if (TryNum(root, "inletLineSizeIn", out var inch) && InRange(inch, 0.5m, 100m))
        {
            s.InletLineSizeIn = inch;
        }

        if (TryNum(root, "inletLineSizeMm", out var mm) && InRange(mm / 25.4m, 0.5m, 100m))
        {
            s.InletLineSizeIn = Math.Round(mm / 25.4m, 4);
        }

        if (TryText(root, "gasType", out var gas))
        {
            var normalized = NormalizeGas(gas);
            if (normalized is not null)
            {
                s.GasType = normalized;
            }
        }

        if (TryNum(root, "operatingTempC", out var temp) && InRange(temp, -50m, 1000m))
        {
            s.OperatingTempC = temp;
        }

        if (TryNum(root, "operatingPressKPa", out var press) && InRange(press, 1m, 10000m))
        {
            s.OperatingPressKPa = press;
        }

        if (TryNum(root, "particleSizeMicron", out var size) && InRange(size, 0.01m, 100000m))
        {
            s.ParticleSizeMicron = size;
        }

        if (TryNum(root, "particleDensityKgm3", out var pd) && InRange(pd, 1m, 25000m))
        {
            s.ParticleDensityKgm3 = pd;
        }

        if (TryNum(root, "bulkDensityKgm3", out var bd) && InRange(bd, 1m, 25000m))
        {
            s.BulkDensityKgm3 = bd;
        }

        if (TryNum(root, "effectiveTurns", out var turns) && InRange(turns, 1m, 20m))
        {
            s.EffectiveTurns = turns;
        }

        if (TryNum(root, "numberOfCyclones", out var count) && InRange(count, 1m, 100m))
        {
            s.NumberOfCyclones = (int)Math.Round(count);
        }

        if (TryNum(root, "safetyFactor", out var sf) && InRange(sf, 0.5m, 3m))
        {
            s.SafetyFactor = sf;
        }

        if (TryNum(root, "shapeFactor", out var shape) && InRange(shape, 0.1m, 1m))
        {
            s.ShapeFactor = shape;
        }
    }

    private static async Task<AgentComputation> ComputeAsync(
        IServiceScopeFactory scopeFactory,
        CycloneDraftState state,
        CancellationToken cancellationToken)
    {
        try
        {
            using var scope = scopeFactory.CreateScope();
            var designs = scope.ServiceProvider.GetRequiredService<IDesignRepository>();
            var calc = scope.ServiceProvider.GetRequiredService<ICyclonCalculation>();

            var types = await designs.GetActiveCycloneTypesAsync().ConfigureAwait(false);
            cancellationToken.ThrowIfCancellationRequested();

            var type = MatchType(types, state.CycloneTypeCode);
            if (type is null)
            {
                var valid = string.Join(", ", types.Select(t => t.Code));
                state.CycloneTypeCode = null;
                return Error($"Unknown cyclone type. Available types: {valid}.");
            }

            state.CycloneTypeCode = type.Code;

            var ratios = calc.ParseRatios(type.DimensionRatiosJson);
            if (ratios is null)
            {
                return Error("Cyclone type configuration error. Contact admin.");
            }

            var revision = BuildRevision(state, type);
            var result = calc.Calculate(revision, ratios);
            var checks = BuildChecks(result, state);
            var summary = BuildSummary(result, state, type, revision);

            return new AgentComputation(
                JsonSerializer.Serialize(summary, ReplyJson),
                JsonSerializer.Serialize(checks, ReplyJson));
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            return Error(ex.Message);
        }
    }

    private static AgentComputation Error(string message) =>
        new(JsonSerializer.Serialize(new { error = message }, ReplyJson), "[]");

    private static List<AgentCheck> BuildChecks(CyclonOutputDto r, CycloneDraftState s)
    {
        var checks = new List<AgentCheck>();

        var dpStatus = r.PressureDropPa >= 2500 ? "FAIL" : r.PressureDropPa >= 1500 ? "WARN" : "PASS";
        checks.Add(new AgentCheck(
            "Pressure drop",
            "inletLineSizeIn",
            dpStatus,
            Inv($"Pressure drop is {r.PressureDropPa:0} Pa (warning above 1500 Pa, critical above 2500 Pa).")));

        string vStatus;
        string vDetail;
        if (r.InletVelocityMs < 12)
        {
            vStatus = "WARN";
            vDetail = Inv($"Inlet velocity {r.InletVelocityMs:0.0} m/s is low; efficiency will suffer and solids may settle.");
        }
        else if (r.InletVelocityMs > 40)
        {
            vStatus = "FAIL";
            vDetail = Inv($"Inlet velocity {r.InletVelocityMs:0.0} m/s is above 40 m/s; erosion and re-entrainment risk.");
        }
        else if (r.InletVelocityMs > 25)
        {
            vStatus = "WARN";
            vDetail = Inv($"Inlet velocity {r.InletVelocityMs:0.0} m/s is above 25 m/s; pressure drop and wear increase.");
        }
        else
        {
            vStatus = "PASS";
            vDetail = Inv($"Inlet velocity {r.InletVelocityMs:0.0} m/s is within range.");
        }

        checks.Add(new AgentCheck("Inlet velocity", "inletLineSizeIn", vStatus, vDetail));

        var effStatus = r.Efficiency < 60 ? "FAIL" : r.Efficiency < 80 ? "WARN" : "PASS";
        checks.Add(new AgentCheck(
            "Collection efficiency",
            "cycloneType",
            effStatus,
            Inv($"Predicted efficiency is {r.Efficiency:0.0}% at the given particle size.")));

        var particle = (double)(s.ParticleSizeMicron ?? 0m);
        if (particle > 0 && r.CutDiameterMicron > particle)
        {
            checks.Add(new AgentCheck(
                "Cut diameter",
                "particleSizeMicron",
                "WARN",
                Inv($"Cut diameter {r.CutDiameterMicron:0.0} µm is larger than the average particle size {particle:0.0} µm.")));
        }
        else
        {
            checks.Add(new AgentCheck(
                "Cut diameter",
                "particleSizeMicron",
                "PASS",
                Inv($"Cut diameter {r.CutDiameterMicron:0.0} µm is below the average particle size.")));
        }

        return checks;
    }

    private static object BuildSummary(
        CyclonOutputDto r,
        CycloneDraftState s,
        CycloneType type,
        DesignRevision revision)
    {
        var assumptions = new List<string>();

        if (s.GasType is null) assumptions.Add("Gas type: Air");
        if (s.OperatingTempC is null) assumptions.Add("Operating temperature: 25 °C");
        if (s.OperatingPressKPa is null) assumptions.Add("Operating pressure: 101.325 kPa");
        if (s.EffectiveTurns is null) assumptions.Add(Inv($"Effective turns: {revision.EffectiveTurns:0.##} (type default)"));
        if (s.BulkDensityKgm3 is null) assumptions.Add("Bulk density: 60% of particle density");
        if (s.NumberOfCyclones is null) assumptions.Add("Cyclones in parallel: 1");
        if (s.SafetyFactor is null) assumptions.Add("Safety factor: 1.0");
        assumptions.Add("Particle D10 / D90: 0.3× / 2.5× of average size");

        return new
        {
            cycloneType = type.Code,
            cycloneTypeName = type.Name,
            flowRateM3hr = r.FlowRateM3hr,
            flowRateCfm = (double)revision.FlowRateCFM,
            inletVelocityMs = r.InletVelocityMs,
            efficiencyPercent = r.Efficiency,
            cutDiameterMicron = r.CutDiameterMicron,
            pressureDropPa = r.PressureDropPa,
            pressureDropMmWc = r.PressureDropMmWc,
            gasDensityKgm3 = r.GasDensityKgm3,
            gasViscosityKgms = r.GasViscosityKgms,
            barrelDiameterMm = r.Dimensions.BarrelDiameterMm,
            totalHeightMm = r.Dimensions.TotalHeightMm,
            inletHeightMm = r.Dimensions.InletHeightMm,
            inletWidthMm = r.Dimensions.InletWidthMm,
            exhaustDiaMm = r.Dimensions.ExhaustDiaMm,
            assumptions
        };
    }

    private static string Inv(FormattableString text) => text.ToString(CultureInfo.InvariantCulture);

    private static bool InRange(decimal value, decimal min, decimal max) => value >= min && value <= max;

    private static bool TryText(JsonElement root, string name, out string value)
    {
        value = string.Empty;

        if (!root.TryGetProperty(name, out var p) || p.ValueKind != JsonValueKind.String)
        {
            return false;
        }

        var text = p.GetString();
        if (string.IsNullOrWhiteSpace(text))
        {
            return false;
        }

        value = text.Trim();
        return true;
    }

    private static bool TryNum(JsonElement root, string name, out decimal value)
    {
        value = 0m;

        if (!root.TryGetProperty(name, out var p))
        {
            return false;
        }

        if (p.ValueKind == JsonValueKind.Number)
        {
            return p.TryGetDecimal(out value);
        }

        if (p.ValueKind == JsonValueKind.String)
        {
            var text = p.GetString()?.Replace(",", string.Empty);
            return decimal.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
        }

        return false;
    }

    private static string NormalizeType(string value)
    {
        var chars = value.Trim().ToUpperInvariant()
            .Select(c => char.IsLetterOrDigit(c) ? c : '_')
            .ToArray();

        return new string(chars);
    }

    private static string? NormalizeGas(string raw)
    {
        var key = new string(raw.Where(char.IsLetterOrDigit).ToArray()).ToLowerInvariant();

        return key switch
        {
            "air" => "Air",
            "n2" or "nitrogen" => "N2",
            "fluegas" or "flue" => "FlueGas",
            _ => null
        };
    }
}
