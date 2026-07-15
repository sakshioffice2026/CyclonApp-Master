using CyclonApp.Database;
using CyclonApp.Model.DTOs;
using System.Text.Json;

public class CyclonCalculationRepository : ICyclonCalculation
{
    private static readonly JsonSerializerOptions _jsonOpts = new()
    {
        PropertyNameCaseInsensitive = true
    };

    // ── MAIN ENTRY POINT ───────────────────────────────────────────────────────

    public CyclonOutputDto Calculate(DesignRevision input, CyclonTypeRatios ratios)
    {
        if (input.FlowRateCFM <= 0)
            throw new Exception("FlowRateCFM must be greater than zero.");

        if (input.InletLineSizeIn <= 0)
            throw new Exception("InletLineSizeIn must be greater than zero.");

        if (input.ParticleDensityKgm3 <= 0)
            throw new Exception("ParticleDensityKgm3 must be greater than zero.");

        if (input.EffectiveTurns <= 0)
            throw new Exception("EffectiveTurns must be greater than zero.");

        if (ratios == null)
            throw new Exception("Cyclone ratios are null.");

        if (ratios.InletHeightRatio <= 0)
            throw new Exception("InletHeightRatio must be greater than zero.");

        if (ratios.InletWidthRatio <= 0)
            throw new Exception("InletWidthRatio must be greater than zero.");

        if (ratios.BarrelHeightRatio <= 0)
            throw new Exception("BarrelHeightRatio must be greater than zero.");

        if (ratios.ConeHeightRatio <= 0)
            throw new Exception("ConeHeightRatio must be greater than zero.");

        if (ratios.OutletDiamRatio <= 0)
            throw new Exception("OutletDiamRatio must be greater than zero.");

        if (ratios.ExhaustLengthRatio <= 0)
            throw new Exception("ExhaustLengthRatio must be greater than zero.");

        if (ratios.BottomOutletRatio <= 0)
            throw new Exception("BottomOutletRatio must be greater than zero.");

        // ── 1. Flow Conversions ──────────────────────────────────────────────
        double Q_cfm = (double)input.FlowRateCFM;
        double Q_m3s = Q_cfm * 0.000471947;          // CFM → m³/s
        double Q_m3hr = Q_cfm * 1.69901;              // CFM → m³/hr

        // ── 2. Inlet Pipe Area & Velocity ────────────────────────────────────
        double d_inlet_m = (double)input.InletLineSizeIn * 0.0254;
        double A_pipe_m2 = Math.PI * Math.Pow(d_inlet_m / 2.0, 2.0);
        double V_inlet_ms = Q_m3s / A_pipe_m2;         // m/s

        // ── 3. Barrel Diameter (Dc) from inlet duct area ─────────────────────
        //  Rectangular inlet: H = h_ratio*Dc, W = w_ratio*Dc
        //  Area = H*W = h_ratio * w_ratio * Dc²  →  Dc = sqrt(Area / (h*w))
        double h = ratios.InletHeightRatio;
        double w = ratios.InletWidthRatio;

        double Dc_m = Math.Sqrt(A_pipe_m2 / (h * w));
        double Dc_in = Dc_m / 0.0254;
        double Dc_mm = Dc_m * 1000.0;

        // ── 4. All Cyclone Dimensions ─────────────────────────────────────────

        double inletH_in = Dc_in * ratios.InletHeightRatio;
        double inletW_in = Dc_in * ratios.InletWidthRatio;
        double barrelH_in = Dc_in * ratios.BarrelHeightRatio;
        double coneH_in = Dc_in * ratios.ConeHeightRatio;
        double exhaustDia_in = Dc_in * ratios.OutletDiamRatio;
        double exhaustLen_in = Dc_in * ratios.ExhaustLengthRatio;
        double bottomOut_in = Dc_in * ratios.BottomOutletRatio;
        double totalH_in = barrelH_in + coneH_in;

        var dims = new CyclonDimensions
        {
            BarrelDiameterIn = Math.Round(Dc_in, 3),
            InletHeightIn = Math.Round(inletH_in, 3),
            InletWidthIn = Math.Round(inletW_in, 3),
            BarrelHeightIn = Math.Round(barrelH_in, 3),
            ConeHeightIn = Math.Round(coneH_in, 3),
            ExhaustDiaIn = Math.Round(exhaustDia_in, 3),
            ExhaustLengthIn = Math.Round(exhaustLen_in, 3),
            BottomOutletIn = Math.Round(bottomOut_in, 3),
            TotalHeightIn = Math.Round(totalH_in, 3),

            BarrelDiameterMm = Math.Round(Dc_mm, 1),
            InletHeightMm = Math.Round(inletH_in * 25.4, 1),
            InletWidthMm = Math.Round(inletW_in * 25.4, 1),
            BarrelHeightMm = Math.Round(barrelH_in * 25.4, 1),
            ConeHeightMm = Math.Round(coneH_in * 25.4, 1),
            ExhaustDiaMm = Math.Round(exhaustDia_in * 25.4, 1),
            ExhaustLengthMm = Math.Round(exhaustLen_in * 25.4, 1),
            BottomOutletMm = Math.Round(bottomOut_in * 25.4, 1),
            TotalHeightMm = Math.Round(totalH_in * 25.4, 1),

            BarrelDiameterM = Math.Round(Dc_m, 5),
        };

        // ── 5. Gas Viscosity ─────────────────────────────────────────────────
        double mu;
        if (!input.ViscosityAutoCalc && (double)input.GasViscosityKgms > 0)
            mu = (double)input.GasViscosityKgms;
        else
            mu = ComputeViscosity((double)input.OperatingTempC, input.GasType);

        // ── 6. Gas Density ───────────────────────────────────────────────────
        double rho_g = ComputeGasDensity(
            (double)input.OperatingTempC,
            (double)input.OperatingPressKPa,
            input.GasType);

        // ── 7. Cut Diameter (Lapple Model) ───────────────────────────────────
        //  Dpc = sqrt( 9 * mu * W / (π * Nt * Vi * ρp) )
        //  where W = inlet width in metres
        double rho_p = (double)input.ParticleDensityKgm3;
        double Nt = (double)input.EffectiveTurns;
        double W_m = inletW_in * 0.0254;             // inlet width in metres
        double Dpc_m = Math.Sqrt((9.0 * mu * W_m) /
                        (Math.PI * Nt * V_inlet_ms * rho_p));
        double Dpc_mic = Dpc_m * 1e6;                   // → microns

        // ── 8. Collection Efficiency at input particle size (Lapple) ─────────
        //  η = 1 / (1 + (Dpc/Dp)²)
        double Dp_mic = (double)input.ParticleSizeMicron;
        double eta = Dp_mic > 0
            ? 1.0 / (1.0 + Math.Pow(Dpc_mic / Dp_mic, 2.0))
            : 0.0;

        // ── 9. Pressure Drop (Shepherd-Lapple) ───────────────────────────────
        //  ΔP = Nh * ½ * ρg * Vi²
        //  Nh = 16 * Hi * Wi / (π * De²)
        double Hi_m = inletH_in * 0.0254;
        double Wi_m = inletW_in * 0.0254;
        double De_m = exhaustDia_in * 0.0254;
        double Nh = (16.0 * Hi_m * Wi_m) / (Math.PI * De_m * De_m);
        double dP_Pa = Nh * 0.5 * rho_g * V_inlet_ms * V_inlet_ms;

        // ── 10. Grade Efficiency Curve ────────────────────────────────────────
        //  Points from 0.1 µm to 1000 µm
        var gradePoints = new List<GradeEfficiencyPoint>();
        double[] sizes = { 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 30, 50,
                           75, 100, 150, 200, 300, 500, 750, 1000 };
        foreach (var sz in sizes)
        {
            double eff_i = 1.0 / (1.0 + Math.Pow(Dpc_mic / sz, 2.0));
            gradePoints.Add(new GradeEfficiencyPoint
            {
                ParticleSizeMicron = sz,
                EfficiencyPercent = Math.Round(eff_i * 100.0, 2)
            });
        }

        // ── 11. Assemble Output ───────────────────────────────────────────────
        return new CyclonOutputDto
        {
            Dimensions = dims,
            CutDiameterMicron = Math.Round(Dpc_mic, 3),
            Efficiency = Math.Round(eta * 100.0, 2),
            PressureDropPa = Math.Round(dP_Pa, 2),
            PressureDropMmWc = Math.Round(dP_Pa / 9.80665, 2),
            PressureDropInWc = Math.Round(dP_Pa / 249.089, 3),
            FlowRateM3hr = Math.Round(Q_m3hr, 3),
            FlowRateM3s = Math.Round(Q_m3s, 5),
            InletVelocityMs = Math.Round(V_inlet_ms, 3),
            GasViscosityKgms = Math.Round(mu, 10),
            GasDensityKgm3 = Math.Round(rho_g, 4),
            InletAreaM2 = Math.Round(A_pipe_m2, 6),
            GradeEfficiencyCurve = gradePoints,
        };
    }

    // ── SUTHERLAND VISCOSITY ──────────────────────────────────────────────────
    //  μ(T) = μ₀ * (T/T₀)^1.5 * (T₀+C)/(T+C)
    //  Air:  μ₀=1.716e-5 Pa·s, T₀=273.15 K, C=110.4 K
    //  N₂:   μ₀=1.663e-5 Pa·s, T₀=273.15 K, C=107 K
    //  CO₂:  μ₀=1.370e-5 Pa·s, T₀=273.15 K, C=222 K

    public double ComputeViscosity(double tempC, string gasType)
    {
        double T = tempC + 273.15;
        double mu0, T0, C;

        switch (gasType.ToUpper())
        {
            case "N2":
            case "NITROGEN":
                mu0 = 1.663e-5; T0 = 273.15; C = 107.0;
                break;
            case "CO2":
            case "FLUEGAS":
                mu0 = 1.370e-5; T0 = 273.15; C = 222.0;
                break;
            default:   // Air
                mu0 = 1.716e-5; T0 = 273.15; C = 110.4;
                break;
        }

        return mu0 * Math.Pow(T / T0, 1.5) * ((T0 + C) / (T + C));
    }

    // ── IDEAL GAS DENSITY ─────────────────────────────────────────────────────
    //  ρ = P*M / (R*T)   where R = 8.314 J/(mol·K)

    private static double ComputeGasDensity(double tempC, double pressKPa, string gasType)
    {
        double T = tempC + 273.15;
        double P = pressKPa * 1000.0;                   // Pa
        double M = gasType.ToUpper() switch
        {
            "N2" or "NITROGEN" => 0.02801,
            "CO2" or "FLUEGAS" => 0.04401,
            _ => 0.02897,            // Air
        };
        return (P * M) / (8.314 * T);
    }

    // ── PARSE RATIOS JSON ─────────────────────────────────────────────────────

    public CyclonTypeRatios? ParseRatios(string json)
    {
        try
        {
            return JsonSerializer.Deserialize<CyclonTypeRatios>(json, _jsonOpts);
        }
        catch
        {
            return null;
        }
    }
}
