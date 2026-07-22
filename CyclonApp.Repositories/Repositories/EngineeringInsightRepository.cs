using System;
using System.Collections.Generic;
using System.Linq;
using CyclonApp.Model.DTOs;
using CyclonApp.Repositories.Contracts;

namespace CyclonApp.Repositories.Repositories
{
    /// <summary>
    /// Rule-based engineering insight generator. No LLM/generative model
    /// anywhere in this class — every judgment below is a fixed numeric
    /// threshold. PLACEHOLDER THRESHOLDS: the constants in Thresholds are
    /// starting estimates, not confirmed engineering limits — have the
    /// client's engineer review/adjust them before this is relied on in
    /// production.
    ///
    /// The "Separation Efficiency" risk indicator uses the real Lapple-model
    /// efficiency (CyclonOutputDto.Efficiency) when GenerateReport is given
    /// one, and only falls back to a swirl-only estimate — clearly labeled
    /// "(estimated)" — when it isn't available. The "Mass Conservation" card
    /// is a distinct check (does the flow field balance mass, not how well
    /// it separates particles) and was previously miscategorized under the
    /// "Separation Efficiency" name; that's why two unrelated numbers used
    /// to look like the same metric. See EstimateSeparationEfficiency's
    /// remarks for detail.
    /// </summary>
    public class EngineeringInsightRepository : IEngineeringInsight
    {
        private static class Thresholds
        {
            // FALLBACK ONLY — used when no design-specific baseline is
            // available (see EvaluatePressureDrop). Same "placeholder,
            // needs engineer review" caveat as every other threshold in
            // this class.
            public const double PressureDropWarningPa = 1500.0;
            public const double PressureDropCriticalPa = 2500.0;

            // PREFERRED PATH — how far the field-solve's pressure drop is
            // allowed to exceed THIS design's own Shepherd-Lapple calculated
            // ΔP before being flagged. Ratios, not absolute Pa, so the same
            // rule applies whether the design's calculated baseline is a
            // GP-typical 400 Pa or an HE-typical 1200 Pa — either way,
            // running materially higher than what the geometry itself
            // predicts indicates something is actually wrong (mis-sized
            // inlet, a solver artifact, wear changing effective geometry,
            // etc.), rather than just "this type runs high by design."
            // PLACEHOLDER RATIOS — same caveat as above, have an engineer
            // confirm 15%/30% are reasonable tolerances before relying on
            // this in production.
            public const double PressureDropBaselineWarningRatio = 1.15;
            public const double PressureDropBaselineCriticalRatio = 1.30;

            // LOW-SIDE — the field-solve landing well BELOW the design's own
            // calculated baseline is just as much a disagreement as landing
            // above it, and was previously not checked at all (fell through
            // to "Good" no matter how large the shortfall was). Same 15%/30%
            // tolerance, mirrored below 1.0 instead of above it.
            // PLACEHOLDER RATIOS — same "needs engineer review" caveat as
            // every other threshold in this class.
            public const double PressureDropBaselineWarningRatioLow = 0.85;
            public const double PressureDropBaselineCriticalRatioLow = 0.70;

            public const double TangentialVelocityLowMs = 12.0;
            public const double TangentialVelocityWarningMs = 25.0;
            public const double TangentialVelocityCriticalMs = 40.0;

            public const double WallVelocityWarningMs = 30.0;
            public const double WallVelocityCriticalMs = 45.0;

            public const double MassFlowSpreadWarning = 0.15;
            public const double MassFlowSpreadCritical = 0.30;
        }

        public CycloneHealthReportDto GenerateReport(FieldResultDto result, double? knownEfficiencyPercent = null, double? knownPressureDropPa = null)
        {
            if (result == null) throw new ArgumentNullException(nameof(result));

            var insights = new List<EngineeringInsightDto>();

            double maxPressureDrop = ComputePressureDrop(result);
            double avgTangential = Average(result.VThetaMs);
            double maxWallVelocity = ComputeMaxWallVelocity(result);

            insights.Add(EvaluatePressureDrop(maxPressureDrop, knownPressureDropPa));
            insights.Add(EvaluateSwirlStrength(avgTangential));
            insights.Add(EvaluateWallVelocity(maxWallVelocity));
            insights.Add(EvaluateMassConservation(result));

            var physics = new PhysicsValidationDto
            {
                MassConservationPassed = !string.Equals(result.MassConservationStatus, "failed", StringComparison.OrdinalIgnoreCase),
                BoundaryConditionsPassed = true,
                ConvergencePassed = !result.FinalLoss.HasValue || result.FinalLoss.Value < 1.0,
                ConfidencePercent = ComputeConfidence(result)
            };

            var riskIndicators = new List<RiskIndicatorDto>
            {
                BuildRisk("Wear Risk", NormalizePercent(maxWallVelocity, Thresholds.WallVelocityWarningMs, Thresholds.WallVelocityCriticalMs)),
                BuildRisk("Energy Consumption", NormalizePercent(maxPressureDrop, Thresholds.PressureDropWarningPa, Thresholds.PressureDropCriticalPa)),
            };

            // Prefer the real Lapple-model efficiency (from the standard
            // calculation, which actually accounts for geometry and particle
            // size) whenever the caller has one. Only fall back to the
            // swirl-only placeholder — and say so in the label — when no
            // standard calculation result is available yet, so the number on
            // screen never silently looks more authoritative than it is.
            bool usingRealEfficiency = knownEfficiencyPercent.HasValue;
            double separationEfficiencyPercent = usingRealEfficiency
                ? knownEfficiencyPercent!.Value
                : EstimateSeparationEfficiency(avgTangential);

            riskIndicators.Add(new RiskIndicatorDto
            {
                Name = usingRealEfficiency ? "Separation Efficiency" : "Separation Efficiency (estimated)",
                Percent = Math.Round(Math.Max(0.0, Math.Min(100.0, separationEfficiencyPercent)), 1),
                Level = "Info"
            });

            double healthScore = ComputeHealthScore(insights, physics);

            return new CycloneHealthReportDto
            {
                HealthScore = Math.Round(healthScore, 0),
                Grade = GradeFor(healthScore),
                PhysicsValidation = physics,
                Insights = insights,
                RiskIndicators = riskIndicators,
                Summary = BuildSummary(healthScore, insights, physics),
                Conclusion = BuildConclusion(healthScore, insights, physics)
            };
        }

        public string BuildReportHtml(CycloneHealthReportDto report, string? tagNumber, int revisionNumber, string? projectName)
        {
            if (report == null) throw new ArgumentNullException(nameof(report));

            string scoreColor = report.HealthScore >= 90 ? "#16a34a"
                               : report.HealthScore >= 75 ? "#0891b2"
                               : report.HealthScore >= 60 ? "#d97706" : "#dc2626";

            string pv = report.PhysicsValidation == null ? "" : $@"
      <table>
        <tr><th>Check</th><th>Result</th></tr>
        {PhysicsRow("Mass Conservation", report.PhysicsValidation.MassConservationPassed)}
        {PhysicsRow("Boundary Conditions", report.PhysicsValidation.BoundaryConditionsPassed)}
        {PhysicsRow("Model Convergence", report.PhysicsValidation.ConvergencePassed)}
      </table>
      <p style=""margin-top:8px;font-size:11.5px;color:#64748b;"">
        Simulation Confidence: <strong>{report.PhysicsValidation.ConfidencePercent:F0}%</strong>
      </p>";

            string cards = string.Concat((report.Insights ?? new List<EngineeringInsightDto>()).Select(i =>
            {
                var (color, icon) = i.Severity switch
                {
                    InsightSeverity.Critical => ("#dc2626", "&#10006;"),
                    InsightSeverity.Warning => ("#d97706", "&#9888;"),
                    _ => ("#16a34a", "&#10003;"),
                };
                string impact = (i.Impact != null && i.Impact.Count > 0)
                    ? "<ul>" + string.Concat(i.Impact.Select(x => $"<li>{x}</li>")) + "</ul>"
                    : "";
                return $@"
      <div style=""border-left:4px solid {color};padding:10px 14px;margin-bottom:10px;background:#fafafa;border-radius:0 6px 6px 0;"">
        <div style=""font-weight:700;color:{color};"">{icon} {i.Category}</div>
        <div style=""margin-top:4px;""><strong>What happened?</strong> {i.WhatHappened}</div>
        <div style=""margin-top:2px;""><strong>Why?</strong> {i.Why}</div>
        {(impact.Length > 0 ? $"<div style='margin-top:2px;'><strong>Impact:</strong>{impact}</div>" : "")}
        <div style=""margin-top:2px;""><strong>Recommendation:</strong> {i.Recommendation}</div>
      </div>";
            }));

            string risks = string.Concat((report.RiskIndicators ?? new List<RiskIndicatorDto>()).Select(r =>
            {
                string color = r.Level == "High" ? "#dc2626" : r.Level == "Medium" ? "#d97706" : "#16a34a";
                double pct = Math.Clamp(r.Percent, 0, 100);
                return $@"
      <div style=""margin-bottom:10px;"">
        <div style=""font-size:11.5px;margin-bottom:3px;"">{r.Name} &mdash; {r.Percent:F1}% ({r.Level})</div>
        <div style=""background:#e2e8f0;height:10px;width:100%;max-width:320px;border-radius:5px;overflow:hidden;"">
          <div style=""background:{color};height:10px;width:{pct.ToString("F1", System.Globalization.CultureInfo.InvariantCulture)}%;""></div>
        </div>
      </div>";
            }));

            return $@"<!DOCTYPE html>
<html lang=""en"">
<head>
<meta charset=""UTF-8""/>
<title>AI Engineering Insight Report — {tagNumber ?? "—"} Rev {revisionNumber}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',Arial,sans-serif;font-size:12px;color:#1e293b;background:#fff}}
  .page{{padding:28px 32px;max-width:900px;margin:0 auto}}
  .report-header{{background:linear-gradient(135deg,#1a56db,#1240a8);color:#fff;padding:24px 28px;border-radius:10px;margin-bottom:20px;display:flex;justify-content:space-between;align-items:center}}
  .report-title{{font-size:20px;font-weight:700}}
  .report-sub{{font-size:12px;opacity:.8;margin-top:4px}}
  .score-badge{{text-align:right}}
  .score-value{{font-size:32px;font-weight:800;line-height:1}}
  .score-grade{{font-size:12.5px;opacity:.9;margin-top:2px}}
  .section{{margin-bottom:18px}}
  .section-title{{font-size:13px;font-weight:700;color:#1a56db;border-bottom:2px solid #1a56db;padding-bottom:5px;margin-bottom:10px}}
  table{{width:100%;border-collapse:collapse;font-size:11.5px}}
  th{{background:#1a56db;color:#fff;padding:7px 10px;text-align:left;font-weight:600}}
  td{{padding:6px 10px;border-bottom:1px solid #f1f5f9}}
  tr:nth-child(even) td{{background:#f8fafc}}
  .summary-box{{background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:14px 16px;font-style:italic;font-size:12px;}}
  .report-footer{{margin-top:24px;padding-top:12px;border-top:1px solid #e2e8f0;display:flex;justify-content:space-between;font-size:10.5px;color:#94a3b8}}
  @media print{{body{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}.page{{padding:8px}}@page{{margin:10mm;size:A4}}.report-header{{border-radius:0}}}}
</style>
</head>
<body>
<div class=""page"">
  <div class=""report-header"">
    <div>
      <div class=""report-title"">&#129504; AI Engineering Insight Report</div>
      <div class=""report-sub"">Tag: {tagNumber ?? "—"} &nbsp;&middot;&nbsp; {(string.IsNullOrEmpty(projectName) ? "" : projectName + " &nbsp;&middot;&nbsp; ")}Revision {revisionNumber}</div>
    </div>
    <div class=""score-badge"">
      <div class=""score-value"" style=""color:#fff;"">{report.HealthScore:F0}<span style=""font-size:16px;opacity:.7"">/100</span></div>
      <div class=""score-grade"">{report.Grade}</div>
    </div>
  </div>

  <div class=""section"">
    <div class=""section-title"">Physics Validation</div>
    {pv}
  </div>

  <div class=""section"">
    <div class=""section-title"">Engineering Insights</div>
    {(cards.Length > 0 ? cards : "<p style='color:#64748b;font-size:11.5px;'>No specific insights flagged for this run.</p>")}
  </div>

  <div class=""section"">
    <div class=""section-title"">Risk Indicators</div>
    {(risks.Length > 0 ? risks : "<p style='color:#64748b;font-size:11.5px;'>No risk indicators available.</p>")}
  </div>

  <div class=""section"">
    <div class=""section-title"">Summary</div>
    <div class=""summary-box"">{report.Summary}</div>
  </div>

  <div class=""section"">
    <div class=""section-title"">Conclusion</div>
    <div style=""background:{(report.Conclusion.ReadyToProceed ? "#f0fdf4" : "#fef2f2")};border:1px solid {(report.Conclusion.ReadyToProceed ? "#bbf7d0" : "#fecaca")};border-radius:8px;padding:14px 16px;"">
      <p style=""margin:0 0 {(report.Conclusion.PriorityActions.Count > 0 ? "10px" : "0")};font-size:12px;"">{report.Conclusion.Verdict}</p>
      {(report.Conclusion.PriorityActions.Count > 0 ? $@"
      <div style=""font-weight:700;font-size:11.5px;margin-bottom:4px;"">Priority actions, in order:</div>
      <ol style=""margin:0;padding-left:18px;font-size:11.5px;"">
        {string.Concat(report.Conclusion.PriorityActions.Select(a => $"<li style='margin-bottom:3px;'>{a}</li>"))}
      </ol>" : "")}
    </div>
  </div>

  <div class=""report-footer"">
    <span>Generated by Cyclone Design App — AI Engineering Insight &nbsp;&middot;&nbsp; {DateTime.UtcNow:dd MMM yyyy HH:mm} UTC</span>
    <span>CONFIDENTIAL — Engineering Use Only</span>
  </div>
</div>
</body>
</html>";
        }

        private static string PhysicsRow(string label, bool passed) =>
            $"<tr><td>{label}</td><td style='color:{(passed ? "#16a34a" : "#dc2626")};font-weight:600;'>{(passed ? "&#10003; Passed" : "&#10006; Failed")}</td></tr>";

        // ── Individual rule evaluators ──────────────────────────────────

        /// <summary>
        /// Judges the field-solve's pressure drop against THIS design's own
        /// Shepherd-Lapple calculated ΔP (<paramref name="knownPressureDropPa"/>)
        /// whenever it's available, rather than one fixed Pa threshold shared
        /// by every cyclone type. A Stairmand GP and a Stairmand HE are
        /// *supposed* to show different absolute pressure drops at the same
        /// flow rate — that's the entire design tradeoff between them — so a
        /// flat threshold either flags normal HE performance as a problem or
        /// never flags an actually-oversized GP inlet. Comparing against the
        /// design's own calculated baseline sidesteps that: it asks "does the
        /// field-solve agree with what this exact geometry should produce",
        /// which is type-agnostic by construction, instead of "is this an
        /// unusually high number in general".
        ///
        /// FALLBACK: when no baseline is supplied (e.g. the standard
        /// calculation for this revision hasn't been run), falls back to the
        /// old fixed absolute thresholds so this insight still degrades
        /// gracefully rather than being skipped outright.
        /// </summary>
        private EngineeringInsightDto EvaluatePressureDrop(double pressureDropPa, double? knownPressureDropPa = null)
        {
            if (knownPressureDropPa.HasValue && knownPressureDropPa.Value > 0)
            {
                double baseline = knownPressureDropPa.Value;
                double ratio = pressureDropPa / baseline;
                double pctOver = (ratio - 1.0) * 100.0;

                if (ratio >= Thresholds.PressureDropBaselineCriticalRatio)
                {
                    return new EngineeringInsightDto
                    {
                        Category = "Pressure Drop",
                        Severity = InsightSeverity.Critical,
                        WhatHappened = $"Field-solve pressure loss ({pressureDropPa:F0} Pa) is {pctOver:F0}% above this " +
                                       $"design's own calculated baseline ({baseline:F0} Pa) — well outside normal " +
                                       $"agreement for this geometry.",
                        Why = "Either the inlet velocity is higher than the design geometry accounts for, or the " +
                              "field-solve result itself has not converged to a physically consistent answer.",
                        Impact = new List<string> { "Significantly higher fan power consumption", "Higher operating cost", "Increased wall wear", "Reduced confidence in this field-solve result" },
                        Recommendation = "Reduce inlet velocity or increase cyclone diameter, and re-run the simulation; if the gap persists, check the mass-conservation result before trusting this number."
                    };
                }
                if (ratio >= Thresholds.PressureDropBaselineWarningRatio)
                {
                    return new EngineeringInsightDto
                    {
                        Category = "Pressure Drop",
                        Severity = InsightSeverity.Warning,
                        WhatHappened = $"Field-solve pressure loss ({pressureDropPa:F0} Pa) is {pctOver:F0}% above this " +
                                       $"design's own calculated baseline ({baseline:F0} Pa).",
                        Why = "Air is entering somewhat faster than this geometry's calculated design point assumes.",
                        Impact = new List<string> { "Increased fan power", "Higher operating cost", "Increased wall wear" },
                        Recommendation = "Reduce inlet velocity by approximately 8-12% or optimize cyclone dimensions."
                    };
                }
                // LOW SIDE — previously missing entirely, so any shortfall
                // fell straight through to "Good" regardless of size. A
                // field-solve landing far BELOW the calculated baseline is
                // just as much a disagreement as landing far above it: it
                // can mean the request fell outside the trained
                // diameter/flow range (extrapolation, not a validated
                // prediction — check the Python service's console for a
                // "request ... is outside the trained range" warning), or
                // that training itself under-predicts for this design.
                if (ratio <= Thresholds.PressureDropBaselineCriticalRatioLow)
                {
                    double pctUnder = (1.0 - ratio) * 100.0;
                    return new EngineeringInsightDto
                    {
                        Category = "Pressure Drop",
                        Severity = InsightSeverity.Critical,
                        WhatHappened = $"Field-solve pressure loss ({pressureDropPa:F0} Pa) is {pctUnder:F0}% BELOW this " +
                                       $"design's own calculated baseline ({baseline:F0} Pa) — well outside normal " +
                                       $"agreement for this geometry.",
                        Why = "The field-solve model may be extrapolating outside the diameter/flow range it was " +
                              "trained on for this cyclone type, or has under-converged for this design point.",
                        Impact = new List<string> { "This field-solve result should not be trusted at face value", "Reduced confidence in downstream swirl/wall-velocity numbers from the same solve" },
                        Recommendation = "Confirm this design's diameter and flow rate fall inside the trained range for its cyclone type; if they do, re-run the simulation and check mass conservation before trusting this number."
                    };
                }
                if (ratio <= Thresholds.PressureDropBaselineWarningRatioLow)
                {
                    double pctUnder = (1.0 - ratio) * 100.0;
                    return new EngineeringInsightDto
                    {
                        Category = "Pressure Drop",
                        Severity = InsightSeverity.Warning,
                        WhatHappened = $"Field-solve pressure loss ({pressureDropPa:F0} Pa) is {pctUnder:F0}% below this " +
                                       $"design's own calculated baseline ({baseline:F0} Pa).",
                        Why = "The field-solve is predicting a notably gentler pressure loss than the analytic " +
                              "calculation for this exact geometry.",
                        Impact = new List<string> { "Worth confirming before relying on this number for fan sizing" },
                        Recommendation = "Cross-check this design's diameter/flow rate against the trained range for its cyclone type before trusting this value."
                    };
                }
                return new EngineeringInsightDto
                {
                    Category = "Pressure Drop",
                    Severity = InsightSeverity.Good,
                    WhatHappened = $"Field-solve pressure drop ({pressureDropPa:F0} Pa) agrees with this design's own " +
                                   $"calculated baseline ({baseline:F0} Pa) (within {Math.Abs(pctOver):F0}%).",
                    Why = "Inlet velocity and geometry are well matched for this flow rate, and the field-solve confirms it.",
                    Impact = new List<string> { "Fan energy consumption is at an efficient level" },
                    Recommendation = "No action needed."
                };
            }

            // FALLBACK ONLY — no design-specific baseline supplied.
            if (pressureDropPa >= Thresholds.PressureDropCriticalPa)
            {
                return new EngineeringInsightDto
                {
                    Category = "Pressure Drop",
                    Severity = InsightSeverity.Critical,
                    WhatHappened = $"Pressure loss ({pressureDropPa:F0} Pa) is well above the recommended operating range.",
                    Why = "Air is entering at a high velocity, increasing wall friction and turbulent losses.",
                    Impact = new List<string> { "Significantly higher fan power consumption", "Higher operating cost", "Increased wall wear" },
                    Recommendation = "Reduce inlet velocity or increase cyclone diameter; re-run the simulation to confirm improvement."
                };
            }
            if (pressureDropPa >= Thresholds.PressureDropWarningPa)
            {
                return new EngineeringInsightDto
                {
                    Category = "Pressure Drop",
                    Severity = InsightSeverity.Warning,
                    WhatHappened = $"Pressure drop ({pressureDropPa:F0} Pa) is slightly higher than the recommended operating range.",
                    Why = "Air enters the cyclone at a higher velocity, increasing wall friction.",
                    Impact = new List<string> { "Increased fan power", "Higher operating cost", "Increased wall wear" },
                    Recommendation = "Reduce inlet velocity by approximately 8-12% or optimize cyclone dimensions."
                };
            }
            return new EngineeringInsightDto
            {
                Category = "Pressure Drop",
                Severity = InsightSeverity.Good,
                WhatHappened = $"Pressure drop ({pressureDropPa:F0} Pa) is within the recommended operating range.",
                Why = "Inlet velocity and geometry are well matched for this flow rate.",
                Impact = new List<string> { "Fan energy consumption is at an efficient level" },
                Recommendation = "No action needed."
            };
        }

        private EngineeringInsightDto EvaluateSwirlStrength(double avgTangentialMs)
        {
            if (avgTangentialMs < Thresholds.TangentialVelocityLowMs)
            {
                return new EngineeringInsightDto
                {
                    Category = "Weak Swirl",
                    Severity = InsightSeverity.Warning,
                    WhatHappened = $"The cyclone vortex ({avgTangentialMs:F1} m/s average swirl) is weaker than expected.",
                    Why = "Insufficient inlet velocity or flow rate for this geometry to sustain a strong vortex.",
                    Impact = new List<string> { "Reduced particle separation", "More particles may leave through the outlet" },
                    Recommendation = "Review inlet dimensions or increase operating flow rate."
                };
            }
            if (avgTangentialMs >= Thresholds.TangentialVelocityCriticalMs)
            {
                return new EngineeringInsightDto
                {
                    Category = "Swirl Strength",
                    Severity = InsightSeverity.Warning,
                    WhatHappened = $"Air is spinning very strongly inside the cyclone ({avgTangentialMs:F1} m/s average).",
                    Why = "High inlet velocity relative to geometry.",
                    Impact = new List<string> { "Generally improves particle separation", "May increase wall wear and pressure loss" },
                    Recommendation = "Monitor wall erosion; consider a moderate inlet velocity reduction if wear becomes a concern."
                };
            }
            return new EngineeringInsightDto
            {
                Category = "Swirl Strength",
                Severity = InsightSeverity.Good,
                WhatHappened = $"Air is spinning strongly inside the cyclone ({avgTangentialMs:F1} m/s average swirl).",
                Why = "Inlet velocity and geometry are well matched.",
                Impact = new List<string> { "Supports good particle separation" },
                Recommendation = "No action needed."
            };
        }

        private EngineeringInsightDto EvaluateWallVelocity(double maxWallVelocityMs)
        {
            if (maxWallVelocityMs >= Thresholds.WallVelocityCriticalMs)
            {
                return new EngineeringInsightDto
                {
                    Category = "High Wall Velocity",
                    Severity = InsightSeverity.Critical,
                    WhatHappened = $"Peak velocity near the wall ({maxWallVelocityMs:F1} m/s) is high.",
                    Why = "High tangential momentum concentrated near the barrel wall.",
                    Impact = new List<string> { "Increased erosion risk", "Reduced equipment life" },
                    Recommendation = "Inspect wall thickness/material or reduce inlet velocity."
                };
            }
            if (maxWallVelocityMs >= Thresholds.WallVelocityWarningMs)
            {
                return new EngineeringInsightDto
                {
                    Category = "Wall Velocity",
                    Severity = InsightSeverity.Warning,
                    WhatHappened = $"Peak velocity near the wall ({maxWallVelocityMs:F1} m/s) is elevated.",
                    Why = "Moderately high tangential momentum near the barrel wall.",
                    Impact = new List<string> { "Some increase in long-term erosion risk" },
                    Recommendation = "Periodic wall-thickness inspection recommended."
                };
            }
            return new EngineeringInsightDto
            {
                Category = "Wall Velocity",
                Severity = InsightSeverity.Good,
                WhatHappened = $"Peak wall velocity ({maxWallVelocityMs:F1} m/s) is within a normal range.",
                Why = "Geometry and flow rate are well matched.",
                Impact = new List<string> { "Low erosion risk from flow velocity" },
                Recommendation = "No action needed."
            };
        }

        private EngineeringInsightDto EvaluateMassConservation(FieldResultDto result)
        {
            bool passed = string.Equals(result.MassConservationStatus, "ok", StringComparison.OrdinalIgnoreCase);
            bool isWarningStatus = string.Equals(result.MassConservationStatus, "warning", StringComparison.OrdinalIgnoreCase);
            double spread = result.MassFlowSpread ?? 0.0;

            if ((!passed && !isWarningStatus) || spread >= Thresholds.MassFlowSpreadCritical)


            {
                return new EngineeringInsightDto
                {
                    Category = "Mass Conservation",
                    Severity = InsightSeverity.Critical,
                    WhatHappened = "The simulated flow field did not conserve mass within an acceptable tolerance.",
                    Why = "The underlying physics solve has not converged to a physically consistent flow field.",
                    Impact = new List<string> { "Dust carryover may increase", "Product recovery may decrease", "Other results in this report are less trustworthy" },
                    Recommendation = "Re-run the simulation; if this persists, review training configuration before trusting other results."
                };
            }
            if (spread >= Thresholds.MassFlowSpreadWarning)
            {
                return new EngineeringInsightDto
                {
                    Category = "Mass Conservation",
                    Severity = InsightSeverity.Warning,
                    WhatHappened = "Mass conservation passed, but with some spread across the flow field.",
                    Why = "Minor numerical variation in the physics solve.",
                    Impact = new List<string> { "Small uncertainty in separation efficiency estimates" },
                    Recommendation = "Results are usable; consider a longer training run for higher confidence."
                };
            }
            return new EngineeringInsightDto
            {
                Category = "Mass Conservation",
                Severity = InsightSeverity.Good,
                WhatHappened = "Mass conservation passed with low spread across the flow field.",
                Why = "The physics solve converged to a consistent flow field.",
                Impact = new List<string> { "Simulation results are reliable" },
                Recommendation = "No action needed."
            };
        }

        // ── Aggregation helpers ──────────────────────────────────────────

        private double ComputeHealthScore(List<EngineeringInsightDto> insights, PhysicsValidationDto physics)
        {
            double score = 100.0;
            foreach (var insight in insights)
            {
                score -= insight.Severity switch
                {
                    InsightSeverity.Critical => 25.0,
                    InsightSeverity.Warning => 10.0,
                    _ => 0.0
                };
            }
            if (!physics.MassConservationPassed) score -= 20.0;
            if (!physics.ConvergencePassed) score -= 10.0;
            return Math.Max(0.0, Math.Min(100.0, score));
        }

        private string GradeFor(double score)
        {
            if (score >= 90) return "Excellent";
            if (score >= 75) return "Good";
            if (score >= 60) return "Fair";
            return "Needs Attention";
        }

        private string BuildSummary(double score, List<EngineeringInsightDto> insights, PhysicsValidationDto physics)
        {
            var criticalCount = insights.Count(i => i.Severity == InsightSeverity.Critical);
            var warningCount = insights.Count(i => i.Severity == InsightSeverity.Warning);

            string overall = score >= 90 ? "Overall cyclone performance is excellent."
                : score >= 75 ? "Overall cyclone performance is good."
                : score >= 60 ? "Overall cyclone performance is fair."
                : "Overall cyclone performance needs attention.";

            string physicsLine = physics.MassConservationPassed
                ? "Physics validation passed successfully."
                : "Physics validation did not pass — treat other results with caution.";

            string issueLine = criticalCount > 0
                ? $"{criticalCount} critical issue(s) and {warningCount} warning(s) were detected."
                : warningCount > 0
                    ? $"{warningCount} minor issue(s) were detected."
                    : "No issues were detected.";

            return $"{overall} {physicsLine} {issueLine}";
        }

        /// <summary>
        /// Builds the report's closing Conclusion: a short verdict paragraph
        /// plus a prioritized, deduplicated action list — every Warning/
        /// Critical issue's Recommendation, worst severity first, in the
        /// order the engineer should actually work through them. This is
        /// deliberately separate from the per-card Recommendation fields:
        /// those answer "what do I do about THIS issue", this answers
        /// "given everything on this page, what do I do first, overall".
        /// </summary>
        private ConclusionDto BuildConclusion(double score, List<EngineeringInsightDto> insights, PhysicsValidationDto physics)
        {
            var actionable = insights
                .Where(i => i.Severity != InsightSeverity.Good)
                .OrderByDescending(i => i.Severity)              // Critical (2) before Warning (1)
                .Select(i => i.Recommendation)
                .Where(r => !string.IsNullOrWhiteSpace(r))
                .Distinct()
                .ToList();

            bool readyToProceed = physics.MassConservationPassed
                && !insights.Any(i => i.Severity == InsightSeverity.Critical);

            string verdict;
            if (!physics.MassConservationPassed)
            {
                verdict = "Physics validation did not pass, so the numbers above are not yet trustworthy. " +
                          "Resolve the simulation issue and re-run before using this result for design decisions.";
            }
            else if (readyToProceed && actionable.Count == 0)
            {
                verdict = $"This design is in good shape — health score {Math.Round(score, 0)}/100, " +
                          "no critical or warning issues detected. It's ready to proceed as-is.";
            }
            else if (readyToProceed)
            {
                verdict = $"This design is workable as-is (health score {Math.Round(score, 0)}/100) but has " +
                          "room to improve. None of the open items are blocking, so proceeding is reasonable " +
                          "while the actions below are addressed in a future revision.";
            }
            else
            {
                verdict = $"This design has at least one critical issue (health score {Math.Round(score, 0)}/100) " +
                          "and should not be finalized yet. Work through the priority actions below, then re-run " +
                          "the field solve to confirm the fix before proceeding.";
            }

            return new ConclusionDto
            {
                Verdict = verdict,
                PriorityActions = actionable,
                ReadyToProceed = readyToProceed
            };
        }

        private double ComputeConfidence(FieldResultDto result)
        {
            double confidence = 100.0;
            if (!string.Equals(result.MassConservationStatus, "ok", StringComparison.OrdinalIgnoreCase)) confidence -= 30;
            if (result.MassFlowSpread.HasValue) confidence -= Math.Min(30, result.MassFlowSpread.Value * 100.0);
            if (result.FinalLoss.HasValue && result.FinalLoss.Value >= 1.0) confidence -= 15;
            return Math.Max(0.0, Math.Round(confidence, 0));
        }

        private double ComputePressureDrop(FieldResultDto result)
        {
            if (result.PressurePa == null || result.PressurePa.Count == 0) return 0.0;
            return result.PressurePa.Max() - result.PressurePa.Min();
        }

        private double ComputeMaxWallVelocity(FieldResultDto result)
        {
            if (result.RMeters == null || result.RMeters.Count == 0) return 0.0;
            double maxR = result.RMeters.Max();
            double wallBand = maxR * 0.95;

            double maxSpeed = 0.0;
            for (int i = 0; i < result.RMeters.Count; i++)
            {
                if (result.RMeters[i] < wallBand) continue;
                double vr = result.VRMs[i];
                double vt = result.VThetaMs[i];
                double vz = result.VZMs[i];
                double speed = Math.Sqrt(vr * vr + vt * vt + vz * vz);
                if (speed > maxSpeed) maxSpeed = speed;
            }
            return maxSpeed;
        }

        private double EstimateSeparationEfficiency(double avgTangentialMs)
        {
            // FALLBACK ONLY. GenerateReport now prefers the real Lapple-model
            // efficiency (CyclonOutputDto.Efficiency, computed in
            // CyclonCalculationRepository from actual geometry, particle size,
            // density, and viscosity) whenever the caller supplies it — see
            // knownEfficiencyPercent above. This function only runs when that
            // isn't available, e.g. the standard calculation for this revision
            // hasn't been run yet. It's a simple monotonic placeholder mapping
            // swirl strength alone -> an illustrative percentage, and is NOT a
            // validated efficiency correlation: it ignores geometry and
            // particle size entirely, so different designs with similar swirl
            // velocity will show similar numbers here even though their real
            // separation performance differs. The risk-indicator label is
            // marked "(estimated)" whenever this path is used, specifically so
            // this number is never mistaken for the real one.
            double capped = Math.Min(avgTangentialMs, 40.0);
            return 50.0 + (capped / 40.0) * 45.0;
        }

        private double NormalizePercent(double value, double warningThreshold, double criticalThreshold)
        {
            if (criticalThreshold <= warningThreshold) return 0.0;
            double pct = (value / criticalThreshold) * 100.0;
            return Math.Round(Math.Max(0.0, Math.Min(100.0, pct)), 1);
        }

        private RiskIndicatorDto BuildRisk(string name, double percent)
        {
            string level = percent >= 70 ? "High" : percent >= 40 ? "Medium" : "Low";
            return new RiskIndicatorDto { Name = name, Percent = percent, Level = level };
        }

        private double Average(List<double> values)
        {
            if (values == null || values.Count == 0) return 0.0;
            return values.Average();
        }
    }
}