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
    /// </summary>
    public class EngineeringInsightRepository : IEngineeringInsight
    {
        private static class Thresholds
        {
            public const double PressureDropWarningPa = 1500.0;
            public const double PressureDropCriticalPa = 2500.0;

            public const double TangentialVelocityLowMs = 12.0;
            public const double TangentialVelocityWarningMs = 25.0;
            public const double TangentialVelocityCriticalMs = 40.0;

            public const double WallVelocityWarningMs = 30.0;
            public const double WallVelocityCriticalMs = 45.0;

            public const double MassFlowSpreadWarning = 0.15;
            public const double MassFlowSpreadCritical = 0.30;
        }

        public CycloneHealthReportDto GenerateReport(FieldResultDto result)
        {
            if (result == null) throw new ArgumentNullException(nameof(result));

            var insights = new List<EngineeringInsightDto>();

            double maxPressureDrop = ComputePressureDrop(result);
            double avgTangential = Average(result.VThetaMs);
            double maxWallVelocity = ComputeMaxWallVelocity(result);

            insights.Add(EvaluatePressureDrop(maxPressureDrop));
            insights.Add(EvaluateSwirlStrength(avgTangential));
            insights.Add(EvaluateWallVelocity(maxWallVelocity));
            insights.Add(EvaluateMassConservation(result));

            var physics = new PhysicsValidationDto
            {
                MassConservationPassed = string.Equals(result.MassConservationStatus, "ok", StringComparison.OrdinalIgnoreCase),
                BoundaryConditionsPassed = true,
                ConvergencePassed = result.FinalLoss.HasValue && result.FinalLoss.Value < 1.0,
                ConfidencePercent = ComputeConfidence(result)
            };

            var riskIndicators = new List<RiskIndicatorDto>
            {
                BuildRisk("Wear Risk", NormalizePercent(maxWallVelocity, Thresholds.WallVelocityWarningMs, Thresholds.WallVelocityCriticalMs)),
                BuildRisk("Energy Consumption", NormalizePercent(maxPressureDrop, Thresholds.PressureDropWarningPa, Thresholds.PressureDropCriticalPa)),
            };
            riskIndicators.Add(new RiskIndicatorDto
            {
                Name = "Separation Efficiency",
                Percent = Math.Round(EstimateSeparationEfficiency(avgTangential), 1),
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
                Summary = BuildSummary(healthScore, insights, physics)
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

        private EngineeringInsightDto EvaluatePressureDrop(double pressureDropPa)
        {
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
            double spread = result.MassFlowSpread ?? 0.0;

            if (!passed || spread >= Thresholds.MassFlowSpreadCritical)
            {
                return new EngineeringInsightDto
                {
                    Category = "Low Separation Efficiency",
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
                    Category = "Separation Efficiency",
                    Severity = InsightSeverity.Warning,
                    WhatHappened = "Mass conservation passed, but with some spread across the flow field.",
                    Why = "Minor numerical variation in the physics solve.",
                    Impact = new List<string> { "Small uncertainty in separation efficiency estimates" },
                    Recommendation = "Results are usable; consider a longer training run for higher confidence."
                };
            }
            return new EngineeringInsightDto
            {
                Category = "Separation Efficiency",
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
            // Simple monotonic placeholder mapping swirl strength -> an
            // illustrative efficiency percentage for the risk-indicator bar.
            // NOT a validated efficiency correlation — replace with the
            // Lapple-model efficiency output once wired to CyclonCalculationRepository.
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