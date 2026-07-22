using System;
using System.Collections.Generic;
using System.Linq;
using CyclonApp.Model.DTOs;
using CyclonApp.Repositories.Contracts;

namespace CyclonApp.Repositories.Repositories
{
    /// <summary>
    /// Rule-based engineering insight generator. No LLM/generative model
    /// anywhere in this class — every judgment below is a fixed rule
    /// evaluated against numbers already computed elsewhere.
    ///
    /// ROOT-CAUSE FIX (this revision): the previous version judged every
    /// cyclone type against the SAME fixed absolute thresholds (e.g. one
    /// pressure-drop band in Pa for all types/sizes) and never saw
    /// CycloneType at all. That's why different designs kept producing
    /// near-identical insight cards. This version:
    ///   1) takes CycloneTypeCode via EngineeringInsightRequestDto, and
    ///   2) judges pressure drop / swirl / wall velocity as DIMENSIONLESS
    ///      RATIOS against per-type reference values (Euler number for
    ///      pressure drop, velocity ratios for swirl/wall speed) instead
    ///      of absolute numbers — so the same rule scales correctly across
    ///      barrel sizes and flow rates, not just across types.
    ///
    /// PLACEHOLDER REFERENCE VALUES: the numbers in TypeReferences below
    /// are commonly-cited literature figures for each cyclone family's
    /// typical Euler number / velocity-ratio behavior (Lapple, Stairmand
    /// HE/GP, Swift HE), NOT values pulled from your specific client
    /// standard. Same caveat applies to the Reynolds-number regime check —
    /// it's adapted from general turbulent internal-flow guidance, not a
    /// cyclone-specific published cutoff. Have the client's engineer
    /// review/confirm all of these before this is relied on in production,
    /// same as the previous placeholder-threshold disclaimer.
    /// </summary>
    public class EngineeringInsightRepository : IEngineeringInsight
    {
        private class TypeReference
        {
            /// <summary>Reference Euler number: ΔP / (0.5 * rho * Vinlet^2).</summary>
            public double EulerNumber { get; set; }
            /// <summary>Reference ratio of average tangential velocity to inlet velocity.</summary>
            public double SwirlRatio { get; set; }
            /// <summary>Reference ratio of peak near-wall speed to inlet velocity.</summary>
            public double WallVelocityRatio { get; set; }
        }

        private static readonly Dictionary<string, TypeReference> TypeReferences = new(StringComparer.OrdinalIgnoreCase)
        {
            ["LAPPLE"] = new TypeReference { EulerNumber = 8.0, SwirlRatio = 0.65, WallVelocityRatio = 0.85 },
            ["STAIRMAND_HE"] = new TypeReference { EulerNumber = 6.4, SwirlRatio = 0.75, WallVelocityRatio = 0.90 },
            ["STAIRMAND_GP"] = new TypeReference { EulerNumber = 8.0, SwirlRatio = 0.65, WallVelocityRatio = 0.85 },
            ["SWIFT_HE"] = new TypeReference { EulerNumber = 9.24, SwirlRatio = 0.70, WallVelocityRatio = 0.88 },
        };

        // Same relative bands applied to every type's ratio-to-reference —
        // only the reference point differs per type above.
        private const double WarningRatioToReference = 1.15;
        private const double CriticalRatioToReference = 1.40;

        // Reynolds number regime bands (see class-level disclaimer).
        private const double ReynoldsCriticalLow = 4000.0;
        private const double ReynoldsWarningLow = 10000.0;

        // Generic collection-efficiency quality bands (see class-level
        // disclaimer) — used only when a standard-calculation efficiency
        // is actually available.
        private const double EfficiencyCriticalHigh = 70.0;
        private const double EfficiencyWarningHigh = 85.0;

        public CycloneHealthReportDto GenerateReport(EngineeringInsightRequestDto request)
        {
            if (request == null) throw new ArgumentNullException(nameof(request));
            var result = request.Result ?? throw new ArgumentException("request.Result is required", nameof(request));

            var reference = TypeReferences.TryGetValue(request.CycloneTypeCode ?? string.Empty, out var found)
                ? found
                : TypeReferences["LAPPLE"]; // generic fallback if type unrecognized/unconfigured

            double maxPressureDrop = ComputePressureDrop(result);
            double avgTangential = Average(result.VThetaMs);
            double maxWallVelocity = ComputeMaxWallVelocity(result);
            double barrelRadius = (result.RMeters != null && result.RMeters.Count > 0) ? result.RMeters.Max() : 0.0;
            double reynolds = ComputeReynolds(result, barrelRadius);

            var insights = new List<EngineeringInsightDto>
            {
                EvaluatePressureDrop(maxPressureDrop, result, reference),
                EvaluateSwirlStrength(avgTangential, result, reference),
                EvaluateWallVelocity(maxWallVelocity, result, reference),
                EvaluateMassConservation(result),
                EvaluateReynolds(reynolds)
            };

            if (request.StandardCalculation != null)
            {
                insights.Add(EvaluateEfficiency(request.StandardCalculation));
            }

            var physics = new PhysicsValidationDto
            {
                MassConservationPassed = !string.Equals(result.MassConservationStatus, "failed", StringComparison.OrdinalIgnoreCase),
                BoundaryConditionsPassed = true,
                ConvergencePassed = !result.FinalLoss.HasValue || result.FinalLoss.Value < 1.0,
                ConfidencePercent = ComputeConfidence(result)
            };

            var riskIndicators = new List<RiskIndicatorDto>
            {
                BuildRisk("Wear Risk", NormalizeRatioPercent(maxWallVelocity, result.VInletMs, reference.WallVelocityRatio)),
                BuildRisk("Energy Consumption", NormalizeEulerPercent(maxPressureDrop, result, reference.EulerNumber)),
            };

            bool usingRealEfficiency = request.StandardCalculation != null;
            double separationEfficiencyPercent = usingRealEfficiency
                ? request.StandardCalculation!.Efficiency
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

        // ── Ratio-based rule evaluators (type-aware) ─────────────────────

        private EngineeringInsightDto EvaluatePressureDrop(double pressureDropPa, FieldResultDto result, TypeReference reference)
        {
            double dynamicPressure = 0.5 * result.RhoKgm3 * result.VInletMs * result.VInletMs;
            double actualEuler = dynamicPressure > 0 ? pressureDropPa / dynamicPressure : 0.0;
            double ratioToReference = reference.EulerNumber > 0 ? actualEuler / reference.EulerNumber : 0.0;

            if (ratioToReference >= CriticalRatioToReference)
            {
                return new EngineeringInsightDto
                {
                    Category = "Pressure Drop",
                    Severity = InsightSeverity.Critical,
                    WhatHappened = $"Pressure loss ({pressureDropPa:F0} Pa, Euler number {actualEuler:F1}) is well above the typical reference for this cyclone type (Eu ≈ {reference.EulerNumber:F1}).",
                    Why = "Air is entering at a high velocity relative to this geometry, increasing wall friction and turbulent losses beyond what this cyclone family typically shows.",
                    Impact = new List<string> { "Significantly higher fan power consumption", "Higher operating cost", "Increased wall wear" },
                    Recommendation = "Reduce inlet velocity or increase cyclone diameter; re-run the simulation to confirm improvement."
                };
            }
            if (ratioToReference >= WarningRatioToReference)
            {
                return new EngineeringInsightDto
                {
                    Category = "Pressure Drop",
                    Severity = InsightSeverity.Warning,
                    WhatHappened = $"Pressure drop ({pressureDropPa:F0} Pa, Euler number {actualEuler:F1}) is somewhat higher than the typical reference for this type (Eu ≈ {reference.EulerNumber:F1}).",
                    Why = "Air enters the cyclone at a higher velocity relative to geometry than this family's typical design point.",
                    Impact = new List<string> { "Increased fan power", "Higher operating cost", "Increased wall wear" },
                    Recommendation = "Reduce inlet velocity by approximately 8-12% or optimize cyclone dimensions."
                };
            }
            return new EngineeringInsightDto
            {
                Category = "Pressure Drop",
                Severity = InsightSeverity.Good,
                WhatHappened = $"Pressure drop ({pressureDropPa:F0} Pa, Euler number {actualEuler:F1}) is within the typical range for this cyclone type.",
                Why = "Inlet velocity and geometry are well matched for this flow rate.",
                Impact = new List<string> { "Fan energy consumption is at an efficient level" },
                Recommendation = "No action needed."
            };
        }

        private EngineeringInsightDto EvaluateSwirlStrength(double avgTangentialMs, FieldResultDto result, TypeReference reference)
        {
            double ratio = result.VInletMs > 0 ? avgTangentialMs / result.VInletMs : 0.0;
            double ratioToReference = reference.SwirlRatio > 0 ? ratio / reference.SwirlRatio : 0.0;

            if (ratioToReference <= (1.0 / CriticalRatioToReference))
            {
                return new EngineeringInsightDto
                {
                    Category = "Weak Swirl",
                    Severity = InsightSeverity.Warning,
                    WhatHappened = $"The cyclone vortex ({avgTangentialMs:F1} m/s average, {ratio:F2}x inlet velocity) is weaker than typical for this cyclone type ({reference.SwirlRatio:F2}x expected).",
                    Why = "Insufficient inlet velocity or flow rate for this geometry to sustain a strong vortex.",
                    Impact = new List<string> { "Reduced particle separation", "More particles may leave through the outlet" },
                    Recommendation = "Review inlet dimensions or increase operating flow rate."
                };
            }
            if (ratioToReference >= CriticalRatioToReference)
            {
                return new EngineeringInsightDto
                {
                    Category = "Swirl Strength",
                    Severity = InsightSeverity.Warning,
                    WhatHappened = $"Air is spinning more strongly than typical for this type ({avgTangentialMs:F1} m/s, {ratio:F2}x inlet velocity vs {reference.SwirlRatio:F2}x expected).",
                    Why = "High inlet velocity relative to geometry.",
                    Impact = new List<string> { "Generally improves particle separation", "May increase wall wear and pressure loss" },
                    Recommendation = "Monitor wall erosion; consider a moderate inlet velocity reduction if wear becomes a concern."
                };
            }
            return new EngineeringInsightDto
            {
                Category = "Swirl Strength",
                Severity = InsightSeverity.Good,
                WhatHappened = $"Air is spinning strongly inside the cyclone ({avgTangentialMs:F1} m/s, {ratio:F2}x inlet velocity), in line with this type's typical range.",
                Why = "Inlet velocity and geometry are well matched.",
                Impact = new List<string> { "Supports good particle separation" },
                Recommendation = "No action needed."
            };
        }

        private EngineeringInsightDto EvaluateWallVelocity(double maxWallVelocityMs, FieldResultDto result, TypeReference reference)
        {
            double ratio = result.VInletMs > 0 ? maxWallVelocityMs / result.VInletMs : 0.0;
            double ratioToReference = reference.WallVelocityRatio > 0 ? ratio / reference.WallVelocityRatio : 0.0;

            if (ratioToReference >= CriticalRatioToReference)
            {
                return new EngineeringInsightDto
                {
                    Category = "High Wall Velocity",
                    Severity = InsightSeverity.Critical,
                    WhatHappened = $"Peak velocity near the wall ({maxWallVelocityMs:F1} m/s, {ratio:F2}x inlet velocity) is high relative to this type's typical range ({reference.WallVelocityRatio:F2}x expected).",
                    Why = "High tangential momentum concentrated near the barrel wall.",
                    Impact = new List<string> { "Increased erosion risk", "Reduced equipment life" },
                    Recommendation = "Inspect wall thickness/material or reduce inlet velocity."
                };
            }
            if (ratioToReference >= WarningRatioToReference)
            {
                return new EngineeringInsightDto
                {
                    Category = "Wall Velocity",
                    Severity = InsightSeverity.Warning,
                    WhatHappened = $"Peak velocity near the wall ({maxWallVelocityMs:F1} m/s, {ratio:F2}x inlet velocity) is elevated relative to this type's typical range.",
                    Why = "Moderately high tangential momentum near the barrel wall.",
                    Impact = new List<string> { "Some increase in long-term erosion risk" },
                    Recommendation = "Periodic wall-thickness inspection recommended."
                };
            }
            return new EngineeringInsightDto
            {
                Category = "Wall Velocity",
                Severity = InsightSeverity.Good,
                WhatHappened = $"Peak wall velocity ({maxWallVelocityMs:F1} m/s, {ratio:F2}x inlet velocity) is within a normal range for this type.",
                Why = "Geometry and flow rate are well matched.",
                Impact = new List<string> { "Low erosion risk from flow velocity" },
                Recommendation = "No action needed."
            };
        }

        private EngineeringInsightDto EvaluateReynolds(double reynolds)
        {
            if (reynolds <= 0)
            {
                return new EngineeringInsightDto
                {
                    Category = "Reynolds Number",
                    Severity = InsightSeverity.Warning,
                    WhatHappened = "Reynolds number could not be computed for this run (missing viscosity or geometry data).",
                    Why = "One or more required field-solve outputs (nu, barrel radius, inlet velocity) were not present.",
                    Impact = new List<string> { "Cannot confirm whether classical correlation assumptions apply here" },
                    Recommendation = "Re-run the field solve and confirm rhoKgm3/nuM2s/vInletMs are populated."
                };
            }
            if (reynolds < ReynoldsCriticalLow)
            {
                return new EngineeringInsightDto
                {
                    Category = "Reynolds Number",
                    Severity = InsightSeverity.Critical,
                    WhatHappened = $"Flow Reynolds number ({reynolds:F0}) is low — outside the turbulent regime these cyclone correlations assume.",
                    Why = "Flow rate is low relative to geometry, so the flow may be laminar or transitional.",
                    Impact = new List<string> { "Efficiency/pressure-drop correlations used elsewhere in this report may not be physically reliable" },
                    Recommendation = "Verify flow rate against this cyclone's intended operating range before trusting other results."
                };
            }
            if (reynolds < ReynoldsWarningLow)
            {
                return new EngineeringInsightDto
                {
                    Category = "Reynolds Number",
                    Severity = InsightSeverity.Warning,
                    WhatHappened = $"Flow Reynolds number ({reynolds:F0}) is in a transitional range.",
                    Why = "Flow rate is on the lower side for this geometry's typical operating range.",
                    Impact = new List<string> { "Some uncertainty in how well standard correlations apply" },
                    Recommendation = "Treat efficiency/pressure-drop figures with some caution; consider confirming against a higher flow rate test point."
                };
            }
            return new EngineeringInsightDto
            {
                Category = "Reynolds Number",
                Severity = InsightSeverity.Good,
                WhatHappened = $"Flow Reynolds number ({reynolds:F0}) is comfortably in the turbulent regime these correlations assume.",
                Why = "Flow rate and geometry combine to give fully turbulent flow.",
                Impact = new List<string> { "Standard correlation assumptions are reasonably well supported" },
                Recommendation = "No action needed."
            };
        }

        private EngineeringInsightDto EvaluateEfficiency(CyclonOutputDto standard)
        {
            string cutSizeNote = $" Cut size (d50) for this design is {standard.CutDiameterMicron:F1} microns.";
            if (standard.Efficiency < EfficiencyCriticalHigh)
            {
                return new EngineeringInsightDto
                {
                    Category = "Collection Efficiency",
                    Severity = InsightSeverity.Critical,
                    WhatHappened = $"Collection efficiency ({standard.Efficiency:F1}%) is low for the specified particle size.{cutSizeNote}",
                    Why = "Geometry and operating conditions are not separating this particle size effectively.",
                    Impact = new List<string> { "Significant product/dust carryover", "May not meet emission or recovery targets" },
                    Recommendation = "Reconsider cyclone type/geometry or operating flow rate for this particle size."
                };
            }
            if (standard.Efficiency < EfficiencyWarningHigh)
            {
                return new EngineeringInsightDto
                {
                    Category = "Collection Efficiency",
                    Severity = InsightSeverity.Warning,
                    WhatHappened = $"Collection efficiency ({standard.Efficiency:F1}%) is moderate for the specified particle size.{cutSizeNote}",
                    Why = "Design is functional but leaves room for improvement at this particle size.",
                    Impact = new List<string> { "Some product/dust carryover" },
                    Recommendation = "Consider a higher-efficiency cyclone type if this particle size is critical to recover."
                };
            }
            return new EngineeringInsightDto
            {
                Category = "Collection Efficiency",
                Severity = InsightSeverity.Good,
                WhatHappened = $"Collection efficiency ({standard.Efficiency:F1}%) is good for the specified particle size.{cutSizeNote}",
                Why = "Geometry and operating conditions suit this particle size well.",
                Impact = new List<string> { "Low expected product/dust carryover" },
                Recommendation = "No action needed."
            };
        }

        private EngineeringInsightDto EvaluateMassConservation(FieldResultDto result)
        {
            double spread = result.MassFlowSpread ?? 0.0;
            string spreadPercentText = (spread * 100.0).ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "%";

            bool isOk = string.Equals(result.MassConservationStatus, "ok", StringComparison.OrdinalIgnoreCase);
            bool isWarning = string.Equals(result.MassConservationStatus, "warning", StringComparison.OrdinalIgnoreCase);

            if (isOk)
            {
                return new EngineeringInsightDto
                {
                    Category = "Mass Conservation",
                    Severity = InsightSeverity.Good,
                    WhatHappened = $"Mass conservation passed with low spread across the flow field ({spreadPercentText} variation in volumetric flow across the barrel mid-section).",
                    Why = "The physics solve converged to a consistent flow field.",
                    Impact = new List<string> { "Simulation results are reliable" },
                    Recommendation = "No action needed."
                };
            }
            if (isWarning)
            {
                return new EngineeringInsightDto
                {
                    Category = "Mass Conservation",
                    Severity = InsightSeverity.Warning,
                    WhatHappened = $"Mass conservation passed, but with some spread across the flow field ({spreadPercentText} variation in volumetric flow across the barrel mid-section).",
                    Why = "Minor numerical variation in the physics solve.",
                    Impact = new List<string> { "Small uncertainty in separation efficiency estimates" },
                    Recommendation = "Results are usable; consider a longer training run for higher confidence."
                };
            }
            return new EngineeringInsightDto
            {
                Category = "Mass Conservation",
                Severity = InsightSeverity.Critical,
                WhatHappened = $"The simulated flow field did not conserve mass within an acceptable tolerance (volumetric flow varied by {spreadPercentText} across the barrel mid-section).",
                Why = "The underlying physics solve has not converged to a physically consistent flow field.",
                Impact = new List<string> { "Dust carryover may increase", "Product recovery may decrease", "Other results in this report are less trustworthy" },
                Recommendation = "Re-run the simulation; if this persists, review training configuration before trusting other results."
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

        private ConclusionDto BuildConclusion(double score, List<EngineeringInsightDto> insights, PhysicsValidationDto physics)
        {
            var actionable = insights
                .Where(i => i.Severity != InsightSeverity.Good)
                .OrderByDescending(i => i.Severity)
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

        private double ComputeReynolds(FieldResultDto result, double barrelRadius)
        {
            if (result.NuM2s <= 0 || barrelRadius <= 0 || result.VInletMs <= 0) return 0.0;
            double barrelDiameter = 2.0 * barrelRadius;
            return result.VInletMs * barrelDiameter / result.NuM2s;
        }

        private double EstimateSeparationEfficiency(double avgTangentialMs)
        {
            // FALLBACK ONLY — used only when request.StandardCalculation is
            // null (standard calc hasn't been run yet for this revision).
            // Swirl-only placeholder; ignores geometry and particle size,
            // so different designs with similar swirl show similar numbers
            // here even though their real separation performance differs.
            double capped = Math.Min(avgTangentialMs, 40.0);
            return 50.0 + (capped / 40.0) * 45.0;
        }

        private double NormalizeEulerPercent(double pressureDropPa, FieldResultDto result, double referenceEuler)
        {
            double dynamicPressure = 0.5 * result.RhoKgm3 * result.VInletMs * result.VInletMs;
            double actualEuler = dynamicPressure > 0 ? pressureDropPa / dynamicPressure : 0.0;
            double ratio = referenceEuler > 0 ? actualEuler / referenceEuler : 0.0;
            return Math.Round(Math.Max(0.0, Math.Min(100.0, ratio / CriticalRatioToReference * 100.0)), 1);
        }

        private double NormalizeRatioPercent(double value, double inletVelocity, double referenceRatio)
        {
            double ratio = inletVelocity > 0 ? value / inletVelocity : 0.0;
            double ratioToReference = referenceRatio > 0 ? ratio / referenceRatio : 0.0;
            return Math.Round(Math.Max(0.0, Math.Min(100.0, ratioToReference / CriticalRatioToReference * 100.0)), 1);
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