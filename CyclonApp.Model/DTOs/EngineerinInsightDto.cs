using System;
using System.Collections.Generic;

namespace CyclonApp.Model.DTOs
{
    public enum InsightSeverity
    {
        Good,
        Warning,
        Critical
    }

    public class EngineeringInsightDto
    {
        public string Category { get; set; } = string.Empty;
        public InsightSeverity Severity { get; set; }
        public string WhatHappened { get; set; } = string.Empty;
        public string Why { get; set; } = string.Empty;
        public List<string> Impact { get; set; } = new();
        public string Recommendation { get; set; } = string.Empty;
    }

    public class RiskIndicatorDto
    {
        public string Name { get; set; } = string.Empty;
        public double Percent { get; set; }
        public string Level { get; set; } = string.Empty; // Low | Medium | High
    }

    public class PhysicsValidationDto
    {
        public bool MassConservationPassed { get; set; }
        public bool BoundaryConditionsPassed { get; set; }
        public bool ConvergencePassed { get; set; }
        public double ConfidencePercent { get; set; }
    }

    public class CycloneHealthReportDto
    {
        public double HealthScore { get; set; }
        public string Grade { get; set; } = string.Empty; // Excellent | Good | Fair | Needs Attention
        public PhysicsValidationDto PhysicsValidation { get; set; } = new();
        public List<EngineeringInsightDto> Insights { get; set; } = new();
        public List<RiskIndicatorDto> RiskIndicators { get; set; } = new();
        public string Summary { get; set; } = string.Empty;

        /// <summary>Closing verdict for the whole report — distinct from the
        /// per-issue Recommendation fields above. Answers "given everything
        /// on this page, what should I actually do next, in order?"</summary>
        public ConclusionDto Conclusion { get; set; } = new();
    }

    public class ConclusionDto
    {
        /// <summary>One short paragraph: overall verdict + whether the
        /// results here can be trusted as-is.</summary>
        public string Verdict { get; set; } = string.Empty;

        /// <summary>Deduplicated recommendations from every Warning/Critical
        /// issue, worst severity first, in the order they should be acted
        /// on. Empty when there's nothing to act on (all-Good report).</summary>
        public List<string> PriorityActions { get; set; } = new();

        /// <summary>Whether this design/result is ready to move forward on
        /// as-is (no Critical issues and physics validation passed).</summary>
        public bool ReadyToProceed { get; set; }
    }
}