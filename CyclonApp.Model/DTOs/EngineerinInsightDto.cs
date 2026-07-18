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
    }
}