using System;
using System.Collections.Generic;

namespace CyclonApp.Model.DTOs
{
    public class CyclonePredictionDto
    {
        public double Efficiency { get; set; }              // % — predicted collection efficiency
        public double PressureDropPa { get; set; }           // predicted pressure drop, Pascals

        public double PhysicsResidual { get; set; }          // how far the prediction sits from
                                                             // the Lapple-model relationships
                                                             // (0 = perfectly consistent)

        public bool IsWithinTrustedRange { get; set; }        // false = extrapolating beyond
                                                              // the formulas/data this model
                                                              // has seen — flag for the engineer

        public string? Notes { get; set; }                    // human-readable caveat, e.g.
                                                              // "Particle size below 5 microns —
                                                              //  outside standard correlation range"

        public DateTime PredictedAtUtc { get; set; } = DateTime.UtcNow;
    }
}