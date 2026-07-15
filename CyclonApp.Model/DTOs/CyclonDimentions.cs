using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CyclonApp.Model.DTOs
{
    public class CyclonDimensions
    {
        // Imperial (inches)
        public double BarrelDiameterIn { get; set; }
        public double InletHeightIn { get; set; }
        public double InletWidthIn { get; set; }
        public double BarrelHeightIn { get; set; }
        public double ConeHeightIn { get; set; }
        public double ExhaustDiaIn { get; set; }
        public double ExhaustLengthIn { get; set; }
        public double BottomOutletIn { get; set; }
        public double TotalHeightIn { get; set; }

        // Metric (mm)
        public double BarrelDiameterMm { get; set; }
        public double InletHeightMm { get; set; }
        public double InletWidthMm { get; set; }
        public double BarrelHeightMm { get; set; }
        public double ConeHeightMm { get; set; }
        public double ExhaustDiaMm { get; set; }
        public double ExhaustLengthMm { get; set; }
        public double BottomOutletMm { get; set; }
        public double TotalHeightMm { get; set; }

        // Barrel diameter in metres (for Three.js scene)
        public double BarrelDiameterM { get; set; }
    }
}
