using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CyclonApp.Model.DTOs
{
    public class CyclonOutputDto
    {
        public CyclonDimensions Dimensions { get; set; } = new();
        public double CutDiameterMicron { get; set; }
        public double Efficiency { get; set; }   // % at input particle size
        public double PressureDropPa { get; set; }
        public double PressureDropMmWc { get; set; }
        public double PressureDropInWc { get; set; }
        public double FlowRateM3hr { get; set; }
        public double FlowRateM3s { get; set; }
        public double InletVelocityMs { get; set; }
        public double GasViscosityKgms { get; set; }
        public double GasDensityKgm3 { get; set; }
        public double InletAreaM2 { get; set; }
        public List<GradeEfficiencyPoint> GradeEfficiencyCurve { get; set; } = new();
    }
}
