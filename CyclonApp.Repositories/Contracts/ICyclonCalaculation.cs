using CyclonApp.Database;
using CyclonApp.Model.DTOs;

public interface ICyclonCalculation
{
    CyclonOutputDto Calculate(DesignRevision input, CyclonTypeRatios ratios);
    double ComputeViscosity(double tempC, string gasType);
    CyclonTypeRatios? ParseRatios(string json);
}