using CyclonApp.Model.DTOs;

namespace CyclonApp.Repositories.Contracts
{
    public interface IEngineeringInsight
    {
        CycloneHealthReportDto GenerateReport(FieldResultDto result);
    }
}