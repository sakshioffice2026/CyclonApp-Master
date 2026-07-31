using CyclonApp.Database;
using CyclonApp.Model.ViewModel;

namespace CyclonApp.Repositories.Contracts
{
    public interface IDesignRepository
    {
        Task<List<CycloneDesign>> GetAllDesignsAsync();
        Task<CycloneDesign?> GetDesignByIdAsync(int id);
        Task<CycloneDesign?> GetDesignWithDetailsAsync(int id);
        Task<List<CycloneType>> GetActiveCycloneTypesAsync();
        Task<CycloneType?> GetCycloneTypeByIdAsync(int id);
        Task CreateDesignAsync(CycloneDesign design);
        Task UpdateDesignRevisionAsync(CycloneDesign design, DesignRevision revision);
        Task<DesignRevision?> GetRevisionByIdAsync(int id);
        Task<DesignRevision?> GetRevisionWithDetailsAsync(int id);
        Task SavePredictionAsync(DesignRevision revision, string predictionJson);
        Task SaveCfdImageAsync(DesignRevision revision, string pngUrl);
    }
}