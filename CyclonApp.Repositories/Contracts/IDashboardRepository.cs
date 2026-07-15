using CyclonApp.Database;
using CyclonApp.Model.DTOs;

namespace CyclonApp.Repositories.Contracts
{
    public interface IDashboardRepository
    {
        Task<DashboardDto> GetEngineerDashboardAsync(int tenantId, int userId);
        Task<DashboardDto> GetClientAdminDashboardAsync(int tenantId);
        Task<DashboardDto> GetViewerDashboardAsync(int tenantId);
        Task<DashboardDto> GetSuperAdminDashboardAsync();
    }

    
}
