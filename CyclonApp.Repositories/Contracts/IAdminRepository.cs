using CyclonApp.Database;

namespace CyclonApp.Repositories.Contracts
{
    public interface IAdminRepository
    {
        Task<List<AppUser>> GetAllUsersAsync();
        Task<List<AppUser>> GetUsersByTenantAsync(int tenantId);
        Task<List<Tenant>> GetAllTenantsAsync();
    }
}
