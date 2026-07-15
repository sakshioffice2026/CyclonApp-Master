using CyclonApp.Database;
using CyclonApp.Repositories.Contracts;
using Microsoft.EntityFrameworkCore;

namespace CyclonApp.Repositories.Repositories
{
    public class AdminRepository : IAdminRepository
    {
        private readonly ApplicationDbContext _db;

        public AdminRepository(ApplicationDbContext db)
        {
            _db = db;
        }

        public async Task<List<AppUser>> GetAllUsersAsync()
        {
            return await _db.Users
                .Include(u => u.Tenant)
                .Include(u => u.UserRole)
                .OrderBy(u => u.TenantId)
                .ThenBy(u => u.FirstName)
                .ToListAsync();
        }

        public async Task<List<AppUser>> GetUsersByTenantAsync(int tenantId)
        {
            return await _db.Users
                .Include(u => u.Tenant)
                .Include(u => u.UserRole)
                .Where(u => u.TenantId == tenantId)
                .OrderBy(u => u.FirstName)
                .ToListAsync();
        }

        public async Task<List<Tenant>> GetAllTenantsAsync()
        {
            return await _db.Tenants
                .OrderBy(t => t.Name)
                .ToListAsync();
        }
    }
}
