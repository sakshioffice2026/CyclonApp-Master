using System.Security.Claims;
using CyclonApp.Database;
using CyclonApp.Repositories.Contracts;
using Microsoft.AspNetCore.Http;
using Microsoft.EntityFrameworkCore;

namespace CyclonApp.Repositories.Repositories
{
    public class TenantRepository : ITenant
    {
        private readonly IHttpContextAccessor _httpContextAccessor;
        private readonly ApplicationDbContext _db;

        public TenantRepository(IHttpContextAccessor httpContextAccessor, ApplicationDbContext db)
        {
            _httpContextAccessor = httpContextAccessor;
            _db = db;
        }

        public int CurrentTenantId
        {
            get
            {
                var val = _httpContextAccessor.HttpContext?
                    .User.FindFirstValue("TenantId");
                return int.TryParse(val, out var id) ? id : 0;
            }
        }

        public async Task<Tenant?> GetCurrentTenantAsync()
        {
            var tenantId = CurrentTenantId;
            if (tenantId == 0) return null;
            return await _db.Tenants.FirstOrDefaultAsync(t => t.Id == tenantId);
        }
    }
}