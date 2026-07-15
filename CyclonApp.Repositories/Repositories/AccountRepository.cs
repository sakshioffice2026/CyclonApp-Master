using CyclonApp.Database;
using CyclonApp.Repositories.Contracts;
using Microsoft.EntityFrameworkCore;

namespace CyclonApp.Repositories.Repositories
{
    public class AccountRepository : IAccount
    {
        private readonly ApplicationDbContext _db;

        public AccountRepository(ApplicationDbContext db)
        {
            _db = db;
        }

        public async Task<AppUser?> GetUserByCredentialsAsync(string email, string encryptedPassword)
        {
            return await _db.Users
                .Include(u => u.Tenant)
                .Include(u => u.UserRole)
                .FirstOrDefaultAsync(u => u.Email == email && u.Password == encryptedPassword);
        }

        public async Task<bool> EmailExistsAsync(string email)
        {
            return await _db.Users.AnyAsync(u => u.Email == email);
        }

        public async Task<UserRole?> GetRoleByIdAsync(int roleId)
        {
            return await _db.UserRoles.FirstOrDefaultAsync(r => r.Id == roleId && r.IsActive);
        }

        public async Task<List<UserRole>> GetActiveRolesAsync()
        {
            return await _db.UserRoles
                .Where(r => r.IsActive)
                .OrderBy(r => r.Id)
                .ToListAsync();
        }

        public async Task<List<Tenant>> GetActiveTenantsAsync()
        {
            return await _db.Tenants
                .Where(t => t.IsActive)
                .OrderBy(t => t.Name)
                .ToListAsync();
        }

        public async Task<AppUser?> GetUserByIdAsync(int userId)
        {
            return await _db.Users
                .Include(u => u.Tenant)
                .FirstOrDefaultAsync(u => u.Id == userId);
        }

        public async Task<AppUser?> GetUserByEmailAsync(string email)
        {
            return await _db.Users.FirstOrDefaultAsync(u => u.Email == email);
        }

        public async Task CreateUserAsync(AppUser user)
        {
            _db.Users.Add(user);
            await _db.SaveChangesAsync();
        }

        public async Task UpdateLastLoginAsync(AppUser user)
        {
            user.LastLoginAt = DateTime.UtcNow;
            await _db.SaveChangesAsync();
        }

        public async Task UpdateProfileAsync(AppUser user)
        {
            await _db.SaveChangesAsync();
        }

        public async Task UpdatePasswordAsync(AppUser user, string encryptedPassword)
        {
            user.Password = encryptedPassword;
            await _db.SaveChangesAsync();
        }
    }
}
