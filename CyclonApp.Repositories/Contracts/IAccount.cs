using CyclonApp.Database;
using CyclonApp.Model.ViewModel;

namespace CyclonApp.Repositories.Contracts
{
    public interface IAccount
    {
        Task<AppUser?> GetUserByCredentialsAsync(string email, string encryptedPassword);
        Task<bool> EmailExistsAsync(string email);
        Task<UserRole?> GetRoleByIdAsync(int roleId);
        Task<List<UserRole>> GetActiveRolesAsync();
        Task<List<Tenant>> GetActiveTenantsAsync();
        Task<AppUser?> GetUserByIdAsync(int userId);
        Task<AppUser?> GetUserByEmailAsync(string email);
        Task CreateUserAsync(AppUser user);
        Task UpdateLastLoginAsync(AppUser user);
        Task UpdateProfileAsync(AppUser user);
        Task UpdatePasswordAsync(AppUser user, string encryptedPassword);
    }
}
