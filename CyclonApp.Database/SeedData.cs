using CyclonApp.Utilities;
using Microsoft.EntityFrameworkCore;


namespace CyclonApp.Database
{
    public static class SeedData
    {
        public const string RoleSuperAdmin = "SuperAdmin";
        public const string RoleClientAdmin = "ClientAdmin";
        public const string RoleEngineer = "Engineer";
        public const string RoleViewer = "Viewer";

        public static async Task InitializeAsync(ApplicationDbContext db)
        {
            // Seed default tenant
            var tenant = await db.Tenants.FirstOrDefaultAsync(t => t.Slug == "default");
            if (tenant == null)
            {
                tenant = new Tenant
                {
                    Name = "Platform Administration",
                    Slug = "default",
                    IsActive = true,
                    CreatedAt = DateTime.UtcNow
                };
                db.Tenants.Add(tenant);
                await db.SaveChangesAsync();
            }

            if (!await db.Users.AnyAsync(u => u.Email == "admin@cyclone.com"))
            {
                db.Users.Add(new AppUser
                {
                    Email = "admin@cyclone.com",
                    Password = Encryp_Decrypt.Encryptdata("Admin@123"),
                    FirstName = "Super",
                    LastName = "Admin",
                    Role = RoleSuperAdmin,
                    TenantId = tenant.Id,
                    IsActive = true,
                    CreatedAt = DateTime.UtcNow
                });

                await db.SaveChangesAsync();
            }

        }
    }
}