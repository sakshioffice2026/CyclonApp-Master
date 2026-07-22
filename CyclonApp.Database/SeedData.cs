using CyclonApp.Utilities;
using Microsoft.EntityFrameworkCore;
using System.Text.Json;


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

            await SeedCycloneTypesAsync(db);
        }

        // ── CYCLONE TYPES ────────────────────────────────────────────────
        // Field names in the JSON below must match CyclonApp.Model.DTOs.
        // CyclonTypeRatios exactly (InletHeightRatio, InletWidthRatio,
        // BarrelHeightRatio, ConeHeightRatio, OutletDiamRatio,
        // BottomOutletRatio, ExhaustLengthRatio) — CyclonCalculationRepository
        // .ParseRatios deserializes straight into that DTO.
        //
        // These values must stay in lockstep with LAPPLE_RATIOS /
        // STAIRMAND_RATIOS in CyclonApp.Business/CyclonePredictionService/
        // field_train.py — that file trains the ONNX field model against
        // whichever ratios are used there, and this seed defines the ratios
        // the deterministic Lapple/Shepherd-Lapple calculation
        // (CyclonCalculationRepository) uses. If the two drift apart, the
        // analytic calculation and the trained field model are silently
        // describing two different physical shapes for the same "type".
        private static async Task SeedCycloneTypesAsync(ApplicationDbContext db)
        {
            if (!await db.CycloneTypes.AnyAsync(t => t.Code == "LAPPLE"))
            {
                db.CycloneTypes.Add(new CycloneType
                {
                    Code = "LAPPLE",
                    Name = "Lapple (General Purpose)",
                    Description = "Standard general-purpose cyclone proportions with a lower " +
                                   "pressure drop than high-efficiency designs; a common default " +
                                   "for bulk/coarse dust collection.",
                    DimensionRatiosJson = BuildRatiosJson(
                        inletHeightRatio: 0.50,
                        inletWidthRatio: 0.25,
                        barrelHeightRatio: 2.00,
                        coneHeightRatio: 2.00,
                        outletDiamRatio: 0.50,
                        bottomOutletRatio: 0.25,
                        exhaustLengthRatio: 0.625),
                    DefaultEffectiveTurns = 6,
                    ApplicationNote = "General industrial dust collection where a lower pressure " +
                                       "drop matters more than fine-particle capture.",
                    IsActive = true,
                    SortOrder = 1,
                });
            }

            if (!await db.CycloneTypes.AnyAsync(t => t.Code == "STAIRMAND"))
            {
                db.CycloneTypes.Add(new CycloneType
                {
                    Code = "STAIRMAND",
                    Name = "Stairmand High Efficiency (HE)",
                    Description = "C. J. Stairmand's high-efficiency proportions, optimized for " +
                                   "fine-particle collection via a strong centrifugal field, at the " +
                                   "cost of a higher pressure drop than general-purpose designs. " +
                                   "Widely used as a CFD/research benchmark.",
                    DimensionRatiosJson = BuildRatiosJson(
                        inletHeightRatio: 0.50,
                        inletWidthRatio: 0.20,
                        barrelHeightRatio: 1.50,
                        coneHeightRatio: 2.50,
                        outletDiamRatio: 0.50,
                        bottomOutletRatio: 0.375,
                        exhaustLengthRatio: 0.50),
                    DefaultEffectiveTurns = 6,
                    ApplicationNote = "Fine dust collection where efficiency matters more than fan " +
                                       "power: cement, power stations, metal/chemical processing, " +
                                       "boiler fly ash.",
                    IsActive = true,
                    SortOrder = 2,
                });
            }

            if (!await db.CycloneTypes.AnyAsync(t => t.Code == "STAIRMAND_GP"))
            {
                db.CycloneTypes.Add(new CycloneType
                {
                    Code = "STAIRMAND_GP",
                    Name = "Stairmand General Purpose (GP)",
                    Description = "Lower-pressure-drop variant of Stairmand's proportions, " +
                                   "using a wider inlet than the High Efficiency (HE) design " +
                                   "to trade some fine-particle collection efficiency for " +
                                   "reduced fan power and operating cost.",
                    DimensionRatiosJson = BuildRatiosJson(
                        inletHeightRatio: 0.50,
                        inletWidthRatio: 0.25,
                        barrelHeightRatio: 1.50,
                        coneHeightRatio: 2.50,
                        outletDiamRatio: 0.50,
                        bottomOutletRatio: 0.375,
                        exhaustLengthRatio: 0.50),
                    DefaultEffectiveTurns = 6,
                    ApplicationNote = "Woodworking dust collection, grain handling, foundries, " +
                                       "coarser cement handling, material transfer systems, " +
                                       "general industrial ventilation, and pre-cleaners ahead " +
                                       "of bag filters or scrubbers.",
                    IsActive = true,
                    SortOrder = 3,
                });
            }



            await db.SaveChangesAsync();
        }



        private static string BuildRatiosJson(
            double inletHeightRatio,
            double inletWidthRatio,
            double barrelHeightRatio,
            double coneHeightRatio,
            double outletDiamRatio,
            double bottomOutletRatio,
            double exhaustLengthRatio)
        {
            return JsonSerializer.Serialize(new
            {
                InletHeightRatio = inletHeightRatio,
                InletWidthRatio = inletWidthRatio,
                BarrelHeightRatio = barrelHeightRatio,
                ConeHeightRatio = coneHeightRatio,
                OutletDiamRatio = outletDiamRatio,
                BottomOutletRatio = bottomOutletRatio,
                ExhaustLengthRatio = exhaustLengthRatio,
            });
        }
    }
}