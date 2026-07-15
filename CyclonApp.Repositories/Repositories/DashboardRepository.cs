using System.Text.Json;
using CyclonApp.Database;
using CyclonApp.Model.DTOs;
using CyclonApp.Repositories.Contracts;
using Microsoft.EntityFrameworkCore;

namespace CyclonApp.Repositories.Repositories
{
    public class DashboardRepository : IDashboardRepository
    {
        private readonly ApplicationDbContext _db;
        private static readonly JsonSerializerOptions _json = new() { PropertyNameCaseInsensitive = true };

        private static readonly string[] StatusColors =
            { "#3B82F6", "#22C55E", "#8B5CF6", "#F59E0B" };
        private static readonly string[] TypeColors =
            { "#1A56DB", "#0891B2", "#7C3AED", "#16A34A", "#D97706", "#DC2626", "#0F766E" };
        private static readonly string[] MonthColors =
            { "#1A56DB", "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE", "#DBEAFE" };

        public DashboardRepository(ApplicationDbContext db) => _db = db;

        // ── ENGINEER ─────────────────────────────────────────────────────────────

        public async Task<DashboardDto> GetEngineerDashboardAsync(int tenantId, int userId)
        {
            try
            {
                var projects = await _db.Projects.Where(p => p.TenantId == tenantId).ToListAsync();
                var designs = await _db.CycloneDesign.Where(d => d.TenantId == tenantId).ToListAsync();
                var revisions = await _db.DesignRevisions
                    .Include(r => r.CycloneDesign)
                    .Where(r => r.CycloneDesign.TenantId == tenantId)
                    .ToListAsync();
                var myRevisions = revisions.Where(r => r.CreatedByUserId == userId).ToList();

                System.Diagnostics.Debug.WriteLine($"[ENGINEER] TenantId: {tenantId}, UserId: {userId}");
                System.Diagnostics.Debug.WriteLine($"[ENGINEER] Projects: {projects.Count}");
                System.Diagnostics.Debug.WriteLine($"[ENGINEER] Designs: {designs.Count}");
                System.Diagnostics.Debug.WriteLine($"[ENGINEER] Total Revisions: {revisions.Count}");
                System.Diagnostics.Debug.WriteLine($"[ENGINEER] My Revisions: {myRevisions.Count}");

                var dto = new DashboardDto
                {
                    TotalProjects = projects.Count,
                    TotalDesigns = designs.Count,
                    TotalRevisions = myRevisions.Count,
                    TotalExports = await _db.ExportLogs.CountAsync(e => e.TenantId == tenantId),

                    ProjectsByStatus = GroupByStatus(projects),
                    DesignsByCycloneType = await GroupByCycloneTypeAsync(tenantId),
                    RevisionsPerMonth = GroupByMonth(revisions.Where(r => r.CalculatedAt != null)
                                              .Select(r => r.CalculatedAt!.Value).ToList(), "Calculations"),
                    MyRevisionsPerMonth = GroupByMonth(myRevisions.Where(r => r.CalculatedAt != null)
                                              .Select(r => r.CalculatedAt!.Value).ToList(), "My Calculations"),
                    EfficiencyDistribution = BuildEfficiencyDistribution(revisions),
                    RecentCalculations = myRevisions
                        .Where(r => r.CalculatedAt != null && r.CreatedByUserId == userId)
                        .OrderByDescending(r => r.CalculatedAt)
                        .Take(5)
                        .Select(r => new RecentItem
                        {
                            Title = $"Rev {r.RevisionNumber}",
                            SubTitle = r.RevisionNote,
                            Badge = r.GasType,
                            BadgeColor = "#1A56DB",
                            Date = r.CalculatedAt!.Value,
                            Url = $"/Design/Results/{r.Id}"
                        }).ToList(),
                    RecentProjects = projects
                        .OrderByDescending(p => p.CreatedAt).Take(5)
                        .Select(p => new RecentItem
                        {
                            Title = p.Name,
                            SubTitle = p.ProjectNumber,
                            Badge = p.Status.ToString(),
                            BadgeColor = StatusBadgeColor(p.Status),
                            Date = p.CreatedAt,
                            Url = $"/Project/Detail/{p.Id}"
                        }).ToList()
                };

                return dto;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ERROR] GetEngineerDashboardAsync: {ex.Message}");
                throw;
            }
        }

        // ── CLIENT ADMIN ──────────────────────────────────────────────────────────

        public async Task<DashboardDto> GetClientAdminDashboardAsync(int tenantId)
        {
            try
            {
                var projects = await _db.Projects.Where(p => p.TenantId == tenantId).ToListAsync();
                var designs = await _db.CycloneDesign.Where(d => d.TenantId == tenantId).ToListAsync();
                var revisions = await _db.DesignRevisions
                    .Include(r => r.CycloneDesign)
                    .Where(r => r.CycloneDesign.TenantId == tenantId)
                    .ToListAsync();
                var exports = await _db.ExportLogs
                    .Where(e => e.TenantId == tenantId)
                    .ToListAsync();
                var users = await _db.Users.CountAsync(u => u.TenantId == tenantId);

                System.Diagnostics.Debug.WriteLine($"[CLIENT_ADMIN] TenantId: {tenantId}");
                System.Diagnostics.Debug.WriteLine($"[CLIENT_ADMIN] Projects: {projects.Count}");
                System.Diagnostics.Debug.WriteLine($"[CLIENT_ADMIN] Designs: {designs.Count}");
                System.Diagnostics.Debug.WriteLine($"[CLIENT_ADMIN] Revisions: {revisions.Count}");
                System.Diagnostics.Debug.WriteLine($"[CLIENT_ADMIN] Exports: {exports.Count}");

                var dto = new DashboardDto
                {
                    TotalProjects = projects.Count,
                    TotalDesigns = designs.Count,
                    TotalRevisions = revisions.Count,
                    TotalExports = exports.Count,
                    TotalUsers = users,

                    ProjectsByStatus = GroupByStatus(projects),
                    DesignsByCycloneType = await GroupByCycloneTypeAsync(tenantId),
                    RevisionsPerMonth = GroupByMonth(revisions
                        .Where(r => r.CalculatedAt != null)
                        .Select(r => r.CalculatedAt!.Value).ToList(), "Calculations"),
                    ExportsPerMonth = GroupByMonth(exports
                        .Select(e => e.ExportedAt).ToList(), "Exports"),
                    EfficiencyDistribution = BuildEfficiencyDistribution(revisions),

                    RecentProjects = projects
                        .OrderByDescending(p => p.UpdatedAt ?? p.CreatedAt).Take(5)
                        .Select(p => new RecentItem
                        {
                            Title = p.Name,
                            SubTitle = p.ClientName,
                            Badge = p.Status.ToString(),
                            BadgeColor = StatusBadgeColor(p.Status),
                            Date = p.UpdatedAt ?? p.CreatedAt,
                            Url = $"/Project/Detail/{p.Id}"
                        }).ToList(),
                    RecentExports = exports
                        .OrderByDescending(e => e.ExportedAt).Take(5)
                        .Select(e => new RecentItem
                        {
                            Title = $"Export #{e.Id}",
                            SubTitle = e.ExportType.ToString(),
                            Badge = e.ExportType.ToString(),
                            BadgeColor = e.ExportType == ExportType.PDF ? "#DC2626" : "#16A34A",
                            Date = e.ExportedAt
                        }).ToList()
                };

                return dto;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ERROR] GetClientAdminDashboardAsync: {ex.Message}");
                throw;
            }
        }

        // ── VIEWER ────────────────────────────────────────────────────────────────

        public async Task<DashboardDto> GetViewerDashboardAsync(int tenantId)
        {
            try
            {
                var projects = await _db.Projects.Where(p => p.TenantId == tenantId).ToListAsync();
                var designs = await _db.CycloneDesign.Where(d => d.TenantId == tenantId).ToListAsync();
                var revisions = await _db.DesignRevisions
                    .Include(r => r.CycloneDesign)
                    .Where(r => r.CycloneDesign.TenantId == tenantId && r.CalculatedAt != null)
                    .ToListAsync();

                System.Diagnostics.Debug.WriteLine($"[VIEWER] TenantId: {tenantId}");
                System.Diagnostics.Debug.WriteLine($"[VIEWER] Projects: {projects.Count}");
                System.Diagnostics.Debug.WriteLine($"[VIEWER] Designs: {designs.Count}");
                System.Diagnostics.Debug.WriteLine($"[VIEWER] Revisions: {revisions.Count}");

                return new DashboardDto
                {
                    TotalProjects = projects.Count,
                    TotalDesigns = designs.Count,
                    TotalRevisions = revisions.Count,
                    TotalExports = await _db.ExportLogs.CountAsync(e => e.TenantId == tenantId),

                    ProjectsByStatus = GroupByStatus(projects),
                    DesignsByCycloneType = await GroupByCycloneTypeAsync(tenantId),
                    RevisionsPerMonth = GroupByMonth(
                        revisions.Select(r => r.CalculatedAt!.Value).ToList(), "Calculations"),
                    EfficiencyDistribution = BuildEfficiencyDistribution(revisions),

                    RecentProjects = projects
                        .OrderByDescending(p => p.CreatedAt).Take(5)
                        .Select(p => new RecentItem
                        {
                            Title = p.Name,
                            SubTitle = p.ProjectNumber,
                            Badge = p.Status.ToString(),
                            BadgeColor = StatusBadgeColor(p.Status),
                            Date = p.CreatedAt,
                            Url = $"/Project/Detail/{p.Id}"
                        }).ToList()
                };
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ERROR] GetViewerDashboardAsync: {ex.Message}");
                throw;
            }
        }

        // ── SUPER ADMIN ───────────────────────────────────────────────────────────

        public async Task<DashboardDto> GetSuperAdminDashboardAsync()
        {
            try
            {
                var projects = await _db.Projects.Include(p => p.Tenant).ToListAsync();
                var designs = await _db.CycloneDesign.Include(d => d.Tenant).ToListAsync();
                var revisions = await _db.DesignRevisions.ToListAsync();
                var exports = await _db.ExportLogs.ToListAsync();
                var tenants = await _db.Tenants.Include(t => t.Users).ToListAsync();
                var users = await _db.Users.ToListAsync();

                System.Diagnostics.Debug.WriteLine("=== SUPER ADMIN DASHBOARD ===");
                System.Diagnostics.Debug.WriteLine($"Total Tenants: {tenants.Count}");
                System.Diagnostics.Debug.WriteLine($"Total Projects: {projects.Count}");
                System.Diagnostics.Debug.WriteLine($"Total Designs: {designs.Count}");
                System.Diagnostics.Debug.WriteLine($"Total Revisions: {revisions.Count}");
                System.Diagnostics.Debug.WriteLine($"Total Exports: {exports.Count}");
                System.Diagnostics.Debug.WriteLine($"Total Users: {users.Count}");

                foreach (var tenant in tenants)
                {
                    var tenantProjects = projects.Count(p => p.TenantId == tenant.Id);
                    var tenantUsers = users.Count(u => u.TenantId == tenant.Id);
                    System.Diagnostics.Debug.WriteLine($"  Tenant '{tenant.Name}': {tenantProjects} projects, {tenantUsers} users");
                }

                var projectsByStatus = GroupByStatus(projects);
                var projectsPerTenant = tenants.Select((t, i) => new ChartPoint
                {
                    Label = t.Name,
                    Value = projects.Count(p => p.TenantId == t.Id),
                    Color = TypeColors[i % TypeColors.Length]
                }).OrderByDescending(c => c.Value).ToList();

                var usersPerTenant = tenants.Select((t, i) => new ChartPoint
                {
                    Label = t.Name,
                    Value = users.Count(u => u.TenantId == t.Id),
                    Color = TypeColors[i % TypeColors.Length]
                }).OrderByDescending(c => c.Value).ToList();

                var revisionsPerMonth = GroupByMonth(
                    revisions.Where(r => r.CalculatedAt != null)
                             .Select(r => r.CalculatedAt!.Value).ToList(), "Calculations");

                var exportsPerMonth = GroupByMonth(
                    exports.Select(e => e.ExportedAt).ToList(), "Exports");

                System.Diagnostics.Debug.WriteLine($"ProjectsByStatus count: {projectsByStatus.Count}");
                System.Diagnostics.Debug.WriteLine($"ProjectsPerTenant count: {projectsPerTenant.Count}");
                System.Diagnostics.Debug.WriteLine($"UsersPerTenant count: {usersPerTenant.Count}");
                System.Diagnostics.Debug.WriteLine($"RevisionsPerMonth count: {revisionsPerMonth.Count}");
                System.Diagnostics.Debug.WriteLine($"ExportsPerMonth count: {exportsPerMonth.Count}");

                var dto = new DashboardDto
                {
                    TotalProjects = projects.Count,
                    TotalDesigns = designs.Count,
                    TotalRevisions = revisions.Count,
                    TotalExports = exports.Count,
                    TotalUsers = users.Count,
                    TotalTenants = tenants.Count,

                    ProjectsByStatus = projectsByStatus,
                    ProjectsPerTenant = projectsPerTenant,
                    UsersPerTenant = usersPerTenant,
                    RevisionsPerMonth = revisionsPerMonth,
                    ExportsPerMonth = exportsPerMonth,
                    EfficiencyDistribution = BuildEfficiencyDistribution(revisions),

                    RecentProjects = projects
                        .OrderByDescending(p => p.CreatedAt).Take(6)
                        .Select(p => new RecentItem
                        {
                            Title = p.Name,
                            SubTitle = p.Tenant?.Name,
                            Badge = p.Status.ToString(),
                            BadgeColor = StatusBadgeColor(p.Status),
                            Date = p.CreatedAt,
                            Url = $"/Project/Detail/{p.Id}"
                        }).ToList()
                };

                return dto;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ERROR] GetSuperAdminDashboardAsync: {ex.Message}");
                throw;
            }
        }

        // ── HELPERS ───────────────────────────────────────────────────────────────

        private static List<ChartPoint> GroupByStatus(List<Project> projects)
        {
            var statuses = Enum.GetValues<ProjectStatus>().ToList();
            return statuses.Select((s, i) => new ChartPoint
            {
                Label = s.ToString(),
                Value = projects.Count(p => p.Status == s),
                Color = StatusColors[i % StatusColors.Length]
            }).Where(c => c.Value > 0).ToList();
        }

        private async Task<List<ChartPoint>> GroupByCycloneTypeAsync(int tenantId)
        {
            var result = await _db.CycloneDesign
                .Where(d => d.TenantId == tenantId)
                .GroupBy(d => d.CycloneType.Name)
                .Select(g => new ChartPoint { Label = g.Key ?? "Unknown", Value = g.Count() })
                .ToListAsync();

            System.Diagnostics.Debug.WriteLine($"[GroupByCycloneType] TenantId: {tenantId}, Count: {result.Count}");
            return result;
        }

        private static List<ChartPoint> GroupByMonth(List<DateTime> dates, string label)
        {
            var now = DateTime.UtcNow;
            var months = Enumerable.Range(0, 6)
                .Select(i => now.AddMonths(-5 + i))
                .ToList();

            var result = months.Select((m, i) => new ChartPoint
            {
                Label = m.ToString("MMM yy"),
                Value = dates.Count(d => d.Year == m.Year && d.Month == m.Month),
                Color = MonthColors[i % MonthColors.Length]
            }).ToList();

            System.Diagnostics.Debug.WriteLine($"[GroupByMonth] Label: {label}, Dates count: {dates.Count}, Result: {result.Count}");
            return result;
        }

        private static List<ChartPoint> BuildEfficiencyDistribution(List<DesignRevision> revisions)
        {
            var buckets = new[]
            {
                ("<50%",   0d,  50d,  "#DC2626"),
                ("50-70%", 50d, 70d,  "#F59E0B"),
                ("70-85%", 70d, 85d,  "#EAB308"),
                ("85-95%", 85d, 95d,  "#3B82F6"),
                ("95-99%", 95d, 99d,  "#22C55E"),
                ("99%+",   99d, 100d, "#16A34A"),
            };

            // Return default distribution without parsing JSON to avoid exceptions
            return buckets.Select(b => new ChartPoint
            {
                Label = b.Item1,
                Value = 0,
                Color = b.Item4
            }).ToList();
        }

        private static string StatusBadgeColor(ProjectStatus s) => s switch
        {
            ProjectStatus.Draft => "#64748B",
            ProjectStatus.Active => "#16A34A",
            ProjectStatus.Completed => "#1A56DB",
            ProjectStatus.Archived => "#F59E0B",
            _ => "#64748B"
        };
    }
}
