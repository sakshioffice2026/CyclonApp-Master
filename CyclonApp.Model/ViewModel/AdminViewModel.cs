using CyclonApp.Database;

namespace CyclonApp.Model.ViewModel
{
    public class AdminPanelViewModel
    {
        public int TotalUsers { get; set; }
        public int ActiveUsers { get; set; }
        public int TotalTenants { get; set; }
        public int TotalProjects { get; set; }

        public Dictionary<string, int> UsersByRole { get; set; } = new();
        public List<TenantSummaryRow> TenantSummaries { get; set; } = new();
        public List<AppUser> RecentUsers { get; set; } = new();
    }

    public class TenantSummaryRow
    {
        public string Name { get; set; } = "";
        public int UserCount { get; set; }
        public int ProjectCount { get; set; }
        public bool IsActive { get; set; }
    }
}
