using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CyclonApp.Model.DTOs
{
    public class DashboardDto
    {
        // KPI cards
        public int TotalProjects { get; set; }
        public int TotalDesigns { get; set; }
        public int TotalRevisions { get; set; }
        public int TotalExports { get; set; }
        public int TotalUsers { get; set; }
        public int TotalTenants { get; set; }

        // Charts
        public List<ChartPoint> ProjectsByStatus { get; set; } = new();
        public List<ChartPoint> DesignsByCycloneType { get; set; } = new();
        public List<ChartPoint> RevisionsPerMonth { get; set; } = new();
        public List<ChartPoint> ExportsPerMonth { get; set; } = new();
        public List<ChartPoint> EfficiencyDistribution { get; set; } = new();
        public List<ChartPoint> ProjectsPerTenant { get; set; } = new();
        public List<ChartPoint> UsersPerTenant { get; set; } = new();
        public List<ChartPoint> MyRevisionsPerMonth { get; set; } = new();

        // Recent activity
        public List<RecentItem> RecentProjects { get; set; } = new();
        public List<RecentItem> RecentCalculations { get; set; } = new();
        public List<RecentItem> RecentExports { get; set; } = new();
    }

    public class ChartPoint
    {
        public string Label { get; set; } = "";
        public double Value { get; set; }
        public string? Color { get; set; }
    }

    public class RecentItem
    {
        public string Title { get; set; } = "";
        public string? SubTitle { get; set; }
        public string? Badge { get; set; }
        public string? BadgeColor { get; set; }
        public DateTime Date { get; set; }
        public string? Url { get; set; }
    }
}


