namespace CyclonApp.Database
{
    public class ExportLog
    {
        public int Id { get; set; }

        public int TenantId { get; set; }
        public Tenant Tenant { get; set; } = null!;

        public int DesignRevisionId { get; set; }
        public DesignRevision DesignRevision { get; set; } = null!;

        public ExportType ExportType { get; set; }

        public int? ExportedByUserId { get; set; }
        public AppUser? ExportedBy { get; set; }

        public DateTime ExportedAt { get; set; } = DateTime.UtcNow;
    }

    public enum ExportType { PDF, Excel }
}