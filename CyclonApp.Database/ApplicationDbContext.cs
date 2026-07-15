using Microsoft.EntityFrameworkCore;

namespace CyclonApp.Database
{
    public class ApplicationDbContext : DbContext
    {
        public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options)
            : base(options) { }

        public DbSet<exceptionhandler> exceptionhandler { get; set; }
        public DbSet<UserRole> UserRoles { get; set; }
        public DbSet<AppUser> Users => Set<AppUser>();
        public DbSet<Tenant> Tenants => Set<Tenant>();
        public DbSet<CycloneType> CycloneTypes => Set<CycloneType>();
        public DbSet<Project> Projects => Set<Project>();
        public DbSet<CycloneDesign> CycloneDesign => Set<CycloneDesign>();
        public DbSet<DesignRevision> DesignRevisions => Set<DesignRevision>();
        public DbSet<ExportLog> ExportLogs => Set<ExportLog>();

        protected override void OnModelCreating(ModelBuilder builder)
        {
            base.OnModelCreating(builder);

            // ── AppUser ───────────────────────────────────────────────────────
            builder.Entity<AppUser>()
                .Ignore(u => u.Role);

            builder.Entity<AppUser>()
                .HasIndex(u => u.Email).IsUnique();

            builder.Entity<AppUser>()
                .HasOne(u => u.UserRole)
                .WithMany(r => r.Users)
                .HasForeignKey(u => u.UserRoleId)
                .OnDelete(DeleteBehavior.Restrict);

            builder.Entity<AppUser>()
                .HasOne(u => u.Tenant)
                .WithMany(t => t.Users)
                .HasForeignKey(u => u.TenantId)
                .OnDelete(DeleteBehavior.Restrict);

            // ── Project ───────────────────────────────────────────────────────
            builder.Entity<Project>()
                .Property(p => p.Status)
                .HasConversion<string>();

            builder.Entity<Project>()
                .HasOne(p => p.CreatedBy)
                .WithMany()
                .HasForeignKey(p => p.CreatedByUserId)
                .HasConstraintName("FK_Projects_Users_CreatedByUserId")
                .OnDelete(DeleteBehavior.Restrict);

            builder.Entity<Project>()
                .HasOne<AppUser>()
                .WithMany()
                .HasForeignKey(p => p.LastModifiedByUserId)
                .OnDelete(DeleteBehavior.Restrict);

            builder.Entity<Project>()
                .HasIndex(p => new { p.TenantId, p.ProjectNumber });

            // ── CycloneDesign ─────────────────────────────────────────────────
            builder.Entity<CycloneDesign>(entity =>
            {
                entity.ToTable("cyclonedesign");  
                entity.HasKey(d => d.Id);

                entity.Property(d => d.CreatedByUserId)
                      .HasColumnName("CreatedByUserId");

                entity.HasOne(d => d.Project)
                      .WithMany(p => p.Designs)
                      .HasForeignKey(d => d.ProjectId)
                      .OnDelete(DeleteBehavior.Cascade);

                entity.HasOne(d => d.Tenant)
                      .WithMany(t => t.CycloneDesign)
                      .HasForeignKey(d => d.TenantId)
                      .OnDelete(DeleteBehavior.Restrict);

                entity.HasOne(d => d.CycloneType)
                      .WithMany(ct => ct.Designs)
                      .HasForeignKey(d => d.CycloneTypeId)
                      .OnDelete(DeleteBehavior.Restrict);

                entity.HasOne(d => d.CreatedBy)
                      .WithMany()
                      .HasForeignKey(d => d.CreatedByUserId)
                      .HasConstraintName("FK_CycloneDesign_Users_CreatedByUserId")
                      .OnDelete(DeleteBehavior.Restrict);

                entity.HasIndex(d => new { d.TenantId, d.ProjectId });
            });
            builder.Entity<CycloneType>(entity =>
            {
                entity.ToTable("cyclonetypes");

                entity.HasKey(x => x.Id);

                entity.Property(x => x.Code)
                      .HasMaxLength(20)
                      .IsRequired();

                entity.Property(x => x.Name)
                      .HasMaxLength(100)
                      .IsRequired();
            });
            // ── DesignRevision ────────────────────────────────────────────────
            builder.Entity<DesignRevision>(entity =>
            {
                entity.ToTable("designrevision"); 
                entity.HasKey(r => r.Id);

                entity.Property(r => r.InletShape)
                      .HasConversion<string>();

                entity.Property(r => r.CreatedByUserId)
                      .HasColumnName("CreatedByUserId");

                entity.HasOne(r => r.CycloneDesign)
                      .WithMany(d => d.Revisions)
                      .HasForeignKey(r => r.CycloneDesignId)
                      .OnDelete(DeleteBehavior.Cascade);

                entity.HasOne(r => r.CreatedBy)
                      .WithMany()
                      .HasForeignKey(r => r.CreatedByUserId)
                      .HasConstraintName("FK_DesignRevision_Users_CreatedByUserId")
                      .OnDelete(DeleteBehavior.SetNull);

                entity.HasIndex(r => new { r.CycloneDesignId, r.RevisionNumber })
                      .IsUnique();
            });

            // ── ExportLog ─────────────────────────────────────────────────────
            builder.Entity<ExportLog>(entity =>
            {
                entity.HasKey(e => e.Id);

                entity.Property(e => e.ExportType)
                      .HasConversion<string>();

                entity.HasOne(e => e.DesignRevision)
                      .WithMany(r => r.ExportLogs)
                      .HasForeignKey(e => e.DesignRevisionId)
                      .OnDelete(DeleteBehavior.Cascade);

                entity.HasOne(e => e.ExportedBy)
                      .WithMany()
                      .HasForeignKey(e => e.ExportedByUserId)
                      .OnDelete(DeleteBehavior.SetNull);

                entity.HasOne(e => e.Tenant)
                      .WithMany()
                      .HasForeignKey(e => e.TenantId)
                      .OnDelete(DeleteBehavior.Restrict);
            });

            // ── Tenant ────────────────────────────────────────────────────────
            builder.Entity<Tenant>()
                .HasIndex(t => t.Slug).IsUnique();
        }
    }
}
