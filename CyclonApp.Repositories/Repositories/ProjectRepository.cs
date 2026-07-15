using CyclonApp.Database;
using CyclonApp.Model.ViewModel;
using Microsoft.EntityFrameworkCore;

namespace CyclonApp.Repositories
{
    public class ProjectRepository
    {
        private readonly ApplicationDbContext _db;

        public ProjectRepository(ApplicationDbContext db)
        {
            _db = db;
        }

        // ── GET ALL / INDEX ───────────────────────────────────────────────────
        public async Task<ProjectIndexViewModel> GetProjectIndexDataAsync(string? status, string? search)
        {
            var query = _db.Projects
                .Include(p => p.CreatedBy)
                .AsQueryable();

            // 🔹 Status Filter
            if (!string.IsNullOrEmpty(status) &&
                Enum.TryParse<ProjectStatus>(status, out var statusEnum))
            {
                query = query.Where(p => p.Status == statusEnum);
            }

            var rawProjects = await query.ToListAsync();

            // 🔹 Search Filter (post-query to avoid EF issues with string.Contains)
            if (!string.IsNullOrEmpty(search))
            {
                search = search.Trim().ToLower();
                rawProjects = rawProjects.Where(p =>
                    p.Name.ToLower().Contains(search) ||
                    (p.ProjectNumber != null && p.ProjectNumber.ToLower().Contains(search)) ||
                    (p.ClientName != null && p.ClientName.ToLower().Contains(search)))
                    .ToList();
            }

            var projects = rawProjects
                .OrderByDescending(p => p.UpdatedAt ?? p.CreatedAt)
                .ToList();

            // 🔹 Load design counts separately — avoids EF chain join bug
            var designCounts = await _db.CycloneDesign
                .GroupBy(d => d.ProjectId)
                .Select(g => new { ProjectId = g.Key, Count = g.Count() })
                .ToDictionaryAsync(x => x.ProjectId, x => x.Count);

            var allProjects = await _db.Projects.ToListAsync();

            return new ProjectIndexViewModel
            {
                Projects = projects.Select(p => new ProjectListItemViewModel
                {
                    Id = p.Id,
                    ProjectNumber = p.ProjectNumber ?? $"PRJ-{p.Id:D4}",
                    Name = p.Name,
                    ClientName = p.ClientName,
                    Location = p.Location,
                    Status = p.Status,
                    DesignCount = designCounts.GetValueOrDefault(p.Id, 0),
                    CreatedBy = p.CreatedBy?.DisplayName ?? "Unknown",
                    CreatedAt = p.CreatedAt,
                    UpdatedAt = p.UpdatedAt
                }).ToList(),

                TotalProjects = allProjects.Count,
                ActiveProjects = allProjects.Count(p => p.Status == ProjectStatus.Active),
                DraftProjects = allProjects.Count(p => p.Status == ProjectStatus.Draft),
                CompletedProjects = allProjects.Count(p => p.Status == ProjectStatus.Completed),
                TotalDesigns = await _db.CycloneDesign.CountAsync(),
                FilterStatus = status,
                SearchTerm = search
            };
        }

        // ── GET DETAIL ────────────────────────────────────────────────────────
        public async Task<ProjectDetailViewModel?> GetProjectDetailAsync(int id)
        {
            var project = await _db.Projects
                .Include(p => p.CreatedBy)
                .Include(p => p.Designs)
                    .ThenInclude(d => d.CycloneType)
                .Include(p => p.Designs)
                    .ThenInclude(d => d.Revisions)
                .FirstOrDefaultAsync(p => p.Id == id);

            if (project == null)
                return null;

            return new ProjectDetailViewModel
            {
                Id = project.Id,
                ProjectNumber = project.ProjectNumber ?? $"PRJ-{project.Id:D4}",
                Name = project.Name,
                ClientName = project.ClientName,
                Location = project.Location,
                Description = project.Description,
                Status = project.Status,
                CreatedBy = project.CreatedBy?.DisplayName ?? "Unknown",
                CreatedAt = project.CreatedAt,
                UpdatedAt = project.UpdatedAt,
                Designs = project.Designs
                    .OrderByDescending(d => d.CreatedAt)
                    .Select(d => new DesignSummaryViewModel
                    {
                        Id = d.Id,
                        TagNumber = d.TagNumber ?? $"DES-{d.Id:D3}",
                        Name = d.Name,
                        CycloneType = d.CycloneType?.Name ?? "—",
                        CurrentRevision = d.CurrentRevision,
                        CreatedAt = d.CreatedAt,
                        HasResults = d.Revisions.Any(r => r.CalculatedAt != null)
                    }).ToList()
            };
        }

        // ── GET FOR EDIT ──────────────────────────────────────────────────────
        public async Task<ProjectFormViewModel?> GetProjectForEditAsync(int id)
        {
            return await _db.Projects
                .Where(p => p.Id == id)
                .Select(p => new ProjectFormViewModel
                {
                    Id = p.Id,
                    ProjectNumber = p.ProjectNumber,
                    Name = p.Name,
                    ClientName = p.ClientName,
                    Location = p.Location,
                    Description = p.Description,
                    Status = p.Status
                })
                .FirstOrDefaultAsync();
        }

        // ── GENERATE PROJECT NUMBER ───────────────────────────────────────────
        public async Task<string> GenerateProjectNumberAsync()
        {
            var lastProject = await _db.Projects
                .OrderByDescending(p => p.Id)
                .FirstOrDefaultAsync();

            int nextNum = (lastProject?.Id ?? 0) + 1;
            return $"PRJ-{nextNum:D4}";
        }

        // ── CHECK DUPLICATE PROJECT NUMBER ────────────────────────────────────
        public async Task<bool> IsProjectNumberExistsAsync(string? projectNumber, int? projectId = null)
        {
            return await _db.Projects.AnyAsync(p =>
                p.ProjectNumber == projectNumber &&
                (!projectId.HasValue || p.Id != projectId.Value));
        }

        // ── CREATE PROJECT ────────────────────────────────────────────────────
        public async Task<int> CreateProjectAsync(ProjectFormViewModel vm, int tenantId, int userId)
        {
            var project = new Project
            {
                TenantId = tenantId,
                ProjectNumber = vm.ProjectNumber,
                Name = vm.Name,
                ClientName = vm.ClientName,
                Location = vm.Location,
                Description = vm.Description,
                Status = vm.Status,
                CreatedByUserId = userId,
                CreatedAt = DateTime.UtcNow
            };

            _db.Projects.Add(project);
            await _db.SaveChangesAsync();

            return project.Id;
        }

        // ── UPDATE PROJECT ────────────────────────────────────────────────────
        public async Task<bool> UpdateProjectAsync(int id, ProjectFormViewModel vm, int userId)
        {
            var project = await _db.Projects.FindAsync(id);
            if (project == null)
                return false;

            project.ProjectNumber = vm.ProjectNumber;
            project.Name = vm.Name;
            project.ClientName = vm.ClientName;
            project.Location = vm.Location;
            project.Description = vm.Description;
            project.Status = vm.Status;
            project.LastModifiedByUserId = userId;
            project.UpdatedAt = DateTime.UtcNow;

            await _db.SaveChangesAsync();
            return true;
        }

        // ── CHANGE STATUS ─────────────────────────────────────────────────────
        public async Task<bool> ChangeStatusAsync(int id, ProjectStatus status, int userId)
        {
            var project = await _db.Projects.FindAsync(id);
            if (project == null)
                return false;

            project.Status = status;
            project.LastModifiedByUserId = userId;
            project.UpdatedAt = DateTime.UtcNow;

            if (status == ProjectStatus.Archived)
                project.ArchivedAt = DateTime.UtcNow;

            await _db.SaveChangesAsync();
            return true;
        }

        // ── DELETE PROJECT ────────────────────────────────────────────────────
        public async Task<(bool Success, string? Message)> DeleteProjectAsync(int id)
        {
            var project = await _db.Projects
                .Include(p => p.Designs)
                .FirstOrDefaultAsync(p => p.Id == id);

            if (project == null)
                return (false, "Project not found.");

            if (project.Designs.Any())
                return (false, $"Cannot delete project — it has {project.Designs.Count} design(s). Archive it instead, or delete all designs first.");

            _db.Projects.Remove(project);
            await _db.SaveChangesAsync();

            return (true, project.Name);
        }

        // ── GET BY ID (simplified) ────────────────────────────────────────────
        public async Task<Project?> GetProjectByIdAsync(int id)
        {
            return await _db.Projects.FindAsync(id);
        }

        // ── CHECK CAN EDIT ────────────────────────────────────────────────────
        public async Task<(bool CanEdit, string? Message)> CanEditProjectAsync(int id)
        {
            var project = await GetProjectByIdAsync(id);
            if (project == null)
                return (false, "Project not found.");

            if (project.Status == ProjectStatus.Archived)
                return (false, "Archived projects cannot be edited. Please restore them first.");

            return (true, null);
        }
    }
}