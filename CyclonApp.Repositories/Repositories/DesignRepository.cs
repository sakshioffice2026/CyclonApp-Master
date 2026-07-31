using CyclonApp.Database;
using CyclonApp.Repositories.Contracts;
using Microsoft.EntityFrameworkCore;

namespace CyclonApp.Repositories.Repositories
{
    public class DesignRepository : IDesignRepository
    {
        private readonly ApplicationDbContext _db;

        public DesignRepository(ApplicationDbContext db)
        {
            _db = db;
        }

        public async Task<List<CycloneDesign>> GetAllDesignsAsync()
        {
            return await _db.CycloneDesign
                .Include(d => d.Project)
                .Include(d => d.CycloneType)
                .Include(d => d.Revisions)
                .OrderByDescending(d => d.CreatedAt)
                .ToListAsync();
        }

        public async Task<CycloneDesign?> GetDesignByIdAsync(int id)
        {
            return await _db.CycloneDesign.FindAsync(id);
        }

        public async Task<CycloneDesign?> GetDesignWithDetailsAsync(int id)
        {
            var design = await _db.CycloneDesign
                .Include(d => d.Project)
                .Include(d => d.CycloneType)
                .Include(d => d.Revisions)
                .FirstOrDefaultAsync(d => d.Id == id);

            return design;
        }

        public async Task<List<CycloneType>> GetActiveCycloneTypesAsync()
        {
            return await _db.CycloneTypes
                .Where(ct => ct.IsActive)
                .OrderBy(ct => ct.SortOrder)
                .ToListAsync();
        }

        public async Task<CycloneType?> GetCycloneTypeByIdAsync(int id)
        {
            return await _db.CycloneTypes.FindAsync(id);
        }

        public async Task CreateDesignAsync(CycloneDesign design)
        {
            _db.CycloneDesign.Add(design);
            await _db.SaveChangesAsync();
        }

        public async Task UpdateDesignRevisionAsync(CycloneDesign design, DesignRevision revision)
        {
            _db.DesignRevisions.Add(revision);
            design.CurrentRevision = revision.RevisionNumber;
            design.UpdatedAt = DateTime.UtcNow;
            await _db.SaveChangesAsync();
        }

        public async Task<DesignRevision?> GetRevisionByIdAsync(int id)
        {
            return await _db.DesignRevisions.FindAsync(id);
        }

        public async Task<DesignRevision?> GetRevisionWithDetailsAsync(int id)
        {
            return await _db.DesignRevisions
                .Include(r => r.CycloneDesign)
                    .ThenInclude(d => d.Project)
                .Include(r => r.CycloneDesign)
                    .ThenInclude(d => d.CycloneType)
                .FirstOrDefaultAsync(r => r.Id == id);
        }
        public async Task SavePredictionAsync(DesignRevision revision, string predictionJson)
        {
            revision.PredictionJson = predictionJson;
            revision.PredictedAt = DateTime.UtcNow;
            await _db.SaveChangesAsync();
        }

        public async Task SaveCfdImageAsync(DesignRevision revision, string pngUrl)
        {
            revision.CfdImageUrl = pngUrl;
            revision.CfdImageGeneratedAt = DateTime.UtcNow;
            await _db.SaveChangesAsync();
        }
    }
}