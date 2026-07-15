using System.Text.Json;
using ClosedXML.Excel;
using CyclonApp.Database;
using CyclonApp.Model.DTOs;
using CyclonApp.Repositories.Contracts;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace CyclonApp.Repositories.Repositories
{
    public class ExportRepository : IExport
    {
        private readonly ApplicationDbContext _db;
        private readonly ILogger<ExportRepository> _logger;

        private static readonly JsonSerializerOptions _jsonOpts =
            new() { PropertyNameCaseInsensitive = true };

        // Brand colours
        private const string ColPrimary = "1A56DB";
        private const string ColDark = "1E3A5F";
        private const string ColMuted = "64748B";
        private const string ColLightBlue = "F0F7FF";
        private const string ColSurface = "F8FAFC";
        private const string ColBorder = "E2E8F0";
        private const string ColSuccess = "16A34A";
        private const string ColWarning = "D97706";
        private const string ColDanger = "DC2626";

        public ExportRepository(ApplicationDbContext db, ILogger<ExportRepository> logger)
        {
            _db = db;
            _logger = logger;
        }

        // ── LOAD ─────────────────────────────────────────────────────────────────

        public async Task<(DesignRevision Revision, CyclonOutputDto? Output)> GetRevisionForExportAsync(int revisionId)
        {
            var rev = await _db.DesignRevisions
                .Include(r => r.CycloneDesign)
                    .ThenInclude(d => d.Project)
                        .ThenInclude(p => p.Tenant)
                .Include(r => r.CycloneDesign)
                    .ThenInclude(d => d.CycloneType)
                .Include(r => r.CreatedBy)
                .FirstOrDefaultAsync(r => r.Id == revisionId)
                ?? throw new KeyNotFoundException($"Revision {revisionId} not found.");

            CyclonOutputDto? output = null;
            if (!string.IsNullOrEmpty(rev.EfficiencyJson))
                output = JsonSerializer.Deserialize<CyclonOutputDto>(rev.EfficiencyJson, _jsonOpts);

            return (rev, output);
        }

        // ── EXPORT LOGS ───────────────────────────────────────────────────────────

        public async Task<List<ExportLog>> GetExportLogsAsync(int? tenantId = null, int take = 200)
        {
            var query = _db.ExportLogs
                .Include(e => e.DesignRevision)
                    .ThenInclude(r => r.CycloneDesign)
                        .ThenInclude(d => d.Project)
                .Include(e => e.ExportedBy)
                .AsQueryable();

            if (tenantId.HasValue)
                query = query.Where(e => e.TenantId == tenantId.Value);

            return await query
                .OrderByDescending(e => e.ExportedAt)
                .Take(take)
                .ToListAsync();
        }

        public async Task<List<ExportLog>> GetExportLogsByDesignAsync(int designId)
        {
            return await _db.ExportLogs
                .Include(e => e.ExportedBy)
                .Where(e => e.DesignRevision.CycloneDesign.Id == designId)
                .OrderByDescending(e => e.ExportedAt)
                .ToListAsync();
        }

        public async Task LogExportAsync(int revisionId, int tenantId, int? exportedByUserId, ExportType type)
        {
            _db.ExportLogs.Add(new ExportLog
            {
                TenantId = tenantId,
                DesignRevisionId = revisionId,
                ExportType = type,
                ExportedByUserId = exportedByUserId,
                ExportedAt = DateTime.UtcNow
            });
            await _db.SaveChangesAsync();
            _logger.LogInformation("Export logged: RevisionId={RevisionId} Type={Type}", revisionId, type);
        }

        // ═══════════════════════════════════════════════════════════════════════════
        //  PDF — returns UTF-8 HTML bytes (swap in DinkToPdf to get real PDF)
        // ═══════════════════════════════════════════════════════════════════════════

        public Task<byte[]> GeneratePdfAsync(DesignRevision rev, CyclonOutputDto? output)
        {
            _logger.LogInformation("PDF export: DesignId={Id} Rev={Rev}", rev.CycloneDesignId, rev.RevisionNumber);
            return Task.FromResult(System.Text.Encoding.UTF8.GetBytes(BuildPdfHtml(rev, output)));
        }

        private static string BuildPdfHtml(DesignRevision rev, CyclonOutputDto? o)
        {
            var d = rev.CycloneDesign;
            var proj = d.Project;
            var dims = o?.Dimensions ?? new CyclonDimensions();

            var effColor = (o?.Efficiency ?? 0) >= 95 ? "#16a34a"
                         : (o?.Efficiency ?? 0) >= 80 ? "#d97706" : "#dc2626";

            return $@"<!DOCTYPE html>
<html lang=""en"">
<head>
<meta charset=""UTF-8""/>
<title>Cyclone Design Report — {d.TagNumber} Rev {rev.RevisionNumber}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',Arial,sans-serif;font-size:12px;color:#1e293b;background:#fff}}
  .page{{padding:28px 32px;max-width:900px;margin:0 auto}}
  .report-header{{background:linear-gradient(135deg,#1a56db,#1240a8);color:#fff;padding:24px 28px;border-radius:10px;margin-bottom:20px;display:flex;justify-content:space-between;align-items:center}}
  .report-title{{font-size:20px;font-weight:700}}
  .report-sub{{font-size:12px;opacity:.8;margin-top:4px}}
  .report-meta{{text-align:right;font-size:11px;opacity:.85;line-height:1.8}}
  .kpi-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}}
  .kpi{{border:1px solid #e2e8f0;border-radius:8px;padding:12px 14px}}
  .kpi-label{{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:#64748b}}
  .kpi-value{{font-size:20px;font-weight:700;margin-top:3px;line-height:1}}
  .kpi-sub{{font-size:10.5px;color:#64748b;margin-top:3px}}
  .section{{margin-bottom:18px}}
  .section-title{{font-size:13px;font-weight:700;color:#1a56db;border-bottom:2px solid #1a56db;padding-bottom:5px;margin-bottom:10px}}
  table{{width:100%;border-collapse:collapse;font-size:11.5px}}
  th{{background:#1a56db;color:#fff;padding:7px 10px;text-align:left;font-weight:600}}
  td{{padding:6px 10px;border-bottom:1px solid #f1f5f9}}
  tr:nth-child(even) td{{background:#f8fafc}}
  .val{{font-family:Consolas,'Courier New',monospace;font-weight:600}}
  .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
  .report-footer{{margin-top:24px;padding-top:12px;border-top:1px solid #e2e8f0;display:flex;justify-content:space-between;font-size:10.5px;color:#94a3b8}}
  @media print{{body{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}.page{{padding:8px}}@page{{margin:10mm;size:A4}}.kpi-row{{grid-template-columns:repeat(4,1fr)}}.two-col{{grid-template-columns:1fr 1fr}}.report-header{{border-radius:0}}}}
</style>
</head>
<body>
<div class=""page"">
  <div class=""report-header"">
    <div>
      <div class=""report-title"">🌀 Cyclone Design Report</div>
      <div class=""report-sub"">Tag: {d.TagNumber ?? "—"} &nbsp;·&nbsp; {d.CycloneType?.Name ?? "—"} &nbsp;·&nbsp; Revision {rev.RevisionNumber}</div>
      {(rev.RevisionNote != null ? $"<div class='report-sub' style='margin-top:6px;opacity:.7'>Note: {rev.RevisionNote}</div>" : "")}
    </div>
    <div class=""report-meta"">
      <div><strong>{proj?.Name ?? "—"}</strong></div>
      <div>Project No: {proj?.ProjectNumber ?? "—"}</div>
      <div>Client: {proj?.ClientName ?? "—"}</div>
      <div>Calculated: {rev.CalculatedAt?.ToString("dd MMM yyyy HH:mm") ?? "—"}</div>
      <div>Tenant: {proj?.Tenant?.Name ?? "—"}</div>
    </div>
  </div>

  <div class=""kpi-row"">
    <div class=""kpi"">
      <div class=""kpi-label"">Collection Efficiency</div>
      <div class=""kpi-value"" style=""color:{effColor}"">{(o?.Efficiency ?? 0):F2}%</div>
      <div class=""kpi-sub"">At {rev.ParticleSizeMicron} µm</div>
    </div>
    <div class=""kpi"">
      <div class=""kpi-label"">Cut Diameter (Dpc)</div>
      <div class=""kpi-value"" style=""color:#1a56db"">{(o?.CutDiameterMicron ?? 0):F3} µm</div>
      <div class=""kpi-sub"">50% separation point</div>
    </div>
    <div class=""kpi"">
      <div class=""kpi-label"">Pressure Drop</div>
      <div class=""kpi-value"" style=""color:#d97706"">{(o?.PressureDropPa ?? 0):F1} Pa</div>
      <div class=""kpi-sub"">{(o?.PressureDropMmWc ?? 0):F2} mm WC</div>
    </div>
    <div class=""kpi"">
      <div class=""kpi-label"">Barrel Diameter (Dc)</div>
      <div class=""kpi-value"" style=""color:#0891b2"">{dims.BarrelDiameterIn:F3}''</div>
      <div class=""kpi-sub"">{dims.BarrelDiameterMm:F1} mm</div>
    </div>
  </div>

  <div class=""two-col"">
    <div class=""section"">
      <div class=""section-title"">Cyclone Dimensions</div>
      <table>
        <tr><th>Dimension</th><th>Inches</th><th>mm</th></tr>
        {DimRow("Barrel Dia (Dc)", dims.BarrelDiameterIn, dims.BarrelDiameterMm)}
        {DimRow("Inlet Height (H)", dims.InletHeightIn, dims.InletHeightMm)}
        {DimRow("Inlet Width (W)", dims.InletWidthIn, dims.InletWidthMm)}
        {DimRow("Barrel Height (Lb)", dims.BarrelHeightIn, dims.BarrelHeightMm)}
        {DimRow("Cone Height (Lc)", dims.ConeHeightIn, dims.ConeHeightMm)}
        {DimRow("Exhaust Dia (De)", dims.ExhaustDiaIn, dims.ExhaustDiaMm)}
        {DimRow("Exhaust Length", dims.ExhaustLengthIn, dims.ExhaustLengthMm)}
        {DimRow("Bottom Outlet (Dd)", dims.BottomOutletIn, dims.BottomOutletMm)}
        {DimRow("Total Height", dims.TotalHeightIn, dims.TotalHeightMm)}
      </table>
    </div>
    <div class=""section"">
      <div class=""section-title"">Calculated Results</div>
      <table>
        <tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>
        {ResRow("Collection Efficiency", (o?.Efficiency ?? 0).ToString("F2"), "%")}
        {ResRow("Cut Diameter (Dpc)", (o?.CutDiameterMicron ?? 0).ToString("F3"), "µm")}
        {ResRow("Pressure Drop", (o?.PressureDropPa ?? 0).ToString("F1"), "Pa")}
        {ResRow("Pressure Drop", (o?.PressureDropMmWc ?? 0).ToString("F2"), "mm WC")}
        {ResRow("Pressure Drop", (o?.PressureDropInWc ?? 0).ToString("F3"), "in WC")}
        {ResRow("Flow Rate", (o?.FlowRateM3hr ?? 0).ToString("F2"), "m³/hr")}
        {ResRow("Inlet Velocity", (o?.InletVelocityMs ?? 0).ToString("F3"), "m/s")}
        {ResRow("Gas Density", (o?.GasDensityKgm3 ?? 0).ToString("F4"), "kg/m³")}
        {ResRow("Gas Viscosity", (o?.GasViscosityKgms ?? 0).ToString("E3"), "kg/m·s")}
      </table>
    </div>
  </div>

  <div class=""section"">
    <div class=""section-title"">Design Inputs</div>
    <div class=""two-col"">
      <table>
        <tr><th colspan=""2"">Process Parameters</th></tr>
        {InRow("Flow Rate", $"{rev.FlowRateCFM} CFM / {rev.FlowRateM3hr:F2} m³/hr")}
        {InRow("Inlet Line Size", $"{rev.InletLineSizeIn} inches Ø")}
        {InRow("Gas Type", rev.GasType)}
        {InRow("Temperature", $"{rev.OperatingTempC} °C")}
        {InRow("Pressure", $"{rev.OperatingPressKPa} kPa")}
        {InRow("Inlet Shape", rev.InletShape.ToString())}
      </table>
      <table>
        <tr><th colspan=""2"">Particle / Design Parameters</th></tr>
        {InRow("Particle Size (avg)", $"{rev.ParticleSizeMicron} µm")}
        {InRow("D10 / D50 / D90", $"{rev.ParticleSizeD10} / {rev.ParticleSizeD50} / {rev.ParticleSizeD90} µm")}
        {InRow("Particle Density", $"{rev.ParticleDensityKgm3} kg/m³")}
        {InRow("Bulk Density", $"{rev.BulkDensityKgm3} kg/m³")}
        {InRow("Effective Turns", $"{rev.EffectiveTurns}")}
        {InRow("No. of Cyclones", $"{rev.NumberOfCyclones}")}
        {InRow("Safety Factor", $"{rev.SafetyFactor}")}
      </table>
    </div>
  </div>

  <div class=""report-footer"">
    <span>Generated by Cyclone Design App &nbsp;·&nbsp; {DateTime.UtcNow:dd MMM yyyy HH:mm} UTC</span>
    <span>By: {rev.CreatedBy?.DisplayName ?? "—"} &nbsp;·&nbsp; Tenant: {proj?.Tenant?.Name ?? "—"}</span>
    <span>CONFIDENTIAL — Engineering Use Only</span>
  </div>
</div>
</body>
</html>";
        }

        private static string DimRow(string label, double inch, double mm) =>
            $"<tr><td>{label}</td><td class='val'>{inch:F3}\"</td><td class='val'>{mm:F1}</td></tr>";
        private static string ResRow(string label, string val, string unit) =>
            $"<tr><td>{label}</td><td class='val'>{val}</td><td>{unit}</td></tr>";
        private static string InRow(string label, string val) =>
            $"<tr><td style='color:#64748b'>{label}</td><td class='val'>{val}</td></tr>";

        // ═══════════════════════════════════════════════════════════════════════════
        //  EXCEL — ClosedXML workbook
        // ═══════════════════════════════════════════════════════════════════════════

        public Task<byte[]> GenerateExcelAsync(DesignRevision rev, CyclonOutputDto? output)
        {
            var d = rev.CycloneDesign;
            var proj = d.Project;
            var dims = output?.Dimensions ?? new CyclonDimensions();

            using var wb = new XLWorkbook();
            wb.Properties.Author = "Cyclone Design App";
            wb.Properties.Title = $"Cyclone Design Report — {d.TagNumber} Rev {rev.RevisionNumber}";
            wb.Properties.Subject = proj?.Name ?? "Cyclone Design";

            BuildCoverSheet(wb, rev, output, proj, d);
            BuildDimensionsSheet(wb, rev, dims, d);
            BuildResultsSheet(wb, rev, output, d);
            BuildInputsSheet(wb, rev, d, proj);

            if (output?.GradeEfficiencyCurve?.Any() == true)
                BuildEfficiencySheet(wb, output, rev);

            using var ms = new MemoryStream();
            wb.SaveAs(ms);

            _logger.LogInformation("Excel export: DesignId={Id} Rev={Rev}", rev.CycloneDesignId, rev.RevisionNumber);
            return Task.FromResult(ms.ToArray());
        }

        // ── COVER ─────────────────────────────────────────────────────────────────

        private static void BuildCoverSheet(XLWorkbook wb, DesignRevision rev,
            CyclonOutputDto? o, Project? proj, CycloneDesign d)
        {
            var ws = wb.Worksheets.Add("Report Cover");
            ws.ShowGridLines = false;

            ws.Cell("B2").Value = "CYCLONE DESIGN REPORT";
            ws.Cell("B2").Style.Font.FontSize = 22;
            ws.Cell("B2").Style.Font.Bold = true;
            ws.Cell("B2").Style.Font.FontColor = XLColor.FromHtml(ColPrimary);
            ws.Range("B2:G2").Merge();

            ws.Cell("B3").Value = $"{d.CycloneType?.Name ?? "—"} · Tag: {d.TagNumber ?? "—"} · Revision {rev.RevisionNumber}";
            ws.Cell("B3").Style.Font.FontSize = 12;
            ws.Cell("B3").Style.Font.FontColor = XLColor.FromHtml(ColMuted);
            ws.Range("B3:G3").Merge();

            ws.Row(4).Height = 4;
            ws.Range("B4:G4").Style.Fill.BackgroundColor = XLColor.FromHtml(ColPrimary);

            int r = 6;
            var infoRows = new[]
            {
                ("Project",      proj?.Name ?? "—"),
                ("Project No.",  proj?.ProjectNumber ?? "—"),
                ("Client",       proj?.ClientName ?? "—"),
                ("Location",     proj?.Location ?? "—"),
                ("Cyclone Type", d.CycloneType?.Name ?? "—"),
                ("Tag Number",   d.TagNumber ?? "—"),
                ("Revision",     $"{rev.RevisionNumber}"),
                ("Rev. Note",    rev.RevisionNote ?? "—"),
                ("Calculated",   rev.CalculatedAt?.ToString("dd MMM yyyy HH:mm") ?? "—"),
                ("Prepared By",  rev.CreatedBy?.DisplayName ?? "—"),
                ("Tenant",       proj?.Tenant?.Name ?? "—"),
            };

            foreach (var (label, value) in infoRows)
            {
                ws.Cell(r, 2).Value = label;
                ws.Cell(r, 2).Style.Font.Bold = true;
                ws.Cell(r, 2).Style.Font.FontColor = XLColor.FromHtml(ColMuted);
                ws.Cell(r, 2).Style.Font.FontSize = 11;
                ws.Cell(r, 3).Value = value;
                ws.Cell(r, 3).Style.Font.FontSize = 11;
                r++;
            }

            r += 2;
            AddSectionHeader(ws, r, 2, "KEY RESULTS SUMMARY", 6); r += 2;

            var kpis = new[]
            {
                ("Collection Efficiency", $"{(o?.Efficiency ?? 0):F2} %"),
                ("Cut Diameter (Dpc)",    $"{(o?.CutDiameterMicron ?? 0):F3} µm"),
                ("Pressure Drop",         $"{(o?.PressureDropPa ?? 0):F1} Pa  /  {(o?.PressureDropMmWc ?? 0):F2} mm WC"),
                ("Barrel Diameter (Dc)",  $"{(o?.Dimensions.BarrelDiameterIn ?? 0):F3}\"  /  {(o?.Dimensions.BarrelDiameterMm ?? 0):F1} mm"),
                ("Total Cyclone Height",  $"{(o?.Dimensions.TotalHeightIn ?? 0):F3}\"  /  {(o?.Dimensions.TotalHeightMm ?? 0):F1} mm"),
                ("Inlet Velocity",        $"{(o?.InletVelocityMs ?? 0):F3} m/s"),
            };

            bool shade = false;
            foreach (var (label, value) in kpis)
            {
                if (shade) ws.Range(r, 2, r, 7).Style.Fill.BackgroundColor = XLColor.FromHtml("F0F7FF");
                ws.Cell(r, 2).Value = label;
                ws.Cell(r, 2).Style.Font.Bold = true;
                ws.Range(r, 2, r, 4).Merge();
                ws.Cell(r, 5).Value = value;
                ws.Cell(r, 5).Style.Font.FontColor = XLColor.FromHtml(ColPrimary);
                ws.Cell(r, 5).Style.Font.Bold = true;
                ws.Range(r, 5, r, 7).Merge();
                r++; shade = !shade;
            }

            ws.Column(1).Width = 3;
            ws.Column(2).Width = 22;
            ws.Column(3).Width = 28;
            ws.Column(4).Width = 4;
            ws.Column(5).Width = 20;
            ws.Column(6).Width = 20;
            ws.Column(7).Width = 20;
        }

        // ── DIMENSIONS ────────────────────────────────────────────────────────────

        private static void BuildDimensionsSheet(XLWorkbook wb, DesignRevision rev,
            CyclonDimensions dims, CycloneDesign d)
        {
            var ws = wb.Worksheets.Add("Dimensions");
            ws.ShowGridLines = false;

            int r = 1;
            AddSheetTitle(ws, r, $"CYCLONE DIMENSIONS — {d.TagNumber}  Rev {rev.RevisionNumber}"); r += 2;

            ws.Range(r, 1, r, 5).Style.Fill.BackgroundColor = XLColor.FromHtml(ColPrimary);
            ws.Range(r, 1, r, 5).Style.Font.FontColor = XLColor.White;
            ws.Range(r, 1, r, 5).Style.Font.Bold = true;
            ws.Cell(r, 1).Value = "Dimension";
            ws.Cell(r, 2).Value = "Symbol";
            ws.Cell(r, 3).Value = "Ratio (×Dc)";
            ws.Cell(r, 4).Value = "Value (inches)";
            ws.Cell(r, 5).Value = "Value (mm)";
            r++;

            var dimRows = new[]
            {
                ("Barrel Diameter",   "Dc", "—",          dims.BarrelDiameterIn,  dims.BarrelDiameterMm),
                ("Inlet Height",      "H",  "Dc × H/Dc",  dims.InletHeightIn,     dims.InletHeightMm),
                ("Inlet Width",       "W",  "Dc × W/Dc",  dims.InletWidthIn,      dims.InletWidthMm),
                ("Barrel Height",     "Lb", "Dc × Lb/Dc", dims.BarrelHeightIn,    dims.BarrelHeightMm),
                ("Cone Height",       "Lc", "Dc × Lc/Dc", dims.ConeHeightIn,      dims.ConeHeightMm),
                ("Exhaust Pipe Dia",  "De", "Dc × De/Dc", dims.ExhaustDiaIn,      dims.ExhaustDiaMm),
                ("Exhaust Length",    "Le", "Dc × Le/Dc", dims.ExhaustLengthIn,   dims.ExhaustLengthMm),
                ("Bottom Outlet",     "Dd", "Dc × Dd/Dc", dims.BottomOutletIn,    dims.BottomOutletMm),
                ("Total Height",      "HT", "Lb + Lc",    dims.TotalHeightIn,     dims.TotalHeightMm),
            };

            bool shade = false;
            foreach (var (label, sym, ratio, inch, mm) in dimRows)
            {
                if (shade) ws.Range(r, 1, r, 5).Style.Fill.BackgroundColor = XLColor.FromHtml(ColLightBlue);
                ws.Cell(r, 1).Value = label;
                ws.Cell(r, 2).Value = sym;
                ws.Cell(r, 2).Style.Font.Bold = true;
                ws.Cell(r, 2).Style.Font.FontColor = XLColor.FromHtml(ColPrimary);
                ws.Cell(r, 3).Value = ratio;
                ws.Cell(r, 3).Style.Font.FontColor = XLColor.FromHtml(ColMuted);
                ws.Cell(r, 4).Value = inch;
                ws.Cell(r, 4).Style.NumberFormat.Format = "0.000\"\"\"\"";
                ws.Cell(r, 4).Style.Font.Bold = true;
                ws.Cell(r, 5).Value = mm;
                ws.Cell(r, 5).Style.NumberFormat.Format = "0.0";
                ws.Cell(r, 5).Style.Font.Bold = true;
                SetRowBorder(ws, r, 1, 5);
                r++; shade = !shade;
            }

            ws.Column(1).Width = 22;
            ws.Column(2).Width = 10;
            ws.Column(3).Width = 18;
            ws.Column(4).Width = 18;
            ws.Column(5).Width = 18;
        }

        // ── RESULTS ───────────────────────────────────────────────────────────────

        private static void BuildResultsSheet(XLWorkbook wb, DesignRevision rev,
            CyclonOutputDto? o, CycloneDesign d)
        {
            var ws = wb.Worksheets.Add("Calculated Results");
            ws.ShowGridLines = false;

            int r = 1;
            AddSheetTitle(ws, r, $"CALCULATED RESULTS — {d.TagNumber}  Rev {rev.RevisionNumber}"); r += 2;

            var sections = new[]
            {
                ("Separation Performance", new[]
                {
                    ("Collection Efficiency", $"{(o?.Efficiency ?? 0):F2}",         "%",     "At specified particle size"),
                    ("Cut Diameter (Dpc)",    $"{(o?.CutDiameterMicron ?? 0):F3}",  "µm",    "50% separation point (Lapple)"),
                }),
                ("Pressure Drop", new[]
                {
                    ("Pressure Drop",         $"{(o?.PressureDropPa ?? 0):F2}",     "Pa",    "Shepherd-Lapple model"),
                    ("Pressure Drop",         $"{(o?.PressureDropMmWc ?? 0):F2}",   "mm WC", ""),
                    ("Pressure Drop",         $"{(o?.PressureDropInWc ?? 0):F3}",   "in WC", ""),
                }),
                ("Flow & Velocity", new[]
                {
                    ("Flow Rate",             $"{rev.FlowRateCFM}",                 "CFM",   "Input"),
                    ("Flow Rate",             $"{(o?.FlowRateM3hr ?? 0):F2}",       "m³/hr", "Converted"),
                    ("Inlet Velocity",        $"{(o?.InletVelocityMs ?? 0):F3}",    "m/s",   "At inlet pipe"),
                    ("Inlet Area",            $"{(o?.InletAreaM2 ?? 0):F6}",        "m²",    "Circular pipe cross-section"),
                }),
                ("Gas Properties", new[]
                {
                    ("Gas Density",           $"{(o?.GasDensityKgm3 ?? 0):F4}",     "kg/m³", "Ideal gas law"),
                    ("Gas Viscosity",         $"{(o?.GasViscosityKgms ?? 0):E4}",   "kg/m·s","Sutherland eq."),
                }),
            };

            foreach (var (sectionName, rows) in sections)
            {
                AddSectionHeader(ws, r, 1, sectionName, 4); r++;
                AddResultsTableHeader(ws, r); r++;

                bool shade = false;
                foreach (var (label, val, unit, note) in rows)
                {
                    if (shade) ws.Range(r, 1, r, 4).Style.Fill.BackgroundColor = XLColor.FromHtml(ColLightBlue);
                    ws.Cell(r, 1).Value = label;
                    ws.Cell(r, 2).Value = val;
                    ws.Cell(r, 2).Style.Font.Bold = true;
                    ws.Cell(r, 2).Style.Font.FontColor = XLColor.FromHtml(ColPrimary);
                    ws.Cell(r, 3).Value = unit;
                    ws.Cell(r, 3).Style.Font.FontColor = XLColor.FromHtml(ColMuted);
                    ws.Cell(r, 4).Value = note;
                    ws.Cell(r, 4).Style.Font.FontColor = XLColor.FromHtml(ColMuted);
                    ws.Cell(r, 4).Style.Font.Italic = true;
                    SetRowBorder(ws, r, 1, 4);
                    r++; shade = !shade;
                }
                r++;
            }

            ws.Column(1).Width = 28;
            ws.Column(2).Width = 18;
            ws.Column(3).Width = 12;
            ws.Column(4).Width = 32;
        }

        // ── INPUTS ────────────────────────────────────────────────────────────────

        private static void BuildInputsSheet(XLWorkbook wb, DesignRevision rev,
            CycloneDesign d, Project? proj)
        {
            var ws = wb.Worksheets.Add("Design Inputs");
            ws.ShowGridLines = false;

            int r = 1;
            AddSheetTitle(ws, r, $"DESIGN INPUTS — {d.TagNumber}  Rev {rev.RevisionNumber}"); r += 2;

            var sections = new[]
            {
                ("Process Parameters", new[]
                {
                    ("Gas Flow Rate",       rev.FlowRateCFM.ToString(),       "CFM"),
                    ("Flow Rate (m³/hr)",   rev.FlowRateM3hr.ToString("F2"),  "m³/hr"),
                    ("Inlet Line Size",     rev.InletLineSizeIn.ToString(),   "inches Ø"),
                    ("Inlet Shape",         rev.InletShape.ToString(),        "—"),
                    ("Gas Type",            rev.GasType,                      "—"),
                    ("Operating Temp.",     rev.OperatingTempC.ToString(),    "°C"),
                    ("Operating Pressure",  rev.OperatingPressKPa.ToString(), "kPa"),
                    ("Gas Viscosity",       rev.ViscosityAutoCalc
                                               ? "Auto (Sutherland)"
                                               : rev.GasViscosityKgms.ToString("E4"),
                                            rev.ViscosityAutoCalc ? "" : "kg/m·s"),
                }),
                ("Particle / Solids", new[]
                {
                    ("Particle Size (avg)", rev.ParticleSizeMicron.ToString(), "µm"),
                    ("Particle Size D10",   rev.ParticleSizeD10.ToString(),    "µm"),
                    ("Particle Size D50",   rev.ParticleSizeD50.ToString(),    "µm"),
                    ("Particle Size D90",   rev.ParticleSizeD90.ToString(),    "µm"),
                    ("Particle Density",    rev.ParticleDensityKgm3.ToString(),"kg/m³"),
                    ("Bulk Density",        rev.BulkDensityKgm3.ToString(),    "kg/m³"),
                    ("Shape Factor",        rev.ShapeFactor.ToString(),        "—  (1.0 = sphere)"),
                }),
                ("Design Parameters", new[]
                {
                    ("Effective Turns (Nt)", rev.EffectiveTurns.ToString(),   "—"),
                    ("No. of Cyclones",      rev.NumberOfCyclones.ToString(), "parallel"),
                    ("Safety Factor",        rev.SafetyFactor.ToString(),     "—"),
                    ("Cyclone Type",         d.CycloneType?.Name ?? "—",      "—"),
                    ("Cyclone Code",         d.CycloneType?.Code ?? "—",      "—"),
                }),
            };

            foreach (var (sectionName, rows) in sections)
            {
                AddSectionHeader(ws, r, 1, sectionName, 3); r++;
                bool shade = false;
                foreach (var (label, val, unit) in rows)
                {
                    if (shade) ws.Range(r, 1, r, 3).Style.Fill.BackgroundColor = XLColor.FromHtml(ColSurface);
                    ws.Cell(r, 1).Value = label;
                    ws.Cell(r, 1).Style.Font.FontColor = XLColor.FromHtml(ColMuted);
                    ws.Cell(r, 2).Value = val;
                    ws.Cell(r, 2).Style.Font.Bold = true;
                    ws.Cell(r, 3).Value = unit;
                    ws.Cell(r, 3).Style.Font.FontColor = XLColor.FromHtml(ColMuted);
                    SetRowBorder(ws, r, 1, 3);
                    r++; shade = !shade;
                }
                r++;
            }

            ws.Column(1).Width = 26;
            ws.Column(2).Width = 24;
            ws.Column(3).Width = 20;
        }

        // ── GRADE EFFICIENCY ──────────────────────────────────────────────────────

        private static void BuildEfficiencySheet(XLWorkbook wb, CyclonOutputDto o, DesignRevision rev)
        {
            var ws = wb.Worksheets.Add("Grade Efficiency Curve");
            ws.ShowGridLines = false;

            int r = 1;
            AddSheetTitle(ws, r, $"GRADE EFFICIENCY CURVE — Rev {rev.RevisionNumber}"); r += 2;

            ws.Cell(r, 1).Value = $"Cut Diameter (Dpc): {o.CutDiameterMicron:F3} µm";
            ws.Cell(r, 1).Style.Font.Bold = true;
            ws.Cell(r, 1).Style.Font.FontColor = XLColor.FromHtml(ColPrimary);
            r += 2;

            ws.Range(r, 1, r, 3).Style.Fill.BackgroundColor = XLColor.FromHtml(ColPrimary);
            ws.Range(r, 1, r, 3).Style.Font.FontColor = XLColor.White;
            ws.Range(r, 1, r, 3).Style.Font.Bold = true;
            ws.Cell(r, 1).Value = "Particle Size (µm)";
            ws.Cell(r, 2).Value = "Efficiency (%)";
            ws.Cell(r, 3).Value = "Ratio (Dp / Dpc)";
            r++;

            bool shade = false;
            foreach (var pt in o.GradeEfficiencyCurve)
            {
                if (shade) ws.Range(r, 1, r, 3).Style.Fill.BackgroundColor = XLColor.FromHtml(ColLightBlue);
                ws.Cell(r, 1).Value = pt.ParticleSizeMicron;
                ws.Cell(r, 1).Style.NumberFormat.Format = "0.0##";
                ws.Cell(r, 2).Value = pt.EfficiencyPercent;
                ws.Cell(r, 2).Style.NumberFormat.Format = "0.00";
                ws.Cell(r, 2).Style.Font.Bold = true;
                ws.Cell(r, 2).Style.Font.FontColor = XLColor.FromHtml(
                    pt.EfficiencyPercent >= 95 ? ColSuccess :
                    pt.EfficiencyPercent >= 50 ? ColWarning : ColDanger);
                ws.Cell(r, 3).Value = pt.ParticleSizeMicron / o.CutDiameterMicron;
                ws.Cell(r, 3).Style.NumberFormat.Format = "0.00##";
                SetRowBorder(ws, r, 1, 3);
                r++; shade = !shade;
            }

            ws.Column(1).Width = 22;
            ws.Column(2).Width = 18;
            ws.Column(3).Width = 26;
        }

        // ── SHEET HELPERS ─────────────────────────────────────────────────────────

        private static void AddSheetTitle(IXLWorksheet ws, int row, string title)
        {
            ws.Cell(row, 1).Value = title;
            ws.Cell(row, 1).Style.Font.FontSize = 14;
            ws.Cell(row, 1).Style.Font.Bold = true;
            ws.Cell(row, 1).Style.Font.FontColor = XLColor.FromHtml(ColPrimary);
            ws.Row(row + 1).Height = 3;
            ws.Range(row + 1, 1, row + 1, 7).Style.Fill.BackgroundColor = XLColor.FromHtml(ColPrimary);
        }

        private static void AddSectionHeader(IXLWorksheet ws, int row, int startCol, string title, int span)
        {
            var range = ws.Range(row, startCol, row, startCol + span - 1);
            range.Merge();
            range.Style.Fill.BackgroundColor = XLColor.FromHtml(ColDark);
            range.Style.Font.FontColor = XLColor.White;
            range.Style.Font.Bold = true;
            range.Style.Font.FontSize = 11;
            ws.Cell(row, startCol).Value = title;
        }

        private static void AddResultsTableHeader(IXLWorksheet ws, int row)
        {
            ws.Range(row, 1, row, 4).Style.Fill.BackgroundColor = XLColor.FromHtml("374151");
            ws.Range(row, 1, row, 4).Style.Font.FontColor = XLColor.White;
            ws.Range(row, 1, row, 4).Style.Font.Bold = true;
            ws.Cell(row, 1).Value = "Parameter";
            ws.Cell(row, 2).Value = "Value";
            ws.Cell(row, 3).Value = "Unit";
            ws.Cell(row, 4).Value = "Notes";
        }

        private static void SetRowBorder(IXLWorksheet ws, int row, int c1, int c2)
        {
            ws.Range(row, c1, row, c2).Style.Border.BottomBorder = XLBorderStyleValues.Thin;
            ws.Range(row, c1, row, c2).Style.Border.BottomBorderColor = XLColor.FromHtml(ColBorder);
        }
    }
}