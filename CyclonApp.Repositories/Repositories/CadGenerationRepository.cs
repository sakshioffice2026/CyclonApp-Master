using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading.Tasks;
using CyclonApp.Database;
using CyclonApp.Model.DTOs;
using CyclonApp.Repositories.Contracts;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;

namespace CyclonApp.Repositories.Repositories
{
    public class CadGenerationRepository : ICadGeneration
    {
        private readonly ApplicationDbContext _db;
        private readonly IHttpClientFactory _httpClientFactory;
        private readonly string _baseUrl;

        private static readonly JsonSerializerOptions _readOpts =
            new() { PropertyNameCaseInsensitive = true };

        // Same reasoning as CyclonePredictionRepository.OutgoingJsonOptions:
        // PostAsJsonAsync defaults to camelCase, but app.py's Pydantic
        // request model only accepts its declared PascalCase aliases.
        private static readonly JsonSerializerOptions OutgoingJsonOptions = new()
        {
            PropertyNamingPolicy = null
        };

        public CadGenerationRepository(
            ApplicationDbContext db,
            IHttpClientFactory httpClientFactory,
            IConfiguration configuration)
        {
            _db = db;
            _httpClientFactory = httpClientFactory;
            // Same config key + same Python host as CyclonePredictionRepository
            // — one FastAPI service serves both /predict_field/* and
            // /generate_cad, so there is only one base URL to configure.
            _baseUrl = configuration["CyclonePredictionService:BaseUrl"]
                       ?? "http://localhost:8000";
        }

        public async Task<CadGenerationResultDto> GenerateCadAsync(int revisionId)
        {
            var rev = await _db.DesignRevisions
                .Include(r => r.CycloneDesign)
                .FirstOrDefaultAsync(r => r.Id == revisionId)
                ?? throw new KeyNotFoundException($"Revision {revisionId} not found.");

            if (string.IsNullOrEmpty(rev.EfficiencyJson))
                throw new InvalidOperationException(
                    $"Revision {revisionId} has no calculated dimensions yet. " +
                    "Run the design calculation before generating CAD.");

            var output = JsonSerializer.Deserialize<CyclonOutputDto>(rev.EfficiencyJson, _readOpts)
                         ?? throw new InvalidOperationException(
                             $"Revision {revisionId}'s calculation output could not be read.");

            var dims = output.Dimensions
                       ?? throw new InvalidOperationException(
                           $"Revision {revisionId} has no dimensions in its calculation output.");

            var request = new GenerateCadRequest
            {
                RevisionId = revisionId,
                BarrelDiameterMm = dims.BarrelDiameterMm,
                BarrelHeightMm = dims.BarrelHeightMm,
                ConeHeightMm = dims.ConeHeightMm,
                ExhaustDiaMm = dims.ExhaustDiaMm,
                ExhaustLengthMm = dims.ExhaustLengthMm,
                BottomOutletMm = dims.BottomOutletMm,
                InletHeightMm = dims.InletHeightMm,
                InletWidthMm = dims.InletWidthMm
            };

            var client = _httpClientFactory.CreateClient("CyclonePrediction");
            client.BaseAddress = new Uri(_baseUrl);
            // FreeCAD subprocess can legitimately take up to the service's
            // own 120s internal timeout — give the HTTP call enough room.
            client.Timeout = TimeSpan.FromSeconds(150);

            var response = await client.PostAsJsonAsync("/generate_cad", request, OutgoingJsonOptions);

            if (!response.IsSuccessStatusCode)
            {
                var detail = await TryReadErrorDetailAsync(response);
                throw new Exception(
                    $"CAD generation service returned {(int)response.StatusCode} " +
                    $"{response.StatusCode}: {detail ?? "(no error detail in response body)"}");
            }

            var wire = await response.Content.ReadFromJsonAsync<GenerateCadResponse>()
                       ?? throw new Exception("CAD generation service returned an empty response.");

            // All returned paths are relative (e.g. "/cad-exports/12/cyclone.step")
            // — resolve against _baseUrl, same as PngUrl in CyclonePredictionRepository.
            string? Resolve(string? relative) =>
                string.IsNullOrEmpty(relative) ? null : new Uri(new Uri(_baseUrl), relative).ToString();

            return new CadGenerationResultDto
            {
                StepUrl = Resolve(wire.StepUrl),
                DxfUrl = Resolve(wire.DxfUrl),
                PdfUrl = Resolve(wire.PdfUrl),
                ObjUrl = Resolve(wire.ObjUrl),
                AllPartsDxfUrl = Resolve(wire.AllPartsDxfUrl)
            };
        }

        private static async Task<string?> TryReadErrorDetailAsync(HttpResponseMessage response)
        {
            try
            {
                var body = await response.Content.ReadFromJsonAsync<Dictionary<string, object>>();
                return body != null && body.TryGetValue("detail", out var detail) ? detail?.ToString() : null;
            }
            catch
            {
                return null;
            }
        }

        // ── Wire-format classes matching app.py's GenerateCadRequest/Response ──
        private class GenerateCadRequest
        {
            public int RevisionId { get; set; }
            public double BarrelDiameterMm { get; set; }
            public double BarrelHeightMm { get; set; }
            public double ConeHeightMm { get; set; }
            public double ExhaustDiaMm { get; set; }
            public double ExhaustLengthMm { get; set; }
            public double BottomOutletMm { get; set; }
            public double InletHeightMm { get; set; }
            public double InletWidthMm { get; set; }
        }

        private class GenerateCadResponse
        {
            public string? StepUrl { get; set; }
            public string? DxfUrl { get; set; }
            public string? PdfUrl { get; set; }
            public string? ObjUrl { get; set; }
            public string? AllPartsDxfUrl { get; set; }
        }
    }
}
