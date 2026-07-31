using CyclonApp.Business.CyclonePrediction;
using CyclonApp.Database;
using CyclonApp.Repositories;
using CyclonApp.Repositories.Contracts;
using CyclonApp.Repositories.Repositories;
using DocumentFormat.OpenXml.Office2016.Drawing.ChartDrawing;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.EntityFrameworkCore;
using System.Linq;
using System.Text.Json;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllersWithViews();


// Database
builder.Services.AddDbContext<ApplicationDbContext>(options =>
    options.UseMySql(
        builder.Configuration.GetConnectionString("DefaultConnection"),
        ServerVersion.AutoDetect(builder.Configuration.GetConnectionString("DefaultConnection"))
    ));

// Cookie auth
builder.Services.AddAuthentication(CookieAuthenticationDefaults.AuthenticationScheme)
    .AddCookie(options =>
    {
        options.LoginPath = "/Account/Login";
        options.LogoutPath = "/Account/Logout";
        options.AccessDeniedPath = "/Account/AccessDenied";
        options.ExpireTimeSpan = TimeSpan.FromHours(8);
    });

builder.Services.AddAuthorization();
builder.Services.AddHttpContextAccessor();

// Repositories
builder.Services.AddScoped<IAccount, AccountRepository>();
builder.Services.AddScoped<IAdminRepository, AdminRepository>();
builder.Services.AddScoped<IDesignRepository, DesignRepository>();
builder.Services.AddScoped<IExport, ExportRepository>();
builder.Services.AddScoped<ProjectRepository>();
builder.Services.AddScoped<ITenant, TenantRepository>();
builder.Services.AddScoped<ICyclonCalculation, CyclonCalculationRepository>();
builder.Services.AddScoped<IUnitOfWork, UnitOfWorks>();
builder.Services.AddScoped<IDashboardRepository, DashboardRepository>();
builder.Services.AddScoped<ICyclonePrediction, CyclonePredictionRepository>();
builder.Services.AddScoped<ExceptionHandlerRepository>();
builder.Services.AddHttpClient("CyclonePrediction"); // name must match CreateClient("CyclonePrediction") in the repository
builder.Services.AddScoped<ICyclonePrediction, CyclonePredictionRepository>();
builder.Services.AddScoped<IEngineeringInsight, EngineeringInsightRepository>();
builder.Services.AddSingleton<CycloneFieldOnnxPredictorProvider>(sp =>
{
    var env = sp.GetRequiredService<IWebHostEnvironment>();
    var config = sp.GetRequiredService<IConfiguration>();

    // Per-cyclone-type model files, e.g.:
    //   "CyclonePredictionService:OnnxModelPathsByType": {
    //     "LAPPLE": "cyclone_model.onnx",
    //     "STAIRMAND": "cyclone_model_stairmand.onnx"
    //   }
    // Each cyclone family (Lapple, Stairmand, ...) is trained separately
    // (see field_train.py's LAPPLE_RATIOS / STAIRMAND_RATIOS) and exported
    // to its own .onnx file, so one model file cannot serve every type.
    var modelPathsByType = config
        .GetSection("CyclonePredictionService:OnnxModelPathsByType")
        .GetChildren()
        .ToDictionary(c => c.Key, c => c.Value ?? "", StringComparer.OrdinalIgnoreCase);

    // Falls back to "cyclone_model.onnx" for any type without an explicit
    // entry — preserves the original single-model behavior (Lapple) if the
    // new config section hasn't been set up yet.
    var defaultFileName = config["CyclonePredictionService:OnnxModelPath"]
                           ?? "cyclone_model.onnx";

    return new CycloneFieldOnnxPredictorProvider(
        Path.Combine(env.ContentRootPath, "Models"),
        modelPathsByType,
        defaultFileName);
});
builder.Services
    .AddControllersWithViews()
    .AddJsonOptions(options =>
    {
        options.JsonSerializerOptions.PropertyNamingPolicy =
            JsonNamingPolicy.CamelCase;
        // Root-cause fix: InsightSeverity (Good/Warning/Critical) was being
        // sent as its raw int (0/1/2). _EngineeringInsights.cshtml's JS
        // compares item.severity === "Critical"/"Warning" as strings, so
        // that comparison always failed and every insight card silently
        // rendered with the "Good" (green check) styling regardless of its
        // real severity — the text said "Critical" but the icon/color said
        // "fine". Serializing enums by name fixes the mismatch.
        options.JsonSerializerOptions.Converters.Add(
            new System.Text.Json.Serialization.JsonStringEnumConverter());
    });
var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();
app.UseRouting();
app.UseAuthentication();
app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Account}/{action=Login}/{id?}");


// Create schema (no EF migrations exist yet) and seed default admin
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();
    // NOTE: EnsureCreated() builds tables directly from the current model,
    // with no migration history. Do NOT mix this with EF migrations later —
    // if you ever run `dotnet ef migrations add`, switch this to
    // db.Database.Migrate() instead and drop EnsureCreated().
    await db.Database.EnsureCreatedAsync();
    await SeedData.InitializeAsync(db);
}

app.Run();