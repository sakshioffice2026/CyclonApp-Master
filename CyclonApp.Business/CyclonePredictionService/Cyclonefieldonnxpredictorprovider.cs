using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;

namespace CyclonApp.Business.CyclonePrediction
{
    /// <summary>
    /// Resolves the trained ONNX field model for a given cyclone type
    /// (e.g. "LAPPLE", "STAIRMAND") and caches one CycloneFieldOnnxPredictor
    /// per type for the lifetime of the app.
    ///
    /// Each cyclone family has its own checkpoint because
    /// CycloneFieldPINN's parametric inputs are (r, z, diameter_m,
    /// flow_rate_cfm) only — the dimension RATIOS that define a family's
    /// shape (see field_train.py's LAPPLE_RATIOS / STAIRMAND_RATIOS) are
    /// baked in at training time, not passed at inference time. So a model
    /// trained on Lapple-shaped geometry cannot be reused for Stairmand
    /// geometry, and vice versa — each type needs its own .onnx file here.
    ///
    /// Registered as a singleton (see Program.cs); sessions are created
    /// lazily on first use per type and reused after that.
    /// </summary>
    public sealed class CycloneFieldOnnxPredictorProvider : IDisposable
    {
        private readonly string _modelsDirectory;
        private readonly IReadOnlyDictionary<string, string> _modelFileNamesByTypeCode;
        private readonly string _defaultFileName;
        private readonly ConcurrentDictionary<string, CycloneFieldOnnxPredictor> _predictorsByTypeCode = new();

        /// <param name="modelsDirectory">Absolute path to the folder containing the .onnx files (typically ContentRootPath/Models).</param>
        /// <param name="modelFileNamesByTypeCode">Maps CycloneType.Code (e.g. "STAIRMAND") to its .onnx file name. Keys are matched case-insensitively.</param>
        /// <param name="defaultFileName">Used when a type code has no explicit entry — kept for backward compatibility with the original single-model setup.</param>
        public CycloneFieldOnnxPredictorProvider(
            string modelsDirectory,
            IReadOnlyDictionary<string, string> modelFileNamesByTypeCode,
            string defaultFileName = "cyclone_model.onnx")
        {
            _modelsDirectory = modelsDirectory ?? throw new ArgumentNullException(nameof(modelsDirectory));
            _modelFileNamesByTypeCode = new Dictionary<string, string>(
                modelFileNamesByTypeCode ?? new Dictionary<string, string>(),
                StringComparer.OrdinalIgnoreCase);
            _defaultFileName = defaultFileName;
        }

        /// <summary>
        /// Returns the predictor for the given cyclone type code, loading and
        /// caching its ONNX session on first request. Throws
        /// FileNotFoundException with a message naming the missing model file
        /// if that type hasn't been trained/exported yet (e.g. Stairmand
        /// before its cyclone_model_stairmand.onnx has been produced by
        /// export_onnx.py) — callers should surface this as a clear
        /// "not available for this cyclone type yet" error rather than a
        /// generic 500.
        /// </summary>
        public CycloneFieldOnnxPredictor GetPredictor(string cycloneTypeCode)
        {
            if (string.IsNullOrWhiteSpace(cycloneTypeCode))
                throw new ArgumentException("cycloneTypeCode must be provided.", nameof(cycloneTypeCode));

            return _predictorsByTypeCode.GetOrAdd(cycloneTypeCode.Trim().ToUpperInvariant(), key =>
            {
                var fileName = _modelFileNamesByTypeCode.TryGetValue(key, out var mapped)
                    ? mapped
                    : _defaultFileName;

                var fullPath = Path.Combine(_modelsDirectory, fileName);
                return new CycloneFieldOnnxPredictor(fullPath);
            });
        }

        public void Dispose()
        {
            foreach (var predictor in _predictorsByTypeCode.Values)
                predictor.Dispose();
            _predictorsByTypeCode.Clear();
        }
    }
}