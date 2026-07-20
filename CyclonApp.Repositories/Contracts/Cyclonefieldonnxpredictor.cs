using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace CyclonApp.Business.CyclonePrediction
{
    /// <summary>
    /// Result of a single field-prediction query: one value per input (r, z)
    /// point. All arrays are the same length as the r/z arrays passed in.
    /// </summary>
    public sealed class CycloneFieldResult
    {
        public float[] VR { get; init; } = Array.Empty<float>();
        public float[] VTheta { get; init; } = Array.Empty<float>();
        public float[] VZ { get; init; } = Array.Empty<float>();
        public float[] P { get; init; } = Array.Empty<float>();
        public float[] K { get; init; } = Array.Empty<float>();
        public float[] Eps { get; init; } = Array.Empty<float>();
    }

    /// <summary>
    /// CPU inference wrapper around cyclone_model.onnx, exported from the
    /// Python CycloneFieldPINN (see field_model.py / the Colab ONNX export
    /// cell). No Python or training happens here — this loads the frozen
    /// weights + scaler constants baked in at export time and evaluates the
    /// network directly.
    ///
    /// IMPORTANT: this checkpoint (and therefore this .onnx file) is trained
    /// for ONE fixed geometry/operating point (see field_train.py's
    /// "PRODUCTION INFERENCE MODE" note). diameter_m/flow_rate_cfm are
    /// still required inputs because CycloneFieldPINN takes them as explicit
    /// conditioning inputs, but passing values far from what this specific
    /// checkpoint was trained on (see field_model.py's D_min/D_max/Q_min/Q_max
    /// normalization window) will extrapolate, not interpolate, and is not
    /// guaranteed to be physically meaningful.
    ///
    /// Input tensor names (must match torch.onnx.export's input_names):
    ///   "r", "z", "diameter_m", "flow_rate_cfm" — all float32, shape [N],
    ///   same N across all four (diameter_m/flow_rate_cfm are typically the
    ///   same scalar value repeated N times, matching how field_model.py's
    ///   evaluate_grid() builds them in Python).
    ///
    /// Output tensor names (must match torch.onnx.export's output_names):
    ///   "v_r", "v_theta", "v_z", "p", "k", "eps" — all float32, shape [N].
    /// </summary>
    public sealed class CycloneFieldOnnxPredictor : IDisposable
    {
        private readonly InferenceSession _session;

        // Must exactly match input_names / output_names passed to
        // torch.onnx.export in the Colab conversion cell.
        private const string InputR = "r";
        private const string InputZ = "z";
        private const string InputDiameterM = "diameter_m";
        private const string InputFlowRateCfm = "flow_rate_cfm";

        private const string OutputVR = "v_r";
        private const string OutputVTheta = "v_theta";
        private const string OutputVZ = "v_z";
        private const string OutputP = "p";
        private const string OutputK = "k";
        private const string OutputEps = "eps";

        public CycloneFieldOnnxPredictor(string onnxModelPath)
        {
            if (string.IsNullOrWhiteSpace(onnxModelPath))
                throw new ArgumentException("onnxModelPath must be provided.", nameof(onnxModelPath));
            if (!System.IO.File.Exists(onnxModelPath))
                throw new System.IO.FileNotFoundException(
                    $"ONNX model not found at '{onnxModelPath}'. Confirm the file was " +
                    $"downloaded from Drive and deployed alongside this service.",
                    onnxModelPath);

            // Default SessionOptions run on CPU, which is the point of this
            // ONNX path (no GPU/Python needed at inference time).
            _session = new InferenceSession(onnxModelPath);
        }

        /// <summary>
        /// Evaluates the field at every (r[i], z[i]) point for a single
        /// fixed (diameter_m, flow_rate_cfm) design. r and z must be the
        /// same length; diameter_m/flow_rate_cfm are broadcast to that same
        /// length internally (mirroring evaluate_grid's
        /// torch.full_like(r_valid, diameter_m) in field_model.py).
        /// </summary>
        public CycloneFieldResult Predict(float[] r, float[] z, float diameterM, float flowRateCfm)
        {
            if (r is null) throw new ArgumentNullException(nameof(r));
            if (z is null) throw new ArgumentNullException(nameof(z));
            if (r.Length != z.Length)
                throw new ArgumentException(
                    $"r and z must be the same length (got r.Length={r.Length}, z.Length={z.Length}).");
            if (r.Length == 0)
            {
                return new CycloneFieldResult(); // empty in, empty out — mirrors evaluate_grid's degenerate-domain case
            }

            int n = r.Length;
            var diameterArr = new float[n];
            var flowRateArr = new float[n];
            Array.Fill(diameterArr, diameterM);
            Array.Fill(flowRateArr, flowRateCfm);

            var rTensor = new DenseTensor<float>(r, new[] { n });
            var zTensor = new DenseTensor<float>(z, new[] { n });
            var dTensor = new DenseTensor<float>(diameterArr, new[] { n });
            var qTensor = new DenseTensor<float>(flowRateArr, new[] { n });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(InputR, rTensor),
                NamedOnnxValue.CreateFromTensor(InputZ, zTensor),
                NamedOnnxValue.CreateFromTensor(InputDiameterM, dTensor),
                NamedOnnxValue.CreateFromTensor(InputFlowRateCfm, qTensor),
            };

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = _session.Run(inputs);

            return new CycloneFieldResult
            {
                VR = ExtractOutput(outputs, OutputVR),
                VTheta = ExtractOutput(outputs, OutputVTheta),
                VZ = ExtractOutput(outputs, OutputVZ),
                P = ExtractOutput(outputs, OutputP),
                K = ExtractOutput(outputs, OutputK),
                Eps = ExtractOutput(outputs, OutputEps),
            };
        }

        private static float[] ExtractOutput(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, string name)
        {
            DisposableNamedOnnxValue? match = outputs.FirstOrDefault(o => o.Name == name);
            if (match is null)
                throw new InvalidOperationException(
                    $"ONNX model output did not contain expected tensor '{name}'. " +
                    $"Confirm this .onnx was exported with output_names matching " +
                    $"CycloneFieldOnnxPredictor's expected names.");

            return match.AsEnumerable<float>().ToArray();
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}