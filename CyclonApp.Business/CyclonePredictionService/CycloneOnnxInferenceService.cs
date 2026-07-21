// CyclonApp/Services/CycloneOnnxInferenceService.cs
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Extensions.Options;

namespace CyclonApp.Services
{
    public sealed class CycloneOnnxInferenceService : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly ModelSettings _settings;

        public CycloneOnnxInferenceService(IOptions<ModelSettings> settings)
        {
            _settings = settings.Value;

            var fullPath = Path.Combine(AppContext.BaseDirectory, _settings.OnnxModelPath);
            if (!File.Exists(fullPath))
                throw new FileNotFoundException($"ONNX model not found at {fullPath}. " +
                    "Verify Copy to Output Directory is set on cyclone_model.onnx.");

            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                IntraOpNumThreads = 1 // small point counts per grid — see note below
            };

            _session = new InferenceSession(fullPath, options);
        }

        /// <summary>
        /// r, z: flattened grid coordinate arrays (metres), equal length N.
        /// diameterM, flowRateCfm: scalar operating params, replicated to N internally.
        /// Returns 6 output arrays each of length N: (vR, vTheta, vZ, p, k, eps).
        /// </summary>
        public (float[] vR, float[] vTheta, float[] vZ, float[] p, float[] k, float[] eps)
            Predict(float[] r, float[] z, float diameterM, float flowRateCfm)
        {
            int n = r.Length;
            if (z.Length != n)
                throw new ArgumentException("r and z must be equal length.");

            var diameterArr = new float[n];
            var flowRateArr = new float[n];
            Array.Fill(diameterArr, diameterM);
            Array.Fill(flowRateArr, flowRateCfm);

            var rTensor = new DenseTensor<float>(r, new[] { n, 1 });
            var zTensor = new DenseTensor<float>(z, new[] { n, 1 });
            var dTensor = new DenseTensor<float>(diameterArr, new[] { n, 1 });
            var qTensor = new DenseTensor<float>(flowRateArr, new[] { n, 1 });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_settings.InputNames.R, rTensor),
                NamedOnnxValue.CreateFromTensor(_settings.InputNames.Z, zTensor),
                NamedOnnxValue.CreateFromTensor(_settings.InputNames.Diameter, dTensor),
                NamedOnnxValue.CreateFromTensor(_settings.InputNames.FlowRate, qTensor)
            };

            using var results = _session.Run(inputs, new[] { _settings.OutputName });
            var outTensor = results.First(x => x.Name == _settings.OutputName).AsTensor<float>();
            // Expected shape: [n, 6] -> (v_r, v_theta, v_z, p, k, eps)

            var vR = new float[n]; var vTheta = new float[n]; var vZ = new float[n];
            var p = new float[n]; var k = new float[n]; var eps = new float[n];

            for (int i = 0; i < n; i++)
            {
                vR[i] = outTensor[i, 0];
                vTheta[i] = outTensor[i, 1];
                vZ[i] = outTensor[i, 2];
                p[i] = outTensor[i, 3];
                k[i] = outTensor[i, 4];
                eps[i] = outTensor[i, 5];
            }

            return (vR, vTheta, vZ, p, k, eps);
        }

        public void Dispose() => _session.Dispose();
    }

    public sealed class ModelSettings
    {
        public string BaseUrl { get; set; } = "http://localhost:8000"; // untouched, still used by HTTP repo
        public string OnnxModelPath { get; set; } = "cyclone_model.onnx";
        public InputNameSettings InputNames { get; set; } = new();
        public string OutputName { get; set; } = "output";
    }

    public sealed class InputNameSettings
    {
        public string R { get; set; } = "r";
        public string Z { get; set; } = "z";
        public string Diameter { get; set; } = "diameter_m";
        public string FlowRate { get; set; } = "flow_rate_cfm";
    }
}