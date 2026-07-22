"""
test_onnx_geometries.py
────────────────────────
Quick sanity check for cyclone_model_stairmand_fixed.onnx (or any
parametric export from export_onnx.py).

Feeds the SAME (r, z) sample points through two different
(diameter_m, flow_rate_cfm) geometries and prints v_r/v_theta/v_z/p/k/eps
for each. If the parametric fix is working, the two output sets should
differ meaningfully (not be identical, and not be NaN/zero).

Requires: pip install onnxruntime numpy --break-system-packages
(or just: pip install onnxruntime numpy)

Usage:
    python test_onnx_geometries.py cyclone_model_stairmand_fixed.onnx
"""
import sys
import numpy as np
import onnxruntime as ort

ONNX_PATH = sys.argv[1] if len(sys.argv) > 1 else "cyclone_model_stairmand_fixed.onnx"

# Two clearly different geometries/flow rates to compare.
# (diameter_m, flow_rate_cfm) — pick values inside the checkpoint's
# trained range: D in [0.15, 0.75] m, Q in [300, 13000] cfm.
CASE_A = {"name": "small/low-flow",  "diameter_m": 0.20, "flow_rate_cfm": 500.0}
CASE_B = {"name": "large/high-flow", "diameter_m": 0.60, "flow_rate_cfm": 10000.0}

# Same (r, z) sample points for both cases, so any difference in the
# outputs is coming purely from diameter_m/flow_rate_cfm, not from r/z.
N = 5
R = np.linspace(0.01, 0.30, N).astype(np.float32)
Z = np.linspace(0.0, 0.60, N).astype(np.float32)

OUTPUT_NAMES = ["v_r", "v_theta", "v_z", "p", "k", "eps"]


def run_case(session: ort.InferenceSession, case: dict) -> dict:
    diameter_m = np.full(N, case["diameter_m"], dtype=np.float32)
    flow_rate_cfm = np.full(N, case["flow_rate_cfm"], dtype=np.float32)
    outputs = session.run(
        OUTPUT_NAMES,
        {
            "r": R,
            "z": Z,
            "diameter_m": diameter_m,
            "flow_rate_cfm": flow_rate_cfm,
        },
    )
    return dict(zip(OUTPUT_NAMES, outputs))


def main() -> None:
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    results = {}
    for case in (CASE_A, CASE_B):
        results[case["name"]] = run_case(session, case)
        print(f"\n=== {case['name']}  (D={case['diameter_m']} m, Q={case['flow_rate_cfm']} cfm) ===")
        for name in OUTPUT_NAMES:
            vals = results[case["name"]][name]
            print(f"  {name:8s}: {np.array2string(vals, precision=4)}")

    print("\n=== Difference check (large minus small) ===")
    any_nan = False
    all_identical = True
    for name in OUTPUT_NAMES:
        a = results[CASE_A["name"]][name]
        b = results[CASE_B["name"]][name]
        diff = b - a
        if np.isnan(a).any() or np.isnan(b).any():
            any_nan = True
        if not np.allclose(a, b, atol=1e-6):
            all_identical = False
        print(f"  {name:8s} diff: {np.array2string(diff, precision=4)}")

    print()
    if any_nan:
        print("FAIL: NaNs found in outputs — something is broken.")
    elif all_identical:
        print("FAIL: outputs are identical across geometries — parametric "
              "inputs are being ignored (the bug we were checking for).")
    else:
        print("PASS: outputs differ meaningfully across geometries — "
              "the per-geometry parametric fix is working end-to-end.")


if __name__ == "__main__":
    main()