"""
export_onnx.py
──────────────
Loads a trained CycloneFieldPINN checkpoint (.pth, as saved by
field_train.py: model_state_dict / hidden / n_layers / scaler_state_dict)
and exports it to cyclone_model.onnx for the CPU-only C# inference path
(CyclonApp.Repositories.Contracts.CycloneFieldOnnxPredictor).

The exported graph bakes in the trained weights AND the FieldScaler
constants (L, U, P, K, E, D_min/max, Q_min/max) captured at training time,
so the .onnx file takes raw physical units (r, z in meters; diameter_m in
meters; flow_rate_cfm in cfm) and returns raw physical units directly --
matching CycloneFieldOnnxPredictor's documented contract exactly:

  inputs:  r, z, diameter_m, flow_rate_cfm   (float32, shape [N])
  outputs: v_r, v_theta, v_z, p, k, eps      (float32, shape [N])

Usage:
    python export_onnx.py <checkpoint.pth> [output.onnx]
"""
from __future__ import annotations
import sys
import torch
import torch.nn as nn

from field_model import CycloneFieldPINN, FieldScaler


class _OnnxExportWrapper(nn.Module):
    """Bakes a fixed FieldScaler into the forward pass so the exported
    graph's inputs/outputs are raw physical units, not normalized ones --
    the scaler itself has no learnable parameters, so this only fixes the
    (already-trained) normalization constants, it does not retrain anything.
    """

    def __init__(self, model: CycloneFieldPINN, scaler: FieldScaler):
        super().__init__()
        self.model = model
        self.scaler = scaler

    def forward(self, r, z, diameter_m, flow_rate_cfm):
        out = self.model(r, z, diameter_m, flow_rate_cfm, self.scaler)
        # Order here fixes the order ONNX assigns output_names to below.
        return out["v_r"], out["v_theta"], out["v_z"], out["p"], out["k"], out["eps"]


def export(checkpoint_path: str, output_path: str = "cyclone_model.onnx") -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = CycloneFieldPINN(hidden=ckpt["hidden"], n_layers=ckpt["n_layers"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = FieldScaler.from_state_dict(ckpt["scaler_state_dict"])

    wrapper = _OnnxExportWrapper(model, scaler)
    wrapper.eval()

    # Dummy trace inputs -- values don't matter (no data-dependent control
    # flow in the network), only shape/dtype do. N=3 (not 1) so dynamic_axes
    # on dim 0 aren't accidentally inferred as fixed-size-1.
    n = 3
    dummy_r = torch.linspace(0.0, scaler.L, n, dtype=torch.float32)
    dummy_z = torch.linspace(0.0, scaler.L * 2, n, dtype=torch.float32)
    dummy_d = torch.full((n,), (scaler.D_min + scaler.D_max) / 2.0, dtype=torch.float32)
    dummy_q = torch.full((n,), ckpt.get("flow_rate_cfm", (scaler.Q_min + scaler.Q_max) / 2.0), dtype=torch.float32)

    input_names = ["r", "z", "diameter_m", "flow_rate_cfm"]
    output_names = ["v_r", "v_theta", "v_z", "p", "k", "eps"]
    dynamic_axes = {name: {0: "N"} for name in input_names + output_names}

    torch.onnx.export(
        wrapper,
        (dummy_r, dummy_z, dummy_d, dummy_q),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported {output_path}")
    print(f"  hidden={ckpt['hidden']} n_layers={ckpt['n_layers']}")
    print(f"  scaler: L={scaler.L} U={scaler.U} P={scaler.P} K={scaler.K} E={scaler.E}")
    print(f"  D range=({scaler.D_min}, {scaler.D_max}) Q range=({scaler.Q_min}, {scaler.Q_max})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_onnx.py <checkpoint.pth> [output.onnx]")
        sys.exit(1)
    ckpt_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "cyclone_model.onnx"
    export(ckpt_path, out_path)
