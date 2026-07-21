"""
export_onnx.py
──────────────
Loads a trained CycloneFieldPINN checkpoint (.pth, as saved by
field_train.py: either train_field_model's single-geometry
save_field_checkpoint, or train_parametric_field_model's
save_parametric_field_checkpoint) and exports it to .onnx for the
CPU-only C# inference path
(CyclonApp.Repositories.Contracts.CycloneFieldOnnxPredictor).

  inputs:  r, z, diameter_m, flow_rate_cfm   (float32, shape [N])
  outputs: v_r, v_theta, v_z, p, k, eps      (float32, shape [N])

TWO DIFFERENT WRAPPERS, ONE PER CHECKPOINT KIND — NOT INTERCHANGEABLE:

  * Single-geometry checkpoints (no "checkpoint_kind" key, or
    checkpoint_kind != "parametric") carry a scaler_state_dict with the
    REAL trained L/U/P/K/E for the one fixed geometry they cover.
    _FixedScaleWrapper bakes that scaler in as-is. This is unchanged from
    before and is what produced the currently-deployed, verified-correct
    cyclone_model.onnx (LAPPLE).

  * Parametric checkpoints (checkpoint_kind == "parametric") carry a
    scaler_state_dict whose L/U/P/K/E are the template_scaler's
    PLACEHOLDER values (length_scale_m=1.0, velocity_scale_ms=1.0,
    rho=1.0 — see train_parametric_field_model's docstring: "its L/U/P/K/E
    are meaningless here and are NOT saved [for inference] — app.py
    recomputes them per-request via FieldScaler.with_scales"). Baking
    those placeholders in directly (as the previous version of this file
    did) produces a graph that returns physically meaningless numbers for
    every query. _ParametricScaleWrapper instead recomputes the correct
    per-geometry L (=barrel radius), U (=inlet velocity), P/K/E (derived
    from U, L and the checkpoint's fixed rho) INSIDE the traced graph,
    from diameter_m/flow_rate_cfm — the same relationship
    train_parametric_field_model computes fresh every epoch via
    geometry_mm_from_diameter + inlet_velocity_ms + FieldScaler.with_scales
    — so the exported graph is a genuine function of only the four
    documented inputs, matching CycloneFieldOnnxPredictor.cs, for ANY
    (diameter_m, flow_rate_cfm) inside the checkpoint's trained range —
    not just the one design a fixed scaler would silently assume.

Usage:
    python export_onnx.py <checkpoint.pth> [output.onnx]
"""
from __future__ import annotations
import sys
import torch
import torch.nn as nn

from field_model import CycloneFieldPINN, FieldScaler
from field_physics import CFM_TO_M3S, EPS, fluid_properties, gas_type_to_onehot


class _FixedScaleWrapper(nn.Module):
    """Single-geometry checkpoints: bake the checkpoint's own (real,
    trained-for-this-one-geometry) FieldScaler in as-is. Unchanged
    behavior from before this fix — this is the path LAPPLE's currently
    deployed, verified-correct cyclone_model.onnx went through."""

    def __init__(self, model: CycloneFieldPINN, scaler: FieldScaler):
        super().__init__()
        self.model = model
        self.scaler = scaler

    def forward(self, r, z, diameter_m, flow_rate_cfm):
        out = self.model(r, z, diameter_m, flow_rate_cfm, self.scaler)
        return out["v_r"], out["v_theta"], out["v_z"], out["p"], out["k"], out["eps"]


class _ParametricScaleWrapper(nn.Module):
    """Parametric checkpoints: recompute the real per-geometry L/U/P/K/E
    from (diameter_m, flow_rate_cfm) inside the graph, instead of trusting
    the checkpoint's placeholder scaler_state_dict. inlet_h_ratio/
    inlet_w_ratio/rho are fixed constants captured at export time (from
    the checkpoint's saved `ratios`/`operating_temp_c`/
    `operating_press_kpa`/`gas_type` — the same fixed assumptions
    train_parametric_field_model trained under), NOT learned parameters."""

    def __init__(
        self,
        model: CycloneFieldPINN,
        d_min: float, d_max: float,
        q_min: float, q_max: float,
        inlet_height_ratio: float,
        inlet_width_ratio: float,
        rho: float,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("d_min", torch.tensor(float(d_min)))
        self.register_buffer("d_max", torch.tensor(float(d_max)))
        self.register_buffer("q_min", torch.tensor(float(q_min)))
        self.register_buffer("q_max", torch.tensor(float(q_max)))
        self.register_buffer("inlet_h_ratio", torch.tensor(float(inlet_height_ratio)))
        self.register_buffer("inlet_w_ratio", torch.tensor(float(inlet_width_ratio)))
        self.register_buffer("rho", torch.tensor(float(rho)))

    def forward(self, r, z, diameter_m, flow_rate_cfm):
        # Same derivation geometry_mm_from_diameter + inlet_velocity_ms
        # use, just elementwise on tensors so it traces into the graph:
        # r_barrel = D/2; inlet dims scale off D by the fixed ratios; V =
        # Q / inlet_area.
        r_barrel = diameter_m / 2.0
        inlet_h_m = diameter_m * self.inlet_h_ratio
        inlet_w_m = diameter_m * self.inlet_w_ratio
        q_m3s = flow_rate_cfm * CFM_TO_M3S
        v_inlet = q_m3s / (inlet_h_m * inlet_w_m + EPS)
        v_inlet = torch.clamp(v_inlet, min=1e-6)

        scaler = FieldScaler(
            length_scale_m=r_barrel,
            velocity_scale_ms=v_inlet,
            rho=self.rho,
            diameter_range_m=(self.d_min, self.d_max),
            flow_rate_range_cfm=(self.q_min, self.q_max),
        )
        out = self.model(r, z, diameter_m, flow_rate_cfm, scaler)
        return out["v_r"], out["v_theta"], out["v_z"], out["p"], out["k"], out["eps"]


def export(checkpoint_path: str, output_path: str = "cyclone_model.onnx") -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    is_parametric = ckpt.get("checkpoint_kind") == "parametric"

    model = CycloneFieldPINN(hidden=ckpt["hidden"], n_layers=ckpt["n_layers"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = FieldScaler.from_state_dict(ckpt["scaler_state_dict"])

    if is_parametric:
        ratios = ckpt.get("ratios")
        if not ratios:
            raise RuntimeError(
                f"'{checkpoint_path}' is a parametric checkpoint but has no "
                f"saved 'ratios' — cannot compute per-geometry inlet "
                f"dimensions without them. Re-save it with "
                f"save_parametric_field_checkpoint(..., ratios=...)."
            )
        gas_onehot = torch.tensor([gas_type_to_onehot(ckpt.get("gas_type") or "Air")])
        rho_t, _ = fluid_properties(
            torch.tensor([float(ckpt.get("operating_temp_c", 25.0))]),
            torch.tensor([float(ckpt.get("operating_press_kpa", 101.325))]),
            gas_onehot,
        )
        wrapper = _ParametricScaleWrapper(
            model=model,
            d_min=scaler.D_min, d_max=scaler.D_max,
            q_min=scaler.Q_min, q_max=scaler.Q_max,
            inlet_height_ratio=ratios["InletHeightRatio"],
            inlet_width_ratio=ratios["InletWidthRatio"],
            rho=rho_t.item(),
        )
        dummy_d = torch.full((3,), (scaler.D_min + scaler.D_max) / 2.0, dtype=torch.float32)
        dummy_q = torch.full((3,), (scaler.Q_min + scaler.Q_max) / 2.0, dtype=torch.float32)
        dummy_r = torch.linspace(0.0, float(dummy_d[0]) / 2.0, 3, dtype=torch.float32)
        dummy_z = torch.linspace(0.0, float(dummy_d[0]), 3, dtype=torch.float32)
    else:
        wrapper = _FixedScaleWrapper(model, scaler)
        dummy_r = torch.linspace(0.0, scaler.L, 3, dtype=torch.float32)
        dummy_z = torch.linspace(0.0, scaler.L * 2, 3, dtype=torch.float32)
        dummy_d = torch.full((3,), (scaler.D_min + scaler.D_max) / 2.0, dtype=torch.float32)
        dummy_q = torch.full((3,), ckpt.get("flow_rate_cfm", (scaler.Q_min + scaler.Q_max) / 2.0), dtype=torch.float32)

    wrapper.eval()

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
    print(f"Exported {output_path}  (checkpoint_kind={'parametric' if is_parametric else 'single'})")
    print(f"  hidden={ckpt['hidden']} n_layers={ckpt['n_layers']}")
    if is_parametric:
        print(f"  per-geometry L/U/P/K/E now computed in-graph from "
              f"diameter_m/flow_rate_cfm (rho={wrapper.rho.item():.5f} kg/m3, "
              f"InletHeightRatio={wrapper.inlet_h_ratio.item()}, "
              f"InletWidthRatio={wrapper.inlet_w_ratio.item()})")
    else:
        print(f"  scaler: L={scaler.L} U={scaler.U} P={scaler.P} K={scaler.K} E={scaler.E}")
    print(f"  D range=({scaler.D_min}, {scaler.D_max}) Q range=({scaler.Q_min}, {scaler.Q_max})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_onnx.py <checkpoint.pth> [output.onnx]")
        sys.exit(1)
    ckpt_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "cyclone_model.onnx"
    export(ckpt_path, out_path)