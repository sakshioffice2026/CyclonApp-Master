"""
debug_pressure_breakdown.py
────────────────────────────
Instruments compute_pressure_drop's inlet/outlet averaging with a full
component breakdown -- static pressure, and each of v_r/v_theta/v_z
separately -- for ONE (diameter, flow) case, so the field/baseline ratio
gap seen in validate_pressure_drop.py can be diagnosed from real trained-
model numbers instead of hand-derived algebra.

Usage:
    python debug_pressure_breakdown.py cyclone_model_parametric.pth
    python debug_pressure_breakdown.py cyclone_model_parametric.pth --diameter-mm 400 --flow-cfm 3000
"""
from __future__ import annotations

import argparse
import math

import torch

from field_train import load_parametric_field_checkpoint, LAPPLE_RATIOS
from field_physics import (
    fluid_properties, gas_type_to_onehot, geometry_from_dimensions_mm,
    inlet_velocity_ms, inlet_axial_velocity_ms, CFM_TO_M3S,
)
from field_model import evaluate_grid
from sanity_check import _group_by_z

OPERATING_TEMP_C = 25.0
OPERATING_PRESS_KPA = 101.325
GAS_TYPE = "Air"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--diameter-mm", type=float, default=400.0)
    parser.add_argument("--flow-cfm", type=float, default=3000.0)
    args = parser.parse_args()

    state = load_parametric_field_checkpoint(args.checkpoint)
    model, scaler = state["model"], state["scaler"]
    ratios = LAPPLE_RATIOS

    diameter_mm, flow_cfm = args.diameter_mm, args.flow_cfm
    geometry = geometry_from_dimensions_mm(
        barrel_diameter_mm=diameter_mm,
        barrel_height_mm=diameter_mm * ratios["BarrelHeightRatio"],
        cone_height_mm=diameter_mm * ratios["ConeHeightRatio"],
        exhaust_dia_mm=diameter_mm * ratios["OutletDiamRatio"],
        exhaust_length_mm=diameter_mm * ratios["ExhaustLengthRatio"],
        bottom_outlet_mm=diameter_mm * ratios["BottomOutletRatio"],
    )

    gas_onehot = torch.tensor([gas_type_to_onehot(GAS_TYPE)])
    rho_t, _nu_t = fluid_properties(
        torch.tensor([OPERATING_TEMP_C]), torch.tensor([OPERATING_PRESS_KPA]), gas_onehot,
    )
    rho = rho_t.item()

    v_inlet = inlet_velocity_ms(
        torch.tensor([flow_cfm]),
        torch.tensor([diameter_mm * ratios["InletHeightRatio"] * 1e-3]),
        torch.tensor([diameter_mm * ratios["InletWidthRatio"] * 1e-3]),
    ).item()
    v_z_inlet_theory = inlet_axial_velocity_ms(
        torch.tensor([flow_cfm]), r_barrel_m=geometry.r_barrel, r_exhaust_m=geometry.r_exhaust,
    ).item()

    eval_scaler = scaler.with_scales(
        length_scale_m=geometry.r_barrel, velocity_scale_ms=max(v_inlet, 1e-6), rho=rho,
    )
    grid = evaluate_grid(
        model, eval_scaler, geometry,
        diameter_m=diameter_mm * 1e-3, flow_rate_cfm=flow_cfm,
    )

    groups = _group_by_z(grid)
    z0 = min(groups.keys())
    pts = groups[z0]

    inlet_pts = [(r, vr, vt, vz, p) for (r, vr, vt, vz, p) in pts if r >= geometry.r_exhaust]
    outlet_pts = [(r, vr, vt, vz, p) for (r, vr, vt, vz, p) in pts if r < geometry.r_exhaust]

    def summarize(name: str, side_pts: list[tuple]) -> None:
        n = len(side_pts)
        p_avg = sum(p for (_r, _vr, _vt, _vz, p) in side_pts) / n
        vr_avg = sum(vr for (_r, vr, _vt, _vz, _p) in side_pts) / n
        vt_avg = sum(vt for (_r, _vr, vt, _vz, _p) in side_pts) / n
        vz_avg = sum(vz for (_r, _vr, _vt, vz, _p) in side_pts) / n
        vr_rms = math.sqrt(sum(vr**2 for (_r, vr, _vt, _vz, _p) in side_pts) / n)
        vt_rms = math.sqrt(sum(vt**2 for (_r, _vr, vt, _vz, _p) in side_pts) / n)
        vz_rms = math.sqrt(sum(vz**2 for (_r, _vr, _vt, vz, _p) in side_pts) / n)
        dyn_vr = 0.5 * rho * (vr_rms ** 2)
        dyn_vt = 0.5 * rho * (vt_rms ** 2)
        dyn_vz = 0.5 * rho * (vz_rms ** 2)
        total_p = p_avg + dyn_vr + dyn_vt + dyn_vz

        print(f"\n-- {name} (n={n} pts, z={z0:.6f}) --")
        print(f"  static p_avg        = {p_avg:10.2f} Pa")
        print(f"  v_r  : avg={vr_avg:8.3f} m/s  rms={vr_rms:8.3f} m/s  -> dyn = {dyn_vr:9.2f} Pa")
        print(f"  v_th : avg={vt_avg:8.3f} m/s  rms={vt_rms:8.3f} m/s  -> dyn = {dyn_vt:9.2f} Pa")
        print(f"  v_z  : avg={vz_avg:8.3f} m/s  rms={vz_rms:8.3f} m/s  -> dyn = {dyn_vz:9.2f} Pa")
        print(f"  TOTAL pressure (static + all dynamic) = {total_p:10.2f} Pa")
        return total_p

    print(f"Case: D={diameter_mm}mm  Q={flow_cfm}cfm  rho={rho:.4f} kg/m3")
    print(f"Theoretical v_inlet (tangential BC target)   = {v_inlet:.3f} m/s")
    print(f"Theoretical v_z_inlet (smeared axial BC target) = {v_z_inlet_theory:.3f} m/s")
    one_head_pa = 0.5 * rho * v_inlet * v_inlet
    print(f"One velocity head (0.5*rho*v_inlet^2)        = {one_head_pa:.2f} Pa")

    inlet_total = summarize("INLET RING", inlet_pts)
    outlet_total = summarize("OUTLET BORE", outlet_pts)

    print(f"\nField pressure_drop_pa (inlet_total - outlet_total) = {inlet_total - outlet_total:.2f} Pa")
    print(f"  = {(inlet_total - outlet_total) / one_head_pa:.3f} velocity heads")

    Hi_m = diameter_mm * 1e-3 * ratios["InletHeightRatio"]
    Wi_m = diameter_mm * 1e-3 * ratios["InletWidthRatio"]
    De_m = diameter_mm * 1e-3 * ratios["OutletDiamRatio"]
    Nh = (16.0 * Hi_m * Wi_m) / (math.pi * De_m * De_m)
    baseline_pa = Nh * one_head_pa
    print(f"\nBaseline Nh = {Nh:.3f} heads -> baseline dP = {baseline_pa:.2f} Pa")
    print(f"ratio = {(inlet_total - outlet_total) / baseline_pa:.4f}")


if __name__ == "__main__":
    main()