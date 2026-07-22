"""
validate_pressure_drop.py
──────────────────────────
Batch-checks the PINN field solver's pressure drop against the analytical
Shepherd-Lapple baseline across many (diameter, flow rate) designs at once,
so this doesn't have to be checked by hand one design at a time.

Extends the same spirit as test_onnx_geometries.py (quick, script-not-app
sanity check you run after retraining/redeploying) but compares against
the REAL baseline formula instead of just checking outputs differ.

Both sides of the comparison are built from field_physics.py /
sanity_check.py's own functions (fluid_properties, inlet geometry,
compute_pressure_drop) rather than re-derived from scratch, specifically
so this script cannot silently drift from what app.py actually serves.
The ONLY formula re-implemented here by hand is the Shepherd-Lapple
baseline itself (Nh = 16*H*W/(pi*De^2), dP = Nh*0.5*rho*V_inlet^2) --
copied verbatim from CyclonCalculationRepository.cs's comments so the two
implementations are trivially diffable against each other; if that C#
file's formula ever changes, this function must be updated to match or
this script will silently compare against a stale baseline.

Usage:
    python validate_pressure_drop.py cyclone_model.pth
    python validate_pressure_drop.py cyclone_model_stairmand_gp.pth --ratios STAIRMAND_GP

Requires the checkpoint to be a PARAMETRIC checkpoint (the kind
train_*_colab.py scripts produce) -- a single-geometry checkpoint isn't
supported here (see load_parametric_field_checkpoint's own error message
if you pass the wrong kind).
"""
from __future__ import annotations

import argparse
import math

import torch

from field_train import (
    load_parametric_field_checkpoint,
    LAPPLE_RATIOS,
    STAIRMAND_RATIOS,
    STAIRMAND_GP_RATIOS,
    SWIFT_HE_RATIOS,
)
from field_physics import (
    fluid_properties,
    gas_type_to_onehot,
    geometry_from_dimensions_mm,
    inlet_velocity_ms,
    CFM_TO_M3S,
)
from field_model import evaluate_grid
from sanity_check import compute_pressure_drop

RATIO_PRESETS = {
    "LAPPLE": LAPPLE_RATIOS,
    "STAIRMAND": STAIRMAND_RATIOS,
    "STAIRMAND_GP": STAIRMAND_GP_RATIOS,
    "SWIFT_HE": SWIFT_HE_RATIOS,
}

# (diameter_mm, flow_rate_cfm) pairs to check. Deliberately spans small/
# low-flow to large/high-flow, all inside the standard [150,750]mm /
# [300,13000]cfm trained window (see train_*_colab.py CONFIG sections) --
# if you retrained on a different window, update these or pass --cases.
DEFAULT_CASES: list[tuple[float, float]] = [
    (200.0, 500.0),
    (300.0, 1500.0),
    (400.0, 3000.0),
    (500.0, 6000.0),
    (600.0, 9000.0),
    (700.0, 12000.0),
]

OPERATING_TEMP_C = 25.0
OPERATING_PRESS_KPA = 101.325
GAS_TYPE = "Air"

# Loose sanity bounds on field/baseline ratio -- NOT a tight correctness
# proof (the PINN is an approximation, not an exact solver), just enough
# to catch the kind of systematic ~2x-or-worse mismatch this script was
# written to catch. Tighten once you have a track record of what a
# well-trained checkpoint's ratio spread actually looks like.
RATIO_WARN_LOW = 0.5
RATIO_WARN_HIGH = 2.0


def shepherd_lapple_baseline_pa(
    diameter_mm: float,
    flow_rate_cfm: float,
    ratios: dict[str, float],
    rho_kgm3: float,
) -> dict:
    """Copied verbatim (formula-wise) from CyclonCalculationRepository.cs's
    Calculate(): Nh = 16*Hi*Wi/(pi*De^2), dP_Pa = Nh*0.5*rho*V_inlet^2.
    Uses the SAME rho the field solve uses for this design (passed in),
    so any difference in the two pressure-drop numbers is coming from the
    pressure-drop definitions/models themselves, not a rho mismatch.
    """
    Dc_m = diameter_mm * 1e-3
    Hi_m = Dc_m * ratios["InletHeightRatio"]
    Wi_m = Dc_m * ratios["InletWidthRatio"]
    De_m = Dc_m * ratios["OutletDiamRatio"]

    Q_m3s = flow_rate_cfm * CFM_TO_M3S
    V_inlet_ms = Q_m3s / (Hi_m * Wi_m)

    Nh = (16.0 * Hi_m * Wi_m) / (math.pi * De_m * De_m)
    dP_Pa = Nh * 0.5 * rho_kgm3 * V_inlet_ms * V_inlet_ms

    return {"pressure_drop_pa": dP_Pa, "v_inlet_ms": V_inlet_ms, "Nh": Nh}


def field_solve_pa(
    diameter_mm: float,
    flow_rate_cfm: float,
    ratios: dict[str, float],
    model,
    scaler,
) -> dict:
    """Same pipeline as app.py's _run_field_job: build geometry -> fluid
    properties -> inlet velocity -> evaluate_grid -> compute_pressure_drop.
    """
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
        torch.tensor([OPERATING_TEMP_C]),
        torch.tensor([OPERATING_PRESS_KPA]),
        gas_onehot,
    )
    rho = rho_t.item()

    v_inlet = inlet_velocity_ms(
        torch.tensor([flow_rate_cfm]),
        torch.tensor([diameter_mm * ratios["InletHeightRatio"] * 1e-3]),
        torch.tensor([diameter_mm * ratios["InletWidthRatio"] * 1e-3]),
    ).item()

    eval_scaler = scaler.with_scales(
        length_scale_m=geometry.r_barrel,
        velocity_scale_ms=max(v_inlet, 1e-6),
        rho=rho,
    )

    grid = evaluate_grid(
        model, eval_scaler, geometry,
        diameter_m=diameter_mm * 1e-3,
        flow_rate_cfm=flow_rate_cfm,
    )
    pdrop = compute_pressure_drop(grid, r_exhaust_m=geometry.r_exhaust)
    return {"rho": rho, "v_inlet_ms": v_inlet, **pdrop}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to a parametric .pth checkpoint")
    parser.add_argument(
        "--ratios", default="LAPPLE", choices=sorted(RATIO_PRESETS.keys()),
        help="Which ratio family this checkpoint was trained on (default: LAPPLE). "
             "Must match the checkpoint, not just the filename -- if unsure, check "
             "the checkpoint's own saved 'ratios' dict, printed below at startup.",
    )
    parser.add_argument(
        "--cases", default=None,
        help="Comma-separated diameter_mm:flow_cfm pairs, e.g. "
             "'250:1000,450:5000'. Defaults to DEFAULT_CASES if omitted.",
    )
    args = parser.parse_args()

    state = load_parametric_field_checkpoint(args.checkpoint)
    model, scaler = state["model"], state["scaler"]
    ratios = RATIO_PRESETS[args.ratios]

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint's own saved ratios: {state.get('ratios')}")
    print(f"Using ratio preset '{args.ratios}' for the baseline comparison: {ratios}")
    if state.get("ratios") is not None and state["ratios"] != ratios:
        print(
            "WARNING: --ratios preset does NOT match the ratios this checkpoint "
            "was actually trained with (printed above). The comparison below will "
            "be comparing the field solve against the WRONG baseline geometry -- "
            "pass the matching --ratios preset instead."
        )
    print(
        f"Trained diameter range: [{scaler.D_min}, {scaler.D_max}] m, "
        f"flow range: [{scaler.Q_min}, {scaler.Q_max}] cfm\n"
    )

    if args.cases:
        cases = []
        for pair in args.cases.split(","):
            d_str, q_str = pair.split(":")
            cases.append((float(d_str), float(q_str)))
    else:
        cases = DEFAULT_CASES

    header = (
        f"{'D(mm)':>7} {'Q(cfm)':>8} {'baseline(Pa)':>13} {'field(Pa)':>11} "
        f"{'ratio':>7}  status"
    )
    print(header)
    print("-" * len(header))

    ratios_seen = []
    for diameter_mm, flow_cfm in cases:
        diameter_m = diameter_mm * 1e-3
        out_of_range = not (scaler.D_min <= diameter_m <= scaler.D_max) or \
                        not (scaler.Q_min <= flow_cfm <= scaler.Q_max)

        field = field_solve_pa(diameter_mm, flow_cfm, ratios, model, scaler)
        baseline = shepherd_lapple_baseline_pa(
            diameter_mm, flow_cfm, ratios, rho_kgm3=field["rho"]
        )

        if field["pressure_drop_pa"] is None:
            status = "FAIL (field solve returned no pressure drop -- see detail)"
            print(
                f"{diameter_mm:7.0f} {flow_cfm:8.0f} "
                f"{baseline['pressure_drop_pa']:13.1f} {'--':>11} {'--':>7}  {status}"
            )
            print(f"           detail: {field['detail']}")
            continue

        ratio = field["pressure_drop_pa"] / baseline["pressure_drop_pa"]
        ratios_seen.append(ratio)

        if out_of_range:
            status = "SKIP (outside trained range -- extrapolation, not evaluated)"
        elif RATIO_WARN_LOW <= ratio <= RATIO_WARN_HIGH:
            status = "OK"
        else:
            status = "WARN (ratio outside loose sanity bounds)"

        print(
            f"{diameter_mm:7.0f} {flow_cfm:8.0f} "
            f"{baseline['pressure_drop_pa']:13.1f} {field['pressure_drop_pa']:11.1f} "
            f"{ratio:7.3f}  {status}"
        )

    if ratios_seen:
        avg_ratio = sum(ratios_seen) / len(ratios_seen)
        spread = max(ratios_seen) - min(ratios_seen)
        print(
            f"\nAcross {len(ratios_seen)} evaluated cases: "
            f"avg ratio={avg_ratio:.3f}, spread={spread:.3f} "
            f"(min={min(ratios_seen):.3f}, max={max(ratios_seen):.3f})"
        )
        print(
            "A ratio consistently near 1.0 across very different designs means "
            "the field solve and baseline agree; a consistent OFFSET (all ratios "
            "clustered around some other constant, e.g. ~0.45 or ~2.2) points to "
            "a remaining systematic/definitional issue rather than per-design "
            "under-training; a WIDE, inconsistent spread instead points to "
            "per-design training quality (extrapolation, under-trained regions "
            "of the (D,Q) window) rather than a single fixed bug."
        )
    else:
        print("\nNo cases produced a valid field-solve pressure drop -- see FAILs above.")


if __name__ == "__main__":
    main()