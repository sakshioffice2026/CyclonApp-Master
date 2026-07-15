"""
dataset.py
──────────
Two data sources feed training:

1. Synthetic collocation points — sampled across the realistic operating
   envelope of the cyclones this app designs. These have no "measured"
   efficiency; they exist purely so the physics-residual (monotonicity)
   loss in train.py has somewhere to evaluate gradients. This is standard
   PINN practice: collocation points don't need labels because the loss
   applied to them comes from the governing relationships, not from data.

2. Real records (optional) — a CSV of actual commissioned/tested cyclones
   with a measured efficiency and/or pressure drop. This is what the
   correction head is actually FOR. Until real records exist, training
   runs with zero real rows, and the network is regularized to output a
   near-zero correction — i.e. it defers entirely to the physics baseline,
   which is the honest behavior when there's no data to justify deviating
   from Lapple/Shepherd-Lapple.

CSV schema for real data (header row required):
FlowRateCFM,InletLineSizeIn,OperatingTempC,OperatingPressKPa,GasType,
ParticleSizeMicron,ParticleDensityKgm3,EffectiveTurns,InletHeightRatio,
InletWidthRatio,OutletDiamRatio,MeasuredEfficiencyPercent,MeasuredPressureDropPa

Either measured column may be left blank for a given row if only one was
recorded; blank measured columns are excluded from that row's data loss.
"""
from __future__ import annotations
import csv
import torch
from physics import gas_type_to_onehot

FEATURE_RANGES = {
    "flow_cfm":              (200.0, 20000.0),
    "inlet_line_size_in":    (2.0, 24.0),
    "temp_c":                (-10.0, 250.0),
    "press_kpa":             (90.0, 350.0),
    "particle_size_micron":  (0.5, 300.0),
    "particle_density_kgm3": (400.0, 4000.0),
    "effective_turns":       (2.0, 10.0),
    "inlet_height_ratio":    (0.35, 0.75),
    "inlet_width_ratio":     (0.15, 0.35),
    "outlet_diam_ratio":     (0.3, 0.6),
}

_FIELD_ORDER = list(FEATURE_RANGES.keys())


def sample_synthetic_batch(n: int, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """Uniformly sample n collocation points across the operating envelope."""
    out = {}
    for key, (lo, hi) in FEATURE_RANGES.items():
        u = torch.rand(n, generator=generator)
        out[key] = lo + u * (hi - lo)

    # Gas mix: mostly air, some N2/CO2, to reflect typical usage.
    gas_choices = torch.multinomial(
        torch.tensor([0.7, 0.15, 0.15]), n, replacement=True, generator=generator
    )
    onehot = torch.zeros(n, 3)
    onehot[torch.arange(n), gas_choices] = 1.0
    out["gas_onehot"] = onehot

    return out


def load_real_csv(path: str) -> dict[str, torch.Tensor]:
    """
    Loads a real-data CSV (schema in the module docstring) into the same
    tensor dict shape as sample_synthetic_batch, plus measured_efficiency /
    measured_pressure_drop_pa (NaN where a row has no measurement) and a
    boolean mask for each.
    """
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    n = len(rows)
    out = {k: torch.zeros(n) for k in FEATURE_RANGES}
    onehot = torch.zeros(n, 3)
    meas_eff = torch.full((n,), float("nan"))
    meas_dp = torch.full((n,), float("nan"))

    key_map = {
        "flow_cfm": "FlowRateCFM",
        "inlet_line_size_in": "InletLineSizeIn",
        "temp_c": "OperatingTempC",
        "press_kpa": "OperatingPressKPa",
        "particle_size_micron": "ParticleSizeMicron",
        "particle_density_kgm3": "ParticleDensityKgm3",
        "effective_turns": "EffectiveTurns",
        "inlet_height_ratio": "InletHeightRatio",
        "inlet_width_ratio": "InletWidthRatio",
        "outlet_diam_ratio": "OutletDiamRatio",
    }

    for i, r in enumerate(rows):
        for tk, ck in key_map.items():
            out[tk][i] = float(r[ck])
        onehot[i] = torch.tensor(gas_type_to_onehot(r.get("GasType", "Air")))
        if r.get("MeasuredEfficiencyPercent", "").strip():
            meas_eff[i] = float(r["MeasuredEfficiencyPercent"]) / 100.0
        if r.get("MeasuredPressureDropPa", "").strip():
            meas_dp[i] = float(r["MeasuredPressureDropPa"])

    out["gas_onehot"] = onehot
    out["measured_efficiency"] = meas_eff
    out["measured_pressure_drop_pa"] = meas_dp
    return out
