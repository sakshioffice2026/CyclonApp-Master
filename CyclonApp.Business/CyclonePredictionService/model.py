"""
model.py
────────
Design choice, stated plainly: this network never predicts efficiency or
pressure drop from scratch. It predicts a small, BOUNDED CORRECTION on top
of the Lapple / Shepherd-Lapple physics baseline computed by physics.py:

    efficiency_pred      = clamp(eff_physics + correction_eff, 0, 1)
    pressure_drop_pred   = dp_physics * (1 + correction_dp_frac)

correction_eff       ∈ [-MAX_EFF_CORRECTION, +MAX_EFF_CORRECTION]   (tanh-bounded)
correction_dp_frac   ∈ [-MAX_DP_CORRECTION_FRAC, +MAX_DP_CORRECTION_FRAC]

Why this shape, not a free-form regressor:
- It structurally cannot diverge wildly from the trusted physics engine —
  it can only nudge it — which is what makes the .NET side's trusted-range
  check (CyclonePredictionRepository) a meaningful safety net rather than
  an afterthought.
- With no real measured data yet (see dataset.py), training regularizes
  both corrections toward zero, so out of the box this service reproduces
  the deterministic calculation almost exactly. That's intentional and
  honest: there is currently nothing to justify deviating from Lapple.
  The moment real commissioning/test data is added via train.py --data,
  the correction heads start learning real deviations, still bounded.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from physics import lapple_forward

MAX_EFF_CORRECTION = 0.08          # +/- 8 percentage points of efficiency
MAX_DP_CORRECTION_FRAC = 0.15      # +/- 15% of the physics pressure drop

# Raw feature order fed to the network (must match dataset.FEATURE_RANGES
# order + 3 one-hot gas columns appended).
RAW_FEATURES = [
    "flow_cfm", "inlet_line_size_in", "temp_c", "press_kpa",
    "particle_size_micron", "particle_density_kgm3", "effective_turns",
    "inlet_height_ratio", "inlet_width_ratio", "outlet_diam_ratio",
]


class FeatureScaler:
    """Simple min-max scaler fit on the sampling ranges (dataset.FEATURE_RANGES),
    not on a batch — keeps scaling stable/reproducible regardless of what a
    given training run happens to sample."""

    def __init__(self, ranges: dict[str, tuple[float, float]]):
        self.lo = torch.tensor([ranges[k][0] for k in RAW_FEATURES])
        self.hi = torch.tensor([ranges[k][1] for k in RAW_FEATURES])

    def transform(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raw = torch.stack([batch[k] for k in RAW_FEATURES], dim=1)
        scaled = (raw - self.lo) / (self.hi - self.lo)
        return torch.cat([scaled, batch["gas_onehot"]], dim=1)

    def state_dict(self):
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist()}

    @classmethod
    def from_state_dict(cls, sd):
        obj = cls.__new__(cls)
        obj.lo = torch.tensor(sd["lo"])
        obj.hi = torch.tensor(sd["hi"])
        return obj


class CyclonePINN(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        n_in = len(RAW_FEATURES) + 3  # + one-hot gas
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),  # [correction_eff_raw, correction_dp_raw]
        )

    def forward(self, batch: dict[str, torch.Tensor], scaler: FeatureScaler) -> dict[str, torch.Tensor]:
        physics = lapple_forward(
            flow_cfm=batch["flow_cfm"],
            inlet_line_size_in=batch["inlet_line_size_in"],
            temp_c=batch["temp_c"],
            press_kpa=batch["press_kpa"],
            gas_onehot=batch["gas_onehot"],
            particle_size_micron=batch["particle_size_micron"],
            particle_density_kgm3=batch["particle_density_kgm3"],
            effective_turns=batch["effective_turns"],
            inlet_height_ratio=batch["inlet_height_ratio"],
            inlet_width_ratio=batch["inlet_width_ratio"],
            outlet_diam_ratio=batch["outlet_diam_ratio"],
        )

        x = scaler.transform(batch)
        raw = self.net(x)
        correction_eff = torch.tanh(raw[:, 0]) * MAX_EFF_CORRECTION
        correction_dp_frac = torch.tanh(raw[:, 1]) * MAX_DP_CORRECTION_FRAC

        efficiency_pred = torch.clamp(physics["efficiency"] + correction_eff, 0.0, 1.0)
        pressure_drop_pred = physics["pressure_drop_pa"] * (1.0 + correction_dp_frac)

        return {
            "efficiency_pred": efficiency_pred,
            "pressure_drop_pred": pressure_drop_pred,
            "correction_eff": correction_eff,
            "correction_dp_frac": correction_dp_frac,
            "physics_efficiency": physics["efficiency"],
            "physics_pressure_drop_pa": physics["pressure_drop_pa"],
        }