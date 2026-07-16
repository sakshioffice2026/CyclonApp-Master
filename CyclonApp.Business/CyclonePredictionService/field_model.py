"""
field_model.py
───────────────
Fully-connected network predicting the axisymmetric flow field
(v_r, v_theta, v_z, p) as a continuous function of (r, z). Inputs are
scaled by characteristic geometry/velocity scales so training is stable
regardless of a given cyclone's absolute size — this is standard PINN
practice, not a physics assumption.

Architecture per your rules (tanh activations, FC network) — deeper than
model.py's correction network because this one has to represent an actual
spatial field, not a small bounded scalar nudge.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class FieldScaler:
    """Non-dimensionalizes (r, z) by a characteristic length and outputs by
    a characteristic velocity/pressure — keeps network inputs/outputs in a
    well-conditioned range regardless of the cyclone's absolute size."""

    def __init__(self, length_scale_m: float, velocity_scale_ms: float, rho: float):
        self.L = length_scale_m
        self.U = velocity_scale_ms
        self.P = rho * velocity_scale_ms ** 2  # dynamic-pressure scale

    def scale_inputs(self, r: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.stack([r / self.L, z / self.L], dim=1)

    def unscale_outputs(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "v_r": raw[:, 0] * self.U,
            "v_theta": raw[:, 1] * self.U,
            "v_z": raw[:, 2] * self.U,
            "p": raw[:, 3] * self.P,
        }

    def state_dict(self):
        return {"L": self.L, "U": self.U, "P": self.P}

    @classmethod
    def from_state_dict(cls, sd):
        obj = cls.__new__(cls)
        obj.L, obj.U, obj.P = sd["L"], sd["U"], sd["P"]
        return obj


class CycloneFieldPINN(nn.Module):
    def __init__(self, hidden: int = 64, n_layers: int = 6):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 4)]  # v_r, v_theta, v_z, p (scaled)
        self.net = nn.Sequential(*layers)

    def forward(self, r: torch.Tensor, z: torch.Tensor, scaler: FieldScaler) -> dict[str, torch.Tensor]:
        x = scaler.scale_inputs(r, z)
        raw = self.net(x)
        return scaler.unscale_outputs(raw)

    def as_model_fn(self, scaler: FieldScaler):
        """Returns a plain (r, z) -> dict callable, the interface
        field_physics.navier_stokes_residuals and field_boundary_conditions
        expect."""
        def model_fn(r: torch.Tensor, z: torch.Tensor) -> dict[str, torch.Tensor]:
            return self.forward(r, z, scaler)
        return model_fn