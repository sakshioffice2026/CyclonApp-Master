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
import torch.nn.functional as F


class FieldScaler:
    """Non-dimensionalizes (r, z) by a characteristic length and outputs by
    a characteristic velocity/pressure/turbulence scale — keeps network
    inputs/outputs in a well-conditioned range regardless of the cyclone's
    absolute size.

    Turbulence scales (K, E) are the standard dimensional estimates for
    turbulence kinetic energy (~ velocity^2) and dissipation rate
    (~ velocity^3 / length) — see e.g. Pope, "Turbulent Flows", Ch. 5.
    """

    def __init__(self, length_scale_m: float, velocity_scale_ms: float, rho: float):
        self.L = length_scale_m
        self.U = velocity_scale_ms
        self.P = rho * velocity_scale_ms ** 2  # dynamic-pressure scale
        self.K = velocity_scale_ms ** 2  # turbulence kinetic energy scale
        self.E = (velocity_scale_ms ** 3) / length_scale_m  # dissipation-rate scale

    def scale_inputs(self, r: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.stack([r / self.L, z / self.L], dim=1)

    def unscale_outputs(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "v_r": raw[:, 0] * self.U,
            "v_theta": raw[:, 1] * self.U,
            "v_z": raw[:, 2] * self.U,
            "p": raw[:, 3] * self.P,
            # softplus guarantees k, eps > 0 everywhere (structurally, not
            # via a clamp) — required since both are physically
            # non-negative quantities and appear as denominators in the
            # eddy-viscosity closure (field_turbulence.eddy_viscosity).
            "k": F.softplus(raw[:, 4]) * self.K,
            "eps": F.softplus(raw[:, 5]) * self.E,
        }

    def state_dict(self):
        return {"L": self.L, "U": self.U, "P": self.P, "K": self.K, "E": self.E}

    @classmethod
    def from_state_dict(cls, sd):
        obj = cls.__new__(cls)
        obj.L, obj.U, obj.P = sd["L"], sd["U"], sd["P"]
        obj.K, obj.E = sd["K"], sd["E"]
        return obj


class CycloneFieldPINN(nn.Module):
    def __init__(self, hidden: int = 64, n_layers: int = 6):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        # v_r, v_theta, v_z, p, k_raw, eps_raw (last two softplus-mapped
        # to positive k, eps by FieldScaler.unscale_outputs)
        layers += [nn.Linear(hidden, 6)]
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


def evaluate_grid(
    model: "CycloneFieldPINN",
    scaler: FieldScaler,
    geometry,
    n_r: int = 40,
    n_z: int = 60,
) -> dict[str, list[float]]:
    """
    Turns a trained CycloneFieldPINN into a queryable field result: samples
    a regular (r, z) grid over the fluid domain, evaluates the network at
    every point, and returns flat parallel lists (r_m, z_m, v_r_ms,
    v_theta_ms, v_z_ms, pressure_pa) — one entry per grid point that is
    actually inside the fluid domain.

    Points outside the fluid domain (outside the tapered outer wall, or
    beyond total_height) are dropped rather than returned as zeros/NaN, so
    every list is the same length and every entry is a physically valid
    query point. This is what app.py's /predict_field/status endpoint
    serializes into FieldResultDto for the .NET client.

    A regular grid (not the random collocation points used during
    training) is used here because the client needs a queryable, roughly
    uniform sampling of the field for visualization/analysis — training
    and evaluation deliberately use different sampling strategies for
    different purposes.
    """
    device = next(model.parameters()).device

    r_lin = torch.linspace(0.0, geometry.r_barrel, n_r, device=device)
    z_lin = torch.linspace(0.0, geometry.total_height, n_z, device=device)
    r_grid, z_grid = torch.meshgrid(r_lin, z_lin, indexing="ij")
    r_flat = r_grid.reshape(-1)
    z_flat = z_grid.reshape(-1)

    valid = geometry.is_fluid(r_flat, z_flat)
    r_valid = r_flat[valid]
    z_valid = z_flat[valid]

    if r_valid.numel() == 0:
        # Degenerate geometry (shouldn't happen given CycloneAxisymGeometry's
        # own validation) — return empty lists rather than raising, since the
        # caller (app.py) already wraps this in a try/except that reports
        # job failure with a clear message.
        empty: list[float] = []
        return {
            "r_m": empty, "z_m": empty,
            "v_r_ms": empty, "v_theta_ms": empty, "v_z_ms": empty,
            "pressure_pa": empty,
        }

    model.eval()
    with torch.no_grad():
        out = model(r_valid, z_valid, scaler)

    return {
        "r_m": r_valid.cpu().tolist(),
        "z_m": z_valid.cpu().tolist(),
        "v_r_ms": out["v_r"].cpu().tolist(),
        "v_theta_ms": out["v_theta"].cpu().tolist(),
        "v_z_ms": out["v_z"].cpu().tolist(),
        "pressure_pa": out["p"].cpu().tolist(),
    }