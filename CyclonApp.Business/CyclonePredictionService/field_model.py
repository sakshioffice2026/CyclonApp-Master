"""
field_model.py
───────────────
Fully-connected network predicting the axisymmetric flow field
(v_r, v_theta, v_z, p) as a continuous function of (r, z, barrel_diameter,
flow_rate). Inputs are scaled by characteristic geometry/velocity scales so
training is stable regardless of a given cyclone's absolute size — this is
standard PINN practice, not a physics assumption.

PARAMETRIC INPUTS (barrel_diameter_m, flow_rate_cfm) — added to support
training across a range of LAPPLE-type cyclone sizes instead of one fixed
geometry. Two things had to change together, not separately:
  1. FieldScaler's length/velocity scales (previously fixed once from a
     single geometry) are now recomputed per training step from whichever
     geometry/flow rate that step is currently sampling — see
     scale_inputs' diameter_m/flow_rate_cfm args.
  2. The (normalized) diameter and flow rate are ALSO fed to the network
     as explicit inputs, separate from normalization. Position alone
     (r/L, z/L) looks identical at any absolute size — normalizing alone
     would erase the actual scale. Real cyclone physics is not perfectly
     self-similar across sizes (viscous effects in particular don't just
     rescale), so the network needs the actual size as a signal, not just
     a normalization convenience.

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

    length_scale_m/velocity_scale_ms are still passed in per-call (see
    scale_inputs) rather than fixed at construction, since parametric
    training recomputes them for whichever geometry/flow rate is currently
    being sampled. diameter_range_m/flow_rate_range_cfm are fixed at
    construction — they define the min-max normalization window for the
    two new explicit parametric inputs, and must match whatever range
    training data is actually sampled from.

    Turbulence scales (K, E) are the standard dimensional estimates for
    turbulence kinetic energy (~ velocity^2) and dissipation rate
    (~ velocity^3 / length) — see e.g. Pope, "Turbulent Flows", Ch. 5.
    """

    def __init__(
        self,
        length_scale_m: float,
        velocity_scale_ms: float,
        rho: float,
        diameter_range_m: tuple[float, float] = (0.150, 0.750),
        flow_rate_range_cfm: tuple[float, float] = (300.0, 13000.0),
    ):
        self.L = length_scale_m
        self.U = velocity_scale_ms
        self.P = rho * velocity_scale_ms ** 2  # dynamic-pressure scale
        self.K = velocity_scale_ms ** 2  # turbulence kinetic energy scale
        self.E = (velocity_scale_ms ** 3) / length_scale_m  # dissipation-rate scale
        self.D_min, self.D_max = diameter_range_m
        self.Q_min, self.Q_max = flow_rate_range_cfm

    def _normalize(self, value: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        """Min-max normalizes to roughly [-1, 1] — standard practice for a
        parametric input the network must treat as an explicit signal
        (not just a numerical-conditioning convenience like r/L)."""
        mid = (lo + hi) / 2.0
        half_span = max((hi - lo) / 2.0, 1e-9)
        return (value - mid) / half_span

    def scale_inputs(
        self,
        r: torch.Tensor,
        z: torch.Tensor,
        diameter_m: torch.Tensor,
        flow_rate_cfm: torch.Tensor,
    ) -> torch.Tensor:
        """diameter_m/flow_rate_cfm must be broadcastable to the same shape
        as r/z — typically a scalar-per-training-step tensor expanded to
        match the batch of sampled (r, z) points, since one training step
        samples many points from a single randomly-chosen geometry."""
        d_norm = self._normalize(diameter_m, self.D_min, self.D_max)
        q_norm = self._normalize(flow_rate_cfm, self.Q_min, self.Q_max)
        return torch.stack(
            [r / self.L, z / self.L, d_norm.expand_as(r), q_norm.expand_as(r)],
            dim=1,
        )

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
        return {
            "L": self.L, "U": self.U, "P": self.P, "K": self.K, "E": self.E,
            "D_min": self.D_min, "D_max": self.D_max,
            "Q_min": self.Q_min, "Q_max": self.Q_max,
        }

    @classmethod
    def from_state_dict(cls, sd):
        obj = cls.__new__(cls)
        obj.L, obj.U, obj.P = sd["L"], sd["U"], sd["P"]
        obj.K, obj.E = sd["K"], sd["E"]
        # Fall back to this module's defaults for checkpoints saved before
        # the parametric-input change, so old single-geometry checkpoints
        # still load (they just won't have meaningful D/Q normalization,
        # which is fine — they were never trained to vary those anyway).
        obj.D_min = sd.get("D_min", 0.150)
        obj.D_max = sd.get("D_max", 0.750)
        obj.Q_min = sd.get("Q_min", 300.0)
        obj.Q_max = sd.get("Q_max", 13000.0)
        return obj


class CycloneFieldPINN(nn.Module):
    def __init__(self, hidden: int = 64, n_layers: int = 6):
        super().__init__()
        # Input: [r/L, z/L, normalized_diameter, normalized_flow_rate] — 4
        # features (previously 2; see module docstring for why both the
        # normalization AND the explicit diameter/flow-rate inputs matter).
        layers: list[nn.Module] = [nn.Linear(4, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        # v_r, v_theta, v_z, p, k_raw, eps_raw (last two softplus-mapped
        # to positive k, eps by FieldScaler.unscale_outputs)
        layers += [nn.Linear(hidden, 6)]
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        r: torch.Tensor,
        z: torch.Tensor,
        diameter_m: torch.Tensor,
        flow_rate_cfm: torch.Tensor,
        scaler: FieldScaler,
    ) -> dict[str, torch.Tensor]:
        x = scaler.scale_inputs(r, z, diameter_m, flow_rate_cfm)
        raw = self.net(x)
        return scaler.unscale_outputs(raw)

    def as_model_fn(self, scaler: FieldScaler, diameter_m: torch.Tensor, flow_rate_cfm: torch.Tensor):
        """Returns a plain (r, z) -> dict callable, the interface
        field_physics.navier_stokes_residuals and field_boundary_conditions
        expect. diameter_m/flow_rate_cfm are fixed for the training step
        this closure is built for — every point sampled within one step
        comes from the same randomly-chosen geometry/flow rate, so they
        don't need to vary per-point within a single call."""
        def model_fn(r: torch.Tensor, z: torch.Tensor) -> dict[str, torch.Tensor]:
            d = diameter_m if torch.is_tensor(diameter_m) else torch.full_like(r, float(diameter_m))
            q = flow_rate_cfm if torch.is_tensor(flow_rate_cfm) else torch.full_like(r, float(flow_rate_cfm))
            return self.forward(r, z, d, q, scaler)
        return model_fn


def evaluate_grid(
    model: "CycloneFieldPINN",
    scaler: FieldScaler,
    geometry,
    diameter_m: float,
    flow_rate_cfm: float,
    n_r: int = 40,
    n_z: int = 60,
) -> dict[str, list[float]]:
    """
    Turns a trained CycloneFieldPINN into a queryable field result: samples
    a regular (r, z) grid over the fluid domain, evaluates the network at
    every point, and returns flat parallel lists (r_m, z_m, v_r_ms,
    v_theta_ms, v_z_ms, pressure_pa) — one entry per grid point that is
    actually inside the fluid domain.

    diameter_m/flow_rate_cfm: the specific design being queried — passed
    through to the network as the explicit parametric inputs it was
    trained to use (see field_model.py module docstring). geometry must
    already be built to match diameter_m (e.g. via geometry_from_dimensions_mm
    using the LAPPLE ratios) — this function does not derive one from the
    other, that's the caller's responsibility.

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

    d_valid = torch.full_like(r_valid, float(diameter_m))
    q_valid = torch.full_like(r_valid, float(flow_rate_cfm))

    model.eval()
    with torch.no_grad():
        out = model(r_valid, z_valid, d_valid, q_valid, scaler)

    return {
        "r_m": r_valid.cpu().tolist(),
        "z_m": z_valid.cpu().tolist(),
        "v_r_ms": out["v_r"].cpu().tolist(),
        "v_theta_ms": out["v_theta"].cpu().tolist(),
        "v_z_ms": out["v_z"].cpu().tolist(),
        "pressure_pa": out["p"].cpu().tolist(),
    } 