"""
field_boundary_conditions.py
─────────────────────────────
Boundary-condition residuals for the axisymmetric cyclone flow field.
Each function returns a per-point residual tensor (target - predicted, or
a governing constraint) that a training loop squares and means into a loss
term — no labeled data, purely physics/geometry-derived targets.

Boundary conditions applied (see field_physics.CycloneAxisymGeometry for
where each surface is sampled):

  Outer wall (barrel + cone) : no-slip            -> v_r = v_theta = v_z = 0
  Axis (r = 0)               : symmetry            -> v_r = 0, v_theta = 0,
                                                        d(v_z)/dr = 0
  Inlet ring (z = 0,
    r in [r_exhaust, r_barrel]) : prescribed swirl -> v_theta = v_inlet,
                                                        v_r = 0,
                                                        v_z = v_z_inlet
                                (tangential-entry approximation — see
                                 field_physics.sample_inlet_ring docstring.
                                 v_z_inlet is a physically-derived non-zero
                                 axial inflow, see
                                 field_physics.inlet_axial_velocity_ms —
                                 without it, no mass ever enters the domain
                                 through this boundary.)
  Bottom outlet (dust exit)  : pressure reference,  -> p = 0 (gauge),
    Top exhaust outlet          zero-gradient outflow  d(v_r)/dz = d(v_theta)/dz
                                                        = d(v_z)/dz = 0

RANS turbulence quantities (k, epsilon) add their own boundary conditions,
consistent with the Launder-Sharma low-Re closure in field_turbulence.py:

  Outer wall                 : k = 0, epsilon = 0 exactly (the defining
                                property of a low-Re model that integrates
                                to the wall without wall functions).
  Axis (r = 0)                : symmetry -> d(k)/dr = 0, d(epsilon)/dr = 0.
  Inlet ring                  : prescribed via the turbulence-intensity
                                method (field_turbulence.
                                inlet_turbulence_quantities).
  Bottom/top outlets           : zero-gradient outflow, same convention as
                                the velocity/pressure outlet BCs above.
"""
from __future__ import annotations
import torch

from field_physics import _grad, EPS


def wall_noslip_residual(model_fn, r: torch.Tensor, z: torch.Tensor) -> dict[str, torch.Tensor]:
    out = model_fn(r, z)
    return {
        "wall_vr": out["v_r"],
        "wall_vtheta": out["v_theta"],
        "wall_vz": out["v_z"],
        "wall_k": out["k"],
        "wall_eps": out["eps"],
    }


def axis_symmetry_residual(model_fn, r: torch.Tensor, z: torch.Tensor) -> dict[str, torch.Tensor]:
    out = model_fn(r, z)
    dvz_dr = _grad(out["v_z"], r)
    dk_dr = _grad(out["k"], r)
    deps_dr = _grad(out["eps"], r)
    return {
        "axis_vr": out["v_r"],
        "axis_vtheta": out["v_theta"],
        "axis_dvz_dr": dvz_dr,
        "axis_dk_dr": dk_dr,
        "axis_deps_dr": deps_dr,
    }


def inlet_ring_residual(model_fn, r: torch.Tensor, z: torch.Tensor,
                         v_inlet: torch.Tensor,
                         v_z_inlet: torch.Tensor,
                         k_inlet: torch.Tensor,
                         eps_inlet: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    v_inlet: tangential inlet speed (m/s) — swirl momentum injection.
    v_z_inlet: axial inflow speed (m/s), see
        field_physics.inlet_axial_velocity_ms — this is the mass-injection
        fix; previously v_z was targeted at 0 here, which meant no net mass
        ever entered the domain through this boundary (root cause of the
        Q(z) mass-conservation failure — see field_physics.py docstring).
    k_inlet, eps_inlet: turbulence quantities from
        field_turbulence.inlet_turbulence_quantities.
    Each may be a scalar (broadcast) or one value per point.
    """
    out = model_fn(r, z)
    return {
        "inlet_vtheta": out["v_theta"] - v_inlet,
        "inlet_vr": out["v_r"],
        "inlet_vz": out["v_z"] - v_z_inlet,
        "inlet_k": out["k"] - k_inlet,
        "inlet_eps": out["eps"] - eps_inlet,
    }


def outlet_residual(model_fn, r: torch.Tensor, z: torch.Tensor) -> dict[str, torch.Tensor]:
    out = model_fn(r, z)
    dvr_dz = _grad(out["v_r"], z)
    dvt_dz = _grad(out["v_theta"], z)
    dvz_dz = _grad(out["v_z"], z)
    dk_dz = _grad(out["k"], z)
    deps_dz = _grad(out["eps"], z)
    return {
        "outlet_p": out["p"],          # gauge pressure reference = 0
        "outlet_dvr_dz": dvr_dz,
        "outlet_dvtheta_dz": dvt_dz,
        "outlet_dvz_dz": dvz_dz,
        "outlet_dk_dz": dk_dz,
        "outlet_deps_dz": deps_dz,
    }


def assemble_bc_losses(
    model_fn,
    geometry,
    v_inlet: torch.Tensor,
    v_z_inlet: torch.Tensor,
    k_inlet: torch.Tensor,
    eps_inlet: torch.Tensor,
    n_wall: int = 256,
    n_axis: int = 128,
    n_inlet: int = 128,
    n_outlet: int = 128,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Samples fresh collocation points on each boundary and returns the mean-
    squared residual for every BC term, keyed by name, ready to be summed
    (with weights) into a total loss by the training loop.
    """
    losses: dict[str, torch.Tensor] = {}

    r_w, z_w = geometry.sample_outer_wall(n_wall, device=device)
    for k, v in wall_noslip_residual(model_fn, r_w, z_w).items():
        losses[k] = (v ** 2).mean()

    r_a, z_a = geometry.sample_axis(n_axis, device=device)
    for k, v in axis_symmetry_residual(model_fn, r_a, z_a).items():
        losses[k] = (v ** 2).mean()

    r_i, z_i = geometry.sample_inlet_ring(n_inlet, device=device)
    v_inlet_pt = v_inlet if torch.is_tensor(v_inlet) else torch.full((n_inlet,), float(v_inlet), device=device)
    if v_inlet_pt.numel() == 1:
        v_inlet_pt = v_inlet_pt.expand(n_inlet)
    v_z_inlet_pt = v_z_inlet if torch.is_tensor(v_z_inlet) else torch.full((n_inlet,), float(v_z_inlet), device=device)
    if v_z_inlet_pt.numel() == 1:
        v_z_inlet_pt = v_z_inlet_pt.expand(n_inlet)
    k_inlet_pt = k_inlet if torch.is_tensor(k_inlet) else torch.full((n_inlet,), float(k_inlet), device=device)
    if k_inlet_pt.numel() == 1:
        k_inlet_pt = k_inlet_pt.expand(n_inlet)
    eps_inlet_pt = eps_inlet if torch.is_tensor(eps_inlet) else torch.full((n_inlet,), float(eps_inlet), device=device)
    if eps_inlet_pt.numel() == 1:
        eps_inlet_pt = eps_inlet_pt.expand(n_inlet)
    for k, v in inlet_ring_residual(model_fn, r_i, z_i, v_inlet_pt, v_z_inlet_pt, k_inlet_pt, eps_inlet_pt).items():
        losses[k] = (v ** 2).mean()

    r_bo, z_bo = geometry.sample_bottom_outlet(n_outlet, device=device)
    r_to, z_to = geometry.sample_top_exhaust_outlet(n_outlet, device=device)
    r_out = torch.cat([r_bo, r_to])
    z_out = torch.cat([z_bo, z_to])
    for k, v in outlet_residual(model_fn, r_out, z_out).items():
        losses[k] = (v ** 2).mean()

    return losses