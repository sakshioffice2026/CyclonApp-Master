"""
field_train.py
────────────────
Reusable training routine for CycloneFieldPINN. This is the SINGLE source
of truth for training the field-solving model — both the CLI entry point
(`python field_train.py ...`, see bottom of this file) and app.py's async
/predict_field job call the same `run_field_prediction_job` /
`train_field_model` functions. Do not duplicate this loop, or the
geometry/fluid-property glue in `run_field_prediction_job`, anywhere else.

Two training modes live in this file:

  * train_field_model / run_field_prediction_job — trains one small
    network per request, for exactly one fixed geometry/flow rate. Simple,
    reliable, and still what app.py's /predict_field job uses. Geometry
    and operating condition are baked in as fixed constants at training
    time for this mode.

  * train_parametric_field_model — trains ONE network across a whole
    family of LAPPLE cyclone sizes/flow rates at once, using domain
    randomization (a different sampled geometry/flow rate every epoch).
    CycloneFieldPINN's inputs are (r, z, diameter_m, flow_rate_cfm) — see
    field_model.py's module docstring — specifically so this mode is
    possible: train once, then query any diameter/flow rate inside the
    trained range instantly, no per-request training. This is harder to
    get to converge well than the single-geometry mode (see that
    function's docstring), which is why both modes are kept rather than
    deleting the simpler one.

Training recipe, in order:
  1. Adam phase — bulk convergence. Uses an exponentially decaying LR
     (fast progress early, stable fine descent later) and gradient-norm
     clipping (PINN loss landscapes with 4th-order derivative terms in
     the PDE residual are prone to occasional large gradients that can
     destabilize a plain fixed-LR Adam run).
  2. L-BFGS phase — standard PINN fine-tuning step. Second-order method
     squeezes out the last bit of residual once Adam has gotten the
     network into a good basin; running L-BFGS from a random init
     directly tends to get stuck in a bad local minimum.

Collocation points are re-sampled fresh every epoch (not cached once at
the start) — this is standard PINN practice (equivalent to an infinite
stream of training data over the domain) and also directly implements
"bigger collocation batch per step": n_interior below is intentionally
larger than a token/minimal batch so each gradient step sees good domain
coverage.

TURBULENCE INTEGRATION (this revision — root-cause fix):
Previously this file called field_physics.navier_stokes_residuals (laminar
only) and field_boundary_conditions.assemble_bc_losses with the old
(v_inlet)-only signature, while field_model.CycloneFieldPINN already
outputs k/eps and field_boundary_conditions.py already requires
(v_inlet, k_inlet, eps_inlet). That mismatch crashed with:
    TypeError: assemble_bc_losses() missing 2 required positional
    arguments: 'k_inlet' and 'eps_inlet'
— and even past that crash, training was still laminar underneath, since
navier_stokes_residuals never reads the network's k/eps outputs or applies
the Boussinesq eddy-viscosity correction. Fixed below by switching to
field_turbulence.rans_field_residuals for the PDE residual and computing
k_inlet/eps_inlet via field_turbulence.inlet_turbulence_quantities, so the
turbulence closure that field_model.py and field_boundary_conditions.py
were already built for is actually exercised during training.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Optional

import torch

from field_model import CycloneFieldPINN, FieldScaler, evaluate_grid
from field_physics import (
    CycloneAxisymGeometry,
    geometry_from_dimensions_mm,
    fluid_properties,
    inlet_velocity_ms,
    inlet_axial_velocity_ms,
    gas_type_to_onehot,
)
from field_turbulence import (
    rans_field_residuals,
    inlet_turbulence_quantities,
    hydraulic_diameter_rect_m,
)
from field_boundary_conditions import assemble_bc_losses
from sanity_check import mass_conservation_metrics

# ─────────────────────────────────────────────────────────────────────────
# Loss term weights. PDE residuals, BC residuals, and now the k/epsilon
# transport-equation residuals are not naturally on the same scale (BC
# residuals are direct velocity/pressure/turbulence errors; PDE and
# turbulence-equation residuals involve second derivatives and can dominate
# numerically if left unweighted). These weights are a starting point
# carried over from the laminar-only version — PDE_LOSS_WEIGHT/
# BC_LOSS_WEIGHT were validated stable for the laminar case; TURB_LOSS_WEIGHT
# is a first guess, NOT yet empirically validated the way the other two
# were, since this is the first revision that actually exercises the
# k/epsilon equations. Revisit if turbulence quantities fail to converge
# or dominate/get swamped by the momentum terms.
# ─────────────────────────────────────────────────────────────────────────
PDE_LOSS_WEIGHT = 1.0
TURB_LOSS_WEIGHT = 1.0
BC_LOSS_WEIGHT = 10.0

# Direct aggregate mass-conservation regularizer (see _mass_flow_loss below).
# Added on top of the pointwise continuity PDE residual because that single
# residual term is heavily outweighed by the ~21 individually-weighted BC
# residual terms above (each x BC_LOSS_WEIGHT=10) once summed, which lets
# the network satisfy wall/axis/inlet/outlet BCs almost exactly while still
# badly violating continuity in aggregate through the interior — exactly
# the "passes wall/axis checks, fails Q(z) constancy" failure mode seen in
# sanity_check.check_mass_conservation.
#
# ROOT-CAUSE FIX (repeated Q(z) failures after the first mass-flow term):
# the original loss divided by mean(Q). For a reverse-flow cyclone the
# physically correct net mid-plane flow is ~0 (gas leaves via the top
# exhaust, not the bottom), so mean(Q)→0 makes the relative loss explode
# and actively fight the correct solution — training either destabilizes
# or settles on a spurious finite, z-varying Q. Scale by the known design
# volumetric flow instead, pin interior net Q→0, and match exhaust
# outflow to Q_design so mass has a real exit path.
MASS_FLOW_LOSS_WEIGHT = 25.0
# Extra multiplier on the continuity PDE residual alone (momentum/turb
# still use PDE_LOSS_WEIGHT / TURB_LOSS_WEIGHT). Continuity is one term
# against many BC terms; without this it is chronically under-weighted.
CONTINUITY_LOSS_WEIGHT = 5.0

OnProgressFn = Callable[[int, int, float], None]

# ─────────────────────────────────────────────────────────────────────────
# PRODUCTION INFERENCE CHECKPOINT (train once, e.g. in Colab, deploy the
# checkpoint, serve inference-only in app.py — no training after deploy).
#
# NOTE — this only produces correct results for the exact geometry/
# operating point it was trained on. CycloneFieldPINN takes only (r, z) as
# input; geometry and operating condition are baked in as fixed constants
# at training time (see module docstring above). A single checkpoint is
# therefore tied to one design, not a general-purpose model — this is a
# deliberate, explicit tradeoff, not an oversight. Saving state_dict()
# alone is NOT sufficient to reproduce correct predictions: the network's
# inputs/outputs are only meaningful relative to the exact FieldScaler
# (L/U/P/K/E) and geometry it was trained against, so both are stored
# alongside the weights.
# ─────────────────────────────────────────────────────────────────────────

def save_field_checkpoint(
    path: str,
    model: "CycloneFieldPINN",
    scaler: FieldScaler,
    geometry: CycloneAxisymGeometry,
    rho: float,
    nu: float,
    v_inlet: float,
    v_z_inlet: float,
    k_inlet: float,
    eps_inlet: float,
    flow_rate_cfm: float,
    hidden: int,
    n_layers: int,
) -> None:
    """Saves everything app.py needs to run inference with zero training:
    model weights + the exact scaler/geometry/fluid constants that give
    those weights meaning. `geometry` is a plain-attribute object (no
    tensors/handles), so it pickles safely via torch.save.

    flow_rate_cfm is stored explicitly because CycloneFieldPINN.forward
    takes it as an explicit conditioning input (alongside diameter_m) —
    without it here, app.py's inference path has no way to reconstruct
    the value the network was actually trained against."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden": hidden,
            "n_layers": n_layers,
            "scaler_state_dict": scaler.state_dict(),
            "geometry": geometry,
            "rho": rho,
            "nu": nu,
            "v_inlet": v_inlet,
            "v_z_inlet": v_z_inlet,
            "k_inlet": k_inlet,
            "eps_inlet": eps_inlet,
            "flow_rate_cfm": flow_rate_cfm,
        },
        path,
    )

# ─────────────────────────────────────────────────────────────────────────
# RESIDUAL NON-DIMENSIONALIZATION (root-cause fix for loss reaching
# 1e18-1e23 at realistic inlet velocities — confirmed empirically, not
# theoretical: a freshly-initialized network at v_inlet=78.66 m/s produces
# eps ~ 3.6e6 and k ~ 4.2e3 purely from FieldScaler's own K/E scales, so
# the eps-equation's destruction term (C2_EPS * eps^2 / k) evaluates to
# ~6e9 BEFORE any training happens — squared into the loss, that alone is
# ~3.6e19, matching exactly the magnitude seen stuck in real training runs.
#
# FieldScaler already non-dimensionalizes the network's INPUTS (r, z by L)
# and OUTPUTS (v by U, p by P, k by K, eps by E) — see field_model.py. That
# was necessary but not sufficient: the PDE/turbulence residuals computed
# FROM those outputs are still raw, dimensional physical quantities, and
# their natural scale is extremely sensitive to velocity because the
# dissipation-rate scale E = U^3/L grows with velocity CUBED. A cyclone
# design with a high inlet velocity (confirmed: doubling v_inlet from
# ~78 m/s to ~157 m/s took the loss from ~1e19 to ~1e23) will blow up the
# loss by construction, regardless of how well the network is training,
# unless the residuals themselves are non-dimensionalized the same way
# the outputs already are. This is standard PINN practice (see e.g. Raissi
# et al. 2019 discussions of residual scaling; also standard CFD practice
# of solving in non-dimensional form) — it was simply missing here because
# turbulence was added incrementally on top of an already-working laminar
# residual that didn't have this sensitivity (laminar residuals scale with
# U^2/L at worst, not U^3/L).
#
# Each residual is divided by its own characteristic physical scale before
# squaring, so all loss terms are O(1) in magnitude regardless of the
# absolute velocity/geometry scale of a given design:
#   continuity        : units of 1/s              -> scale U/L
#   r/theta/z momentum : units of m/s^2            -> scale U^2/L
#   k_equation         : units of m^2/s^3 (= E)    -> scale E
#   eps_equation       : units of m^2/s^4 (= E*U/L)-> scale E*U/L
# ─────────────────────────────────────────────────────────────────────────

def _pde_residual_scales(scaler: FieldScaler) -> dict[str, float]:
    U, L, E = scaler.U, scaler.L, scaler.E
    return {
        "continuity": U / L,
        "r_momentum": U ** 2 / L,
        "theta_momentum": U ** 2 / L,
        "z_momentum": U ** 2 / L,
        "k_equation": E,
        "eps_equation": E * U / L,
    }


# BC residuals are direct value/gradient differences (not PDE operators),
# so their natural magnitude is already tied to the quantity being
# compared (e.g. "inlet_eps" = predicted_eps - eps_inlet, which for a high-
# velocity design can itself be ~1e4-1e5 before squaring) rather than
# velocity-cubed PDE terms — but they are just as un-scaled as the PDE
# residuals were, so the same non-dimensionalization is applied for
# consistency and to keep BC_LOSS_WEIGHT meaningful across designs of very
# different scale. Value terms scale by the quantity's own characteristic
# scale; gradient terms (name ends in _dr/_dz) scale by (quantity scale)/L.
_BC_VELOCITY_KEYS = {
    "wall_vr", "wall_vtheta", "wall_vz",
    "axis_vr", "axis_vtheta", "axis_dvz_dr",
    "inlet_vr", "inlet_vtheta", "inlet_vz",
    "outlet_dvr_dz", "outlet_dvtheta_dz", "outlet_dvz_dz",
}
_BC_K_KEYS = {"wall_k", "axis_dk_dr", "inlet_k", "outlet_dk_dz"}
_BC_EPS_KEYS = {"wall_eps", "axis_deps_dr", "inlet_eps", "outlet_deps_dz"}
_BC_PRESSURE_KEYS = {"outlet_p"}


def _bc_residual_scale(key: str, scaler: FieldScaler) -> float:
    is_gradient = key.endswith("_dr") or key.endswith("_dz")
    if key in _BC_EPS_KEYS:
        return (scaler.E / scaler.L) if is_gradient else scaler.E
    if key in _BC_K_KEYS:
        return (scaler.K / scaler.L) if is_gradient else scaler.K
    if key in _BC_PRESSURE_KEYS:
        return scaler.P
    if key in _BC_VELOCITY_KEYS:
        return (scaler.U / scaler.L) if is_gradient else scaler.U
    # Should never happen given the fixed set of keys assemble_bc_losses
    # returns — fail loudly rather than silently skip scaling a term.
    raise KeyError(f"Unrecognized BC residual key '{key}' — add it to one "
                    f"of the _BC_*_KEYS sets in field_train.py so it gets "
                    f"properly non-dimensionalized.")

# rans_field_residuals divides by r and r^2 (with only a 1e-9 numerical
# floor) in the r- and theta-momentum residuals. geometry.sample_interior
# draws r uniformly across the full [0, wall_r] range, so with realistic
# batch sizes some points land within microns of the axis, producing 1/r^2
# terms in the billions that swamp the loss and destabilize both the Adam
# and L-BFGS phases (confirmed empirically: min r in a 2000-point batch was
# ~6e-6 m, giving a 1/r^2 term of ~3e10). The axis itself is already
# covered by the dedicated axis boundary condition
# (geometry.sample_axis + field_boundary_conditions.axis_symmetry_residual,
# enforcing v_r=0, v_theta=0, d(v_z)/dr=0, d(k)/dr=0, d(eps)/dr=0), so
# excluding a thin band around r=0 from interior PDE sampling loses no
# physics coverage — it just keeps collocation points out of a region
# already handled by a different, more numerically appropriate constraint.
AXIS_EXCLUSION_FRAC = 0.01  # exclude r < 1% of the barrel radius


def _sample_interior_away_from_axis(
    geometry: CycloneAxisymGeometry, n: int, device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    r_min = AXIS_EXCLUSION_FRAC * geometry.r_barrel
    pts_r, pts_z = [], []
    got = 0
    while got < n:
        batch_n = max(n - got, 64)
        r, z = geometry.sample_interior(max(batch_n * 2, 64), device=device)
        keep = r >= r_min
        r_keep, z_keep = r[keep], z[keep]
        pts_r.append(r_keep)
        pts_z.append(z_keep)
        got += r_keep.shape[0]
    r_cat = torch.cat(pts_r)[:n]
    z_cat = torch.cat(pts_z)[:n]
    # r_cat/z_cat are the result of boolean-mask indexing + cat on top of
    # sample_interior's leaf tensors, so they are themselves NON-leaf
    # tensors carrying that indexing/cat graph history — even though
    # requires_grad is already True (inherited), so calling
    # .requires_grad_(True) on them silently no-ops rather than erroring.
    # If reused across more than one backward() call (as the LBFGS phase's
    # fixed batch does), the first backward() frees those upstream
    # indexing/cat buffers, and a second backward() through the same
    # tensors then fails with "Trying to backward through the graph a
    # second time" — confirmed by hitting exactly that error in testing.
    # Detaching and re-leafing gives a genuine leaf tensor with no
    # inherited graph, safe to reuse across any number of forward/backward
    # passes, which is what these collocation points are meant to be.
    r_cat = r_cat.detach().clone().requires_grad_(True)
    z_cat = z_cat.detach().clone().requires_grad_(True)
    return r_cat, z_cat


def _pde_and_turb_loss(model_fn, geometry: CycloneAxisymGeometry, rho: torch.Tensor,
                        nu: torch.Tensor, scaler: FieldScaler, n_interior: int,
                        device: str) -> torch.Tensor:
    r, z = _sample_interior_away_from_axis(geometry, n_interior, device)
    res = rans_field_residuals(model_fn, r, z, rho, nu)
    scales = _pde_residual_scales(scaler)
    continuity = ((res["continuity"] / scales["continuity"]) ** 2).mean()
    pde = (
        CONTINUITY_LOSS_WEIGHT * continuity
        + ((res["r_momentum"] / scales["r_momentum"]) ** 2).mean()
        + ((res["theta_momentum"] / scales["theta_momentum"]) ** 2).mean()
        + ((res["z_momentum"] / scales["z_momentum"]) ** 2).mean()
    )
    turb = (
        (res["k_equation"] / scales["k_equation"]) ** 2
    ).mean() + (
        (res["eps_equation"] / scales["eps_equation"]) ** 2
    ).mean()
    return PDE_LOSS_WEIGHT * pde + TURB_LOSS_WEIGHT * turb


def _bc_loss(model_fn, geometry: CycloneAxisymGeometry, v_inlet: torch.Tensor,
             v_z_inlet: torch.Tensor, k_inlet: torch.Tensor, eps_inlet: torch.Tensor,
             scaler: FieldScaler, device: str) -> torch.Tensor:
    bc_losses = assemble_bc_losses(model_fn, geometry, v_inlet, v_z_inlet, k_inlet, eps_inlet, device=device)
    # bc_losses values are ALREADY mean-squared (see assemble_bc_losses),
    # so we divide by scale**2 here, not scale, to non-dimensionalize them
    # the same way the PDE/turbulence residuals above are.
    total = 0.0
    for key, mean_sq_residual in bc_losses.items():
        scale = _bc_residual_scale(key, scaler)
        total = total + mean_sq_residual / (scale ** 2)
    return total


def _axial_volume_flow(
    model_fn,
    r_max: float,
    z: float,
    n_r: int,
    device: str,
) -> torch.Tensor:
    """Q = 2*pi*integral_0^{r_max} v_z(r,z) * r dr at one axial plane."""
    r = torch.linspace(0.0, float(r_max), n_r, device=device)
    zz = torch.full((n_r,), float(z), device=device)
    vz = model_fn(r, zz)["v_z"]
    return 2.0 * torch.pi * torch.trapz(vz * r, r)


def _mass_flow_loss(
    model,
    scaler,
    geometry: CycloneAxisymGeometry,
    q_design: float,
    device: str,
    diameter_m: torch.Tensor,
    flow_rate_cfm: torch.Tensor,
    n_planes: int = 8,
    n_r: int = 64,
) -> torch.Tensor:
    """
    Aggregate mass-conservation regularizer (complements pointwise
    continuity in _pde_and_turb_loss; same Q(z) sanity_check measures).

    For impermeable side walls, integrated continuity requires Q(z) constant
    on interior planes, and for a gas cyclone that constant equals the
    bottom-outlet throughput (~0) because nearly all mass leaves through
    the top exhaust. Separately, exhaust outflow must match the design
    inlet rate so mass has a real exit path (pointwise outlet BCs alone
    only enforce zero-gradient / p=0 — they do not fix the flux).

    Critical scaling: residuals are divided by q_design (known, stable),
    NEVER by mean(Q). Dividing by mean(Q) blows up as reverse flow drives
    net Q→0 — the physically correct answer — and was the root cause of
    repeated mass-conservation training failures.

    Each interior plane integrates only out to geometry.outer_wall_radius(z)
    (cone taper), not a fixed r_barrel — integrating past the wall would
    sample outside the fluid domain and corrupt Q(z).
    """
    model_fn = model.as_model_fn(scaler, diameter_m, flow_rate_cfm)
    q_scale = max(abs(float(q_design)), 1e-6)

    z_planes = torch.linspace(
        0.15 * geometry.total_height,
        0.85 * geometry.total_height,
        n_planes,
        device=device,
    )

    q_values = []
    for z in z_planes:
        r_max = float(geometry.outer_wall_radius(z.unsqueeze(0)).squeeze(0))
        q_values.append(_axial_volume_flow(model_fn, r_max, float(z), n_r, device))
    q_values = torch.stack(q_values)
    q_mean = q_values.mean()

    # Constancy across interior planes, relative to design flow scale.
    constancy = ((q_values - q_mean.detach()) / q_scale).pow(2).mean()
    # Net mid-plane through-flow ≈ 0 (gas exits via top exhaust).
    level = (q_mean / q_scale).pow(2)

    # Exhaust outflow at z=0, r in [0, r_exhaust]. Domain outward normal at
    # the top is -z, so outward volume flux = -integral(v_z * 2*pi*r dr).
    q_out_exhaust = -_axial_volume_flow(
        model_fn, geometry.r_exhaust, 0.0, n_r, device,
    )
    exhaust_match = ((q_out_exhaust - q_scale) / q_scale).pow(2)

    # Bottom dust outlet should carry ~0 gas volume flow.
    q_out_bottom = _axial_volume_flow(
        model_fn, geometry.r_bottom_outlet, geometry.total_height, n_r, device,
    )
    bottom_match = (q_out_bottom / q_scale).pow(2)

    return constancy + level + exhaust_match + bottom_match


def _total_loss(model, scaler, geometry, rho, nu, v_inlet, v_z_inlet, k_inlet, eps_inlet,
                 q_design: float, n_interior: int, device: str,
                 diameter_m: torch.Tensor, flow_rate_cfm: torch.Tensor) -> torch.Tensor:
    model_fn = model.as_model_fn(scaler, diameter_m, flow_rate_cfm)
    pde_turb = _pde_and_turb_loss(model_fn, geometry, rho, nu, scaler, n_interior, device)
    bc = _bc_loss(model_fn, geometry, v_inlet, v_z_inlet, k_inlet, eps_inlet, scaler, device)
    mass = _mass_flow_loss(model, scaler, geometry, q_design, device, diameter_m, flow_rate_cfm)
    return pde_turb + BC_LOSS_WEIGHT * bc + MASS_FLOW_LOSS_WEIGHT * mass


def train_field_model(
    geometry: CycloneAxisymGeometry,
    rho: float,
    nu: float,
    v_inlet: float,
    v_z_inlet: float,
    k_inlet: float,
    eps_inlet: float,
    flow_rate_cfm: float,
    epochs_adam: int = 3000,
    epochs_lbfgs: int = 300,
    n_interior: int = 2048,
    hidden: int = 64,
    n_layers: int = 6,
    lr_adam_start: float = 3e-3,
    lr_adam_end: float = 1e-4,
    lr_lbfgs: float = 0.5,
    grad_clip_norm: float = 1.0,
    device: str = "cpu",
    seed: Optional[int] = 0,
    on_progress: Optional[OnProgressFn] = None,
    progress_every: int = 25,
) -> tuple[CycloneFieldPINN, FieldScaler, dict]:
    """
    Trains a fresh CycloneFieldPINN (velocity/pressure/k/epsilon) for one
    fixed geometry/operating point, using the RANS (Launder-Sharma
    k-epsilon) closure in field_turbulence.py.

    Args:
        geometry: fluid domain (see field_physics.geometry_from_dimensions_mm)
        rho, nu: fluid density (kg/m3) and kinematic (molecular) viscosity (m2/s)
        v_inlet: inlet tangential velocity (m/s), see field_physics.inlet_velocity_ms
        v_z_inlet: inlet axial (mass-carrying) velocity (m/s), see
            field_physics.inlet_axial_velocity_ms — this is what actually
            ties the trained field to the design flow rate; see the
            root-cause note in field_physics.py / field_boundary_conditions.py.
        k_inlet, eps_inlet: inlet turbulence kinetic energy (m2/s2) and
            dissipation rate (m2/s3), see field_turbulence.inlet_turbulence_quantities
        flow_rate_cfm: design flow rate in CFM. Used two ways: (a) to derive
            q_design for _mass_flow_loss (via v_z_inlet, as before), and
            (b) as CycloneFieldPINN's explicit "flow_rate_cfm" conditioning
            input alongside diameter_m (= 2*geometry.r_barrel) — see
            field_model.py's module docstring for why the network needs
            these as separate inputs rather than only via normalization.
        epochs_adam: Adam phase epoch count
        epochs_lbfgs: L-BFGS fine-tune phase step count
        n_interior: PDE collocation points sampled fresh every epoch
        hidden, n_layers: CycloneFieldPINN architecture
        lr_adam_start / lr_adam_end: exponential LR decay bounds over the
            Adam phase (decaying LR — fast early progress, stable late fit)
        lr_lbfgs: L-BFGS phase learning rate
        grad_clip_norm: max gradient norm during the Adam phase
        device: "cpu" or "cuda"
        seed: RNG seed for reproducible collocation sampling + init;
            None to skip seeding
        on_progress: optional callback(epoch, total_epochs, loss_value),
            called every `progress_every` epochs/steps and once at the end
            of each phase
        progress_every: how often (in epochs/steps) to invoke on_progress

    Returns:
        (model, scaler, history) where model is the trained CycloneFieldPINN
        in eval() mode, scaler is the matching FieldScaler, and history is a
        dict with "adam_loss" / "lbfgs_loss" loss curves (python floats) and
        "final_loss" / "wall_time_s" summary values.
    """
    if seed is not None:
        torch.manual_seed(seed)

    rho_t = torch.as_tensor(float(rho), device=device)
    nu_t = torch.as_tensor(float(nu), device=device)
    v_inlet_t = torch.as_tensor(float(v_inlet), device=device)
    v_z_inlet_t = torch.as_tensor(float(v_z_inlet), device=device)
    k_inlet_t = torch.as_tensor(float(k_inlet), device=device)
    eps_inlet_t = torch.as_tensor(float(eps_inlet), device=device)
    # Design volumetric flow implied by the inlet-ring axial BC — stable
    # scale for _mass_flow_loss (must NOT use mean(Q), see docstring).
    q_design = float(v_z_inlet) * math.pi * (
        geometry.r_barrel ** 2 - geometry.r_exhaust ** 2
    )
    # Explicit conditioning inputs the network needs on every forward pass
    # (see CycloneFieldPINN.forward / as_model_fn) — fixed for this whole
    # training call since this function trains one geometry/flow rate.
    diameter_m_t = torch.as_tensor(2.0 * geometry.r_barrel, device=device)
    flow_rate_cfm_t = torch.as_tensor(float(flow_rate_cfm), device=device)

    length_scale = geometry.r_barrel
    velocity_scale = max(float(v_inlet), 1e-6)
    scaler = FieldScaler(length_scale_m=length_scale, velocity_scale_ms=velocity_scale, rho=float(rho))

    model = CycloneFieldPINN(hidden=hidden, n_layers=n_layers).to(device)
    model.train()

    start_time = time.time()
    adam_loss_history: list[float] = []
    lbfgs_loss_history: list[float] = []
    total_epochs = epochs_adam + epochs_lbfgs

    # ── Phase 1: Adam, decaying LR, gradient clipping ──────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_adam_start)
    # Exponential decay from lr_adam_start to lr_adam_end over epochs_adam steps.
    decay_gamma = (lr_adam_end / lr_adam_start) ** (1.0 / max(epochs_adam, 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_gamma)

    for epoch in range(1, epochs_adam + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _total_loss(model, scaler, geometry, rho_t, nu_t, v_inlet_t, v_z_inlet_t,
                            k_inlet_t, eps_inlet_t, q_design, n_interior, device,
                            diameter_m_t, flow_rate_cfm_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step()

        loss_value = float(loss.item())
        adam_loss_history.append(loss_value)

        if on_progress is not None and (epoch % progress_every == 0 or epoch == epochs_adam):
            on_progress(epoch, total_epochs, loss_value)

    # ── Phase 2: L-BFGS fine-tune ────────────────────────────────────────
    # Unlike the Adam phase, collocation points are sampled ONCE here and
    # held fixed for the whole phase. L-BFGS builds its curvature (inverse
    # Hessian) approximation from the change in gradient between
    # consecutive steps — if the objective itself changes underneath it
    # (as it would with fresh random points every step, the way Adam
    # samples), that curvature estimate is measuring noise, not the true
    # loss landscape, and the strong-Wolfe line search can be driven to
    # wildly wrong step sizes (confirmed empirically: this caused the loss
    # to explode to 1e23 before switching to a fixed batch here). A larger,
    # fixed batch is used to compensate for losing per-step resampling
    # coverage.
    lbfgs_interior_n = n_interior * 4
    r_fixed, z_fixed = _sample_interior_away_from_axis(geometry, lbfgs_interior_n, device)

    def closure():
        optimizer_lbfgs.zero_grad(set_to_none=True)
        model_fn = model.as_model_fn(scaler, diameter_m_t, flow_rate_cfm_t)
        res = rans_field_residuals(model_fn, r_fixed, z_fixed, rho_t, nu_t)
        scales = _pde_residual_scales(scaler)
        continuity = ((res["continuity"] / scales["continuity"]) ** 2).mean()
        pde_turb_loss = (
            PDE_LOSS_WEIGHT * (
                CONTINUITY_LOSS_WEIGHT * continuity
                + ((res["r_momentum"] / scales["r_momentum"]) ** 2).mean()
                + ((res["theta_momentum"] / scales["theta_momentum"]) ** 2).mean()
                + ((res["z_momentum"] / scales["z_momentum"]) ** 2).mean()
            )
            + TURB_LOSS_WEIGHT * (
                ((res["k_equation"] / scales["k_equation"]) ** 2).mean()
                + ((res["eps_equation"] / scales["eps_equation"]) ** 2).mean()
            )
        )
        bc = _bc_loss(model_fn, geometry, v_inlet_t, v_z_inlet_t, k_inlet_t, eps_inlet_t, scaler, device)
        mass = _mass_flow_loss(model, scaler, geometry, q_design, device, diameter_m_t, flow_rate_cfm_t)
        loss = pde_turb_loss + BC_LOSS_WEIGHT * bc + MASS_FLOW_LOSS_WEIGHT * mass
        loss.backward()
        closure_history["losses"].append(float(loss.item()))
        return loss

    closure_history: dict[str, list[float]] = {"losses": []}

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=lr_lbfgs,
        max_iter=max(epochs_lbfgs, 1),   # internal iterations against the fixed batch
        history_size=10,
        line_search_fn="strong_wolfe",
    )
    optimizer_lbfgs.step(closure)

    lbfgs_loss_history = closure_history["losses"]
    if on_progress is not None:
        for i, loss_value in enumerate(lbfgs_loss_history, start=1):
            if i % progress_every == 0 or i == len(lbfgs_loss_history):
                on_progress(epochs_adam + i, epochs_adam + len(lbfgs_loss_history), loss_value)

    model.eval()

    final_loss = lbfgs_loss_history[-1] if lbfgs_loss_history else (
        adam_loss_history[-1] if adam_loss_history else float("nan")
    )
    history = {
        "adam_loss": adam_loss_history,
        "lbfgs_loss": lbfgs_loss_history,
        "final_loss": final_loss,
        "wall_time_s": time.time() - start_time,
    }

    return model, scaler, history


# ─────────────────────────────────────────────────────────────────────────
# SHARED ORCHESTRATION — geometry/fluid-property glue + train + evaluate
# ─────────────────────────────────────────────────────────────────────────
# This is the SINGLE place that turns raw request fields (mm dimensions +
# process conditions, exactly what PredictFieldStartRequest/the CLI accept)
# into a trained field and a queryable grid. app.py's _run_field_job and
# the CLI entry point below both call this — the glue (geometry_from_
# dimensions_mm -> fluid_properties -> inlet_velocity_ms -> inlet
# turbulence quantities) previously lived only inline in app.py; it is
# factored out here so there is exactly one implementation for both callers
# to stay in sync with.

def run_field_prediction_job(
    barrel_diameter_mm: float,
    barrel_height_mm: float,
    cone_height_mm: float,
    exhaust_dia_mm: float,
    exhaust_length_mm: float,
    bottom_outlet_mm: float,
    inlet_height_mm: float,
    inlet_width_mm: float,
    flow_rate_cfm: float,
    operating_temp_c: float = 25.0,
    operating_press_kpa: float = 101.325,
    gas_type: str = "Air",
    epochs_adam: int = 3000,
    epochs_lbfgs: int = 300,
    on_progress: Optional[OnProgressFn] = None,
    save_checkpoint_path: Optional[str] = None,
    **train_kwargs,
) -> dict:
    """
    End-to-end: geometry + fluid properties + inlet velocity + inlet
    turbulence quantities -> train -> evaluate on a grid. Takes exactly the
    field set app.py's PredictFieldStartRequest and the CLI both expose, so
    both callers can pass their parsed request straight through without any
    per-caller glue.

    Args:
        barrel_diameter_mm..bottom_outlet_mm: geometry, see
            field_physics.geometry_from_dimensions_mm
        inlet_height_mm, inlet_width_mm, flow_rate_cfm: inlet sizing/flow,
            see field_physics.inlet_velocity_ms and
            field_turbulence.hydraulic_diameter_rect_m /
            inlet_turbulence_quantities
        operating_temp_c, operating_press_kpa, gas_type: process conditions,
            see field_physics.fluid_properties / field_physics.gas_type_to_onehot
        epochs_adam, epochs_lbfgs: forwarded to train_field_model
        on_progress: forwarded to train_field_model
        save_checkpoint_path: if set, calls save_field_checkpoint() with
            this path after training completes — this is the "train once,
            deploy the file" step for production inference (see app.py).
        **train_kwargs: any other train_field_model kwarg (n_interior,
            hidden, n_layers, lr_adam_start, ..., device, seed, ...)

    Returns:
        dict with keys:
            geometry (CycloneAxisymGeometry), rho (float), nu (float),
            v_inlet (float), k_inlet (float), eps_inlet (float),
            model (CycloneFieldPINN, eval mode), scaler (FieldScaler),
            history (dict, see train_field_model),
            grid (dict: r_m/z_m/v_r_ms/v_theta_ms/v_z_ms/pressure_pa lists)
    """
    geometry = geometry_from_dimensions_mm(
        barrel_diameter_mm=barrel_diameter_mm,
        barrel_height_mm=barrel_height_mm,
        cone_height_mm=cone_height_mm,
        exhaust_dia_mm=exhaust_dia_mm,
        exhaust_length_mm=exhaust_length_mm,
        bottom_outlet_mm=bottom_outlet_mm,
    )

    gas_onehot = torch.tensor([gas_type_to_onehot(gas_type)])
    rho_t, nu_t = fluid_properties(
        torch.tensor([operating_temp_c]),
        torch.tensor([operating_press_kpa]),
        gas_onehot,
    )
    rho, nu = rho_t.item(), nu_t.item()

    v_inlet = inlet_velocity_ms(
        torch.tensor([flow_rate_cfm]),
        torch.tensor([inlet_height_mm * 1e-3]),
        torch.tensor([inlet_width_mm * 1e-3]),
    ).item()

    # Root-cause fix: the axial mass-injection velocity that actually ties
    # the trained field to flow_rate_cfm — see
    # field_physics.inlet_axial_velocity_ms and the Q(z) mass-conservation
    # note in field_boundary_conditions.py.
    v_z_inlet = inlet_axial_velocity_ms(
        torch.tensor([flow_rate_cfm]),
        r_barrel_m=geometry.r_barrel,
        r_exhaust_m=geometry.r_exhaust,
    ).item()

    hydraulic_diameter_m = hydraulic_diameter_rect_m(
        height_m=inlet_height_mm * 1e-3, width_m=inlet_width_mm * 1e-3,
    )
    k_inlet_t, eps_inlet_t = inlet_turbulence_quantities(
        v_inlet=torch.tensor([v_inlet]),
        hydraulic_diameter_m=torch.tensor([hydraulic_diameter_m]),
        nu=torch.tensor([nu]),
    )
    k_inlet, eps_inlet = k_inlet_t.item(), eps_inlet_t.item()

    model, scaler, history = train_field_model(
        geometry, rho, nu, v_inlet, v_z_inlet, k_inlet, eps_inlet,
        flow_rate_cfm=flow_rate_cfm,
        epochs_adam=epochs_adam,
        epochs_lbfgs=epochs_lbfgs,
        on_progress=on_progress,
        **train_kwargs,
    )

    grid = evaluate_grid(
        model, scaler, geometry,
        diameter_m=2.0 * geometry.r_barrel,
        flow_rate_cfm=flow_rate_cfm,
    )

    q_design = float(v_z_inlet) * math.pi * (
        geometry.r_barrel ** 2 - geometry.r_exhaust ** 2
    )

    # Root-cause fix: massConservationStatus/massFlowSpread/finalLoss were
    # defined on both the Python and .NET DTOs (FieldResultDto) but never
    # actually computed or populated here, so app.py always sent them as
    # null/None and EngineeringInsightRepository.EvaluateMassConservation
    # on the .NET side treated every job as "failed" regardless of solve
    # quality. sanity_check.mass_conservation_metrics already had the exact
    # math (Q(z) spread across the barrel mid-section) — it just wasn't
    # wired into the live /predict_field job path. Attaching it to `grid`
    # (rather than adding new top-level dict keys) means app.py's existing
    # `grid = result["grid"]` line picks it up with no extra plumbing.
    mc = mass_conservation_metrics(grid, q_design=q_design)
    grid["mass_conservation_status"] = mc["status"]
    grid["mass_flow_spread"] = mc["rel_spread"]

    if save_checkpoint_path:
        save_field_checkpoint(
            save_checkpoint_path,
            model=model,
            scaler=scaler,
            geometry=geometry,
            rho=rho,
            nu=nu,
            v_inlet=v_inlet,
            v_z_inlet=v_z_inlet,
            k_inlet=k_inlet,
            eps_inlet=eps_inlet,
            flow_rate_cfm=flow_rate_cfm,
            hidden=train_kwargs.get("hidden", 64),
            n_layers=train_kwargs.get("n_layers", 6),
        )

    return {
        "geometry": geometry,
        "rho": rho,
        "nu": nu,
        "v_inlet": v_inlet,
        "v_z_inlet": v_z_inlet,
        "q_design": q_design,
        "k_inlet": k_inlet,
        "eps_inlet": eps_inlet,
        "model": model,
        "scaler": scaler,
        "history": history,
        "grid": grid,
    }


# ─────────────────────────────────────────────────────────────────────────
# PARAMETRIC (DOMAIN-RANDOMIZATION) TRAINING
#
# Everything above (train_field_model / run_field_prediction_job) trains a
# fresh network for ONE fixed geometry + flow rate — that is still a fully
# supported, valid way to use this file (e.g. a quick single-design check).
#
# What follows is the OTHER mode: train ONE CycloneFieldPINN that covers an
# entire family of LAPPLE-type cyclones (any diameter/flow rate inside the
# sampled ranges), by picking a different geometry/flow rate every epoch
# instead of holding both fixed for the whole run. This is what
# CycloneFieldPINN's 4-input signature (r, z, diameter_m, flow_rate_cfm)
# was built for — see field_model.py's module docstring.
#
# Only Adam is used here, never L-BFGS. L-BFGS estimates curvature from how
# the gradient changes between consecutive steps; that estimate is only
# valid if the objective is the same function from one step to the next.
# Domain randomization changes the geometry AND the flow rate every epoch,
# so the "objective" is a different function every step — L-BFGS's
# curvature estimate would be measuring noise, not the loss landscape (this
# is exactly why train_field_model's own L-BFGS phase above is only safe
# because it samples a fixed batch from a SINGLE geometry).
# ─────────────────────────────────────────────────────────────────────────

# Standard high-efficiency Lapple cyclone proportions (dimension / barrel
# diameter), used ONLY as a fallback default so this module has something
# runnable out of the box. These mirror the field names of the .NET
# CyclonTypeRatios DTO (CyclonApp.Model/DTOs/CyclonTypeRatios.cs) —
# ExhaustLengthRatio doesn't have one universally agreed literature figure,
# so this uses ExhaustDiaRatio's typical companion length.
#
# IMPORTANT: this app already stores per-cyclone-type ratios in the
# CycloneType DB table (that's what CyclonTypeRatios models). Before running
# a real training job, pull the actual "Lapple" row's ratios from that table
# and pass them as the `ratios` argument below instead of relying on this
# default — that guarantees the PINN is trained on the exact same geometry
# family the rest of the app (and any Lapple/Shepherd-Lapple analytic
# calculations it's compared against) uses. Treat the values below as a
# textbook placeholder, not a source of truth for your data.
LAPPLE_RATIOS: dict[str, float] = {
    "InletHeightRatio": 0.50,
    "InletWidthRatio": 0.25,
    "BarrelHeightRatio": 2.00,   # was 1.50 — wrong
    "ConeHeightRatio": 2.00,     # was 2.50 — wrong
    "OutletDiamRatio": 0.50,
    "BottomOutletRatio": 0.25,
    "ExhaustLengthRatio": 0.625,
}

# Standard Stairmand High-Efficiency (HE) cyclone proportions (dimension /
# barrel diameter D), as published in the Stairmand HE geometry spec.
# Unlike LAPPLE_RATIOS above, every one of these 7 values (including
# ExhaustLengthRatio, the vortex-finder length) has a directly documented
# figure — there is no "no universally agreed literature figure" placeholder
# needed here:
#   Inlet height   a  = 0.50 D  -> InletHeightRatio
#   Inlet width    b  = 0.20 D  -> InletWidthRatio
#   Cylinder ht.   h  = 1.50 D  -> BarrelHeightRatio
#   Cone height    Hc = 2.50 D  -> ConeHeightRatio
#   Vortex-finder diameter De = 0.50 D -> OutletDiamRatio
#   Vortex-finder length   S  = 0.50 D -> ExhaustLengthRatio
#   Dust outlet diameter   B  = 0.375 D -> BottomOutletRatio
#
# Same caveat as LAPPLE_RATIOS applies: this is a textbook default for
# running this module standalone. Before a production training run, pull
# the actual "Stairmand" row's ratios from the CycloneType DB table so the
# PINN is trained on exactly the geometry family the rest of the app (and
# any Lapple-cut-size/Shepherd-Lapple analytic comparison) uses for that
# type — do not let this dict silently drift out of sync with that row.
STAIRMAND_RATIOS: dict[str, float] = {
    "InletHeightRatio": 0.50,
    "InletWidthRatio": 0.20,
    "BarrelHeightRatio": 1.50,
    "ConeHeightRatio": 2.50,
    "OutletDiamRatio": 0.50,
    "BottomOutletRatio": 0.375,
    "ExhaustLengthRatio": 0.50,
}


def geometry_mm_from_diameter(
    barrel_diameter_mm: float, ratios: dict[str, float],
) -> dict[str, float]:
    """Scales every other LAPPLE dimension off one sampled barrel diameter,
    using fixed dimension/diameter ratios. Returns a dict with exactly the
    keyword names geometry_from_dimensions_mm (and run_field_prediction_job)
    expect, so it can be splatted straight into either.

    This is what lets domain randomization sample a single scalar (the
    diameter) per epoch and still get a complete, geometrically-consistent
    cyclone — rather than having to independently sample 6+ correlated
    dimensions, most combinations of which wouldn't be a valid/manufacturable
    cyclone at all.
    """
    d = float(barrel_diameter_mm)
    return {
        "barrel_diameter_mm": d,
        "barrel_height_mm": d * ratios["BarrelHeightRatio"],
        "cone_height_mm": d * ratios["ConeHeightRatio"],
        "exhaust_dia_mm": d * ratios["OutletDiamRatio"],
        "exhaust_length_mm": d * ratios["ExhaustLengthRatio"],
        "bottom_outlet_mm": d * ratios["BottomOutletRatio"],
        "inlet_height_mm": d * ratios["InletHeightRatio"],
        "inlet_width_mm": d * ratios["InletWidthRatio"],
    }


def _sample_diameter_and_flow(
    diameter_range_m: tuple[float, float],
    flow_rate_range_cfm: tuple[float, float],
    device: str,
) -> tuple[float, float]:
    """One (diameter_m, flow_rate_cfm) draw, uniform over each range
    independently. Plain Python floats (not tensors) on purpose — this
    value drives geometry construction (geometry_mm_from_diameter,
    geometry_from_dimensions_mm), which is plain Python/CycloneAxisymGeometry
    code, not an autograd graph; it only becomes a tensor once it's fed to
    the network as a conditioning input."""
    d = diameter_range_m[0] + torch.rand(1, device=device).item() * (
        diameter_range_m[1] - diameter_range_m[0]
    )
    q = flow_rate_range_cfm[0] + torch.rand(1, device=device).item() * (
        flow_rate_range_cfm[1] - flow_rate_range_cfm[0]
    )
    return d, q


def train_parametric_field_model(
    rho_fn,
    ratios: dict[str, float] = LAPPLE_RATIOS,
    diameter_range_m: tuple[float, float] = (0.150, 0.750),
    flow_rate_range_cfm: tuple[float, float] = (300.0, 13000.0),
    operating_temp_c: float = 25.0,
    operating_press_kpa: float = 101.325,
    gas_type: str = "Air",
    epochs: int = 20000,
    n_interior: int = 1024,
    hidden: int = 64,
    n_layers: int = 6,
    lr_start: float = 3e-3,
    lr_end: float = 1e-4,
    grad_clip_norm: float = 1.0,
    device: str = "cpu",
    seed: Optional[int] = 0,
    on_progress: Optional[OnProgressFn] = None,
    progress_every: int = 100,
    resume_from: Optional[str] = None,
    checkpoint_every: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
) -> tuple[CycloneFieldPINN, FieldScaler, dict]:
    """
    Trains ONE CycloneFieldPINN across a whole family of LAPPLE cyclones
    instead of a single fixed geometry — see the module-level comment above
    this function for why (domain randomization + why L-BFGS is dropped).

    Every epoch:
      1. sample a diameter and flow rate,
      2. scale the rest of the LAPPLE dimensions off that diameter (`ratios`),
      3. build that epoch's geometry + fluid/inlet/turbulence quantities,
      4. draw fresh collocation/boundary points for THAT geometry,
      5. build model_fn = model.as_model_fn(scaler, diameter_m, flow_rate_cfm)
         so every physics/BC/mass-flow term is evaluated against the
         network's prediction FOR that sampled design (not some other
         geometry's field) — this is item 1/2/3 of the "remaining work":
         every as_model_fn() call and every _mass_flow_loss() call must
         receive the CURRENT epoch's sampled diameter_m/flow_rate_cfm, not
         a stale or default one.
      6. one Adam step.

    Args:
        rho_fn: callable(temp_c, press_kpa, gas_onehot) -> (rho, nu) tensors
            — pass field_physics.fluid_properties. Kept as a parameter
            (rather than importing directly) only to make this function
            trivially testable with a fake fluid model; production callers
            should pass field_physics.fluid_properties.
        ratios: LAPPLE dimension ratios (see geometry_mm_from_diameter) —
            pull these from the CycloneType DB table for "Lapple" rather
            than trusting the LAPPLE_RATIOS placeholder above.
        diameter_range_m / flow_rate_range_cfm: the family of designs this
            one network will learn to cover. Must be inside a physically
            sane LAPPLE size range for `ratios`, and should match (or be a
            subset of) the range you'll actually query at inference time —
            the network was never shown geometries outside this window and
            has no guarantee of extrapolating correctly beyond it. IGNORED
            if `resume_from` is set — see resume_from below.
        epochs: total Adam steps FOR THIS CALL (not cumulative across
            resumes — see resume_from). Needs to be much larger than
            train_field_model's epochs_adam, since each step only sees one
            (of infinitely many) sampled geometries — this is genuinely a
            harder learning problem than fitting one fixed field.
        n_interior: PDE collocation points sampled fresh every epoch for
            that epoch's geometry.
        hidden, n_layers: CycloneFieldPINN architecture. IGNORED if
            `resume_from` is set (the checkpoint's architecture wins, since
            weights must match the loaded state_dict's shape).
        lr_start / lr_end: exponential LR decay bounds over `epochs` for
            THIS call — resuming restarts the decay schedule from lr_start
            rather than picking up mid-schedule (simpler and safe: an Adam
            optimizer's moment estimates are not preserved across resume
            either, so there is no "true" mid-schedule LR to resume at).
        grad_clip_norm: max gradient norm per step.
        device: "cpu" or "cuda".
        seed: RNG seed for reproducible sampling + init; None to skip.
            IGNORED if `resume_from` is set (model weights come from the
            checkpoint, not a fresh seeded init).
        on_progress / progress_every: as in train_field_model.
        resume_from: path to a parametric checkpoint (see
            save_parametric_field_checkpoint) to continue training from,
            instead of a fresh random init. The checkpoint's own
            hidden/n_layers and D_min/D_max/Q_min/Q_max normalization
            window are used (printed if they differ from what was passed
            in) — the network's conditioning-input meaning must stay fixed
            across a resume, the same way it must stay fixed across epochs
            within one run (see FieldScaler.with_scales docstring). This is
            what makes it safe to train in bounded chunks (e.g. across
            multiple Colab sessions) instead of one all-or-nothing run.
        checkpoint_every: if set (with checkpoint_path), saves an
            intermediate parametric checkpoint every this-many epochs (and
            once more at the final epoch) DURING training — so a Colab
            disconnect loses at most checkpoint_every epochs of progress,
            not the whole run. Requires checkpoint_path.
        checkpoint_path: destination for the periodic saves above. The
            caller (field_train.py's CLI) is still responsible for treating
            its own final save as authoritative; this is a safety net
            against losing progress mid-run, not a substitute for it.

    Returns:
        (model, scaler, history). `scaler` is a "template" FieldScaler —
        its D_min/D_max/Q_min/Q_max are the fixed normalization window used
        for every epoch (correct to reuse at inference time via
        evaluate_grid), but its L/U/P/K/E are only from the LAST sampled
        epoch — at inference time, use FieldScaler.with_scales(...) (or
        just field_model.evaluate_grid, which only needs diameter_m/
        flow_rate_cfm, not L/U directly) for the specific design you're
        querying, not this returned scaler's raw L/U.
    """
    if checkpoint_every is not None and not checkpoint_path:
        raise ValueError("checkpoint_every requires checkpoint_path to be set.")

    if seed is not None:
        torch.manual_seed(seed)

    if resume_from:
        loaded = load_parametric_field_checkpoint(resume_from)
        if loaded["hidden"] != hidden or loaded["n_layers"] != n_layers:
            print(
                f"[train_parametric_field_model] NOTE: --hidden/--n-layers "
                f"({hidden}, {n_layers}) ignored — resuming from "
                f"'{resume_from}' architecture ({loaded['hidden']}, "
                f"{loaded['n_layers']})."
            )
        hidden, n_layers = loaded["hidden"], loaded["n_layers"]
        model = loaded["model"].to(device)
        model.train()
        resumed_scaler = loaded["scaler"]
        if (resumed_scaler.D_min, resumed_scaler.D_max) != diameter_range_m or (
            resumed_scaler.Q_min, resumed_scaler.Q_max
        ) != flow_rate_range_cfm:
            print(
                f"[train_parametric_field_model] NOTE: --diameter-*/--flow-* "
                f"range ignored — resuming from '{resume_from}''s trained "
                f"window diameter=[{resumed_scaler.D_min}, "
                f"{resumed_scaler.D_max}] m, "
                f"flow=[{resumed_scaler.Q_min}, {resumed_scaler.Q_max}] CFM. "
                f"Changing this window mid-training would make D_norm/Q_norm "
                f"mean something different than what the network already "
                f"learned (see FieldScaler.with_scales)."
            )
        diameter_range_m = (resumed_scaler.D_min, resumed_scaler.D_max)
        flow_rate_range_cfm = (resumed_scaler.Q_min, resumed_scaler.Q_max)
    else:
        model = CycloneFieldPINN(hidden=hidden, n_layers=n_layers).to(device)
        model.train()

    # Fixed for the whole run — this is what makes D_norm/Q_norm mean the
    # same thing on epoch 1 and epoch 20000 (see FieldScaler.with_scales).
    template_scaler = FieldScaler(
        length_scale_m=1.0, velocity_scale_ms=1.0, rho=1.0,
        diameter_range_m=diameter_range_m,
        flow_rate_range_cfm=flow_rate_range_cfm,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)
    decay_gamma = (lr_end / lr_start) ** (1.0 / max(epochs, 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_gamma)

    start_time = time.time()
    loss_history: list[float] = []
    scaler = template_scaler  # last-used scaler, returned for reference

    for epoch in range(1, epochs + 1):
        diameter_m, flow_rate_cfm = _sample_diameter_and_flow(
            diameter_range_m, flow_rate_range_cfm, device,
        )
        dims_mm = geometry_mm_from_diameter(diameter_m * 1e3, ratios)
        geometry_kwargs = {
            k: v for k, v in dims_mm.items()
            if k not in ("inlet_height_mm", "inlet_width_mm")
        }
        geometry = geometry_from_dimensions_mm(**geometry_kwargs)

        gas_onehot = torch.tensor([gas_type_to_onehot(gas_type)])
        rho_t, nu_t = rho_fn(
            torch.tensor([operating_temp_c]),
            torch.tensor([operating_press_kpa]),
            gas_onehot,
        )
        rho, nu = rho_t.item(), nu_t.item()

        v_inlet = inlet_velocity_ms(
            torch.tensor([flow_rate_cfm]),
            torch.tensor([dims_mm["inlet_height_mm"] * 1e-3]),
            torch.tensor([dims_mm["inlet_width_mm"] * 1e-3]),
        ).item()
        v_z_inlet = inlet_axial_velocity_ms(
            torch.tensor([flow_rate_cfm]),
            r_barrel_m=geometry.r_barrel, r_exhaust_m=geometry.r_exhaust,
        ).item()
        hydraulic_diameter_m = hydraulic_diameter_rect_m(
            height_m=dims_mm["inlet_height_mm"] * 1e-3,
            width_m=dims_mm["inlet_width_mm"] * 1e-3,
        )
        k_inlet_t, eps_inlet_t = inlet_turbulence_quantities(
            v_inlet=torch.tensor([v_inlet], device=device),
            hydraulic_diameter_m=torch.tensor([hydraulic_diameter_m], device=device),
            nu=torch.tensor([nu], device=device),
        )

        q_design = float(v_z_inlet) * math.pi * (
            geometry.r_barrel ** 2 - geometry.r_exhaust ** 2
        )
        # This epoch's physical scale — the D/Q normalization window comes
        # from template_scaler and does NOT change (see with_scales docstring).
        scaler = template_scaler.with_scales(
            length_scale_m=geometry.r_barrel,
            velocity_scale_ms=max(v_inlet, 1e-6),
            rho=rho,
        )

        diameter_m_t = torch.as_tensor(diameter_m, device=device)
        flow_rate_cfm_t = torch.as_tensor(float(flow_rate_cfm), device=device)
        rho_scalar_t = torch.as_tensor(rho, device=device)
        nu_scalar_t = torch.as_tensor(nu, device=device)
        v_inlet_t = torch.as_tensor(v_inlet, device=device)
        v_z_inlet_t = torch.as_tensor(v_z_inlet, device=device)

        optimizer.zero_grad(set_to_none=True)
        loss = _total_loss(
            model, scaler, geometry, rho_scalar_t, nu_scalar_t, v_inlet_t, v_z_inlet_t,
            k_inlet_t, eps_inlet_t, q_design, n_interior, device,
            diameter_m_t, flow_rate_cfm_t,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step()

        loss_value = float(loss.item())
        loss_history.append(loss_value)
        if on_progress is not None and (epoch % progress_every == 0 or epoch == epochs):
            on_progress(epoch, epochs, loss_value)

        if checkpoint_every is not None and (
            epoch % checkpoint_every == 0 or epoch == epochs
        ):
            save_parametric_field_checkpoint(
                checkpoint_path,
                model=model,
                scaler=template_scaler,
                ratios=ratios,
                operating_temp_c=operating_temp_c,
                operating_press_kpa=operating_press_kpa,
                gas_type=gas_type,
                hidden=hidden,
                n_layers=n_layers,
            )
            print(f"[train_parametric_field_model] checkpoint saved at "
                  f"epoch {epoch}/{epochs} -> {checkpoint_path}")

    model.eval()
    history = {
        "loss": loss_history,
        "final_loss": loss_history[-1] if loss_history else float("nan"),
        "wall_time_s": time.time() - start_time,
        "hidden": hidden,
        "n_layers": n_layers,
    }
    return model, template_scaler, history


def save_parametric_field_checkpoint(
    path: str,
    model: "CycloneFieldPINN",
    scaler: FieldScaler,
    ratios: dict[str, float],
    operating_temp_c: float,
    operating_press_kpa: float,
    gas_type: str,
    hidden: int,
    n_layers: int,
) -> None:
    """Saves a checkpoint for the PARAMETRIC training mode
    (train_parametric_field_model) — distinct from save_field_checkpoint,
    which is for train_field_model's single-fixed-geometry mode and
    requires a single baked geometry/rho/nu/v_inlet/etc. that a parametric
    model does not have (it covers a whole family of designs, not one).

    `scaler` must be the FieldScaler returned by train_parametric_field_model
    (the "template" scaler) — its D_min/D_max/Q_min/Q_max are the fixed
    parametric normalization window the network was trained against; its
    L/U/P/K/E are meaningless here and are NOT saved (app.py recomputes
    them per-request via FieldScaler.with_scales for the request's actual
    geometry — see app.py's _run_field_job).

    ratios/operating_temp_c/operating_press_kpa/gas_type are saved so the
    exact LAPPLE family and fluid assumptions this checkpoint was trained
    under are recorded alongside the weights, not left to tribal knowledge.
    """
    torch.save(
        {
            "checkpoint_kind": "parametric",
            "model_state_dict": model.state_dict(),
            "hidden": hidden,
            "n_layers": n_layers,
            "scaler_state_dict": scaler.state_dict(),
            "ratios": ratios,
            "operating_temp_c": operating_temp_c,
            "operating_press_kpa": operating_press_kpa,
            "gas_type": gas_type,
        },
        path,
    )


def load_parametric_field_checkpoint(path: str) -> dict:
    """Shared loader for parametric checkpoints (checkpoint_kind ==
    "parametric") — used by both train_parametric_field_model's
    `resume_from` (continue training) and app.py's serving path (inference
    only), so there is exactly one place that knows this file's format.

    Returns model in eval() mode with requires_grad left as-is on its
    parameters (resume_from re-enables .train() itself before continuing;
    app.py leaves it in eval() for serving).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("checkpoint_kind") != "parametric":
        raise RuntimeError(
            f"'{path}' is not a parametric checkpoint (missing/invalid "
            f"checkpoint_kind). Produce one with "
            f"`python field_train.py --mode parametric --save-checkpoint ...` "
            f"— a single-geometry checkpoint from --mode single is not "
            f"compatible with resume/serve."
        )
    model = CycloneFieldPINN(hidden=ckpt["hidden"], n_layers=ckpt["n_layers"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    scaler = FieldScaler.from_state_dict(ckpt["scaler_state_dict"])
    return {
        "model": model,
        "scaler": scaler,
        "hidden": ckpt["hidden"],
        "n_layers": ckpt["n_layers"],
        "ratios": ckpt.get("ratios"),
        "operating_temp_c": ckpt.get("operating_temp_c"),
        "operating_press_kpa": ckpt.get("operating_press_kpa"),
        "gas_type": ckpt.get("gas_type"),
    }


def evaluate_parametric_interpolation(
    model: CycloneFieldPINN,
    scaler: FieldScaler,
    ratios: dict[str, float],
    diameter_m: float,
    flow_rate_cfm: float,
    rho: float,
    operating_temp_c: float = 25.0,
    operating_press_kpa: float = 101.325,
    gas_type: str = "Air",
) -> dict:
    """
    Item 4 of the remaining work: check that a parametrically-trained model
    gives a physically sane result for a (diameter, flow rate) pair it was
    NOT necessarily trained on directly (any point inside the training
    ranges other than the exact sampled values counts, since domain
    randomization draws continuous, not gridded, values).

    Builds the geometry for the requested diameter, evaluates the trained
    model on it (via evaluate_grid, which only needs diameter_m/
    flow_rate_cfm — no retraining), and runs the same mass-conservation
    check used in production (sanity_check.mass_conservation_metrics) so
    "does it interpolate reasonably" has a concrete pass/fail signal instead
    of just eyeballing a plot.
    """
    dims_mm = geometry_mm_from_diameter(diameter_m * 1e3, ratios)
    geometry_kwargs = {
        k: v for k, v in dims_mm.items()
        if k not in ("inlet_height_mm", "inlet_width_mm")
    }
    geometry = geometry_from_dimensions_mm(**geometry_kwargs)

    gas_onehot = torch.tensor([gas_type_to_onehot(gas_type)])
    _, nu_t = fluid_properties(
        torch.tensor([operating_temp_c]), torch.tensor([operating_press_kpa]), gas_onehot,
    )
    nu = nu_t.item()

    v_z_inlet = inlet_axial_velocity_ms(
        torch.tensor([flow_rate_cfm]),
        r_barrel_m=geometry.r_barrel, r_exhaust_m=geometry.r_exhaust,
    ).item()
    q_design = float(v_z_inlet) * math.pi * (
        geometry.r_barrel ** 2 - geometry.r_exhaust ** 2
    )

    eval_scaler = scaler.with_scales(
        length_scale_m=geometry.r_barrel,
        velocity_scale_ms=max(
            inlet_velocity_ms(
                torch.tensor([flow_rate_cfm]),
                torch.tensor([dims_mm["inlet_height_mm"] * 1e-3]),
                torch.tensor([dims_mm["inlet_width_mm"] * 1e-3]),
            ).item(),
            1e-6,
        ),
        rho=rho,
    )
    grid = evaluate_grid(model, eval_scaler, geometry, diameter_m=diameter_m, flow_rate_cfm=flow_rate_cfm)
    mc = mass_conservation_metrics(grid, q_design=q_design)
    grid["mass_conservation_status"] = mc["status"]
    grid["mass_flow_spread"] = mc["rel_spread"]
    return grid


# ─────────────────────────────────────────────────────────────────────────
# CLI — offline testing/tuning without going through the HTTP service.
# Calls the exact same run_field_prediction_job as app.py, so results seen
# here match what /predict_field/start would produce for the same inputs.
# ─────────────────────────────────────────────────────────────────────────

def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Train a CycloneFieldPINN, offline — no HTTP service required. "
            "--mode single (default) trains one fixed geometry/operating "
            "point and evaluates a grid, matching what /predict_field/start "
            "would produce for the same inputs. --mode parametric trains "
            "ONE network across a whole diameter/flow-rate family (Lapple or "
            "Stairmand — see --ratios-preset) via domain randomization (see "
            "train_parametric_field_model) and "
            "requires --save-checkpoint — this is the actual 'train in "
            "Colab, deploy the checkpoint' step for app.py's production "
            "inference mode."
        ),
    )
    p.add_argument("--mode", choices=["single", "parametric"], default="single")
    p.add_argument("--ratios-preset", choices=["lapple", "stairmand"], default="lapple",
                   help="--mode parametric only: which cyclone family's fixed "
                        "dimension ratios (LAPPLE_RATIOS / STAIRMAND_RATIOS above) "
                        "to scale every sampled diameter against. Each family needs "
                        "its own checkpoint/.onnx — see those dicts' docstrings.")

    geo = p.add_argument_group("geometry (mm) — required for --mode single only")
    geo.add_argument("--barrel-diameter-mm", type=float, default=None)
    geo.add_argument("--barrel-height-mm", type=float, default=None)
    geo.add_argument("--cone-height-mm", type=float, default=None)
    geo.add_argument("--exhaust-dia-mm", type=float, default=None)
    geo.add_argument("--exhaust-length-mm", type=float, default=None)
    geo.add_argument("--bottom-outlet-mm", type=float, default=None)
    geo.add_argument("--inlet-height-mm", type=float, default=None)
    geo.add_argument("--inlet-width-mm", type=float, default=None)

    proc = p.add_argument_group("process conditions")
    proc.add_argument("--flow-rate-cfm", type=float, default=None,
                       help="required for --mode single; ignored for "
                            "--mode parametric (use --flow-min-cfm/--flow-max-cfm)")
    proc.add_argument("--operating-temp-c", type=float, default=25.0)
    proc.add_argument("--operating-press-kpa", type=float, default=101.325)
    proc.add_argument("--gas-type", type=str, default="Air")

    param = p.add_argument_group("parametric range (mm/CFM) — --mode parametric only")
    param.add_argument("--diameter-min-mm", type=float, default=150.0)
    param.add_argument("--diameter-max-mm", type=float, default=750.0)
    param.add_argument("--flow-min-cfm", type=float, default=300.0)
    param.add_argument("--flow-max-cfm", type=float, default=13000.0)

    train = p.add_argument_group("training")
    train.add_argument("--epochs-adam", type=int, default=3000,
                        help="--mode single Adam epoch count")
    train.add_argument("--epochs-lbfgs", type=int, default=300,
                        help="--mode single L-BFGS step count (parametric mode never uses L-BFGS)")
    train.add_argument("--epochs", type=int, default=20000,
                        help="--mode parametric Adam step count — needs to be much "
                             "larger than --epochs-adam since each step only sees "
                             "one sampled geometry (see train_parametric_field_model)")
    train.add_argument("--n-interior", type=int, default=2048)
    train.add_argument("--hidden", type=int, default=64)
    train.add_argument("--n-layers", type=int, default=6)
    train.add_argument("--lr-adam-start", type=float, default=3e-3)
    train.add_argument("--lr-adam-end", type=float, default=1e-4)
    train.add_argument("--lr-lbfgs", type=float, default=0.5)
    train.add_argument("--grad-clip-norm", type=float, default=1.0)
    train.add_argument("--device", type=str, default="cpu")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--progress-every", type=int, default=25)
    train.add_argument("--quiet", action="store_true",
                        help="suppress per-epoch progress printing")

    out = p.add_argument_group("output")
    out.add_argument("--output-json", type=str, default=None,
                      help="--mode single only: write the full grid result + history to this path")
    out.add_argument("--save-checkpoint", type=str, default=None,
                      help="if set, save a production inference checkpoint to this "
                           "path, e.g. cyclone_model.pth — this is the 'train once, "
                           "deploy the file' step; load it in app.py for "
                           "inference-only serving with no training after deploy. "
                           "REQUIRED for --mode parametric.")
    out.add_argument("--checkpoint-every", type=int, default=None,
                      help="--mode parametric only: save an intermediate checkpoint "
                           "to --save-checkpoint every this-many epochs (plus once "
                           "at the final epoch), so a disconnect/crash loses at most "
                           "this many epochs of progress instead of the whole run. "
                           "E.g. --epochs 20000 --checkpoint-every 2000 lets you "
                           "watch the loss curve and stop early once it plateaus, "
                           "picking up the most recent save.")
    out.add_argument("--resume-from", type=str, default=None,
                      help="--mode parametric only: path to an existing parametric "
                           "checkpoint to continue training from (e.g. one saved by "
                           "--checkpoint-every from a previous, interrupted run) "
                           "instead of a fresh random init. The checkpoint's own "
                           "architecture and trained diameter/flow-rate window are "
                           "used; --hidden/--n-layers/--diameter-*/--flow-* are "
                           "ignored with a printed note if they'd otherwise differ.")

    return p


def _cli_progress_printer(epoch: int, total: int, loss: float) -> None:
    print(f"[{epoch:>5}/{total}] loss={loss:.6e}")


def _run_single_mode(args, parser, on_progress) -> None:
    required = {
        "barrel_diameter_mm": args.barrel_diameter_mm,
        "barrel_height_mm": args.barrel_height_mm,
        "cone_height_mm": args.cone_height_mm,
        "exhaust_dia_mm": args.exhaust_dia_mm,
        "exhaust_length_mm": args.exhaust_length_mm,
        "bottom_outlet_mm": args.bottom_outlet_mm,
        "inlet_height_mm": args.inlet_height_mm,
        "inlet_width_mm": args.inlet_width_mm,
        "flow_rate_cfm": args.flow_rate_cfm,
    }
    missing = [f"--{k.replace('_', '-')}-mm" if k != "flow_rate_cfm" else "--flow-rate-cfm"
               for k, v in required.items() if v is None]
    if missing:
        parser.error(f"--mode single requires: {', '.join(missing)}")

    print(
        f"Training field model (single): barrel_d={args.barrel_diameter_mm}mm, "
        f"flow={args.flow_rate_cfm}CFM, gas={args.gas_type}, "
        f"epochs=({args.epochs_adam} Adam + {args.epochs_lbfgs} L-BFGS)"
    )

    result = run_field_prediction_job(
        barrel_diameter_mm=args.barrel_diameter_mm,
        barrel_height_mm=args.barrel_height_mm,
        cone_height_mm=args.cone_height_mm,
        exhaust_dia_mm=args.exhaust_dia_mm,
        exhaust_length_mm=args.exhaust_length_mm,
        bottom_outlet_mm=args.bottom_outlet_mm,
        inlet_height_mm=args.inlet_height_mm,
        inlet_width_mm=args.inlet_width_mm,
        flow_rate_cfm=args.flow_rate_cfm,
        operating_temp_c=args.operating_temp_c,
        operating_press_kpa=args.operating_press_kpa,
        gas_type=args.gas_type,
        epochs_adam=args.epochs_adam,
        epochs_lbfgs=args.epochs_lbfgs,
        n_interior=args.n_interior,
        hidden=args.hidden,
        n_layers=args.n_layers,
        lr_adam_start=args.lr_adam_start,
        lr_adam_end=args.lr_adam_end,
        lr_lbfgs=args.lr_lbfgs,
        grad_clip_norm=args.grad_clip_norm,
        device=args.device,
        seed=args.seed,
        on_progress=on_progress,
        progress_every=args.progress_every,
        save_checkpoint_path=args.save_checkpoint,
    )

    history = result["history"]
    grid = result["grid"]
    print("\n── Done ──────────────────────────────────────────────")
    print(f"rho={result['rho']:.5f} kg/m3  nu={result['nu']:.3e} m2/s  "
          f"v_inlet={result['v_inlet']:.4f} m/s  v_z_inlet={result['v_z_inlet']:.4f} m/s")
    print(f"k_inlet={result['k_inlet']:.5e} m2/s2  eps_inlet={result['eps_inlet']:.5e} m2/s3")
    print(f"final_loss={history['final_loss']:.6e}  "
          f"wall_time_s={history['wall_time_s']:.1f}")
    print(f"grid points evaluated: {len(grid['r_m'])}")

    if args.output_json:
        import json
        payload = {
            "rho_kgm3": result["rho"],
            "nu_m2s": result["nu"],
            "v_inlet_ms": result["v_inlet"],
            "v_z_inlet_ms": result["v_z_inlet"],
            "q_design_m3s": result["q_design"],
            "k_inlet": result["k_inlet"],
            "eps_inlet": result["eps_inlet"],
            "final_loss": history["final_loss"],
            "wall_time_s": history["wall_time_s"],
            "grid": grid,
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f)
        print(f"Wrote grid result to {args.output_json}")

    if args.save_checkpoint:
        print(f"Saved single-geometry production inference checkpoint to {args.save_checkpoint}")


def _run_parametric_mode(args, parser, on_progress) -> None:
    if not args.save_checkpoint:
        parser.error(
            "--mode parametric requires --save-checkpoint — a parametric "
            "training run only produces value once the checkpoint is saved "
            "for app.py to load (there is no single grid/geometry to print, "
            "unlike --mode single)."
        )
    if args.checkpoint_every is not None and args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be a positive integer.")

    ratios = STAIRMAND_RATIOS if args.ratios_preset == "stairmand" else LAPPLE_RATIOS

    if args.resume_from:
        print(f"Resuming parametric training from '{args.resume_from}' "
              f"for {args.epochs} more epochs...")
    else:
        print(
            f"Training field model (parametric, {args.ratios_preset}): "
            f"diameter=[{args.diameter_min_mm}, {args.diameter_max_mm}]mm, "
            f"flow=[{args.flow_min_cfm}, {args.flow_max_cfm}]CFM, gas={args.gas_type}, "
            f"epochs={args.epochs} (Adam only, domain-randomized)"
        )
    if args.checkpoint_every:
        print(f"Intermediate checkpoints every {args.checkpoint_every} epochs "
              f"-> {args.save_checkpoint}")

    model, scaler, history = train_parametric_field_model(
        rho_fn=fluid_properties,
        ratios=ratios,
        diameter_range_m=(args.diameter_min_mm * 1e-3, args.diameter_max_mm * 1e-3),
        flow_rate_range_cfm=(args.flow_min_cfm, args.flow_max_cfm),
        operating_temp_c=args.operating_temp_c,
        operating_press_kpa=args.operating_press_kpa,
        gas_type=args.gas_type,
        epochs=args.epochs,
        n_interior=args.n_interior,
        hidden=args.hidden,
        n_layers=args.n_layers,
        lr_start=args.lr_adam_start,
        lr_end=args.lr_adam_end,
        grad_clip_norm=args.grad_clip_norm,
        device=args.device,
        seed=args.seed,
        on_progress=on_progress,
        progress_every=args.progress_every,
        resume_from=args.resume_from,
        checkpoint_every=args.checkpoint_every,
        checkpoint_path=args.save_checkpoint,
    )

    print("\n── Done ──────────────────────────────────────────────")
    print(f"final_loss={history['final_loss']:.6e}  wall_time_s={history['wall_time_s']:.1f}")

    # Use the ACTUALLY-used architecture (history["hidden"]/["n_layers"]),
    # not args.hidden/args.n_layers directly — if --resume-from overrode
    # them (see train_parametric_field_model docstring), saving args'
    # values here would write a checkpoint whose recorded architecture
    # doesn't match the model's real state_dict shapes.
    save_parametric_field_checkpoint(
        args.save_checkpoint,
        model=model,
        scaler=scaler,
        ratios=ratios,
        operating_temp_c=args.operating_temp_c,
        operating_press_kpa=args.operating_press_kpa,
        gas_type=args.gas_type,
        hidden=history["hidden"],
        n_layers=history["n_layers"],
    )
    print(f"Saved parametric production inference checkpoint to {args.save_checkpoint}")


def main(argv: Optional[list] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    on_progress = None if args.quiet else _cli_progress_printer

    if args.mode == "single":
        _run_single_mode(args, parser, on_progress)
    else:
        _run_parametric_mode(args, parser, on_progress)


if __name__ == "__main__":
    main()