"""
field_train.py
────────────────
Reusable training routine for CycloneFieldPINN. This is the SINGLE source
of truth for training the field-solving model — both the CLI entry point
(`python field_train.py ...`, see bottom of this file) and app.py's async
/predict_field job call the same `run_field_prediction_job` /
`train_field_model` functions. Do not duplicate this loop, or the
geometry/fluid-property glue in `run_field_prediction_job`, anywhere else.

Why train-on-demand: CycloneFieldPINN takes only (r, z) as input, with
geometry and operating condition baked in as fixed constants at training
time (see field_model.CycloneFieldPINN / FieldScaler). Unlike CyclonePINN,
there is no single pre-trained checkpoint that answers field queries for
an arbitrary design — each request trains a small, purpose-built network
for that one geometry/operating point. This is the chosen tradeoff versus
a generalized parametric network (which would take geometry/operating
params as additional inputs, train once, infer instantly, but is
significantly harder to get to converge well and was not chosen here).

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
# sanity_check.check_mass_conservation. This term computes Q(z) at several
# cross-sections directly and penalizes it for deviating from its own mean,
# giving continuity a much stronger, harder-to-ignore gradient signal.
# Not yet empirically tuned relative to BC_LOSS_WEIGHT — start here, and
# increase if Q(z) rel_spread is still high after training with this term.
MASS_FLOW_LOSS_WEIGHT = 5.0

OnProgressFn = Callable[[int, int, float], None]

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
    pde = (
        (res["continuity"] / scales["continuity"]) ** 2
    ).mean() + (
        (res["r_momentum"] / scales["r_momentum"]) ** 2
    ).mean() + (
        (res["theta_momentum"] / scales["theta_momentum"]) ** 2
    ).mean() + (
        (res["z_momentum"] / scales["z_momentum"]) ** 2
    ).mean()
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


def _mass_flow_loss(
    model,
    scaler,
    geometry: CycloneAxisymGeometry,
    device: str,
    n_planes: int = 8,
    n_r: int = 64,
) -> torch.Tensor:
    """
    Direct aggregate mass-conservation regularizer: computes the
    volumetric flow Q(z) = 2*pi*integral(v_z * r dr) at several z
    cross-sections spanning the barrel and cone (avoiding the inlet/outlet
    end regions, where Q(z) is legitimately expected to change), and
    penalizes each Q(z) for deviating from the batch's own mean. For a
    divergence-free field with impermeable side walls, Q(z) must be
    constant between the inlet and outlet boundaries — this term targets
    that aggregate property directly, as a stronger complement to the
    pointwise continuity PDE residual in _pde_and_turb_loss (see
    MASS_FLOW_LOSS_WEIGHT's docstring above for why the pointwise
    residual alone was insufficient — same quantity sanity_check.
    check_mass_conservation independently measures on the trained grid).

    Each z-plane integrates only out to that plane's TRUE local wall
    radius (geometry.outer_wall_radius(z), which tapers linearly through
    the cone), not a fixed r_barrel — integrating past the true wall would
    evaluate the network outside the fluid domain (inside the solid cone
    wall) and corrupt the Q(z) estimate with meaningless extrapolated
    velocity, actively fighting the very constraint this term is meant to
    enforce.
    """
    model_fn = model.as_model_fn(scaler)

    z_planes = torch.linspace(
        0.15 * geometry.total_height,
        0.85 * geometry.total_height,
        n_planes,
        device=device,
    )

    q_values = []
    for z in z_planes:
        r_max = geometry.outer_wall_radius(z.unsqueeze(0)).squeeze(0)
        r = torch.linspace(0.0, float(r_max), n_r, device=device, requires_grad=True)
        zz = torch.full_like(r, float(z))

        pred = model_fn(r, zz)
        vz = pred["v_z"]
        q = 2.0 * torch.pi * torch.trapz(vz * r, r)
        q_values.append(q)

    q_values = torch.stack(q_values)
    target_q = q_values.mean().detach()
    return ((q_values - target_q) / (target_q + 1e-9)).pow(2).mean()


def _total_loss(model, scaler, geometry, rho, nu, v_inlet, v_z_inlet, k_inlet, eps_inlet,
                 n_interior: int, device: str) -> torch.Tensor:
    model_fn = model.as_model_fn(scaler)
    pde_turb = _pde_and_turb_loss(model_fn, geometry, rho, nu, scaler, n_interior, device)
    bc = _bc_loss(model_fn, geometry, v_inlet, v_z_inlet, k_inlet, eps_inlet, scaler, device)
    mass = _mass_flow_loss(model, scaler, geometry, device)
    return pde_turb + BC_LOSS_WEIGHT * bc + MASS_FLOW_LOSS_WEIGHT * mass


def train_field_model(
    geometry: CycloneAxisymGeometry,
    rho: float,
    nu: float,
    v_inlet: float,
    v_z_inlet: float,
    k_inlet: float,
    eps_inlet: float,
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
                            k_inlet_t, eps_inlet_t, n_interior, device)
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
        model_fn = model.as_model_fn(scaler)
        res = rans_field_residuals(model_fn, r_fixed, z_fixed, rho_t, nu_t)
        scales = _pde_residual_scales(scaler)
        pde_turb_loss = (
            PDE_LOSS_WEIGHT * (
                ((res["continuity"] / scales["continuity"]) ** 2).mean()
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
        mass = _mass_flow_loss(model, scaler, geometry, device)
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
        epochs_adam=epochs_adam,
        epochs_lbfgs=epochs_lbfgs,
        on_progress=on_progress,
        **train_kwargs,
    )

    grid = evaluate_grid(model, scaler, geometry)

    return {
        "geometry": geometry,
        "rho": rho,
        "nu": nu,
        "v_inlet": v_inlet,
        "v_z_inlet": v_z_inlet,
        "k_inlet": k_inlet,
        "eps_inlet": eps_inlet,
        "model": model,
        "scaler": scaler,
        "history": history,
        "grid": grid,
    }


# ─────────────────────────────────────────────────────────────────────────
# CLI — offline testing/tuning without going through the HTTP service.
# Calls the exact same run_field_prediction_job as app.py, so results seen
# here match what /predict_field/start would produce for the same inputs.
# ─────────────────────────────────────────────────────────────────────────

def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Train and evaluate a CycloneFieldPINN for one geometry/"
            "operating point, offline — no HTTP service required. Prints a "
            "summary and optionally writes the full grid result to JSON."
        ),
    )
    geo = p.add_argument_group("geometry (mm)")
    geo.add_argument("--barrel-diameter-mm", type=float, required=True)
    geo.add_argument("--barrel-height-mm", type=float, required=True)
    geo.add_argument("--cone-height-mm", type=float, required=True)
    geo.add_argument("--exhaust-dia-mm", type=float, required=True)
    geo.add_argument("--exhaust-length-mm", type=float, required=True)
    geo.add_argument("--bottom-outlet-mm", type=float, required=True)
    geo.add_argument("--inlet-height-mm", type=float, required=True)
    geo.add_argument("--inlet-width-mm", type=float, required=True)

    proc = p.add_argument_group("process conditions")
    proc.add_argument("--flow-rate-cfm", type=float, required=True)
    proc.add_argument("--operating-temp-c", type=float, default=25.0)
    proc.add_argument("--operating-press-kpa", type=float, default=101.325)
    proc.add_argument("--gas-type", type=str, default="Air")

    train = p.add_argument_group("training")
    train.add_argument("--epochs-adam", type=int, default=3000)
    train.add_argument("--epochs-lbfgs", type=int, default=300)
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
                      help="if set, write the full grid result + history to this path")

    return p


def _cli_progress_printer(epoch: int, total: int, loss: float) -> None:
    print(f"[{epoch:>5}/{total}] loss={loss:.6e}")


def main(argv: Optional[list] = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    on_progress = None if args.quiet else _cli_progress_printer

    print(
        f"Training field model: barrel_d={args.barrel_diameter_mm}mm, "
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
            "k_inlet": result["k_inlet"],
            "eps_inlet": result["eps_inlet"],
            "final_loss": history["final_loss"],
            "wall_time_s": history["wall_time_s"],
            "grid": grid,
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f)
        print(f"Wrote grid result to {args.output_json}")


if __name__ == "__main__":
    main()