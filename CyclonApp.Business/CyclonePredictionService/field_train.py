"""
field_train.py
────────────────
Reusable training routine for CycloneFieldPINN. This is the SINGLE source
of truth for training the field-solving model — both the CLI entry point
(`python field_train.py ...`, if/when one is added) and app.py's async
/predict_field job call the same `train_field_model` function. Do not
duplicate this loop anywhere else.

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
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import torch

from field_model import CycloneFieldPINN, FieldScaler
from field_physics import CycloneAxisymGeometry, navier_stokes_residuals
from field_boundary_conditions import assemble_bc_losses

# ─────────────────────────────────────────────────────────────────────────
# Loss term weights. PDE residuals and BC residuals are not naturally on
# the same scale (BC residuals are direct velocity/pressure errors; PDE
# residuals involve second derivatives and can dominate numerically if
# left unweighted). These weights are a starting point validated to
# produce stable, convergent training — not claimed optimal for every
# geometry; revisit if a particular design shape trains poorly.
# ─────────────────────────────────────────────────────────────────────────
PDE_LOSS_WEIGHT = 1.0
BC_LOSS_WEIGHT = 10.0

OnProgressFn = Callable[[int, int, float], None]

# navier_stokes_residuals divides by r^2 (with only a 1e-9 numerical floor)
# in the r- and theta-momentum residuals. geometry.sample_interior draws r
# uniformly across the full [0, wall_r] range, so with realistic batch
# sizes some points land within microns of the axis, producing 1/r^2 terms
# in the billions that swamp the loss and destabilize both the Adam and
# L-BFGS phases (confirmed empirically: min r in a 2000-point batch was
# ~6e-6 m, giving a 1/r^2 term of ~3e10). The axis itself is already
# covered by the dedicated axis boundary condition
# (geometry.sample_axis + field_boundary_conditions.axis_symmetry_residual,
# enforcing v_r=0, v_theta=0, d(v_z)/dr=0), so excluding a thin band around
# r=0 from interior PDE sampling loses no physics coverage — it just keeps
# collocation points out of a region already handled by a different, more
# numerically appropriate constraint.
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


def _pde_loss(model_fn, geometry: CycloneAxisymGeometry, rho: torch.Tensor,
              nu: torch.Tensor, n_interior: int, device: str) -> torch.Tensor:
    r, z = _sample_interior_away_from_axis(geometry, n_interior, device)
    res = navier_stokes_residuals(model_fn, r, z, rho, nu)
    return (
        (res["continuity"] ** 2).mean()
        + (res["r_momentum"] ** 2).mean()
        + (res["theta_momentum"] ** 2).mean()
        + (res["z_momentum"] ** 2).mean()
    )


def _bc_loss(model_fn, geometry: CycloneAxisymGeometry, v_inlet: torch.Tensor,
             device: str) -> torch.Tensor:
    bc_losses = assemble_bc_losses(model_fn, geometry, v_inlet, device=device)
    return sum(bc_losses.values())


def _total_loss(model, scaler, geometry, rho, nu, v_inlet,
                 n_interior: int, device: str) -> torch.Tensor:
    model_fn = model.as_model_fn(scaler)
    pde = _pde_loss(model_fn, geometry, rho, nu, n_interior, device)
    bc = _bc_loss(model_fn, geometry, v_inlet, device)
    return PDE_LOSS_WEIGHT * pde + BC_LOSS_WEIGHT * bc


def train_field_model(
    geometry: CycloneAxisymGeometry,
    rho: float,
    nu: float,
    v_inlet: float,
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
    Trains a fresh CycloneFieldPINN for one fixed geometry/operating point.

    Args:
        geometry: fluid domain (see field_physics.geometry_from_dimensions_mm)
        rho, nu: fluid density (kg/m3) and kinematic viscosity (m2/s)
        v_inlet: inlet tangential velocity (m/s), see field_physics.inlet_velocity_ms
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
        loss = _total_loss(model, scaler, geometry, rho_t, nu_t, v_inlet_t, n_interior, device)
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
        pde = navier_stokes_residuals(model_fn, r_fixed, z_fixed, rho_t, nu_t)
        pde_loss = (
            (pde["continuity"] ** 2).mean()
            + (pde["r_momentum"] ** 2).mean()
            + (pde["theta_momentum"] ** 2).mean()
            + (pde["z_momentum"] ** 2).mean()
        )
        bc = _bc_loss(model_fn, geometry, v_inlet_t, device)
        loss = PDE_LOSS_WEIGHT * pde_loss + BC_LOSS_WEIGHT * bc
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