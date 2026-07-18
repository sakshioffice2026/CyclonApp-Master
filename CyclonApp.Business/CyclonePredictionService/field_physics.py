"""
field_physics.py
─────────────────
Axisymmetric (r, z) domain geometry and steady, incompressible, laminar
Navier–Stokes PDE residuals with swirl, for a field-solving physics-guided
network that predicts v_r, v_theta, v_z, p as continuous functions of (r, z).

Governing equations (classical axisymmetric incompressible Navier–Stokes
with swirl — see e.g. White, "Viscous Fluid Flow", or Schlichting,
"Boundary-Layer Theory"; standard textbook form, not project-specific):

Continuity:
    (1/r) d(r*v_r)/dr + d(v_z)/dz = 0

r-momentum:
    v_r*dv_r/dr + v_z*dv_r/dz - v_theta^2/r
        = -(1/rho)*dp/dr + nu*( d2v_r/dr2 + (1/r)*dv_r/dr - v_r/r^2 + d2v_r/dz2 )

theta-momentum:
    v_r*dv_theta/dr + v_z*dv_theta/dz + v_r*v_theta/r
        = nu*( d2v_theta/dr2 + (1/r)*dv_theta/dr - v_theta/r^2 + d2v_theta/dz2 )

z-momentum:
    v_r*dv_z/dr + v_z*dv_z/dz
        = -(1/rho)*dp/dz + nu*( d2v_z/dr2 + (1/r)*dv_z/dr + d2v_z/dz2 )

Known, documented limitation: this is LAMINAR. Cyclone flow is turbulent;
a RANS closure is deferred to a later phase (see CyclonApp area notes).

FLUID-PROPERTY / UNIT-CONVERSION CONSTANTS (absorbed from the now-deleted
physics.py): sutherland_viscosity, ideal_gas_density, gas_type_to_onehot,
IN_TO_M, and CFM_TO_M3S used to live in physics.py, shared with the old
scalar CyclonePINN correction-model path (model.py/dataset.py/train.py).
That path is retired — only lapple_forward() was specific to it, so it was
NOT carried over. Everything below is otherwise byte-for-byte identical to
the physics.py originals, so values still agree exactly with the rest of
the service (and with CyclonCalculationRepository.cs's own Sutherland/
ideal-gas math, which this remains a faithful torch port of).
"""
from __future__ import annotations
import math
import torch

EPS = 1e-9

GAS_TYPES = ["AIR", "N2", "CO2"]  # FLUEGAS maps to CO2 constants, same as C#

# Sutherland constants: mu0 (Pa.s), T0 (K), C (K)
_SUTHERLAND = {
    "AIR": (1.716e-5, 273.15, 110.4),
    "N2":  (1.663e-5, 273.15, 107.0),
    "CO2": (1.370e-5, 273.15, 222.0),
}
# Molar mass (kg/mol)
_MOLAR_MASS = {
    "AIR": 0.02897,
    "N2":  0.02801,
    "CO2": 0.04401,
}

R_GAS = 8.314  # J/(mol.K)

IN_TO_M = 0.0254
CFM_TO_M3S = 0.000471947


def _gas_lookup(table: dict, gas_onehot: torch.Tensor) -> torch.Tensor:
    """gas_onehot: (N,3) one-hot over GAS_TYPES -> weighted constant per row."""
    values = torch.tensor([table[g] for g in GAS_TYPES], dtype=gas_onehot.dtype,
                           device=gas_onehot.device)
    return gas_onehot @ values


def sutherland_viscosity(temp_c: torch.Tensor, gas_onehot: torch.Tensor) -> torch.Tensor:
    T = temp_c + 273.15
    mu0 = _gas_lookup({k: v[0] for k, v in _SUTHERLAND.items()}, gas_onehot)
    T0 = _gas_lookup({k: v[1] for k, v in _SUTHERLAND.items()}, gas_onehot)
    C = _gas_lookup({k: v[2] for k, v in _SUTHERLAND.items()}, gas_onehot)
    return mu0 * (T / T0) ** 1.5 * ((T0 + C) / (T + C))


def ideal_gas_density(temp_c: torch.Tensor, press_kpa: torch.Tensor,
                       gas_onehot: torch.Tensor) -> torch.Tensor:
    T = temp_c + 273.15
    P = press_kpa * 1000.0
    M = _gas_lookup(_MOLAR_MASS, gas_onehot)
    return (P * M) / (R_GAS * T)


def gas_type_to_onehot(gas_type: str) -> list[float]:
    g = (gas_type or "AIR").upper()
    if g in ("FLUEGAS",):
        g = "CO2"
    if g in ("NITROGEN",):
        g = "N2"
    if g not in GAS_TYPES:
        g = "AIR"
    return [1.0 if g == gt else 0.0 for gt in GAS_TYPES]


# ─────────────────────────────────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────────────────────────────────

class CycloneAxisymGeometry:
    """
    Builds the fluid domain boundary from the same CyclonDimensions fields
    already computed by CyclonCalculationRepository.cs / exposed in
    DesignRevision.DimensionsJson. z is measured from the top of the barrel
    (z=0) downward to the cone apex / bottom outlet (z=total_fluid_height).
    All dimensions in meters.
    """

    def __init__(
        self,
        barrel_diameter_m: float,
        barrel_height_m: float,
        cone_height_m: float,
        exhaust_dia_m: float,
        exhaust_length_m: float,
        bottom_outlet_m: float,
    ):
        self.r_barrel = barrel_diameter_m / 2.0
        self.z_barrel_end = barrel_height_m
        self.z_cone_end = barrel_height_m + cone_height_m
        self.r_exhaust = exhaust_dia_m / 2.0
        self.z_exhaust_end = exhaust_length_m          # how far the vortex
                                                         # finder pipe extends
                                                         # down into the barrel
        self.r_bottom_outlet = bottom_outlet_m / 2.0
        self.total_height = self.z_cone_end

        if self.r_exhaust >= self.r_barrel:
            raise ValueError("exhaust_dia_m must be smaller than barrel_diameter_m")
        if self.r_bottom_outlet >= self.r_barrel:
            raise ValueError("bottom_outlet_m must be smaller than barrel_diameter_m")

    def outer_wall_radius(self, z: torch.Tensor) -> torch.Tensor:
        """Outer solid wall radius at height z (barrel = constant, cone = linear taper)."""
        in_barrel = z <= self.z_barrel_end
        cone_frac = (z - self.z_barrel_end) / (self.z_cone_end - self.z_barrel_end + EPS)
        cone_frac = torch.clamp(cone_frac, 0.0, 1.0)
        r_cone = self.r_barrel + (self.r_bottom_outlet - self.r_barrel) * cone_frac
        return torch.where(in_barrel, torch.full_like(z, self.r_barrel), r_cone)

    def is_inside_exhaust_pipe(self, r: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """True where the point sits inside the solid vortex-finder pipe wall
        (excluded from the fluid domain — it's metal, not gas)."""
        return (z <= self.z_exhaust_end) & (r <= self.r_exhaust) & (r > 0)
        # NOTE: r>0 keeps the exhaust pipe's open bore (its internal gas
        # column) available as a valid fluid point for r in (0, r_exhaust)
        # is actually fluid too — the pipe WALL is a thin shell we do not
        # resolve explicitly at this phase; the interior of the exhaust
        # pipe is treated as fluid domain with an outlet BC at z=0.
        # Left disabled (always False in practice below) — see sample_interior.

    def is_fluid(self, r: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        wall_r = self.outer_wall_radius(z)
        within_bounds = (r >= 0) & (r <= wall_r) & (z >= 0) & (z <= self.total_height)
        return within_bounds

    # ── collocation sampling ──────────────────────────────────────────

    def sample_interior(self, n: int, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Rejection-sample n interior (r, z) points uniformly over the fluid domain."""
        pts_r, pts_z = [], []
        batch = max(n * 3, 256)
        while sum(p.shape[0] for p in pts_r) < n:
            z = torch.rand(batch, device=device) * self.total_height
            wall_r = self.outer_wall_radius(z)
            r = torch.rand(batch, device=device) * wall_r
            keep = self.is_fluid(r, z)
            pts_r.append(r[keep])
            pts_z.append(z[keep])
        r_cat = torch.cat(pts_r)[:n].requires_grad_(True)
        z_cat = torch.cat(pts_z)[:n].requires_grad_(True)
        return r_cat, z_cat

    def sample_outer_wall(self, n: int, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Points on the barrel + cone outer wall (no-slip)."""
        z = torch.rand(n, device=device) * self.total_height
        r = self.outer_wall_radius(z)
        return r.requires_grad_(True), z.requires_grad_(True)

    def sample_axis(self, n: int, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Points on the centerline r=0 (symmetry)."""
        z = torch.rand(n, device=device) * self.total_height
        r = torch.zeros(n, device=device)
        return r.requires_grad_(True), z.requires_grad_(True)

    def sample_inlet_ring(self, n: int, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Practical axisymmetric approximation of the real off-axis rectangular
        inlet: the inlet's tangential momentum is applied as an annular ring
        at the top of the barrel (z=0), spanning r in [r_exhaust, r_barrel].
        This is the standard simplification used when collapsing a
        tangential-entry cyclone to an axisymmetric model (the true 3D inlet
        jet is azimuthally smeared into a ring of equivalent swirl momentum).
        """
        r = self.r_exhaust + torch.rand(n, device=device) * (self.r_barrel - self.r_exhaust)
        z = torch.zeros(n, device=device)
        return r.requires_grad_(True), z.requires_grad_(True)

    def sample_bottom_outlet(self, n: int, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        r = torch.rand(n, device=device) * self.r_bottom_outlet
        z = torch.full((n,), self.total_height, device=device)
        return r.requires_grad_(True), z.requires_grad_(True)

    def sample_top_exhaust_outlet(self, n: int, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        r = torch.rand(n, device=device) * self.r_exhaust
        z = torch.zeros(n, device=device)
        return r.requires_grad_(True), z.requires_grad_(True)


def geometry_from_dimensions_mm(
    barrel_diameter_mm: float,
    barrel_height_mm: float,
    cone_height_mm: float,
    exhaust_dia_mm: float,
    exhaust_length_mm: float,
    bottom_outlet_mm: float,
) -> CycloneAxisymGeometry:
    """Direct mapping from CyclonDimensions (mm fields) to the SI geometry used here."""
    mm = 1e-3
    return CycloneAxisymGeometry(
        barrel_diameter_m=barrel_diameter_mm * mm,
        barrel_height_m=barrel_height_mm * mm,
        cone_height_m=cone_height_mm * mm,
        exhaust_dia_m=exhaust_dia_mm * mm,
        exhaust_length_m=exhaust_length_mm * mm,
        bottom_outlet_m=bottom_outlet_mm * mm,
    )


# ─────────────────────────────────────────────────────────────────────────
# PDE RESIDUALS
# ─────────────────────────────────────────────────────────────────────────

def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                create_graph=True, retain_graph=True)[0]


def navier_stokes_residuals(
    model_fn,
    r: torch.Tensor,
    z: torch.Tensor,
    rho: torch.Tensor,
    nu: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    model_fn: callable (r, z) -> dict with keys 'v_r', 'v_theta', 'v_z', 'p',
              each shape (N,). r, z must require_grad.
    rho, nu:  fluid density (kg/m3) and kinematic viscosity (m2/s), shape (N,)
              or broadcastable scalars — reuse physics.py's property functions
              upstream to compute these.
    Returns the four PDE residuals (continuity, r/theta/z-momentum), each
    ideally ~0 for a physically valid flow field.
    """
    out = model_fn(r, z)
    v_r, v_theta, v_z, p = out["v_r"], out["v_theta"], out["v_z"], out["p"]

    r_safe = r + EPS  # avoid divide-by-zero on the axis

    dvr_dr = _grad(v_r, r)
    dvr_dz = _grad(v_r, z)
    dvr_dr2 = _grad(dvr_dr, r)
    dvr_dz2 = _grad(dvr_dz, z)

    dvt_dr = _grad(v_theta, r)
    dvt_dz = _grad(v_theta, z)
    dvt_dr2 = _grad(dvt_dr, r)
    dvt_dz2 = _grad(dvt_dz, z)

    dvz_dr = _grad(v_z, r)
    dvz_dz = _grad(v_z, z)
    dvz_dr2 = _grad(dvz_dr, r)
    dvz_dz2 = _grad(dvz_dz, z)

    dp_dr = _grad(p, r)
    dp_dz = _grad(p, z)

    continuity = (1.0 / r_safe) * (v_r + r_safe * dvr_dr) + dvz_dz

    r_momentum = (
        v_r * dvr_dr + v_z * dvr_dz - (v_theta ** 2) / r_safe
        + (1.0 / rho) * dp_dr
        - nu * (dvr_dr2 + (1.0 / r_safe) * dvr_dr - v_r / (r_safe ** 2) + dvr_dz2)
    )

    theta_momentum = (
        v_r * dvt_dr + v_z * dvt_dz + (v_r * v_theta) / r_safe
        - nu * (dvt_dr2 + (1.0 / r_safe) * dvt_dr - v_theta / (r_safe ** 2) + dvt_dz2)
    )

    z_momentum = (
        v_r * dvz_dr + v_z * dvz_dz
        + (1.0 / rho) * dp_dz
        - nu * (dvz_dr2 + (1.0 / r_safe) * dvz_dr + dvz_dz2)
    )

    return {
        "continuity": continuity,
        "r_momentum": r_momentum,
        "theta_momentum": theta_momentum,
        "z_momentum": z_momentum,
        "v_r": v_r, "v_theta": v_theta, "v_z": v_z, "p": p,
    }


def fluid_properties(temp_c: torch.Tensor, press_kpa: torch.Tensor,
                      gas_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """rho (kg/m3), nu (m2/s) — reuses physics.py so values agree with the
    rest of the service exactly."""
    mu = sutherland_viscosity(temp_c, gas_onehot)
    rho = ideal_gas_density(temp_c, press_kpa, gas_onehot)
    nu = mu / (rho + EPS)
    return rho, nu


def inlet_velocity_ms(flow_cfm: torch.Tensor, inlet_height_m: torch.Tensor,
                       inlet_width_m: torch.Tensor) -> torch.Tensor:
    """Same convention as physics.lapple_forward: Q / inlet area."""
    q_m3s = flow_cfm * CFM_TO_M3S
    area = inlet_height_m * inlet_width_m + EPS
    return q_m3s / area


def inlet_axial_velocity_ms(flow_cfm: torch.Tensor, r_barrel_m: float,
                             r_exhaust_m: float) -> torch.Tensor:
    """
    ROOT-CAUSE FIX (see field_boundary_conditions.inlet_ring_residual):
    the inlet ring BC previously prescribed v_z = 0, meaning no mass ever
    entered the domain through the one boundary meant to represent the
    inlet — nothing in the physics/BC setup tied the trained field to the
    actual design flow rate, so the network was free to converge to an
    internally-inconsistent field (confirmed: integrated Q(z) varied 69%
    across the barrel/cone, when it must be constant for a divergence-free
    field with impermeable walls). This computes the axial inflow velocity
    needed so that integrating v_z over the inlet ring's annular
    cross-section (pi * (r_barrel^2 - r_exhaust^2)) reproduces the actual
    design volumetric flow rate:

        v_z_inlet * pi * (r_barrel^2 - r_exhaust^2) = Q_total

    This is a real physical mass-injection velocity, not the same number
    as inlet_velocity_ms (which is Q / real-duct-area — the real 3D duct's
    physical cross-section is much smaller than this axisymmetric ring's
    full-circumference area, so the equivalent smeared axial velocity is
    correspondingly smaller; this is the expected/correct effect of
    azimuthally smearing a localized 3D inlet jet into an axisymmetric
    ring, not an error).
    """
    q_m3s = flow_cfm * CFM_TO_M3S
    ring_area = math.pi * (r_barrel_m ** 2 - r_exhaust_m ** 2) + EPS
    return q_m3s / ring_area