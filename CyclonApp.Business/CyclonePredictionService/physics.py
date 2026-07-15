"""
physics.py
──────────
A PyTorch re-implementation of the exact closed-form relationships already
used in CyclonApp's CyclonCalculationRepository.cs (Sutherland viscosity,
ideal-gas density, Lapple cut-diameter/efficiency, Shepherd-Lapple pressure
drop). Written in torch (not numpy) so it is differentiable end-to-end —
that differentiability is what lets the PINN's training loop penalize the
network for violating known physical relationships (see train.py), which is
the actual "physics-informed" part of this service.

Keeping this as a faithful port of the C# is deliberate: the network's
physics baseline and the .NET app's own trusted-range cross-check
(CyclonePredictionRepository.PredictAsync) must agree on what "physics says"
or the trusted-range flag on the .NET side becomes meaningless.
"""
from __future__ import annotations
import torch

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


def lapple_forward(
    flow_cfm: torch.Tensor,
    inlet_line_size_in: torch.Tensor,
    temp_c: torch.Tensor,
    press_kpa: torch.Tensor,
    gas_onehot: torch.Tensor,
    particle_size_micron: torch.Tensor,
    particle_density_kgm3: torch.Tensor,
    effective_turns: torch.Tensor,
    inlet_height_ratio: torch.Tensor,
    inlet_width_ratio: torch.Tensor,
    outlet_diam_ratio: torch.Tensor,
    eps: float = 1e-9,
) -> dict[str, torch.Tensor]:
    """
    Faithful port of CyclonCalculationRepository.Calculate() — just the
    subset of outputs the prediction service cares about (efficiency,
    pressure drop), plus the intermediate quantities used as engineered
    features for the network.
    """
    q_m3s = flow_cfm * CFM_TO_M3S

    d_inlet_m = inlet_line_size_in * IN_TO_M
    a_pipe_m2 = torch.pi * (d_inlet_m / 2.0) ** 2
    v_inlet_ms = q_m3s / (a_pipe_m2 + eps)

    h, w = inlet_height_ratio, inlet_width_ratio
    dc_m = torch.sqrt(a_pipe_m2 / (h * w + eps))
    dc_in = dc_m / IN_TO_M

    inlet_h_in = dc_in * h
    inlet_w_in = dc_in * w
    exhaust_dia_in = dc_in * outlet_diam_ratio

    mu = sutherland_viscosity(temp_c, gas_onehot)
    rho_g = ideal_gas_density(temp_c, press_kpa, gas_onehot)

    rho_p = particle_density_kgm3
    nt = effective_turns
    w_m = inlet_w_in * IN_TO_M

    dpc_m = torch.sqrt(
        (9.0 * mu * w_m) / (torch.pi * nt * v_inlet_ms * rho_p + eps) + eps
    )
    dpc_micron = dpc_m * 1e6

    dp_micron = particle_size_micron
    efficiency = 1.0 / (1.0 + (dpc_micron / (dp_micron + eps)) ** 2)

    hi_m = inlet_h_in * IN_TO_M
    wi_m = inlet_w_in * IN_TO_M
    de_m = exhaust_dia_in * IN_TO_M
    nh = (16.0 * hi_m * wi_m) / (torch.pi * de_m ** 2 + eps)
    dp_pa = nh * 0.5 * rho_g * v_inlet_ms ** 2

    return {
        "v_inlet_ms": v_inlet_ms,
        "mu": mu,
        "rho_g": rho_g,
        "dpc_micron": dpc_micron,
        "efficiency": efficiency,        # fraction 0..1
        "pressure_drop_pa": dp_pa,
    }


def gas_type_to_onehot(gas_type: str) -> list[float]:
    g = (gas_type or "AIR").upper()
    if g in ("FLUEGAS",):
        g = "CO2"
    if g in ("NITROGEN",):
        g = "N2"
    if g not in GAS_TYPES:
        g = "AIR"
    return [1.0 if g == gt else 0.0 for gt in GAS_TYPES]
