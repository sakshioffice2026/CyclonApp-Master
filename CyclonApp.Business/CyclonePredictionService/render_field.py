"""
render_field.py
????????????????
Renders a real CFD-style contour image from a completed field-solve job's
grid � smooth, interpolated velocity-magnitude and pressure contours, not
a schematic mockup.

Unlike AxialFanMVC's render_result.py (PyVista/VTK, needs a real OpenGL
context via WGL � hence that project's whole IPC + Scheduled Task +
"run only when logged on" workaround), this uses matplotlib's Agg
backend, which rasterizes entirely in software. No GPU, no display, no
interactive desktop session required � safe to call in-process, straight
from app.py's worker thread, right after evaluate_grid() produces the
field. That's why this module has no IPC/dispatch counterpart: it needs
none.

INPUT SHAPE: evaluate_grid() (field_model.py) returns flat parallel lists
� one (r_m, z_m, v_r_ms, v_theta_ms, v_z_ms, pressure_pa) tuple per valid
fluid point on an r>=0 half-domain (axisymmetric solve). This module:
  1. Mirrors r -> -r to reconstruct the full two-sided cross-section
     (standard presentation for an axisymmetric result).
  2. Interpolates the scattered half-mirrored point cloud onto a fine
     regular (r, z) grid with scipy.griddata � this is what turns a
     point cloud into the smooth blended contour a real CFD tool shows,
     as opposed to flat schematic bands.
  3. Masks grid cells outside the true cyclone silhouette (barrel +
     linearly tapered cone, from CycloneAxisymGeometry's own
     outer_wall_radius formula � kept in sync with field_physics.py's
     geometry, not re-derived independently) to NaN, so contourf never
     paints color over solid metal.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless � must be set before importing pyplot
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def _outer_wall_radius(z: np.ndarray, r_barrel: float, r_bottom_outlet: float,
                        z_barrel_end: float, z_cone_end: float) -> np.ndarray:
    """Mirrors CycloneAxisymGeometry.outer_wall_radius (field_physics.py) in
    numpy so this module doesn't need to import torch just to know the
    cyclone's outline. Barrel = constant radius; cone = linear taper."""
    cone_frac = np.clip(
        (z - z_barrel_end) / max(z_cone_end - z_barrel_end, 1e-9), 0.0, 1.0
    )
    r_cone = r_barrel + (r_bottom_outlet - r_barrel) * cone_frac
    return np.where(z <= z_barrel_end, r_barrel, r_cone)


def _cyclone_outline_xy(r_barrel: float, r_bottom_outlet: float,
                         z_barrel_end: float, z_cone_end: float,
                         r_exhaust: float, z_exhaust_end: float):
    """Closed outline of the cyclone's solid boundary (both sides of the
    mirrored cross-section), for drawing context on top of the contour �
    matches render_result.py's practice of overlaying real geometry
    rather than leaving the field to float unlabeled."""
    left = [
        (-r_barrel, 0.0), (-r_barrel, z_barrel_end),
        (-r_bottom_outlet, z_cone_end),
    ]
    right = [
        (r_bottom_outlet, z_cone_end), (r_barrel, z_barrel_end),
        (r_barrel, 0.0),
    ]
    outline_x = [p[0] for p in left] + [p[0] for p in right] + [left[0][0]]
    outline_y = [p[1] for p in left] + [p[1] for p in right] + [left[0][1]]

    exhaust_x = [-r_exhaust, -r_exhaust, r_exhaust, r_exhaust]
    exhaust_y = [0.0, z_exhaust_end, z_exhaust_end, 0.0]

    return (outline_x, outline_y), (exhaust_x, exhaust_y)


def _draw_flow_arrows(ax, r_barrel: float, r_exhaust: float,
                       z_barrel_end: float, z_exhaust_end: float) -> None:
    """Schematic INLET/OUTLET flow-direction glyphs — the 2D-axisymmetric
    equivalent of render_result.py's 3D `flow_arrow`/`add_point_labels`.
    Matches its visual language (black arrow, white-boxed black-text
    label) rather than its exact geometry, since a real cyclone's inlet
    is tangential (out of this r-z slice's plane) while the axial solve
    only has r/z components to draw. Placed just outside the drawn
    silhouette so it never overlaps geometry/streamlines, same rule
    render_result.py used for its own arrow placement.
    """
    label_kwargs = dict(
        fontsize=9, color="black", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                   edgecolor="none", alpha=0.85),
    )
    arrow_kwargs = dict(arrowstyle="-|>", mutation_scale=16,
                         color="black", linewidth=1.4)

    # INLET — gas enters tangentially near the barrel top; drawn just
    # outside the barrel wall pointing down into the body.
    inlet_x = r_barrel * 1.18
    ax.annotate("", xy=(inlet_x, 0.18 * z_barrel_end), xytext=(inlet_x, 0.0),
                arrowprops=arrow_kwargs)
    ax.text(inlet_x, -0.06 * z_barrel_end, "INLET (gas in)", **label_kwargs)

    # OUTLET — clean gas exits upward through the vortex finder bore
    # (centerline, z=0), so the arrow points from inside the exhaust
    # duct out toward z=0.
    ax.annotate("", xy=(0.0, -0.10 * z_barrel_end), xytext=(0.0, z_exhaust_end),
                arrowprops=arrow_kwargs)
    ax.text(0.0, z_exhaust_end + 0.12 * z_barrel_end, "OUTLET (clean gas)",
            **label_kwargs)


def render_cyclone_field(
    grid: dict,
    geometry_mm: dict,
    output_dir: str,
    known_efficiency_percent: Optional[float] = None,
    known_pressure_drop_pa: Optional[float] = None,
    filename: str = "cfd_result.png",
    n_interp: int = 220,
) -> str:
    """
    grid: the FieldResultDto-equivalent dict for one completed job �
        r_m, z_m, v_r_ms, v_theta_ms, v_z_ms, pressure_pa (flat parallel
        lists, r>=0 half-domain).
    geometry_mm: this design's own dimensions � barrel_diameter_mm,
        barrel_height_mm, cone_height_mm, exhaust_dia_mm,
        exhaust_length_mm, bottom_outlet_mm � the SAME values passed to
        geometry_from_dimensions_mm in app.py, so the drawn silhouette
        matches the domain the field was actually solved on.

    Returns the PNG path.
    """
    os.makedirs(output_dir, exist_ok=True)

    r_barrel = geometry_mm["barrel_diameter_mm"] / 2000.0
    z_barrel_end = geometry_mm["barrel_height_mm"] / 1000.0
    z_cone_end = z_barrel_end + geometry_mm["cone_height_mm"] / 1000.0
    r_exhaust = geometry_mm["exhaust_dia_mm"] / 2000.0
    z_exhaust_end = geometry_mm["exhaust_length_mm"] / 1000.0
    r_bottom_outlet = geometry_mm["bottom_outlet_mm"] / 2000.0

    r = np.asarray(grid["r_m"], dtype=float)
    z = np.asarray(grid["z_m"], dtype=float)
    v_r = np.asarray(grid["v_r_ms"], dtype=float)
    v_theta = np.asarray(grid["v_theta_ms"], dtype=float)
    v_z = np.asarray(grid["v_z_ms"], dtype=float)
    p = np.asarray(grid["pressure_pa"], dtype=float)

    if r.size == 0:
        raise ValueError("render_cyclone_field: grid has no points to render.")

    v_mag = np.sqrt(v_r ** 2 + v_theta ** 2 + v_z ** 2)

    # Mirror r -> -r to rebuild the full two-sided cross-section from the
    # axisymmetric half-solve. v_mag and pressure are scalars, unchanged
    # under the mirror (this is a visualization convenience, not a second
    # independent solve).
    r_full = np.concatenate([-r, r])
    z_full = np.concatenate([z, z])
    v_full = np.concatenate([v_mag, v_mag])
    p_full = np.concatenate([p, p])
    # v_r is a signed *outward* radial speed at r>=0. On the mirrored
    # (negative-r / negative-x) side, "outward" points toward -x, so the
    # Cartesian x-component there is -v_r, not v_r -- unlike the scalars
    # above (v_mag, p), which are mirror-symmetric and unchanged.
    vr_full = np.concatenate([-v_r, v_r])
    vz_full = np.concatenate([v_z, v_z])

    # Fine regular grid to interpolate onto.
    r_lin = np.linspace(-r_barrel, r_barrel, n_interp)
    z_lin = np.linspace(0.0, z_cone_end, n_interp)
    R, Z = np.meshgrid(r_lin, z_lin)

    V = griddata((r_full, z_full), v_full, (R, Z), method="cubic")
    P = griddata((r_full, z_full), p_full, (R, Z), method="cubic")
    # linear + fill_value=0 (not cubic/NaN like V/P above): matplotlib's
    # streamplot cannot integrate through NaN cells, so cells outside the
    # scattered points' convex hull fall back to "no flow" rather than
    # aborting the whole streamplot call.
    VR = griddata((r_full, z_full), vr_full, (R, Z), method="linear", fill_value=0.0)
    VZ = griddata((r_full, z_full), vz_full, (R, Z), method="linear", fill_value=0.0)

    # Mask outside the true cyclone silhouette so contourf never paints
    # over solid wall � uses the same outer-wall formula as the solver's
    # own geometry, not an independent guess at the shape.
    wall_r = _outer_wall_radius(Z, r_barrel, r_bottom_outlet, z_barrel_end, z_cone_end)
    outside = np.abs(R) > wall_r
    V = np.ma.masked_where(outside, V)
    P = np.ma.masked_where(outside, P)

    # Percentile-clip color limits � same reasoning as AxialFanMVC's
    # render_result.py: a handful of near-wall/near-core outlier cells
    # otherwise dominate the scale and collapse the rest of the field
    # into one flat shade.
    v_valid = V.compressed()
    p_valid = P.compressed()
    v_lo, v_hi = np.percentile(v_valid, [2, 98]) if v_valid.size else (0, 1)
    p_lo, p_hi = np.percentile(p_valid, [2, 98]) if p_valid.size else (0, 1)
    if v_hi <= v_lo:
        v_lo, v_hi = float(v_valid.min()), float(v_valid.max())
    if p_hi <= p_lo:
        p_lo, p_hi = float(p_valid.min()), float(p_valid.max())

    outline_xy, exhaust_xy = _cyclone_outline_xy(
        r_barrel, r_bottom_outlet, z_barrel_end, z_cone_end, r_exhaust, z_exhaust_end
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor="#e8e8e8")

    # --- Panel 1: geometry + velocity streamlines --- mirrors
    # render_result.py's panel 1 (fan STL + streamlines_from_source,
    # colored by U_magnitude, jet cmap) using matplotlib's 2D equivalent:
    # streamplot over the interpolated (VR, VZ) vector field, colored by
    # local speed. r_lin/z_lin are the grid's 1D axes (streamplot wants
    # 1D coordinate vectors, not the 2D R/Z meshgrid).
    ax = axes[0]
    ax.set_facecolor("#e8e8e8")
    speed_for_color = np.nan_to_num(np.ma.filled(V, 0.0), nan=0.0)
    strm = ax.streamplot(r_lin, z_lin, VR, VZ, color=speed_for_color,
                          cmap="jet", norm=plt.Normalize(vmin=v_lo, vmax=v_hi),
                          density=1.3, linewidth=1.1, arrowsize=1.0)
    ax.plot(*outline_xy, color="black", linewidth=1.2)
    ax.plot(*exhaust_xy, color="black", linewidth=1.2)
    _draw_flow_arrows(ax, r_barrel, r_exhaust, z_barrel_end, z_exhaust_end)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("Geometry + Flow Visualization (Streamlines)", fontsize=10)
    ax.set_xlabel("r (m)")
    ax.set_ylabel("z (m, from barrel top)")
    fig.colorbar(strm.lines, ax=ax, label="Velocity magnitude (m/s)")

    # --- Panel 2: pressure slice + geometry context --- mirrors
    # render_result.py's panel 2 (pressure slice, coolwarm, view_xz).
    ax2 = axes[1]
    ax2.set_facecolor("#e8e8e8")
    cf2 = ax2.contourf(R, Z, P, levels=40, cmap="coolwarm", vmin=p_lo, vmax=p_hi)
    ax2.plot(*outline_xy, color="black", linewidth=1.2)
    ax2.plot(*exhaust_xy, color="black", linewidth=1.2)
    _draw_flow_arrows(ax2, r_barrel, r_exhaust, z_barrel_end, z_exhaust_end)
    ax2.invert_yaxis()
    ax2.set_aspect("equal")
    ax2.set_title("Quantitative Pressure Slice (Side View)", fontsize=10)
    ax2.set_xlabel("r (m)")
    fig.colorbar(cf2, ax=ax2, label="Static Pressure (Pa)")

    subtitle_parts = []
    if known_efficiency_percent is not None:
        subtitle_parts.append(f"Calc. efficiency: {known_efficiency_percent:.1f}%")
    if known_pressure_drop_pa is not None:
        subtitle_parts.append(f"Calc. \u0394P: {known_pressure_drop_pa:.0f} Pa")
    if subtitle_parts:
        fig.suptitle("  |  ".join(subtitle_parts), fontsize=9, y=0.02)

    fig.tight_layout()

    png_path = os.path.join(output_dir, filename)
    fig.savefig(png_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    return png_path


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 3:
        print("Usage: render_field.py <result.json> <output_dir>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        payload = json.load(f)

    out_path = render_cyclone_field(
        grid=payload["grid"],
        geometry_mm=payload["geometry_mm"],
        output_dir=sys.argv[2],
        known_efficiency_percent=payload.get("known_efficiency_percent"),
        known_pressure_drop_pa=payload.get("known_pressure_drop_pa"),
    )
    print(out_path)