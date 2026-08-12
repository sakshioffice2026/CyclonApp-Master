"""
add_dxf_dimensions.py
----------------------
Adds engineering dimension MARKS (extension lines + dimension line +
arrowheads + mm text) to the cyclone front-view DXF.

ROOT-CAUSE FIX (earlier revision): ezdxf's add_linear_dim(...).render()
builds real DXF DIMENSION entities, but the actual drawable geometry
(lines/arrows/MTEXT) it creates is written into an ANONYMOUS BLOCK, and
only a DIMENSION entity referencing that block sits in modelspace.
Whether FreeCAD's DXF importer resolves and draws that block content is
inconsistent across FreeCAD/importer versions, so dimensions could
silently never appear. FIX: draw plain LINE + TEXT entities only -
every importer draws these unconditionally.

GEOMETRY-SNAP FIX (earlier revision): dimension anchor points (p1/p2)
are computed from nominal `dims_mm` values, then SNAPPED to the nearest
REAL geometry vertex already present in the DXF (within SNAP_TOLERANCE),
so they line up with whatever cad_generator.py actually drew - including
shell wall thickness / flange-overlap offsets near openings.

THIS REVISION - two more fixes:

1. NO FLOATING / DISCONNECTED DIMENSIONS
   Every extension line's first segment now starts from the EXACT
   snapped geometry point, and a short perpendicular tick mark is drawn
   directly ON that point. Even with the small visual ext_gap (standard
   drafting convention - extension lines don't touch the part outline
   directly), the tick makes the anchor-to-geometry connection explicit
   instead of the line appearing to start from empty space.

2. NO OVERLAPPING / COLLIDING DIMENSIONS
   The four dimensions stacked on the right side (barrel height, cone
   height, exhaust length, inlet height) previously used fixed offsets
   (+80/+80/+130/+180) that could cross or overlap depending on the
   model's actual proportions. They now use a running stack cursor
   (_StackCursor) that guarantees a minimum clear gap between each
   successive dimension line, sized from arrow_size + text_height +
   margin, so labels/lines never collide regardless of model size.

Coordinate system: Front view looking along -Y axis
  X-axis: horizontal (radial, mm)
  Z-axis: vertical (axial, mm) -> mapped to DXF's 2D Y axis
"""
from __future__ import annotations
import math
import ezdxf
from ezdxf.enums import TextEntityAlignment

DIM_LAYER = "DIM_TEXT"

# Max distance (mm) a nominal dimension point may snap to a real geometry
# vertex. Beyond this, the nominal (theoretical) point is used unchanged -
# this keeps the fix safe if a feature genuinely isn't in the geometry.
SNAP_TOLERANCE = 15.0

# Minimum clear gap (mm) enforced between successive stacked dimension
# lines on the same side, on top of each one's own arrow/text footprint.
STACK_MARGIN = 20.0


# ---------------------------------------------------------------------------
# Real-geometry point collection + snapping
# ---------------------------------------------------------------------------

def _collect_geometry_points(msp) -> list[tuple[float, float]]:
    """Read every vertex from the ALREADY-DRAWN front-view geometry (before
    any dimension marks are added), so nominal dimension points can be
    snapped to the real outline instead of trusting recomputed math."""
    points: list[tuple[float, float]] = []
    for entity in msp:
        t = entity.dxftype()
        if t == "LWPOLYLINE":
            for p in entity.get_points():
                points.append((p[0], p[1]))
        elif t == "POLYLINE":
            for v in entity.vertices:
                loc = v.dxf.location
                points.append((loc.x, loc.y))
        elif t == "LINE":
            points.append((entity.dxf.start.x, entity.dxf.start.y))
            points.append((entity.dxf.end.x, entity.dxf.end.y))
    return points


def _snap(point: tuple[float, float], geometry_points: list[tuple[float, float]],
          tolerance: float = SNAP_TOLERANCE) -> tuple[float, float]:
    """Return the nearest real geometry vertex to `point` if one exists
    within `tolerance`; otherwise return `point` unchanged."""
    if not geometry_points:
        return point

    px, py = point
    best = None
    best_dist = tolerance
    for gx, gy in geometry_points:
        dist = math.hypot(gx - px, gy - py)
        if dist < best_dist:
            best_dist = dist
            best = (gx, gy)

    return best if best is not None else point


# ---------------------------------------------------------------------------
# Stacking cursor - prevents overlapping dimension lines on the same side
# ---------------------------------------------------------------------------

class _StackCursor:
    """Hands out non-overlapping offsets along one axis for a run of
    stacked dimensions on the same side of the part.

    Each call to `next(footprint)` returns the next clear offset and
    reserves `footprint` mm of additional clearance (arrow_size +
    text_height, roughly) plus STACK_MARGIN before the following call,
    so consecutive dimension lines/labels never collide."""

    def __init__(self, start: float, step_direction: int = 1):
        self._pos = start
        self._dir = 1 if step_direction >= 0 else -1

    def next(self, footprint: float) -> float:
        offset = self._pos
        self._pos += self._dir * (footprint + STACK_MARGIN)
        return offset


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _tick(msp, point, direction, size, attribs):
    """Short perpendicular tick mark drawn exactly ON a snapped geometry
    point, so the extension line's connection to real geometry is
    visually explicit even though the line itself starts `ext_gap` away
    (standard drafting convention, not a disconnect)."""
    ux, uy = direction
    px, py = -uy, ux
    half = size * 0.5
    msp.add_line(
        (point[0] - px * half, point[1] - py * half),
        (point[0] + px * half, point[1] + py * half),
        dxfattribs=attribs,
    )


def _arrow(msp, tip, direction, size, attribs):
    """Draws a simple chevron arrowhead at `tip`, with the two back legs
    extending backward along -direction. `direction` is a unit-ish
    (dx, dy) tuple pointing FROM the back of the arrow TOWARD the tip."""
    ux, uy = direction
    px, py = -uy, ux  # perpendicular
    back = (tip[0] - ux * size, tip[1] - uy * size)
    left = (back[0] + px * size * 0.4, back[1] + py * size * 0.4)
    right = (back[0] - px * size * 0.4, back[1] - py * size * 0.4)
    msp.add_line(tip, left, dxfattribs=attribs)
    msp.add_line(tip, right, dxfattribs=attribs)


def _linear_dim(
    msp,
    p1,
    p2,
    base,
    angle=0,
    text=None,
    text_height=20,
    attribs=None,
    ext_gap=5,
    ext_overshoot=8,
    arrow_size=10,
    text_offset=15,
):
    """Draws one linear dimension mark using only LINE + TEXT entities.

    p1, p2 = the two measured points (model coordinates, mm) - callers
             should pass these already snapped to real geometry.
    base   = a point that fixes WHERE the dimension line sits
             (its X for angle=90 / vertical dims, its Y for angle=0 /
             horizontal dims)
    angle  = 0 -> horizontal dimension line, 90 -> vertical dimension line
    text   = explicit label; if None, computed as the measured distance
    """
    attribs = attribs or {}

    if angle == 90:
        dim_x = base[0]
        y1, y2 = p1[1], p2[1]
        d1 = (dim_x, y1)
        d2 = (dim_x, y2)

        gap_sign_1 = 1 if dim_x >= p1[0] else -1
        gap_sign_2 = 1 if dim_x >= p2[0] else -1

        # Tick marks anchor the extension lines to the real geometry
        # points themselves - drawn ON p1/p2, not at the offset start.
        _tick(msp, p1, (1, 0), ext_gap * 1.6, attribs)
        _tick(msp, p2, (1, 0), ext_gap * 1.6, attribs)

        msp.add_line(
            (p1[0] + gap_sign_1 * ext_gap, y1),
            (dim_x + gap_sign_1 * ext_overshoot, y1),
            dxfattribs=attribs,
        )
        msp.add_line(
            (p2[0] + gap_sign_2 * ext_gap, y2),
            (dim_x + gap_sign_2 * ext_overshoot, y2),
            dxfattribs=attribs,
        )
        msp.add_line(d1, d2, dxfattribs=attribs)

        dy = 1 if y2 > y1 else -1
        _arrow(msp, d1, (0, -dy), arrow_size, attribs)
        _arrow(msp, d2, (0, dy), arrow_size, attribs)

        value = abs(y2 - y1)
        mid_y = (y1 + y2) / 2.0
        text_pos = (dim_x + text_offset, mid_y)
        rotation = 90
    else:
        dim_y = base[1]
        x1, x2 = p1[0], p2[0]
        d1 = (x1, dim_y)
        d2 = (x2, dim_y)

        gap_sign_1 = 1 if dim_y >= p1[1] else -1
        gap_sign_2 = 1 if dim_y >= p2[1] else -1

        _tick(msp, p1, (0, 1), ext_gap * 1.6, attribs)
        _tick(msp, p2, (0, 1), ext_gap * 1.6, attribs)

        msp.add_line(
            (x1, p1[1] + gap_sign_1 * ext_gap),
            (x1, dim_y + gap_sign_1 * ext_overshoot),
            dxfattribs=attribs,
        )
        msp.add_line(
            (x2, p2[1] + gap_sign_2 * ext_gap),
            (x2, dim_y + gap_sign_2 * ext_overshoot),
            dxfattribs=attribs,
        )
        msp.add_line(d1, d2, dxfattribs=attribs)

        dx = 1 if x2 > x1 else -1
        _arrow(msp, d1, (-dx, 0), arrow_size, attribs)
        _arrow(msp, d2, (dx, 0), arrow_size, attribs)

        value = abs(x2 - x1)
        mid_x = (x1 + x2) / 2.0
        text_pos = (mid_x, dim_y + text_offset)
        rotation = 0

    label = text if text is not None else f"{value:.0f}"
    text_entity = msp.add_text(
        label,
        dxfattribs={**attribs, "height": text_height, "rotation": rotation},
    )
    text_entity.set_placement(text_pos, align=TextEntityAlignment.MIDDLE_CENTER)
    return text_entity


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def add_engineering_dimensions_2d(dxf_path: str, dims_mm: dict, out_path: str | None = None) -> str:
    """Add 2D dimension marks to the CyclonePredictionService front-view DXF
    using plain LINE + TEXT entities only - guaranteed visible on import,
    no DIMENSION entity / anonymous block / dimstyle dependency.

    p1/p2 are snapped to real geometry (no drift near flanges/shell), and
    the four right-side stacked dimensions use a running stack cursor
    (no overlap regardless of model proportions).
    """

    barrel_d = dims_mm["BarrelDiameterMm"]
    barrel_h = dims_mm["BarrelHeightMm"]
    cone_h = dims_mm["ConeHeightMm"]
    exhaust_d = dims_mm["ExhaustDiaMm"]
    exhaust_l = dims_mm["ExhaustLengthMm"]
    bottom_outlet = dims_mm["BottomOutletMm"]
    inlet_h = dims_mm["InletHeightMm"]
    inlet_w = dims_mm["InletWidthMm"]
    barrel_r = barrel_d / 2.0

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # Snapshot the real outline BEFORE adding any dimension marks, so the
    # newly-added LINE/TEXT entities never get picked up as "geometry".
    geometry_points = _collect_geometry_points(msp)

    def snap(point):
        return _snap(point, geometry_points)

    if DIM_LAYER not in doc.layers:
        doc.layers.new(name=DIM_LAYER, dxfattribs={"color": 3})  # green

    attribs = {"layer": DIM_LAYER, "color": 3}
    arrow_size = 10
    text_height = 20
    text_height_small = 15

    # Barrel diameter - horizontal, measured across the barrel/cone
    # junction (z=0), dimension line offset below the cone.
    _linear_dim(
        msp,
        p1=snap((-barrel_r, 0)), p2=snap((barrel_r, 0)),
        base=(0, -cone_h - 100), attribs=attribs,
    )

    # Bottom (dust) outlet diameter - horizontal, at the cone tip.
    _linear_dim(
        msp,
        p1=snap((-bottom_outlet / 2.0, -cone_h)), p2=snap((bottom_outlet / 2.0, -cone_h)),
        base=(0, -cone_h - 150), attribs=attribs,
    )

    # ---- Right-side stack: barrel height, cone height, exhaust length,
    # inlet height. A running cursor guarantees clear spacing between
    # each one - no more fixed +80/+130/+180 that could collide.
    right_stack = _StackCursor(start=barrel_r + 80, step_direction=1)

    barrel_h_base_x = right_stack.next(arrow_size + text_height)
    _linear_dim(
        msp,
        p1=snap((barrel_r, 0)), p2=snap((barrel_r, barrel_h)),
        base=(barrel_h_base_x, 0), angle=90, attribs=attribs,
    )

    # Cone height - p1 sits on the barrel wall (barrel_r), p2 sits on the
    # ACTUAL cone-tip outline. A cone tapers, so at y=-cone_h the real
    # edge is at bottom_outlet_r, not barrel_r - using barrel_r here was
    # never a real point on the body (it floated off the slanted wall).
    # Different x per end is normal: the extension lines simply run from
    # each real point over to the shared vertical dimension line.
    bottom_outlet_r = bottom_outlet / 2.0
    cone_h_base_x = right_stack.next(arrow_size + text_height)
    _linear_dim(
        msp,
        p1=snap((barrel_r, 0)), p2=snap((bottom_outlet_r, -cone_h)),
        base=(cone_h_base_x, -cone_h), angle=90, attribs=attribs,
    )

    # Exhaust (vortex finder) length - both ends belong to the EXHAUST
    # PIPE, whose wall sits at exhaust_r, not barrel_r. Reusing barrel_r
    # anchored this dimension to the wrong feature entirely.
    exhaust_r = exhaust_d / 2.0
    exhaust_l_base_x = right_stack.next(arrow_size + text_height)
    _linear_dim(
        msp,
        p1=snap((exhaust_r, barrel_h - exhaust_l)), p2=snap((exhaust_r, barrel_h)),
        base=(exhaust_l_base_x, barrel_h - exhaust_l), angle=90, attribs=attribs,
    )

    inlet_h_base_x = right_stack.next(arrow_size + text_height_small)
    _linear_dim(
        msp,
        p1=snap((barrel_r, barrel_h - inlet_h - 20)), p2=snap((barrel_r, barrel_h - 20)),
        base=(inlet_h_base_x, barrel_h - inlet_h - 20), angle=90,
        text_height=text_height_small, attribs=attribs,
    )

    # Inlet width W - horizontal, right side, just below the inlet dim.
    _linear_dim(
        msp,
        p1=snap((barrel_r - inlet_w, barrel_h - inlet_h - 40)),
        p2=snap((barrel_r, barrel_h - inlet_h - 40)),
        base=(barrel_r - inlet_w / 2.0, barrel_h - inlet_h - 80),
        text_height=text_height_small, attribs=attribs,
    )

    # Exhaust diameter - horizontal, top.
    _linear_dim(
        msp,
        p1=snap((-exhaust_d / 2.0, barrel_h + 50)), p2=snap((exhaust_d / 2.0, barrel_h + 50)),
        base=(0, barrel_h + 80), attribs=attribs,
    )

    out_path = out_path or dxf_path
    doc.saveas(out_path)
    return out_path


if __name__ == "__main__":
    sample_dims = {
        "BarrelDiameterMm": 300, "BarrelHeightMm": 450, "ConeHeightMm": 600,
        "ExhaustDiaMm": 150, "ExhaustLengthMm": 180, "BottomOutletMm": 100,
        "InletHeightMm": 150, "InletWidthMm": 60,
    }
    add_engineering_dimensions_2d("cyclone_front.dxf", sample_dims)