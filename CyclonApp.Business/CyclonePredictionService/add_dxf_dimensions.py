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
from collections import defaultdict
import ezdxf
from ezdxf.enums import TextEntityAlignment

DIM_LAYER = "DIM_TEXT"

# Max distance (mm) a nominal dimension point may snap to a real geometry
# vertex. Beyond this, the nominal (theoretical) point is used unchanged -
# this keeps the fix safe if a feature genuinely isn't in the geometry.
SNAP_TOLERANCE = 40.0

# Minimum clear gap (mm) enforced between successive stacked dimension
# lines on the same side, on top of each one's own arrow/text footprint.
STACK_MARGIN = 20.0

# LOCALIZED TOLERANCE OVERRIDE: exhaust length, inlet height, and inlet
# width measure to vertices that sit on the inlet duct's own protrusion
# geometry, which lands beyond the global SNAP_TOLERANCE (40mm) from the
# nominal computed point on more extreme model proportions. Widening the
# global tolerance to fix these three would risk snapping OTHER
# dimensions to the wrong (nearby but unrelated) vertex on complex
# geometry, so this larger tolerance is applied only at the specific
# snap() call sites for these three dimensions instead.
LOCAL_SNAP_TOLERANCE = 150.0


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

    # Stash the EXACT measured points (already snapped to real geometry)
    # as XDATA, so a later TEXT->DIMENSION conversion step can use these
    # authoritative points instead of re-guessing p1/p2 from the text
    # label's position (which is not on the geometry and produces
    # disconnected/floating dimensions).
    doc = msp.doc
    if doc is not None and "DIMPOINTS" not in doc.appids:
        doc.appids.new("DIMPOINTS")
    text_entity.set_xdata(
        "DIMPOINTS",
        [
            (1000, f"{p1[0]:.6f},{p1[1]:.6f}"),
            (1000, f"{p2[0]:.6f},{p2[1]:.6f}"),
        ],
    )

    return text_entity


def _local_right_x(geometry_points: list[tuple[float, float]], target_y: float,
                    y_window: float = 25.0, fallback: float = 0.0) -> float:
    """Real wall X at a SPECIFIC height, not the whole body's global max X.

    ROOT-CAUSE FIX: a cyclone body with a side-mounted inlet duct has a
    right-wall X that varies by height - the duct sticks out further than
    the barrel wall only over the duct's own height range. Using one
    global actual_right_x for every dimension (regardless of which height
    it's actually measuring) makes any dimension measured away from the
    duct's height aim at an X that has no real geometry near it there -
    the dimension floats even though SOME vertex in the file is close to
    that X value, just not at the Y this particular dimension cares about.

    Filters geometry points to a `y_window` band around `target_y` and
    returns the max X among just those - the true local wall position -
    falling back to `fallback` (typically the global actual_right_x) if no
    points fall in that band."""
    local_xs = [x for x, y in geometry_points if abs(y - target_y) <= y_window]
    return max(local_xs) if local_xs else fallback


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

    def snap(point, tolerance: float = SNAP_TOLERANCE):
        return _snap(point, geometry_points, tolerance=tolerance)

    # ROOT-CAUSE FIX: flatten_to_front_view_2d's XZ projection shifts the
    # right wall away from nominal +barrel_r (e.g. seam edge lands at +184
    # when barrel_r=159). Using nominal barrel_r as p1/p2 makes every
    # dimension float off the real geometry. Derive actual X/Y extents from
    # the real CycloneBody vertices instead, so snap() finds real edges.
    body_xs = [p[0] for p in geometry_points]
    body_ys = [p[1] for p in geometry_points]
    actual_right_x  = max(body_xs) if body_xs else  barrel_r
    actual_left_x   = min(body_xs) if body_xs else -barrel_r
    actual_top_y    = max(body_ys) if body_ys else  barrel_h
    actual_bottom_y = min(body_ys) if body_ys else -cone_h

    if DIM_LAYER not in doc.layers:
        doc.layers.new(name=DIM_LAYER, dxfattribs={"color": 3})  # green

    attribs = {"layer": DIM_LAYER, "color": 3}
    arrow_size = 10
    text_height = 20
    text_height_small = 15

    # Barrel diameter - horizontal at barrel/cone junction (Y=0).
    # Use actual_left_x / actual_right_x so p1/p2 land on real edges.
    _linear_dim(
        msp,
        p1=snap((actual_left_x, 0)), p2=snap((actual_right_x, 0)),
        base=(0, actual_bottom_y - 100), attribs=attribs,
    )

    # Bottom (dust) outlet diameter - horizontal, at the cone tip.
    _linear_dim(
        msp,
        p1=snap((-bottom_outlet / 2.0, actual_bottom_y)),
        p2=snap(( bottom_outlet / 2.0, actual_bottom_y)),
        base=(0, actual_bottom_y - 150), attribs=attribs,
    )

    # ---- Right-side stack: barrel height, cone height, exhaust length,
    # inlet height. Start from actual_right_x (real wall X), not nominal
    # barrel_r, so extension lines originate from real geometry.
    right_stack = _StackCursor(start=actual_right_x + 80, step_direction=1)

    barrel_h_base_x = right_stack.next(arrow_size + text_height)
    # p2's wall X is looked up AT actual_top_y specifically (not the body's
    # single global max-X) - see _local_right_x. A side-mounted inlet duct
    # can push the global max X out further than the true wall at the top.
    top_right_x = _local_right_x(geometry_points, actual_top_y, fallback=actual_right_x)
    _linear_dim(
        msp,
        p1=snap((actual_right_x, 0)), p2=snap((top_right_x, actual_top_y)),
        base=(barrel_h_base_x, 0), angle=90, attribs=attribs,
    )

    # Cone height - p1 at barrel/cone junction on actual right wall,
    # p2 at actual cone tip (bottom_outlet_r, actual_bottom_y).
    bottom_outlet_r = bottom_outlet / 2.0
    cone_h_base_x = right_stack.next(arrow_size + text_height)
    _linear_dim(
        msp,
        p1=snap((actual_right_x, 0)), p2=snap((bottom_outlet_r, actual_bottom_y)),
        base=(cone_h_base_x, actual_bottom_y), angle=90, attribs=attribs,
    )

# Exhaust (vortex finder) length - both ends on exhaust pipe wall (exhaust_r).
    # DERIVE from geometry: find the actual exhaust pipe wall X position from
    # the collected geometry points, so dimensions snap to the real pipe edge
    # rather than the nominal exhaust_d/2 radius which can drift.
    exhaust_actual_r = max(
        [p[0] for p in geometry_points if abs(p[0] - actual_right_x) < 50], default=exhaust_d / 2.0
    )
    exhaust_l_base_x = right_stack.next(arrow_size + text_height)
    _linear_dim(
        msp,
        p1=snap((exhaust_actual_r, actual_top_y - exhaust_l), tolerance=LOCAL_SNAP_TOLERANCE),
        p2=snap((exhaust_actual_r, actual_top_y), tolerance=LOCAL_SNAP_TOLERANCE),
        base=(exhaust_l_base_x, actual_top_y - exhaust_l), angle=90, attribs=attribs,
    )

    # Inlet height - measure to the inlet duct's own top edge, which may sit
    # further than the body's global top Y. Derive actual X position from the
    # inlet duct's real geometry rather than using nominal inlet_h offset.
    inlet_h_base_x = right_stack.next(arrow_size + text_height_small)
    _linear_dim(
        msp,
        p1=snap((actual_right_x, actual_top_y - inlet_h - 20), tolerance=LOCAL_SNAP_TOLERANCE),
        p2=snap((actual_right_x, actual_top_y - 20), tolerance=LOCAL_SNAP_TOLERANCE),
        base=(inlet_h_base_x, actual_top_y - inlet_h - 20), angle=90,
        text_height=text_height_small, attribs=attribs,
    )

    # Inlet width W - horizontal, right side, just below the inlet dim.
    # Use actual right edge X from geometry rather than nominal inlet_w offset,
    # since the inlet duct's outer edge can sit further than nominal width allows.
    _linear_dim(
        msp,
        p1=snap((actual_right_x - inlet_w, actual_top_y - inlet_h - 40), tolerance=LOCAL_SNAP_TOLERANCE),
        p2=snap((actual_right_x, actual_top_y - inlet_h - 40), tolerance=LOCAL_SNAP_TOLERANCE),
        base=(actual_right_x - inlet_w / 2.0, actual_top_y - inlet_h - 80),
        text_height=text_height_small, attribs=attribs,
    )

    # Exhaust diameter - horizontal, top. p1/p2 are the ACTUAL exhaust rim
    # points (on real geometry) so extension lines connect properly; only
    # the dimension LINE (base) sits 80mm above the body - the previous
    # version incorrectly offset p1/p2 themselves by +50, placing the
    # "measured" points 50mm off the real geometry regardless of snapping.
    _linear_dim(
        msp,
        p1=snap((-exhaust_r, actual_top_y)), p2=snap((exhaust_r, actual_top_y)),
        base=(0, actual_top_y + 80), attribs=attribs,
    )

    out_path = out_path or dxf_path
    doc.saveas(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Dimension TEXT cleanup - removes disconnected dimension labels
# ---------------------------------------------------------------------------

def _create_linear_dimension(msp, p1: tuple, p2: tuple, text_pos: tuple, 
                            dim_text: str, layer: str) -> bool:
    """Create proper DXF LINEAR DIMENSION entity with correct codes.
    
    DXF DIMENSION structure:
    - Code 0: DIMENSION
    - Code 8: Layer
    - Code 10/20: Definition point (insertion point)
    - Code 11/21: Text middle point
    - Code 1: Dimension text
    - Code 70: Dimension type (0=Linear, 1=Aligned, etc)
    
    Args:
        msp: Modelspace object
        p1: First definition point (X, Y)
        p2: Second definition point (X, Y)
        text_pos: Text insertion position (X, Y)
        dim_text: Dimension text value (e.g., "300")
        layer: Layer name
    
    Returns:
        True if successfully created, False otherwise
    """
    try:
        # Calculate dimension line angle
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = math.atan2(dy, dx)
        
        # Base point (where dimension starts)
        base_x, base_y = p1
        
        # Text middle point (offset from dimension line)
        text_x, text_y = text_pos
        text_x = (p1[0] + p2[0]) / 2  # Center text between points
        text_y = text_pos[1]  # Use provided Y
        
        # Create rotated dimension (aligned type)
        dim = msp.add_aligned_dim(
            p1=p1,
            p2=p2,
            distance=30,  # Offset distance from geometry
            text=dim_text,
            dxfattribs={
                'layer': layer,
                'dimstyle': 'Standard'
            }
        )
        return True
    except Exception as e:
        return False


def _parse_text_dimension(text_entity) -> tuple:
    """Extract dimension value and position from TEXT entity.
    
    Returns: (dimension_value: str, x: float, y: float)
    """
    value = text_entity.dxf.text
    x = text_entity.dxf.insert.x
    y = text_entity.dxf.insert.y
    return (value, x, y)


def validate_units(dxf_path: str) -> str:
    """Validate drawing units consistency.
    
    Returns: Unit string (e.g., "mm", "inch") or "unknown"
    """
    doc = ezdxf.readfile(dxf_path)
    
    # Check header for unit settings
    header = doc.header
    if 'INSUNITS' in header:
        units_code = header.get('INSUNITS', 0)
        units_map = {
            0: "unitless",
            1: "inch",
            2: "foot",
            3: "mile",
            4: "mm",
            5: "cm",
            6: "meter",
        }
        return units_map.get(units_code, "unknown")
    return "mm"  # default assumption


def validate_geometry_integrity(dxf_path: str) -> dict:
    """Audit geometry for issues like collapsed dimensions or missing data.
    
    Returns: {view_layer: {'status': 'ok'|'warning', 'bounds': (w, h), 'issue': str}}
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    view_audit = {}
    view_bounds = defaultdict(lambda: {'x': [], 'y': []})
    
    # Collect bounds per view layer
    for entity in msp:
        layer = entity.dxf.layer
        if 'View_' in layer:
            t = entity.dxftype()
            if t == "LWPOLYLINE":
                for p in entity.get_points():
                    view_bounds[layer]['x'].append(p[0])
                    view_bounds[layer]['y'].append(p[1])
            elif t == "POLYLINE":
                for v in entity.vertices:
                    loc = v.dxf.location
                    view_bounds[layer]['x'].append(loc.x)
                    view_bounds[layer]['y'].append(loc.y)
            elif t == "LINE":
                view_bounds[layer]['x'].append(entity.dxf.start.x)
                view_bounds[layer]['y'].append(entity.dxf.start.y)
                view_bounds[layer]['x'].append(entity.dxf.end.x)
                view_bounds[layer]['y'].append(entity.dxf.end.y)
    
    # Audit each view
    for view_layer, bounds in view_bounds.items():
        if bounds['x'] and bounds['y']:
            x_min, x_max = min(bounds['x']), max(bounds['x'])
            y_min, y_max = min(bounds['y']), max(bounds['y'])
            
            width = x_max - x_min
            height = y_max - y_min
            
            # Flag suspicious geometry
            status = "ok"
            issue = None
            
            if height < 1.0 and width > 100:  # collapsed height
                status = "warning"
                issue = f"Collapsed height {height:.1f}mm - verify source"
            elif width < 1.0 and height > 100:  # collapsed width
                status = "warning"
                issue = f"Collapsed width {width:.1f}mm - verify source"
            
            view_audit[view_layer] = {
                'status': status,
                'bounds': (width, height),
                'issue': issue
            }
    
    return view_audit


# ---------------------------------------------------------------------------
# Layout reorganization - aligns scattered component views to standard grid
# ---------------------------------------------------------------------------

def reorganize_component_views(dxf_path: str, out_path: str | None = None) -> dict:
    """Reorganize multi-part assembly drawing to standard 3x3 engineering grid.
    
    Maps 8 component detail views to consistent positions:
    - Row 0: Detail A (cone), Front elevation, Side elevation  
    - Row 1: Detail B (barrel), Top/plan view, (reserved)
    - Row 2: Detail E (inlet), Detail C (air-out pipe), Detail D (dust-outlet)
    
    Grid spacing: 1000mm (cols) × 1200mm (rows), 200mm margins
    
    Returns offset_map: {view_layer_name: {'dx': float, 'dy': float}, ...}
    """
    view_layout = {
        'View_cyclone_front_dxf': {'col': 1, 'row': 0},
        'View_cyclone_top_dxf': {'col': 1, 'row': 1},
        'View_cyclone_side_dxf': {'col': 2, 'row': 0},
        'View_cone_dxf': {'col': 0, 'row': 0},
        'View_barrel_dxf': {'col': 0, 'row': 1},
        'View_inlet_duct_dxf': {'col': 0, 'row': 2},
        'View_air_out_pipe_dxf': {'col': 1, 'row': 2},
        'View_dust_outlet_pipe_dxf': {'col': 2, 'row': 2},
    }
    
    margin = 200
    col_spacing = 1000
    row_spacing = 1200
    
    # Read DXF
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    # Measure bounds of each view layer
    view_bounds = defaultdict(lambda: {'x': [], 'y': []})
    
    for entity in msp:
        layer = entity.dxf.layer
        if 'View_' in layer:
            t = entity.dxftype()
            if t == "LWPOLYLINE":
                for p in entity.get_points():
                    view_bounds[layer]['x'].append(p[0])
                    view_bounds[layer]['y'].append(p[1])
            elif t == "POLYLINE":
                for v in entity.vertices:
                    loc = v.dxf.location
                    view_bounds[layer]['x'].append(loc.x)
                    view_bounds[layer]['y'].append(loc.y)
            elif t == "LINE":
                view_bounds[layer]['x'].append(entity.dxf.start.x)
                view_bounds[layer]['y'].append(entity.dxf.start.y)
                view_bounds[layer]['x'].append(entity.dxf.end.x)
                view_bounds[layer]['y'].append(entity.dxf.end.y)
            elif t == "SPLINE":
                for pt in entity.control_points:
                    view_bounds[layer]['x'].append(pt[0])
                    view_bounds[layer]['y'].append(pt[1])
            elif t in ["CIRCLE", "ARC"]:
                c = entity.dxf.center
                view_bounds[layer]['x'].append(c.x)
                view_bounds[layer]['y'].append(c.y)
    
    # Calculate offset for each view
    offset_map = {}
    for view, layout_info in view_layout.items():
        if view in view_bounds and view_bounds[view]['x'] and view_bounds[view]['y']:
            old_x_min = min(view_bounds[view]['x'])
            old_y_min = min(view_bounds[view]['y'])
            
            col, row = layout_info['col'], layout_info['row']
            new_x = margin + col * col_spacing
            new_y = margin + row * row_spacing
            
            offset_map[view] = {
                'dx': new_x - old_x_min,
                'dy': new_y - old_y_min
            }
    
    # Apply offsets to all entities in View_ layers
    for entity in msp:
        layer = entity.dxf.layer
        if layer in offset_map:
            offset = offset_map[layer]
            dx, dy = offset['dx'], offset['dy']
            
            t = entity.dxftype()
            if t == "LWPOLYLINE":
                new_points = []
                for p in entity.get_points():
                    new_points.append((p[0] + dx, p[1] + dy))
                entity.set_points(new_points)
            elif t == "POLYLINE":
                for v in entity.vertices:
                    loc = v.dxf.location
                    v.dxf.location = (loc.x + dx, loc.y + dy, loc.z)
            elif t == "LINE":
                entity.dxf.start = (entity.dxf.start.x + dx, entity.dxf.start.y + dy)
                entity.dxf.end = (entity.dxf.end.x + dx, entity.dxf.end.y + dy)
            elif t == "SPLINE":
                new_cps = [(pt[0] + dx, pt[1] + dy) for pt in entity.control_points]
                entity.set_control_points(new_cps)
            elif t in ["CIRCLE", "ARC"]:
                c = entity.dxf.center
                entity.dxf.center = (c.x + dx, c.y + dy)
            elif t == "TEXT":
                p = entity.dxf.insert
                entity.dxf.insert = (p.x + dx, p.y + dy)
    
    out_path = out_path or dxf_path
    doc.saveas(out_path)
    return offset_map


# ---------------------------------------------------------------------------
# COMPLETE TEXT to DIMENSION CONVERSION - All Steps No Skipping
# ---------------------------------------------------------------------------

def convert_text_to_dimensions(dxf_path: str, out_path: str | None = None) -> dict:
    """Convert floating TEXT dimension labels to proper DXF DIMENSION objects.
    
    Complete step-by-step process:
    1. Load DXF and collect all geometry points from CycloneBody layer
    2. Extract TEXT dimension entities from DIM_TEXT layer
    3. For each TEXT: parse value, find nearest geometry points
    4. Snap dimension definition points (p1, p2) to geometry vertices
    5. Create proper LINEAR DIMENSION entity with dimension line & arrows
    6. Assign dimension style, layer, color
    7. Delete original TEXT entity
    8. Save converted file
    
    Returns: {converted_count: int, dimension_details: [{...}]}
    """
    import math
    
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    # STEP 1: Collect all geometry points from CycloneBody
    geometry_points = []
    for entity in msp:
        if entity.dxf.layer == 'CycloneBody':
            t = entity.dxftype()
            if t == "LWPOLYLINE":
                for p in entity.get_points():
                    geometry_points.append((p[0], p[1]))
            elif t == "LINE":
                geometry_points.append((entity.dxf.start.x, entity.dxf.start.y))
                geometry_points.append((entity.dxf.end.x, entity.dxf.end.y))
    
    if not geometry_points:
        return {'converted_count': 0, 'error': 'No geometry found in CycloneBody layer'}
    
    # STEP 2: Extract TEXT dimensions and identify candidates for conversion
    text_to_convert = []
    text_entities_by_id = {}
    
    for entity in msp:
        if entity.dxftype() == "TEXT" and entity.dxf.layer == "DIM_TEXT":
            text_to_convert.append({
                'entity': entity,
                'value': entity.dxf.text,
                'x': entity.dxf.insert.x,
                'y': entity.dxf.insert.y,
                'handle': entity.dxf.handle
            })
            text_entities_by_id[entity.dxf.handle] = entity
    
    if not text_to_convert:
        return {'converted_count': 0, 'error': 'No TEXT entities found on DIM_TEXT layer'}
    
    # STEP 3-4: For each TEXT dimension, find nearest geometry points and snap
    def find_nearest_points(text_pos, geometry_points, snap_distance=50):
        """Find 2 nearest geometry points for dimension attachment"""
        distances = []
        for i, gpt in enumerate(geometry_points):
            dist = math.sqrt((text_pos[0] - gpt[0])**2 + (text_pos[1] - gpt[1])**2)
            distances.append((dist, i, gpt))
        
        distances.sort()
        nearest = [d for d in distances if d[0] <= snap_distance][:2]
        
        if len(nearest) >= 2:
            return nearest[0][2], nearest[1][2]  # Return actual points
        elif len(nearest) == 1:
            # Single point - create dimension from point to offset
            pt = nearest[0][2]
            return pt, (pt[0] + 50, pt[1])
        else:
            # No nearby geometry - create dimension at text location with offset
            return (text_pos[0] - 25, text_pos[1]), (text_pos[0] + 25, text_pos[1])
    
    # STEP 5-7: Create DIMENSION objects and delete TEXT
    converted_dimensions = []
    entities_to_delete = []
    
    for text_data in text_to_convert:
        try:
            # ROOT-CAUSE FIX: prefer the EXACT p1/p2 stashed as XDATA by
            # _linear_dim (already snapped to real geometry when the
            # dimension mark was first drawn) over guessing new points
            # from the TEXT label's position. The label-position guess
            # (find_nearest_points) grabs whichever geometry vertices
            # happen to be near the label - unrelated to what's actually
            # being measured - which is what produced dimensions that
            # were visually disconnected from the part outline even after
            # "successful" conversion. XDATA is authoritative when present.
            entity = text_data['entity']
            p1 = p2 = None
            try:
                xdata = entity.get_xdata("DIMPOINTS")
                pts = [tag.value for tag in xdata if tag.code == 1000]
                if len(pts) >= 2:
                    p1 = tuple(float(v) for v in pts[0].split(","))
                    p2 = tuple(float(v) for v in pts[1].split(","))
            except Exception:
                p1 = p2 = None

            if p1 is None or p2 is None:
                # Fallback: no XDATA present (e.g. an older file) - guess
                # from label position, same as before.
                p1, p2 = find_nearest_points((text_data['x'], text_data['y']), geometry_points)
            
            # STEP 5: Create LINEAR DIMENSION
            # Set dimension style
            dim_style = 'Standard'
            
            # Create dimension using ezdxf's add_linear_dim
            # Definition point (p1 = first definition point, p2 = second definition point)
            # mid_point = midpoint on dimension line where text appears
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2 + 30  # offset dimension line 30mm above geometry
            
            dim_style_override = msp.add_linear_dim(
                base=(mid_x, mid_y),  # position of dimension line
                p1=p1,                 # first definition point (snapped to geometry)
                p2=p2,                 # second definition point (snapped to geometry)
                angle=0,               # angle of dimension line
                text=text_data['value'],
                dxfattribs={
                    'layer': 'DIM_TEXT',
                    'color': 7,
                    'dimstyle': dim_style
                }
            )

            # STEP 6: Render dimension geometry (draws the block content)
            dim_style_override.render()
            new_dim = dim_style_override.dimension  # underlying DIMENSION entity

            converted_dimensions.append({
                'text_value': text_data['value'],
                'p1': p1,
                'p2': p2,
                'dimension_at': (mid_x, mid_y),
                'handle': new_dim.dxf.handle
            })
            
            # STEP 7: Mark original TEXT for deletion
            entities_to_delete.append(text_data['entity'])
            
        except Exception as e:
            # If dimension creation fails, keep TEXT and log error
            print(f"Warning: Could not convert TEXT '{text_data['value']}' at ({text_data['x']}, {text_data['y']}): {e}")
            continue
    
    # Delete original TEXT entities
    for entity in entities_to_delete:
        msp.delete_entity(entity)
    
    # STEP 8: Save converted file
    out_path = out_path or dxf_path
    doc.saveas(out_path)
    
    return {
        'converted_count': len(converted_dimensions),
        'dimension_details': converted_dimensions,
        'total_text_found': len(text_to_convert),
        'geometry_points_used': len(geometry_points)
    }


def convert_text_to_radial_dimensions(dxf_path: str, layer_name: str = "DIM_TEXT", 
                                      out_path: str | None = None) -> dict:
    """Convert TEXT labels to RADIAL DIMENSION objects for circular features.
    
    Finds circles/arcs in geometry, matches nearby TEXT entities, creates
    RADIUS or DIAMETER dimension objects pointing to circle center.
    
    Returns: {radial_converted: int, detail: [...]}
    """
    import math
    
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    # Find all circles and their centers
    circles = []
    for entity in msp:
        if entity.dxftype() == "CIRCLE":
            circles.append({
                'center': (entity.dxf.center.x, entity.dxf.center.y),
                'radius': entity.dxf.radius,
                'entity': entity
            })
    
    if not circles:
        return {'radial_converted': 0, 'message': 'No circles found'}
    
    # Find TEXT near circles
    text_entities = []
    for entity in msp:
        if entity.dxftype() == "TEXT" and entity.dxf.layer == layer_name:
            text_entities.append({
                'pos': (entity.dxf.insert.x, entity.dxf.insert.y),
                'value': entity.dxf.text,
                'entity': entity
            })
    
    converted_radial = []
    entities_to_delete = []
    
    # Match TEXT to circles and create radial dimensions
    for text in text_entities:
        for circle in circles:
            dist = math.sqrt(
                (text['pos'][0] - circle['center'][0])**2 + 
                (text['pos'][1] - circle['center'][1])**2
            )
            
            # If TEXT is near circle (within 100mm), create radial dimension
            if dist < 100:
                try:
                    # Create radial dimension
                    radius_dim = msp.add_radius_dim(
                        center=circle['center'],
                        radius=circle['radius'],
                        angle=45,  # angle to dimension line
                        dxfattribs={
                            'layer': layer_name,
                            'color': 7
                        }
                    )
                    radius_dim.dxf.text = text['value']
                    radius_dim.render()
                    
                    converted_radial.append({
                        'text_value': text['value'],
                        'circle_center': circle['center'],
                        'circle_radius': circle['radius']
                    })
                    
                    entities_to_delete.append(text['entity'])
                    break
                except Exception as e:
                    continue
    
    # Delete converted TEXT entities
    for entity in entities_to_delete:
        msp.delete_entity(entity)
    
    out_path = out_path or dxf_path
    doc.saveas(out_path)
    
    return {
        'radial_converted': len(converted_radial),
        'details': converted_radial
    }


if __name__ == "__main__":
    sample_dims = {
        "BarrelDiameterMm": 300, "BarrelHeightMm": 450, "ConeHeightMm": 600,
        "ExhaustDiaMm": 150, "ExhaustLengthMm": 180, "BottomOutletMm": 100,
        "InletHeightMm": 150, "InletWidthMm": 60,
    }
    add_engineering_dimensions_2d("cyclone_front.dxf", sample_dims)