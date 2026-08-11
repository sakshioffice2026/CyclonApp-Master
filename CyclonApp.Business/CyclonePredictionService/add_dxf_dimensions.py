"""
add_dxf_dimensions.py
----------------------
Adds engineering dimension MARKS (extension lines + dimension line +
arrowheads + mm text) to the cyclone front-view DXF.

ROOT-CAUSE FIX (this revision): ezdxf's add_linear_dim(...).render()
builds real DXF DIMENSION entities, but the actual drawable geometry
(lines/arrows/MTEXT) it creates is written into an ANONYMOUS BLOCK, and
only a DIMENSION entity referencing that block sits in modelspace.
Whether FreeCAD's DXF importer resolves and draws that block content is
inconsistent across FreeCAD/importer versions - in practice it very
often does NOT, so the dimensions silently never appear, even though the
DXF file itself is 100% valid and correct.

GUARANTEED FIX: stop relying on the DIMENSION entity type entirely.
Draw the extension lines, dimension line, and arrowheads as plain LINE
entities, and the measurement label as a plain TEXT entity (not MTEXT,
never inside a block). LINE and TEXT are the most basic DXF entity
types - every DXF importer, including FreeCAD's, draws LINE entities
unconditionally with zero configuration. This removes any dependency on
DIMENSION-block resolution or DXF-text-import preferences.

NOTE on TEXT visibility specifically: FreeCAD's DXF importer has an
"Import DXF text as Draft Text objects" preference (Edit -> Preferences
-> Import-Export -> DXF). If it's off, plain TEXT entities may still
import but not always render exactly as expected depending on FreeCAD
version. The LINE geometry (extension lines / dimension lines /
arrowheads) does NOT depend on this setting at all and will always be
visible - so even in the worst case, the dimension marks themselves are
guaranteed visible; only the numeric label's exact rendering could vary
by FreeCAD version/preference.

Coordinate system: Front view looking along -Y axis
  X-axis: horizontal (radial, mm)
  Z-axis: vertical (axial, mm) -> mapped to DXF's 2D Y axis
"""
from __future__ import annotations
import ezdxf
from ezdxf.enums import TextEntityAlignment

DIM_LAYER = "DIM_TEXT"


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

    p1, p2 = the two measured points (model coordinates, mm)
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


def add_engineering_dimensions_2d(dxf_path: str, dims_mm: dict, out_path: str | None = None) -> str:
    """Add 2D dimension marks to the DXF front view using plain LINE +
    TEXT entities only - guaranteed visible on import, no DIMENSION
    entity / anonymous block / dimstyle dependency."""

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

    if DIM_LAYER not in doc.layers:
        doc.layers.new(name=DIM_LAYER, dxfattribs={"color": 3})  # green

    attribs = {"layer": DIM_LAYER, "color": 3}

    # Barrel diameter - horizontal, measured across the barrel/cone
    # junction (z=0), dimension line offset below the cone.
    _linear_dim(msp, p1=(-barrel_r, 0), p2=(barrel_r, 0), base=(0, -cone_h - 100), attribs=attribs)

    # Bottom (dust) outlet diameter - horizontal, at the cone tip.
    _linear_dim(
        msp,
        p1=(-bottom_outlet / 2.0, -cone_h), p2=(bottom_outlet / 2.0, -cone_h),
        base=(0, -cone_h - 150), attribs=attribs,
    )

    # Barrel height H - vertical, right side.
    _linear_dim(msp, p1=(barrel_r, 0), p2=(barrel_r, barrel_h), base=(barrel_r + 80, 0), angle=90, attribs=attribs)

    # Cone height - vertical, right side.
    _linear_dim(msp, p1=(barrel_r, 0), p2=(barrel_r, -cone_h), base=(barrel_r + 80, -cone_h), angle=90, attribs=attribs)

    # Exhaust (vortex finder) length L - vertical, further right.
    _linear_dim(
        msp,
        p1=(barrel_r, barrel_h - exhaust_l), p2=(barrel_r, barrel_h),
        base=(barrel_r + 130, barrel_h - exhaust_l), angle=90, attribs=attribs,
    )

    # Inlet height - vertical, right side (duct is on +X per
    # cad_generator.py: duct_center_x = barrel_r - inlet_w/2, flush with
    # the barrel wall at x=+barrel_r). Placed further out (barrel_r+180)
    # to clear the barrel-height/cone-height/exhaust-length dims already
    # stacked at barrel_r+80 / barrel_r+130.
    _linear_dim(
        msp,
        p1=(barrel_r, barrel_h - inlet_h - 20), p2=(barrel_r, barrel_h - 20),
        base=(barrel_r + 180, barrel_h - inlet_h - 20), angle=90, text_height=15, attribs=attribs,
    )

    # Inlet width W - horizontal, right side, just below the inlet dim.
    _linear_dim(
        msp,
        p1=(barrel_r - inlet_w, barrel_h - inlet_h - 40),
        p2=(barrel_r, barrel_h - inlet_h - 40),
        base=(barrel_r - inlet_w / 2.0, barrel_h - inlet_h - 80),
        text_height=15, attribs=attribs,
    )

    # Exhaust diameter - horizontal, top.
    _linear_dim(
        msp,
        p1=(-exhaust_d / 2.0, barrel_h + 50), p2=(exhaust_d / 2.0, barrel_h + 50),
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