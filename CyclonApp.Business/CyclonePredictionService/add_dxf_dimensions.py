"""
add_dxf_dimensions.py
----------------------
Adds engineering dimension MARKS to the cyclone front-view DXF, written
TWICE, on two separate layers, so the drawing is correct in both FreeCAD
and AutoCAD:

  DIM_TEXT   - plain LINE + TEXT entities (extension lines, dimension
               line, arrowheads, mm label). Guaranteed visible in every
               DXF importer, including FreeCAD, with zero configuration.
  DIM_NATIVE - real ezdxf DIMENSION entities (added via add_linear_dim
               (...).render()). These are true, associative, editable
               CAD dimension objects - the value is derived from the
               measured geometry, and AutoCAD (and most other real CAD
               tools) can select/edit them as native dimensions.

WHY BOTH: ezdxf's DIMENSION entities are correct DXF - but the actual
drawable geometry they produce (lines/arrows/MTEXT) is written into an
ANONYMOUS BLOCK, with only a DIMENSION entity in modelspace referencing
that block. Whether FreeCAD's DXF importer resolves and draws that block
content is inconsistent across FreeCAD/importer versions - in practice it
often does NOT, so native-only dimensions can silently never appear in
FreeCAD, even though the file itself is valid. AutoCAD does not have this
problem and is where "real", editable DIMENSION objects matter most.

Writing both means: FreeCAD users always see the DIM_TEXT geometry (plain
LINE/TEXT, no block-resolution dependency); AutoCAD users get true
DIMENSION objects on DIM_NATIVE they can select/edit/associate, in
addition to the always-visible DIM_TEXT geometry. Either layer can be
frozen/turned off in whichever tool if the visual doubling is unwanted -
this module does not freeze either by default, since "guaranteed visible
everywhere" is the priority.

Coordinate system: Front view looking along -Y axis
  X-axis: horizontal (radial, mm)
  Z-axis: vertical (axial, mm) -> mapped to DXF's 2D Y axis
"""
from __future__ import annotations
import ezdxf
from ezdxf.enums import TextEntityAlignment

DIM_LAYER = "DIM_TEXT"
NATIVE_DIM_LAYER = "DIM_NATIVE"


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


def _add_native_dimension(
    msp,
    p1,
    p2,
    base,
    angle,
    text_height,
    arrow_size,
    ext_gap,
    ext_overshoot,
):
    """Best-effort addition of a REAL ezdxf DIMENSION entity (associative,
    editable in AutoCAD) alongside the guaranteed-visible LINE+TEXT this
    module already draws. Value is derived from p1/p2 geometry by ezdxf
    itself - not passed in - matching the "recommended" native behavior.

    Wrapped in try/except: this is a purely additive enhancement, and a
    failure here must never block or corrupt the guaranteed-visible
    DIM_TEXT geometry drawn by _linear_dim.
    """
    try:
        override = {
            "dimtxt": text_height,
            "dimasz": arrow_size,
            "dimexo": ext_gap,
            "dimexe": ext_overshoot,
            "dimclrd": 3,
            "dimclre": 3,
            "dimclrt": 3,
        }
        dim = msp.add_linear_dim(
            base=base,
            p1=p1,
            p2=p2,
            angle=angle,
            dimstyle="Standard",
            override=override,
            dxfattribs={"layer": NATIVE_DIM_LAYER},
        )
        dim.render()
    except Exception:
        pass


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
    """Draws one linear dimension mark using only LINE + TEXT entities,
    PLUS a native ezdxf DIMENSION entity on NATIVE_DIM_LAYER (see
    _add_native_dimension). The LINE+TEXT geometry below is unchanged
    from before and remains the guaranteed-visible copy.

    p1, p2 = the two measured points (model coordinates, mm)
    base   = a point that fixes WHERE the dimension line sits
             (its X for angle=90 / vertical dims, its Y for angle=0 /
             horizontal dims)
    angle  = 0 -> horizontal dimension line, 90 -> vertical dimension line
    text   = explicit label; if None, computed as the measured distance
    """
    attribs = attribs or {}

    _add_native_dimension(
        msp, p1, p2, base, angle, text_height, arrow_size, ext_gap, ext_overshoot
    )

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
    """Add 2D dimension marks to the DXF front view: guaranteed-visible
    LINE+TEXT on DIM_TEXT, plus real DIMENSION entities on DIM_NATIVE for
    AutoCAD-style associative editing. See module docstring."""

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
    if NATIVE_DIM_LAYER not in doc.layers:
        doc.layers.new(name=NATIVE_DIM_LAYER, dxfattribs={"color": 3})  # green

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