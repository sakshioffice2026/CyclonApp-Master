"""
add_dxf_dimensions.py
----------------------
Adds real engineering dimensions (extension lines + dimension line +
arrowheads + numeric mm text) to a cyclone front-view DXF, matching
standard 2D industrial blueprint style - not TEXT labels pointing at
the drawing.

Why ezdxf, not FreeCAD/TechDraw: TechDraw's own dimension objects need
the GUI (TechDrawGui) to compute/render reliably in headless
freecadcmd, and driving them from Python is brittle. ezdxf works
directly on the exported DXF, is headless-safe, and creates true DXF
DIMENSION entities that AutoCAD/LibreCAD/any CAD viewer renders and
measures like any other dimension - the actual requirement here.

Coordinate system: matches _export_view_dxf's front view
(direction = Vector(0, -1, 0), i.e. looking along -Y), so the DXF
plane is X (radial, mm) horizontal / Z (vertical, mm) vertical -
identical to cad_generator._build_cyclone_shape's own coordinates.
All dimension positions below are re-derived from dims_mm using those
same formulas, so they always match the actual exported geometry
regardless of which optional fields were supplied vs defaulted.

Usage:
    from add_dxf_dimensions import add_engineering_dimensions
    add_engineering_dimensions("cyclone_front.dxf", dims_mm)
"""

from __future__ import annotations
import ezdxf


def add_engineering_dimensions(dxf_path: str, dims_mm: dict, out_path: str | None = None) -> str:
    barrel_d = dims_mm["BarrelDiameterMm"]
    barrel_h = dims_mm["BarrelHeightMm"]
    cone_h = dims_mm["ConeHeightMm"]
    exhaust_d = dims_mm["ExhaustDiaMm"]
    exhaust_l = dims_mm["ExhaustLengthMm"]
    bottom_outlet = dims_mm["BottomOutletMm"]
    inlet_h = dims_mm["InletHeightMm"]

    dust_stub = dims_mm.get("DustOutletPipeLengthMm", 100.0)

    barrel_r = barrel_d / 2.0
    pipe_r = exhaust_d / 2.0
    exhaust_bottom_z = barrel_h - exhaust_l
    exhaust_top_z = barrel_h + 50
    dust_pipe_bottom_z = -cone_h - dust_stub
    duct_z0 = barrel_h - inlet_h - 20

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # ISO-style dimension style: arrows, extension-line offset/extension,
    # text above the dimension line, whole-mm auto text.
    style_name = "CYCLONE-ISO"
    if style_name not in doc.dimstyles:
        ds = doc.dimstyles.new(style_name)
        ds.dxf.dimtxt = 12    # text height, mm
        ds.dxf.dimasz = 8     # arrowhead size, mm
        ds.dxf.dimexo = 5     # extension line offset from the part outline
        ds.dxf.dimexe = 5     # extension line overshoot past the dim line
        ds.dxf.dimtad = 1     # text placed above the dimension line
        ds.dxf.dimdec = 0     # 0 decimal places
        ds.dxf.dimclrd = 7
        ds.dxf.dimclre = 7
        ds.dxf.dimclrt = 7

    def vdim(x, z1, z2, offset_x, text=None):
        """Vertical linear dimension between two heights at a fixed X."""
        dim = msp.add_linear_dim(
            base=(x + offset_x, (z1 + z2) / 2.0),
            p1=(x, z1), p2=(x, z2),
            angle=90, dimstyle=style_name, text=text,
        )
        dim.render()

    def hdim(z, x1, x2, offset_z, text=None):
        """Horizontal linear dimension between two X positions at a fixed Z."""
        dim = msp.add_linear_dim(
            base=((x1 + x2) / 2.0, z + offset_z),
            p1=(x1, z), p2=(x2, z),
            angle=0, dimstyle=style_name, text=text,
        )
        dim.render()

    # ---- Vertical (height) dimensions, staggered to the right ----
    vdim(barrel_r, 0, barrel_h, offset_x=40)                          # barrel height
    vdim(barrel_r, -cone_h, 0, offset_x=40)                           # cone height
    vdim(barrel_r, exhaust_bottom_z, exhaust_top_z, offset_x=80)      # exhaust length
    vdim(bottom_outlet / 2.0, dust_pipe_bottom_z, -cone_h, offset_x=40)  # dust stub
    vdim(barrel_r + 40, duct_z0, duct_z0 + inlet_h, offset_x=40)      # inlet height

    # ---- Horizontal (diameter) dimensions, staggered below/above ----
    hdim(-cone_h - dust_stub - 20, -barrel_r, barrel_r, offset_z=-30,
         text=f"D{barrel_d:.0f}")
    hdim(-cone_h - dust_stub - 60, -bottom_outlet / 2.0, bottom_outlet / 2.0,
         offset_z=-30, text=f"D{bottom_outlet:.0f}")
    hdim(exhaust_top_z + 30, -pipe_r, pipe_r, offset_z=20,
         text=f"D{exhaust_d:.0f}")

    out_path = out_path or dxf_path
    doc.saveas(out_path)
    return out_path


if __name__ == "__main__":
    sample_dims = {
        "BarrelDiameterMm": 300, "BarrelHeightMm": 450, "ConeHeightMm": 600,
        "ExhaustDiaMm": 150, "ExhaustLengthMm": 180, "BottomOutletMm": 100,
        "InletHeightMm": 150, "InletWidthMm": 60,
    }
    add_engineering_dimensions("cyclone_front.dxf", sample_dims)