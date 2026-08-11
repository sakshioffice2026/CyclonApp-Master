"""
add_dxf_dimensions.py
----------------------
Adds real engineering dimensions to the existing cyclone DXF and, as the
composition stage, publishes a single-sheet A3 engineering drawing using the
already-generated principal-view and component/detail DXFs.

The cyclone geometry itself is not regenerated here.  The original combined
DXF is retained as cyclone_assembly.dxf; the public cyclone.dxf becomes the
single-sheet drawing so existing API consumers receive the requested drawing
as the primary DXF deliverable.
"""

from __future__ import annotations

import os
import ezdxf

from drawing_sheet_layout import compose_engineering_sheet


def _add_engineering_dimensions_to_file(dxf_path: str, dims_mm: dict) -> str:
    """Add true DXF DIMENSION entities to an existing flat front-view DXF."""
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

    style_name = "CYCLONE-ISO"
    if style_name not in doc.dimstyles:
        ds = doc.dimstyles.new(style_name)
        ds.dxf.dimtxt = 12
        ds.dxf.dimasz = 8
        ds.dxf.dimexo = 5
        ds.dxf.dimexe = 5
        ds.dxf.dimtad = 1
        ds.dxf.dimdec = 0
        ds.dxf.dimclrd = 7
        ds.dxf.dimclre = 7
        ds.dxf.dimclrt = 7

    def vdim(x, z1, z2, offset_x, text=None):
        dim = msp.add_linear_dim(
            base=(x + offset_x, (z1 + z2) / 2.0),
            p1=(x, z1), p2=(x, z2),
            angle=90, dimstyle=style_name, text=text,
        )
        dim.render()

    def hdim(z, x1, x2, offset_z, text=None):
        dim = msp.add_linear_dim(
            base=((x1 + x2) / 2.0, z + offset_z),
            p1=(x1, z), p2=(x2, z),
            angle=0, dimstyle=style_name, text=text,
        )
        dim.render()

    vdim(barrel_r, 0, barrel_h, 40)
    vdim(barrel_r, -cone_h, 0, 40)
    vdim(barrel_r, exhaust_bottom_z, exhaust_top_z, 80)
    vdim(bottom_outlet / 2.0, dust_pipe_bottom_z, -cone_h, 40)
    vdim(barrel_r + 40, duct_z0, duct_z0 + inlet_h, 40)

    hdim(-cone_h - dust_stub - 20, -barrel_r, barrel_r, -30,
         text=f"D{barrel_d:.0f}")
    hdim(-cone_h - dust_stub - 60, -bottom_outlet / 2.0, bottom_outlet / 2.0, -30,
         text=f"D{bottom_outlet:.0f}")
    hdim(exhaust_top_z + 30, -pipe_r, pipe_r, 20,
         text=f"D{exhaust_d:.0f}")

    doc.saveas(dxf_path)
    return dxf_path


def add_engineering_dimensions(dxf_path: str, dims_mm: dict, out_path: str | None = None) -> str:
    """
    Preserve the existing dimensioned assembly DXF and then compose the
    generated orthographic/detail DXFs into one intentionally arranged sheet.

    The original API path remains the same: cyclone.dxf is replaced by the
    requested single-sheet drawing, while cyclone_assembly.dxf preserves the
    previous combined assembly DXF.
    """
    output_dir = os.path.dirname(os.path.abspath(dxf_path))
    raw_assembly_path = os.path.join(output_dir, "cyclone_assembly.dxf")
    sheet_path = out_path or dxf_path

    # Preserve the existing combined assembly DXF before composition.
    _add_engineering_dimensions_to_file(dxf_path, dims_mm)
    if os.path.abspath(raw_assembly_path) != os.path.abspath(dxf_path):
        os.replace(dxf_path, raw_assembly_path)

    view_paths = {
        "front": os.path.join(output_dir, "cyclone_front.dxf"),
        "top": os.path.join(output_dir, "cyclone_top.dxf"),
        "side": os.path.join(output_dir, "cyclone_side.dxf"),
    }
    section_paths = {
        "barrel": os.path.join(output_dir, "barrel.dxf"),
        "cone": os.path.join(output_dir, "cone.dxf"),
        "air_out_pipe": os.path.join(output_dir, "air_out_pipe.dxf"),
        "dust_outlet_pipe": os.path.join(output_dir, "dust_outlet_pipe.dxf"),
        "inlet_duct": os.path.join(output_dir, "inlet_duct.dxf"),
    }

    missing = [p for p in list(view_paths.values()) + list(section_paths.values()) if not os.path.isfile(p)]
    if missing:
        os.replace(raw_assembly_path, dxf_path)
        raise FileNotFoundError(
            "Single-sheet composition requires all generated view/detail DXFs; "
            f"missing: {', '.join(missing)}"
        )

    revision_id = os.path.basename(os.path.normpath(output_dir))
    compose_engineering_sheet(
        view_paths=view_paths,
        section_paths=section_paths,
        output_path=sheet_path,
        dims=dims_mm,
        revision_id=revision_id,
    )
    return sheet_path


if __name__ == "__main__":
    sample_dims = {
        "BarrelDiameterMm": 300, "BarrelHeightMm": 450, "ConeHeightMm": 600,
        "ExhaustDiaMm": 150, "ExhaustLengthMm": 180, "BottomOutletMm": 100,
        "InletHeightMm": 150, "InletWidthMm": 60,
    }
    add_engineering_dimensions("cyclone.dxf", sample_dims)
