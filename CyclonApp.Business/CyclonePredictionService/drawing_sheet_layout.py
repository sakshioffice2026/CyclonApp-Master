"""Final A3 sheet placement wrapper around the composition-stage implementation.

The underlying cyclone geometry and generated DXFs are untouched.  This small
wrapper supplies the deliberate sheet coordinates and standard scales used for
the final drawing.
"""
from __future__ import annotations

import ezdxf

import drawing_sheet_composer as _c


# Standard mechanical drawing scales and third-angle principal-view arrangement.
_c.ASSEMBLY_SCALE = 0.10  # 1:10
_c.DETAIL_SCALE = 0.05    # 1:20
_C = (95.0, 140.0)
_T = (95.0, 258.0)
_S = (220.0, 140.0)
_DETAILS = {
    "barrel": (35.0, 10.0),
    "cone": (82.0, 10.0),
    "air_out_pipe": (129.0, 10.0),
    "dust_outlet_pipe": (176.0, 10.0),
    "inlet_duct": (218.0, 10.0),
}


def compose_engineering_sheet(*, view_paths, section_paths, output_path, dims, revision_id):
    _c.FRONT_ORIGIN = _C
    _c.TOP_ORIGIN = _T
    _c.SIDE_ORIGIN = _S
    _c.DETAIL_ORIGINS = dict(_DETAILS)

    _c.compose_engineering_sheet(
        view_paths=view_paths,
        section_paths=section_paths,
        output_path=output_path,
        dims=dims,
        revision_id=revision_id,
    )

    # The base composer keeps labels close to each view. On this fixed A3
    # layout, move labels into dedicated annotation bands and normalize the
    # scale note to the standard scales actually used above.
    doc = ezdxf.readfile(output_path)
    for entity in doc.modelspace():
        if entity.dxftype() != "TEXT":
            continue
        text = entity.dxf.text
        if text in {"FRONT ELEVATION", "RIGHT SIDE ELEVATION"}:
            entity.dxf.insert = (entity.dxf.insert.x, 218.0)
        elif text == "TOP / PLAN":
            entity.dxf.insert = (entity.dxf.insert.x, 284.0)
        elif text.startswith("DETAIL "):
            entity.dxf.insert = (entity.dxf.insert.x, 55.0)
        elif text.startswith("PRINCIPAL VIEWS"):
            entity.dxf.text = "PRINCIPAL VIEWS 1:10   |   DETAIL VIEWS 1:20"
    doc.saveas(output_path)
    return output_path
