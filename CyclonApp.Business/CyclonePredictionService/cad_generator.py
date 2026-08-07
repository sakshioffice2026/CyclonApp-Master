"""
cad_generator.py
─────────────────
Headless FreeCAD geometry + drawing generator for CyclonApp.

Place this file inside: CyclonApp.Business/CyclonePredictionService/
(same folder as app.py, field_physics.py, etc.)

CONTRACT:
    generate_cyclone_cad(dimensions_mm: dict, output_dir: str) -> dict
        dimensions_mm keys (mm), matching geometry_from_dimensions_mm():
            BarrelDiameterMm, BarrelHeightMm, ConeHeightMm,
            ExhaustDiaMm, ExhaustLengthMm, BottomOutletMm,
            InletHeightMm, InletWidthMm
        output_dir: folder to write step/dxf/pdf into
                    (caller passes cad-exports/<revisionId>/)

    Returns dict with file paths:
        { "step_path": ..., "dxf_path": ..., "pdf_path": ... }

Run standalone for testing:
    freecadcmd.exe cad_generator.py

FreeCAD install path is auto-detected on Windows; override via
FREECAD_BIN_PATH env var if installed elsewhere.
"""

from __future__ import annotations
import os
import sys
import json

# ── Locate FreeCAD's Python modules ─────────────────────────────────────
def _add_freecad_to_path():
    if os.name == "nt":
        candidates = [
            os.environ.get("FREECAD_BIN_PATH"),
            r"C:\Program Files\FreeCAD 1.1\bin",
            r"C:\Program Files\FreeCAD 1.0\bin",
        ]
        for path in candidates:
            if path and os.path.isdir(path):
                sys.path.append(path)
                return
    # On Linux/inside freecadcmd, FreeCAD is already importable — no-op.


_add_freecad_to_path()

import FreeCAD
import Part
import TechDraw


# ── Geometry builder ─────────────────────────────────────────────────────
def _build_cyclone_shape(dims_mm: dict):
    """
    Builds a simple axisymmetric cyclone body:
      - Barrel: cylinder
      - Cone: tapered cone below barrel
      - Exhaust (vortex finder): cylinder from top, into barrel
      - Inlet: rectangular box, positioned tangentially at barrel top
    Units: FreeCAD's Part primitives use mm natively.
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

    # Barrel (cylinder), base at z=0, extends up
    barrel = Part.makeCylinder(barrel_r, barrel_h)

    # Cone: tapers from barrel_r (top) to bottom_outlet/2 (bottom),
    # placed below the barrel (negative z)
    cone = Part.makeCone(
        barrel_r,
        bottom_outlet / 2.0,
        cone_h,
        FreeCAD.Vector(0, 0, -cone_h),
        FreeCAD.Vector(0, 0, 1),
    )

    body = barrel.fuse(cone)

    # Exhaust pipe (vortex finder): cylinder from top of barrel, going down
    # into the barrel by exhaust_l
    exhaust = Part.makeCylinder(
        exhaust_d / 2.0,
        exhaust_l + 50,  # a bit taller so it clearly protrudes above
        FreeCAD.Vector(0, 0, barrel_h - exhaust_l),
        FreeCAD.Vector(0, 0, 1),
    )

    # Cut exhaust bore out of the body (hollow vortex finder passage)
    body = body.cut(exhaust)

    # Inlet duct: rectangular box, tangential to barrel wall, near barrel top
    inlet_box = Part.makeBox(
        inlet_w,
        barrel_r * 1.5,
        inlet_h,
        FreeCAD.Vector(-inlet_w / 2.0, -barrel_r * 1.5, barrel_h - inlet_h - 20),
    )
    inlet_cut = inlet_box.common(Part.makeCylinder(barrel_r + 1, barrel_h))
    body = body.fuse(inlet_box.cut(inlet_cut))  # duct sits outside, opens into barrel

    return body


# ── TechDraw 2D export ───────────────────────────────────────────────────
def _export_techdraw(doc, shape_obj, output_dir: str, base_name: str):
    page = doc.addObject("TechDraw::DrawPage", "Page")
    template = doc.addObject("TechDraw::DrawSVGTemplate", "Template")

    # Use FreeCAD's bundled A3 landscape template if available.
    # FreeCAD.__file__ isn't set in this build, so use the resource dir API.
    resource_dir = FreeCAD.getResourceDir()
    template_path = os.path.join(
        resource_dir, "Mod", "TechDraw", "Templates", "A3_Landscape.svg"
    )
    if os.path.isfile(template_path):
        template.Template = template_path
    page.Template = template

    view = doc.addObject("TechDraw::DrawViewPart", "FrontView")
    view.Source = [shape_obj]
    view.Direction = FreeCAD.Vector(0, -1, 0)  # front view
    page.addView(view)

    doc.recompute()

    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
    dxf_path = os.path.join(output_dir, f"{base_name}.dxf")

    # PDF export requires TechDrawGui, which only loads when FreeCAD runs
    # with GUI modules available (FreeCAD.exe --console), not freecadcmd.exe.
    try:
        import TechDrawGui
        TechDrawGui.exportPageAsPdf(page, pdf_path)
    except Exception as e:
        print(f"WARNING: PDF export failed (needs FreeCAD.exe --console, not freecadcmd.exe): {e}")
        pdf_path = None

    try:
        import importDXF
        importDXF.export([shape_obj], dxf_path)
    except Exception as e:
        print(f"WARNING: DXF export failed: {e}")
        dxf_path = None

    return pdf_path, dxf_path


# ── Public entry point ───────────────────────────────────────────────────
def generate_cyclone_cad(dimensions_mm: dict, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    doc = FreeCAD.newDocument("Cyclone")
    shape = _build_cyclone_shape(dimensions_mm)

    shape_obj = doc.addObject("Part::Feature", "CycloneBody")
    shape_obj.Shape = shape
    doc.recompute()

    base_name = "cyclone"
    step_path = os.path.join(output_dir, f"{base_name}.step")
    Part.export([shape_obj], step_path)

    pdf_path, dxf_path = _export_techdraw(doc, shape_obj, output_dir, base_name)

    FreeCAD.closeDocument(doc.Name)

    return {
        "step_path": step_path,
        "pdf_path": pdf_path,
        "dxf_path": dxf_path,
    }


# ── Standalone test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    test_dims = {
        "BarrelDiameterMm": 300,
        "BarrelHeightMm": 450,
        "ConeHeightMm": 600,
        "ExhaustDiaMm": 150,
        "ExhaustLengthMm": 180,
        "BottomOutletMm": 100,
        "InletHeightMm": 150,
        "InletWidthMm": 60,
    }
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cad-exports", "test")
    result = generate_cyclone_cad(test_dims, out_dir)
    print(json.dumps(result, indent=2))