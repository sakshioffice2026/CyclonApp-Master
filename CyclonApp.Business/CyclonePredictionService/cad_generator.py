"""
cad_generator.py
-----------------
Headless FreeCAD geometry + drawing generator for CyclonApp.

Place this file inside: CyclonApp.Business/CyclonePredictionService/

IMPORTANT: run via FreeCAD's own Python (freecadcmd.exe), NOT imported
from a regular Python process. FreeCAD's native modules are built
against FreeCAD's own bundled Python and will fail with "Module use of
pythonXXX.dll conflicts with this version of Python" if imported into a
different interpreter (e.g. the uvicorn/FastAPI process).

Input is passed via ENVIRONMENT VARIABLES, not command-line arguments.
freecadcmd.exe's own argument parser tries to interpret every extra
command-line argument as a file to open, which breaks plain string
arguments (dims JSON, output path) - environment variables sidestep
that parser entirely.

    CAD_DIMS_JSON  = JSON string of the 8 dimension fields (mm)
    CAD_OUTPUT_DIR = folder to write step/dxf/pdf into

Usage from app.py (via subprocess):
    env = os.environ.copy()
    env["CAD_DIMS_JSON"] = json.dumps(dims)
    env["CAD_OUTPUT_DIR"] = output_dir
    subprocess.run([FREECAD_CMD_PATH, "cad_generator.py"], env=env, ...)

Standalone test (no env vars set -> uses built-in sample dimensions):
    freecadcmd.exe cad_generator.py
"""

from __future__ import annotations
import os
import sys
import json


# ---- Locate FreeCAD's Python modules --------------------------------
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
    # On Linux/inside freecadcmd, FreeCAD is already importable - no-op.


_add_freecad_to_path()

import FreeCAD
import Part
import TechDraw
import Mesh


# ---- Geometry builder -------------------------------------------------
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

    barrel = Part.makeCylinder(barrel_r, barrel_h)

    cone = Part.makeCone(
        barrel_r,
        bottom_outlet / 2.0,
        cone_h,
        FreeCAD.Vector(0, 0, -cone_h),
        FreeCAD.Vector(0, 0, 1),
    )

    body = barrel.fuse(cone)

    exhaust = Part.makeCylinder(
        exhaust_d / 2.0,
        exhaust_l + 50,
        FreeCAD.Vector(0, 0, barrel_h - exhaust_l),
        FreeCAD.Vector(0, 0, 1),
    )

    body = body.cut(exhaust)

    inlet_box = Part.makeBox(
        inlet_w,
        barrel_r * 1.5,
        inlet_h,
        FreeCAD.Vector(-inlet_w / 2.0, -barrel_r * 1.5, barrel_h - inlet_h - 20),
    )
    inlet_cut = inlet_box.common(Part.makeCylinder(barrel_r + 1, barrel_h))
    body = body.fuse(inlet_box.cut(inlet_cut))

    return body


# ---- TechDraw 2D export -------------------------------------------------
def _export_techdraw(doc, shape_obj, output_dir: str, base_name: str):
    page = doc.addObject("TechDraw::DrawPage", "Page")
    template = doc.addObject("TechDraw::DrawSVGTemplate", "Template")

    resource_dir = FreeCAD.getResourceDir()
    template_path = os.path.join(
        resource_dir, "Mod", "TechDraw", "Templates", "A3_Landscape.svg"
    )
    if os.path.isfile(template_path):
        template.Template = template_path
    page.Template = template

    view = doc.addObject("TechDraw::DrawViewPart", "FrontView")
    view.Source = [shape_obj]
    view.Direction = FreeCAD.Vector(0, -1, 0)
    page.addView(view)

    doc.recompute()

    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
    dxf_path = os.path.join(output_dir, f"{base_name}.dxf")

    # freecadcmd.exe has no GUI, so TechDrawGui (needed for PDF export)
    # is not available here. PDF stays None when run this way - expected,
    # not an error. STEP + DXF are the primary deliverables.
    try:
        import TechDrawGui
        TechDrawGui.exportPageAsPdf(page, pdf_path)
    except Exception as e:
        print(f"WARNING: PDF export skipped (needs FreeCAD GUI modules): {e}", file=sys.stderr)
        pdf_path = None

    try:
        import importDXF
        importDXF.export([shape_obj], dxf_path)
    except Exception as e:
        print(f"WARNING: DXF export failed: {e}", file=sys.stderr)
        dxf_path = None

    return pdf_path, dxf_path


# ---- 3D mesh export (OBJ, browser-viewable, headless-compatible) --------
def _export_obj_mesh(shape, output_dir: str, base_name: str):
    """
    Tessellates the solid into a triangle mesh and exports OBJ. Unlike
    STEP (exact B-rep, not directly renderable in a browser) or PDF/DXF
    (needs GUI modules), OBJ export works fully headless via the Mesh
    module and can be loaded straight into Three.js / <model-viewer>
    (convert OBJ -> glTF client-side, or serve OBJ directly with an
    OBJLoader on the frontend).
    """
    obj_path = os.path.join(output_dir, f"{base_name}.obj")
    try:
        doc_mesh = Mesh.Mesh(shape.tessellate(0.5))  # 0.5mm max deviation
        doc_mesh.write(obj_path)
    except Exception as e:
        print(f"WARNING: OBJ mesh export failed: {e}", file=sys.stderr)
        obj_path = None
    return obj_path


# ---- Public entry point -------------------------------------------------
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

    obj_path = _export_obj_mesh(shape, output_dir, base_name)

    pdf_path, dxf_path = _export_techdraw(doc, shape_obj, output_dir, base_name)

    FreeCAD.closeDocument(doc.Name)

    return {
        "step_path": step_path,
        "pdf_path": pdf_path,
        "dxf_path": dxf_path,
        "obj_path": obj_path,
    }


# ---- Entry point: reads input from environment variables ----------------
if __name__ == "__main__":
    dims_json = os.environ.get("CAD_DIMS_JSON")
    out_dir = os.environ.get("CAD_OUTPUT_DIR")

    if dims_json and out_dir:
        dims = json.loads(dims_json)
    else:
        # Standalone manual test with built-in sample dimensions.
        dims = {
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

    result = generate_cyclone_cad(dims, out_dir)
    # Prefixed marker line so app.py can reliably find the result even if
    # FreeCAD prints extra diagnostic lines (Recompute..., transfer stats,
    # etc.) to stdout before this.
    print("RESULT_JSON:" + json.dumps(result))