"""
cad_generator.py - Headless FreeCAD geometry + drawing generator for CyclonApp.
"""

from __future__ import annotations
import os
import sys
import json
import traceback


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


_add_freecad_to_path()

print("DEBUG: Starting FreeCAD imports...", file=sys.stderr, flush=True)

try:
    import FreeCAD
    print("DEBUG: FreeCAD imported OK", file=sys.stderr, flush=True)
except Exception as e:
    print(f"FATAL: FreeCAD import failed: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

try:
    import Part
    print("DEBUG: Part imported OK", file=sys.stderr, flush=True)
except Exception as e:
    print(f"FATAL: Part import failed: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

TechDraw = None
try:
    import TechDraw
    print("DEBUG: TechDraw imported OK", file=sys.stderr, flush=True)
except Exception as e:
    print(f"WARNING: TechDraw import failed: {e}", file=sys.stderr, flush=True)

Mesh = None
try:
    import Mesh
    print("DEBUG: Mesh imported OK", file=sys.stderr, flush=True)
except Exception as e:
    print(f"WARNING: Mesh import failed: {e}", file=sys.stderr, flush=True)

print("DEBUG: All imports complete", file=sys.stderr, flush=True)


def _build_cyclone_shape(dims_mm: dict):
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


def _export_techdraw(doc, shape_obj, output_dir: str, base_name: str):
    if TechDraw is None:
        print("WARNING: Skipping TechDraw export (module not available)", file=sys.stderr, flush=True)
        return None, None

    try:
        print("DEBUG: Creating TechDraw page...", file=sys.stderr, flush=True)
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
        print("DEBUG: TechDraw page created", file=sys.stderr, flush=True)

        pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
        dxf_path = os.path.join(output_dir, f"{base_name}.dxf")

        try:
            import TechDrawGui
            TechDrawGui.exportPageAsPdf(page, pdf_path)
            print(f"DEBUG: PDF exported", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"WARNING: PDF export failed: {e}", file=sys.stderr, flush=True)
            pdf_path = None

        try:
            import importDXF
            importDXF.export([shape_obj], dxf_path)
            print(f"DEBUG: DXF exported", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"WARNING: DXF export failed: {e}", file=sys.stderr, flush=True)
            dxf_path = None

        return pdf_path, dxf_path

    except Exception as e:
        print(f"ERROR in TechDraw: {e}", file=sys.stderr, flush=True)
        return None, None


def _export_obj_mesh(shape, output_dir: str, base_name: str):
    if Mesh is None:
        print("WARNING: Skipping OBJ export (Mesh unavailable)", file=sys.stderr, flush=True)
        return None

    obj_path = os.path.join(output_dir, f"{base_name}.obj")
    try:
        print("DEBUG: Tessellating mesh...", file=sys.stderr, flush=True)
        doc_mesh = Mesh.Mesh(shape.tessellate(0.5))
        doc_mesh.write(obj_path)
        print(f"DEBUG: OBJ exported", file=sys.stderr, flush=True)
        return obj_path
    except Exception as e:
        print(f"WARNING: OBJ export failed: {e}", file=sys.stderr, flush=True)
        return None


def generate_cyclone_cad(dimensions_mm: dict, output_dir: str) -> dict:
    print("DEBUG: generate_cyclone_cad() entered", file=sys.stderr, flush=True)
    os.makedirs(output_dir, exist_ok=True)
    print("DEBUG: Output dir created", file=sys.stderr, flush=True)

    print("DEBUG: Creating FreeCAD doc...", file=sys.stderr, flush=True)
    doc = FreeCAD.newDocument("Cyclone")
    print("DEBUG: Doc created", file=sys.stderr, flush=True)

    print("DEBUG: Building shape...", file=sys.stderr, flush=True)
    shape = _build_cyclone_shape(dimensions_mm)
    print("DEBUG: Shape built", file=sys.stderr, flush=True)

    shape_obj = doc.addObject("Part::Feature", "CycloneBody")
    shape_obj.Shape = shape
    doc.recompute()
    print("DEBUG: Doc recomputed", file=sys.stderr, flush=True)

    base_name = "cyclone"
    step_path = os.path.join(output_dir, f"{base_name}.step")
    print("DEBUG: Exporting STEP...", file=sys.stderr, flush=True)
    Part.export([shape_obj], step_path)
    print("DEBUG: STEP exported", file=sys.stderr, flush=True)

    print("DEBUG: Exporting OBJ...", file=sys.stderr, flush=True)
    obj_path = _export_obj_mesh(shape, output_dir, base_name)
    print("DEBUG: OBJ export done", file=sys.stderr, flush=True)

    print("DEBUG: Exporting TechDraw...", file=sys.stderr, flush=True)
    pdf_path, dxf_path = _export_techdraw(doc, shape_obj, output_dir, base_name)
    print("DEBUG: TechDraw export done", file=sys.stderr, flush=True)

    print("DEBUG: Closing doc...", file=sys.stderr, flush=True)
    FreeCAD.closeDocument(doc.Name)
    print("DEBUG: Doc closed", file=sys.stderr, flush=True)

    result = {
        "step_path": step_path,
        "pdf_path": pdf_path,
        "dxf_path": dxf_path,
        "obj_path": obj_path,
    }
    print("DEBUG: Result dict created, returning", file=sys.stderr, flush=True)
    return result


if __name__ == "__main__":
    print("DEBUG: __main__ block started", file=sys.stderr, flush=True)
    try:
        dims_json = os.environ.get("CAD_DIMS_JSON")
        out_dir = os.environ.get("CAD_OUTPUT_DIR")
        print(f"DEBUG: Got env vars. dims_json={bool(dims_json)}, out_dir={out_dir}", file=sys.stderr, flush=True)

        if dims_json and out_dir:
            print("DEBUG: Parsing JSON...", file=sys.stderr, flush=True)
            dims = json.loads(dims_json)
            print("DEBUG: JSON parsed", file=sys.stderr, flush=True)
        else:
            print("DEBUG: Using test dimensions", file=sys.stderr, flush=True)
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

        print("DEBUG: Calling generate_cyclone_cad...", file=sys.stderr, flush=True)
        result = generate_cyclone_cad(dims, out_dir)
        print("DEBUG: generate_cyclone_cad returned", file=sys.stderr, flush=True)

        print("DEBUG: Printing RESULT_JSON...", file=sys.stderr, flush=True)
        print("RESULT_JSON:" + json.dumps(result), flush=True)
        print("DEBUG: RESULT_JSON printed", file=sys.stderr, flush=True)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        sys.exit(1)