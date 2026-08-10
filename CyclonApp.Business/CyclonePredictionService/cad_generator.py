"""
cad_generator.py
-----------------
Headless FreeCAD geometry + drawing generator for CyclonApp.
Same usage/env-var contract as before. Only _build_cyclone_shape changed:
body is now a HOLLOW SHEET-METAL SHELL (wall thickness = SheetThicknessMm),
not a solid, matching the "Hollow Sheet Metal" note in the design sketch.

    CAD_DIMS_JSON  = JSON string of dimension fields (mm), now includes
                     optional "SheetThicknessMm" (default 3mm)
    CAD_OUTPUT_DIR = folder to write step/dxf/obj/pdf into
"""

from __future__ import annotations
import os
import sys
import json


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

import FreeCAD
import Part
import TechDraw
import Mesh


# ---- Geometry builder (HOLLOW SHELL) -----------------------------------
def _build_cyclone_shape(dims_mm: dict):
    """
    Builds the cyclone as a hollow sheet-metal shell:
      - Barrel: cylindrical shell
      - Cone: tapered conical shell below barrel
      - Exhaust (vortex finder): open-ended tube through the top, into barrel
      - Inlet: tangential rectangular duct, open into the barrel
      - Dust outlet: open bottom (cone tip, per BottomOutletMm)
    Wall thickness = SheetThicknessMm (default 3mm). Everything else
    is unchanged from the solid version - solid is built first, then
    shelled out with Part.makeThickness, removing the top/bottom faces
    so air can actually pass through (open ends), not sealed solid caps.
    """
    barrel_d = dims_mm["BarrelDiameterMm"]
    barrel_h = dims_mm["BarrelHeightMm"]
    cone_h = dims_mm["ConeHeightMm"]
    exhaust_d = dims_mm["ExhaustDiaMm"]
    exhaust_l = dims_mm["ExhaustLengthMm"]
    bottom_outlet = dims_mm["BottomOutletMm"]
    inlet_h = dims_mm["InletHeightMm"]
    inlet_w = dims_mm["InletWidthMm"]
    thickness = dims_mm.get("SheetThicknessMm", 3.0)

    t = dims_mm.get("WallThicknessMm", 3.0)
    protrusion = dims_mm.get("ExhaustProtrusionMm", 100.0)  # pipe height above roof

    barrel_r = barrel_d / 2.0
    bot_r = bottom_outlet / 2.0

    # --- Barrel: hollow tube, open top & bottom (top gets a roof below,
    # bottom joins the cone's open interior - no caps here) ---
    barrel_outer = Part.makeCylinder(barrel_r, barrel_h)
    barrel_inner = Part.makeCylinder(max(barrel_r - t, 1.0), barrel_h)
    barrel_tube = barrel_outer.cut(barrel_inner)

    # --- Cone: hollow tapered tube, built by revolving a thin quad
    # profile (not by subtracting two solid cones - that produced a
    # self-intersecting, non-manifold mesh near the apex). Open top
    # (joins barrel) & open bottom = Dust Outlet, matching the sketch. ---
    inner_top_r = max(barrel_r - t, 1.0)
    inner_bot_r = max(bot_r - t, 0.5)
    profile_pts = [
        FreeCAD.Vector(barrel_r, 0, 0),
        FreeCAD.Vector(bot_r, 0, -cone_h),
        FreeCAD.Vector(inner_bot_r, 0, -cone_h),
        FreeCAD.Vector(inner_top_r, 0, 0),
        FreeCAD.Vector(barrel_r, 0, 0),
    ]
    profile_wire = Part.makePolygon(profile_pts)
    profile_face = Part.Face(profile_wire)
    cone_tube = profile_face.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 360)

    # --- Roof: solid disc closing the barrel top, pierced only by the
    # exhaust pipe hole (this is the "closed except 3 openings" roof) ---
    roof_disc = Part.makeCylinder(barrel_r, t, FreeCAD.Vector(0, 0, barrel_h))
    roof_hole = Part.makeCylinder(
        exhaust_d / 2.0, t + 2, FreeCAD.Vector(0, 0, barrel_h - 1)
    )
    roof = roof_disc.cut(roof_hole)

    # --- Exhaust pipe: real hollow tube, immersed exhaust_l into the
    # barrel and physically protruding 'protrusion' mm above the roof.
    # Open at both ends -> Air Out opening is the top rim of this pipe. ---
    pipe_bottom_z = barrel_h - exhaust_l
    pipe_top_z = barrel_h + t + protrusion
    pipe_h = pipe_top_z - pipe_bottom_z
    pipe_outer = Part.makeCylinder(
        exhaust_d / 2.0, pipe_h, FreeCAD.Vector(0, 0, pipe_bottom_z), FreeCAD.Vector(0, 0, 1)
    )
    pipe_inner = Part.makeCylinder(
        max(exhaust_d / 2.0 - t, 1.0), pipe_h + 2,
        FreeCAD.Vector(0, 0, pipe_bottom_z - 1), FreeCAD.Vector(0, 0, 1),
    )
    pipe = pipe_outer.cut(pipe_inner)

    # --- Inlet duct: hollow rectangular tube, tangential to the barrel.
    # Open at the outer end (Air In) and open where it meets the barrel. ---
    duct_len = barrel_r * 1.5
    duct_z0 = barrel_h - inlet_h - 20
    duct_outer_box = Part.makeBox(
        inlet_w, duct_len, inlet_h, FreeCAD.Vector(barrel_r - inlet_w, -duct_len, duct_z0)
    )
    bore_x0 = barrel_r - inlet_w + t
    bore_x1 = barrel_r + 5.0  # past the outer barrel surface -> clean through-cut
    bore_z0 = duct_z0 + t
    bore_h = max(inlet_h - 2 * t, 1.0)

    # Bore for hollowing the DUCT ITSELF - extends 1mm past both Y ends
    # so both end caps cut cleanly (open tube). Only touches duct_outer_box.
    duct_bore_box = Part.makeBox(
        bore_x1 - bore_x0, duct_len + 2, bore_h,
        FreeCAD.Vector(bore_x0, -duct_len - 1, bore_z0),
    )
    duct_tube = duct_outer_box.cut(duct_bore_box)

    # Window into the BARREL WALL - clamped to duct_outer_box's exact Y
    # footprint (no +1mm overshoot past Y=0). The overshoot was opening
    # a thin extra slit in the barrel wall beside the real duct opening,
    # on the side duct_outer_box doesn't cover - that was the bug.
    barrel_window_box = Part.makeBox(
        bore_x1 - bore_x0, duct_len, bore_h,
        FreeCAD.Vector(bore_x0, -duct_len, bore_z0),
    )
    barrel_tube = barrel_tube.cut(barrel_window_box)

    # Exactly 3 openings on the finished body: Air In (duct far end),
    # Air Out (pipe top), Dust Outlet (cone tip) - everything else closed.
    body = barrel_tube.fuse(cone_tube).fuse(roof).fuse(pipe).fuse(duct_tube)

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


# ---- 3D mesh export (OBJ) ------------------------------------------------
def _export_obj_mesh(shape, output_dir: str, base_name: str):
    obj_path = os.path.join(output_dir, f"{base_name}.obj")
    try:
        doc_mesh = Mesh.Mesh(shape.tessellate(0.5))
        # tessellate() meshes each BREP face independently - adjacent
        # faces don't share vertex indices at their common edge, so
        # naive export looks "fragmented" even for a perfectly valid
        # solid. Weld those seams before writing.
        doc_mesh.removeDuplicatedPoints()
        doc_mesh.removeDuplicatedFacets()
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

    if not shape.isValid():
        print(
            f"WARNING: generated solid failed shape.isValid() - "
            f"this IS a real geometry defect (not an export artifact). "
            f"Check dimension ratios (wall thickness vs bore sizes).",
            file=sys.stderr,
        )

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


if __name__ == "__main__":
    dims_json = os.environ.get("CAD_DIMS_JSON")
    out_dir = os.environ.get("CAD_OUTPUT_DIR")

    if dims_json and out_dir:
        dims = json.loads(dims_json)
    else:
        dims = {
            "BarrelDiameterMm": 300,
            "BarrelHeightMm": 450,
            "ConeHeightMm": 600,
            "ExhaustDiaMm": 150,
            "ExhaustLengthMm": 180,
            "BottomOutletMm": 100,
            "InletHeightMm": 150,
            "InletWidthMm": 60,
            "WallThicknessMm": 3,
        }
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cad-exports", "test")

    result = generate_cyclone_cad(dims, out_dir)
    print("RESULT_JSON:" + json.dumps(result))