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

    CAD_DIMS_JSON  = JSON string of the dimension fields (mm)
    CAD_OUTPUT_DIR = folder to write step/dxf/pdf into

Usage from app.py (via subprocess):
    env = os.environ.copy()
    env["CAD_DIMS_JSON"] = json.dumps(dims)
    env["CAD_OUTPUT_DIR"] = output_dir
    subprocess.run([FREECAD_CMD_PATH, "cad_generator.py"], env=env, ...)

Standalone test (no env vars set -> uses built-in sample dimensions):
    freecadcmd.exe cad_generator.py

FIX (this revision - dimensions invisible in DXF):
_export_techdraw() and _export_view_dxf() set view.Direction but never
set view.Scale / view.X / view.Y. On doc.recompute() FreeCAD auto-picks
a page scale and placement, so the exported DXF geometry lands in
auto-scaled/auto-offset page space - not raw model mm at origin (0,0).
add_dxf_dimensions_2D.py's add_engineering_dimensions_2d() inserts TEXT
at raw model coordinates (e.g. (0, -cone_h - 100)), assuming a 1:1,
origin-anchored frame. Result: text and geometry end up in different
coordinate frames, so the dimension text is not visibly co-located with
the drawing. Fix: pin view.Scale = 1.0, view.X = 0, view.Y = 0 on both
view objects so the DXF's coordinate frame matches what the dimension
code assumes. No geometry, export logic, or section/view structure was
changed.
"""

from __future__ import annotations
import os
import sys
import json
import math

# Adds real DXF DIMENSION entities (extension lines + arrows + mm text)
# to the front-view DXF. Lives next to this file; import failure must not
# stop CAD generation (STEP/DXF/OBJ still matter without dimensions), so
# this is wrapped in try/except at the call site, not here.
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


# ---- Flange builders ----------------------------------------------------
def _round_flange(radius_inner, z, extend_up, flange_t, flange_width, bolt_dia, bolt_count, overlap=5.0):
    """
    Flat annular ring flange for a round pipe end.
    radius_inner = pipe's OUTER radius (the flange's bore matches the
    pipe's outer surface exactly, so it sits flush - no cap, no
    obstruction to the bore that's already open inside the pipe wall).
    z            = the pipe end's z-coordinate (its open face).
    extend_up    = True -> flange body extends from z upward (Air Out,
                   welded on top of the pipe end);
                   False -> flange body extends from z downward (Dust
                   Outlet, welded below the pipe end).
    overlap      = extra mm the flange reaches BACK into the existing
                   pipe wall (past z, into the shell). Without this the
                   flange only touches the shell along a zero-volume
                   ring and Part.fuse() leaves it as a separate
                   disconnected solid (confirmed via BRepCheck: 4 solids
                   instead of 1) - a flange that's only "touching", not
                   welded, is not a valid mechanical part. The overlap
                   gives fuse() a real shared volume to merge on.
    """
    radius_outer = radius_inner + flange_width
    height = flange_t + overlap
    z0 = (z - overlap) if extend_up else (z - flange_t)
    direction = FreeCAD.Vector(0, 0, 1)

    outer_disc = Part.makeCylinder(radius_outer, height, FreeCAD.Vector(0, 0, z0), direction)
    inner_hole = Part.makeCylinder(
        radius_inner, height + 2, FreeCAD.Vector(0, 0, z0 - 1), direction
    )
    flange = outer_disc.cut(inner_hole)

    bolt_circle_r = (radius_inner + radius_outer) / 2.0
    for i in range(bolt_count):
        ang = 2.0 * math.pi * i / bolt_count
        bx = bolt_circle_r * math.cos(ang)
        by = bolt_circle_r * math.sin(ang)
        bolt = Part.makeCylinder(
            bolt_dia / 2.0, height + 2, FreeCAD.Vector(bx, by, z0 - 1), direction
        )
        flange = flange.cut(bolt)

    return flange


def _rect_flange(center_x, center_z, y, inner_w, inner_h, flange_t, flange_width, bolt_dia, overlap=5.0):
    """
    Flat rectangular plate flange for the inlet duct's far end.
    inner_w/inner_h = the duct's OUTER cross-section (its bore matches
    the duct's outer footprint exactly - same flush-fit logic as the
    round flange). y = the duct end's y-coordinate (its open face);
    the plate extends further out in -Y beyond it, and 'overlap' mm
    back INTO the duct wall so fuse() has real shared volume to merge
    on (same reasoning as _round_flange's overlap param).
    """
    outer_w = inner_w + 2.0 * flange_width
    outer_h = inner_h + 2.0 * flange_width
    depth = flange_t + overlap
    y0 = y - flange_t

    plate = Part.makeBox(
        outer_w, depth, outer_h,
        FreeCAD.Vector(center_x - outer_w / 2.0, y0, center_z - outer_h / 2.0),
    )
    hole = Part.makeBox(
        inner_w, depth + 2, inner_h,
        FreeCAD.Vector(center_x - inner_w / 2.0, y0 - 1, center_z - inner_h / 2.0),
    )
    flange = plate.cut(hole)

    margin = flange_width / 2.0
    for sx in (-1, 1):
        for sz in (-1, 1):
            bx = center_x + sx * (outer_w / 2.0 - margin)
            bz = center_z + sz * (outer_h / 2.0 - margin)
            bolt = Part.makeCylinder(
                bolt_dia / 2.0, depth + 2, FreeCAD.Vector(bx, y0 - 1, bz),
                FreeCAD.Vector(0, 1, 0),
            )
            flange = flange.cut(bolt)

    return flange


# ---- Geometry builder -------------------------------------------------
def _build_cyclone_shape(dims_mm: dict):
    """
    Builds the cyclone as a HOLLOW SHEET-METAL SHELL, matching the
    reference sketch: barrel + cone + a physical air-out pipe that
    protrudes above the roof + a tangential inlet duct, all fused into one
    solid envelope, then shelled to WallThicknessMm - leaving exactly 3
    openings: Air In (inlet duct end), Air Out (top of the vortex-finder
    pipe), and Dust Outlet (bottom tip of the cone). A bolted flange is
    then welded onto each of those 3 openings for real ducting connections.

    Returns (final_shape, sections) where sections is a dict of the
    individual PRE-FUSE solid pieces {"barrel", "cone", "inlet_duct",
    "air_out_pipe", "dust_outlet_pipe"} - used only for the new
    per-section 2D DXF export (generate_cyclone_cad). The combined
    final_shape / STEP / OBJ / single-view DXF+PDF path is completely
    unchanged from before.
    """
    barrel_d = dims_mm["BarrelDiameterMm"]
    barrel_h = dims_mm["BarrelHeightMm"]
    cone_h = dims_mm["ConeHeightMm"]
    exhaust_d = dims_mm["ExhaustDiaMm"]
    exhaust_l = dims_mm["ExhaustLengthMm"]
    bottom_outlet = dims_mm["BottomOutletMm"]
    inlet_h = dims_mm["InletHeightMm"]
    inlet_w = dims_mm["InletWidthMm"]

    # Sheet metal wall thickness - defaults to 3mm if not sent, so older
    # JSON payloads (without this field) keep working unchanged.
    wall_t = dims_mm.get("WallThicknessMm", 3.0)

    # Flange dimensions - all optional, all default so old payloads work.
    flange_t = dims_mm.get("FlangeThicknessMm", 10.0)       # plate thickness
    flange_width = dims_mm.get("FlangeWidthMm", 25.0)       # radial/lateral margin beyond the pipe/duct OD
    bolt_dia = dims_mm.get("FlangeBoltHoleDiaMm", 12.0)
    bolt_count_round = int(dims_mm.get("FlangeBoltCountRound", 4))
    flange_overlap = dims_mm.get("FlangeOverlapMm", 5.0)

    barrel_r = barrel_d / 2.0

    barrel = Part.makeCylinder(barrel_r, barrel_h)

    # Wide face (barrel_r) at z=0, touching the barrel; narrow tip
    # (bottom_outlet/2, the dust outlet) at z=-cone_h.
    cone = Part.makeCone(
        barrel_r,
        bottom_outlet / 2.0,
        cone_h,
        FreeCAD.Vector(0, 0, 0),
        FreeCAD.Vector(0, 0, -1),
    )

    body = barrel.fuse(cone)

    # Dust outlet pipe stub: short straight pipe below the cone tip.
    dust_stub_length = dims_mm.get("DustOutletPipeLengthMm", 100.0)
    dust_pipe_bottom_z = -cone_h - dust_stub_length
    dust_pipe = Part.makeCylinder(
        bottom_outlet / 2.0,
        dust_stub_length,
        FreeCAD.Vector(0, 0, dust_pipe_bottom_z),
        FreeCAD.Vector(0, 0, 1),
    )
    body = body.fuse(dust_pipe)

    # Air-out pipe (vortex finder): straight vertical pipe, protruding
    # above the roof.
    pipe_r = exhaust_d / 2.0
    exhaust_bottom_z = barrel_h - exhaust_l
    exhaust_top_z = barrel_h + 50
    exhaust_pipe = Part.makeCylinder(
        pipe_r,
        exhaust_top_z - exhaust_bottom_z,
        FreeCAD.Vector(0, 0, exhaust_bottom_z),
        FreeCAD.Vector(0, 0, 1),
    )
    body = body.fuse(exhaust_pipe)

    # Tangential inlet duct: offset so its outer wall touches the barrel
    # circle at (barrel_r, 0) - gas enters along the wall, not aimed at
    # the axis, which is what creates the vortex.
    duct_len = barrel_r * 1.5
    duct_far_y = -duct_len
    duct_z0 = barrel_h - inlet_h - 20
    duct_center_x = barrel_r - inlet_w / 2.0
    duct_center_z = duct_z0 + inlet_h / 2.0

    inlet_box = Part.makeBox(
        inlet_w, duct_len, inlet_h,
        FreeCAD.Vector(barrel_r - inlet_w, duct_far_y, duct_z0),
    )
    body = body.fuse(inlet_box)

    # ---- Shell into hollow sheet metal ----
    tol = 1.0  # mm matching tolerance for locating faces by position

    def _is_dust_outlet_face(f):
        com = f.CenterOfMass
        return abs(com.z - dust_pipe_bottom_z) < tol and (
            (com.x ** 2 + com.y ** 2) ** 0.5
        ) < (bottom_outlet / 2.0 + tol)

    def _is_air_out_face(f):
        com = f.CenterOfMass
        return abs(com.z - exhaust_top_z) < tol and (
            com.x ** 2 + com.y ** 2
        ) ** 0.5 < (pipe_r + tol)

    def _is_air_in_face(f):
        com = f.CenterOfMass
        return abs(com.y - duct_far_y) < tol

    open_faces = []
    for f in body.Faces:
        if _is_dust_outlet_face(f) or _is_air_out_face(f) or _is_air_in_face(f):
            open_faces.append(f)

    if open_faces:
        shell = body.makeThickness(open_faces, -wall_t, 1e-3)
    else:
        print(
            "WARNING: could not identify open faces for shelling - "
            "returning a SOLID body instead of hollow sheet metal.",
            file=sys.stderr,
        )
        shell = body

    # ---- Weld a bolted flange onto each of the 3 openings ----
    # Bores match each opening's OUTER dimension exactly (see the
    # _round_flange/_rect_flange docstrings) so the flange sits flush
    # around the opening without capping or narrowing it.
    air_out_flange = _round_flange(
        radius_inner=pipe_r, z=exhaust_top_z, extend_up=True,
        flange_t=flange_t, flange_width=flange_width,
        bolt_dia=bolt_dia, bolt_count=bolt_count_round, overlap=flange_overlap,
    )
    dust_outlet_flange = _round_flange(
        radius_inner=bottom_outlet / 2.0, z=dust_pipe_bottom_z, extend_up=False,
        flange_t=flange_t, flange_width=flange_width,
        bolt_dia=bolt_dia, bolt_count=bolt_count_round, overlap=flange_overlap,
    )
    inlet_flange = _rect_flange(
        center_x=duct_center_x, center_z=duct_center_z, y=duct_far_y,
        inner_w=inlet_w, inner_h=inlet_h,
        flange_t=flange_t, flange_width=flange_width, bolt_dia=bolt_dia, overlap=flange_overlap,
    )

    final_shape = shell.fuse(air_out_flange).fuse(dust_outlet_flange).fuse(inlet_flange)

    # ---- NEW: individual section shapes for per-section 2D DXFs -------
    # Solid (pre-shell) representative shapes - fine for 2D outline
    # drawings, which show the silhouette/outline, not wall thickness.
    # Each pipe/duct section includes its own flange fused on, since a
    # real fabrication drawing for that part would show the flange too.
    sections = {
        "barrel": barrel,
        "cone": cone,
        "air_out_pipe": exhaust_pipe.fuse(air_out_flange),
        "dust_outlet_pipe": dust_pipe.fuse(dust_outlet_flange),
        "inlet_duct": inlet_box.fuse(inlet_flange),
    }

    return final_shape, sections


# ---- TechDraw 2D export (ORIGINAL - unchanged) ---------------------------
def _export_techdraw(doc, shape_obj, output_dir: str, base_name: str):
    """Original single front-view export. Still produces the combined
    cyclone.pdf / cyclone.dxf as before.

    FIX: view.Scale/X/Y are now pinned to 1.0/0/0 so the exported DXF's
    coordinate frame is raw model mm at origin (0,0) - matching what
    add_dxf_dimensions_2D.py assumes when it inserts dimension TEXT at
    coordinates like (0, -cone_h - 100). Previously these were left at
    FreeCAD's auto-computed values, which put the geometry at a different
    scale/offset than the dimension text, making the text land off the
    visible drawing area."""
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
    view.Scale = 1.0   # FIX: was auto-computed by recompute()
    view.X = 0          # FIX: was auto-computed by recompute()
    view.Y = 0           # FIX: was auto-computed by recompute()
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


# ---- NEW: single-direction flattened 2D DXF (used by both new features) --
def _export_view_dxf(doc, shape, direction_vec, output_dir: str, filename: str):
    """
    Exports ONE flattened 2D orthographic view of `shape` (looking along
    direction_vec) as its own DXF file. Tries TechDraw's headless
    projection writer first (a true flattened 2D view, correct for real
    orthographic drawings); falls back to the old raw-3D-wireframe
    importDXF.export used elsewhere in this file if that's unavailable on
    this FreeCAD build - same defensive try/except pattern as the rest of
    this module, so this never hard-crashes CAD generation.

    FIX: view.Scale/X/Y pinned to 1.0/0/0 for the same reason as
    _export_techdraw above - keeps this DXF's coordinate frame at raw
    model mm / origin so dimension text inserted afterward lines up with
    the geometry."""
    dxf_path = os.path.join(output_dir, filename)
    tmp_obj = doc.addObject("Part::Feature", f"Tmp_{filename.replace('.', '_')}")
    tmp_obj.Shape = shape
    page = doc.addObject("TechDraw::DrawPage", f"Page_{filename.replace('.', '_')}")
    template = doc.addObject("TechDraw::DrawSVGTemplate", f"Tpl_{filename.replace('.', '_')}")
    resource_dir = FreeCAD.getResourceDir()
    template_path = os.path.join(
        resource_dir, "Mod", "TechDraw", "Templates", "A3_Landscape.svg"
    )
    if os.path.isfile(template_path):
        template.Template = template_path
    page.Template = template

    view = doc.addObject("TechDraw::DrawViewPart", f"View_{filename.replace('.', '_')}")
    view.Source = [tmp_obj]
    view.Direction = direction_vec
    view.Scale = 1.0   # FIX: was auto-computed by recompute()
    view.X = 0          # FIX: was auto-computed by recompute()
    view.Y = 0           # FIX: was auto-computed by recompute()
    page.addView(view)
    doc.recompute()

    try:
        # Headless flattened-projection DXF writer (no GUI needed, unlike
        # TechDrawGui's PDF export).
        TechDraw.writeDXFView(view, dxf_path)
    except Exception as e:
        print(
            f"WARNING: TechDraw.writeDXFView failed for {filename} "
            f"({e}) - falling back to raw 3D wireframe export.",
            file=sys.stderr,
        )
        try:
            import importDXF
            importDXF.export([shape], dxf_path)
        except Exception as e2:
            print(f"WARNING: fallback DXF export also failed for {filename}: {e2}", file=sys.stderr)
            dxf_path = None

    # Clean up the temp page/view/object so pages don't pile up in the
    # document across many section/view exports.
    for obj in (page, template, view, tmp_obj):
        try:
            doc.removeObject(obj.Name)
        except Exception:
            pass

    return dxf_path


def _export_multi_view_dxfs(doc, shape, output_dir: str, base_name: str) -> dict:
    """NEW: Front / Top / Side views of the WHOLE assembly, each its own DXF."""
    views = {
        "front": FreeCAD.Vector(0, -1, 0),
        "top": FreeCAD.Vector(0, 0, -1),
        "side": FreeCAD.Vector(1, 0, 0),
    }
    result = {}
    for name, direction in views.items():
        filename = f"{base_name}_{name}.dxf"
        result[name] = _export_view_dxf(doc, shape, direction, output_dir, filename)
    return result


def _export_section_dxfs(doc, sections: dict, output_dir: str) -> dict:
    """NEW: one DXF per physical section (front view each)."""
    result = {}
    for name, shape in sections.items():
        filename = f"{name}.dxf"
        result[name] = _export_view_dxf(
            doc, shape, FreeCAD.Vector(0, -1, 0), output_dir, filename
        )
    return result


# ---- 3D mesh export (OBJ, browser-viewable, headless-compatible) --------
def _export_obj_mesh(shape, output_dir: str, base_name: str):
    obj_path = os.path.join(output_dir, f"{base_name}.obj")
    try:
        doc_mesh = Mesh.Mesh(shape.tessellate(0.5))  # 0.5mm max deviation
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
    shape, sections = _build_cyclone_shape(dimensions_mm)

    shape_obj = doc.addObject("Part::Feature", "CycloneBody")
    shape_obj.Shape = shape
    doc.recompute()

    if not shape.isValid():
        print(
            "WARNING: generated solid failed shape.isValid() - this IS a "
            "real geometry defect (not an export artifact). Check "
            "dimension ratios (wall/flange thickness vs bore sizes).",
            file=sys.stderr,
        )

    base_name = "cyclone"
    step_path = os.path.join(output_dir, f"{base_name}.step")
    Part.export([shape_obj], step_path)

    obj_path = _export_obj_mesh(shape, output_dir, base_name)

    # ORIGINAL combined single-view export - unchanged (aside from the
    # view.Scale/X/Y fix inside _export_techdraw above).
    pdf_path, dxf_path = _export_techdraw(doc, shape_obj, output_dir, base_name)

    # NEW: multi-view (Front/Top/Side) of the whole assembly.
    view_dxf_paths = _export_multi_view_dxfs(doc, shape, output_dir, base_name)

    # NEW: one DXF per physical section.
    section_dxf_paths = _export_section_dxfs(doc, sections, output_dir)

    FreeCAD.closeDocument(doc.Name)

    return {
        "step_path": step_path,
        "pdf_path": pdf_path,
        "dxf_path": dxf_path,
        "obj_path": obj_path,
        "views": view_dxf_paths,        # {"front":..., "top":..., "side":...}
        "sections": section_dxf_paths,  # {"barrel":..., "cone":..., ...}
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
            "WallThicknessMm": 3,
            "FlangeThicknessMm": 10,
            "FlangeWidthMm": 25,
            "FlangeBoltHoleDiaMm": 12,
            "FlangeBoltCountRound": 4,
        }
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cad-exports", "test")

    result = generate_cyclone_cad(dims, out_dir)
    # Prefixed marker line so app.py can reliably find the result even if
    # FreeCAD prints extra diagnostic lines (Recompute..., transfer stats,
    # etc.) to stdout before this.
    print("RESULT_JSON:" + json.dumps(result))