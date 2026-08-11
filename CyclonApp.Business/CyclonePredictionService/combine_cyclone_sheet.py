"""
combine_cyclone_sheet.py
-------------------------
Merges the ALREADY-GENERATED per-part and per-view 2D DXFs into ONE
single DXF "sheet", arranged in a grid, each part labelled with its
name. This is purely additive - it does not change cad_generator.py,
app.py, flatten_dxf_front_view.py, or add_dxf_dimensions.py, and it does
not touch/replace any of their existing output files. It only READS DXFs
that already exist on disk and WRITES one new combined file.

LAYER SEPARATION (mechanical drawing convention): every part gets its
OWN uniquely-named layer - PART_<NAME> for body geometry, DIM_<NAME> for
that part's dimension lines/text. Source files all reuse the same
generic layer names ("CycloneBody", "DIM_TEXT"), so importing them
as-is would silently merge every part onto one shared layer, making it
impossible to isolate/hide an individual part in AutoCAD/FreeCAD. Every
imported entity is re-layered by PART NAME here specifically to avoid
that.

WHERE THE INPUT FILES COME FROM (see cad_generator.py):
  result = generate_cyclone_cad(...)
  result["views"]    = {"front": ..., "top": ..., "side": ...}
  result["sections"] = {"barrel": ..., "cone": ..., "air_out_pipe": ...,
                         "dust_outlet_pipe": ..., "inlet_duct": ...}
These 8 files already exist after generation - app.py just never reads
`views`/`sections` today. This module reads them and lays them out on
one page.

WHY A SIMPLE TRANSLATE-INTO-A-GRID APPROACH:
Each per-view/per-section DXF comes from TechDraw.writeDXFView, and (per
app.py's own comments) each one uses its OWN page-relative coordinate
frame - they are NOT guaranteed to share a common origin with each
other. That's fine for this purpose: we only need each part's shape to
be internally correct (it is), so for every file we compute its own
bounding box and shift ONLY that file's entities so its bottom-left
corner lands at the next open grid cell. No assumption about any shared
origin between files is made or needed.

USAGE:
    combine_all_into_one_sheet(
        {**result["views"], **result["sections"]},
        out_path="/path/to/cyclone_all_parts.dxf",
    )
"""
from __future__ import annotations
import os
import re
import ezdxf
from ezdxf.addons import importer as ezdxf_importer

LABEL_LAYER = "PART_LABELS"

_LAYER_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 30, 40, 90, 140]  # cycled per part, distinct on screen


def _safe_layer_token(name: str) -> str:
    """DXF layer names can't contain <>/\\":;?*|=`  - sanitize the part
    name (e.g. "air_out_pipe" -> "AIR_OUT_PIPE") into a clean token."""
    token = re.sub(r"[^A-Za-z0-9_\-]", "_", name).strip("_").upper()
    return token or "PART"


def _bbox_of_entities(entities):
    xs, ys = [], []
    for e in entities:
        t = e.dxftype()
        if t == "LINE":
            xs += [e.dxf.start.x, e.dxf.end.x]
            ys += [e.dxf.start.y, e.dxf.end.y]
        elif t == "LWPOLYLINE":
            for p in e.get_points():
                xs.append(p[0]); ys.append(p[1])
        elif t in ("CIRCLE",):
            c, r = e.dxf.center, e.dxf.radius
            xs += [c.x - r, c.x + r]; ys += [c.y - r, c.y + r]
        elif t in ("ARC",):
            c, r = e.dxf.center, e.dxf.radius
            xs += [c.x - r, c.x + r]; ys += [c.y - r, c.y + r]
        elif t == "TEXT":
            ins = e.dxf.insert
            xs.append(ins.x); ys.append(ins.y)
        elif t == "SPLINE":
            for cp in e.control_points:
                xs.append(cp[0]); ys.append(cp[1])
    if not xs:
        return None
    return min(xs), max(xs), min(ys), max(ys)


def _ensure_layer(doc, name: str, color: int):
    if name not in doc.layers:
        doc.layers.add(name, color=color)


def combine_all_into_one_sheet(
    view_paths: dict,
    out_path: str,
    gap_x: float = 120.0,
    gap_y: float = 150.0,
    max_row_width: float = 2500.0,
    label_height: float = 30.0,
) -> str:
    """Reads every {name: dxf_path} in view_paths (skips missing/None
    entries so it degrades gracefully if a given view failed upstream),
    imports each one's geometry into a new combined document ONTO A
    LAYER NAMED FOR THAT PART (PART_<NAME> for body geometry, DIM_<NAME>
    for that part's own dimension lines/text - keeps every part
    independently toggleable, per standard mechanical-drawing practice),
    translates each part into its own grid cell so nothing overlaps, and
    writes one labelled DXF sheet to out_path. Returns out_path.
    """
    out_doc = ezdxf.new("R2010")
    out_msp = out_doc.modelspace()
    _ensure_layer(out_doc, LABEL_LAYER, color=7)

    cursor_x, cursor_y = 0.0, 0.0
    row_height = 0.0
    placed = 0
    skipped = []

    for idx, (name, path) in enumerate(view_paths.items()):
        if not path or not os.path.exists(path):
            skipped.append(name)
            continue

        token = _safe_layer_token(name)
        body_layer = f"PART_{token}"
        dim_layer = f"DIM_{token}"
        part_color = _LAYER_COLORS[idx % len(_LAYER_COLORS)]
        _ensure_layer(out_doc, body_layer, color=part_color)
        _ensure_layer(out_doc, dim_layer, color=part_color)

        src_doc = ezdxf.readfile(path)

        # Import this file's entities (+ needed layers/linetypes/styles)
        # into out_doc, tracking exactly which new entities arrived so we
        # can bbox + move + re-layer only THIS part, not the whole sheet
        # so far.
        handles_before = {e.dxf.handle for e in out_msp}
        imp = ezdxf_importer.Importer(src_doc, out_doc)
        imp.import_modelspace()
        imp.finalize()
        new_entities = [e for e in out_msp if e.dxf.handle not in handles_before]

        if not new_entities:
            skipped.append(name)
            continue

        # Re-layer by part: anything that came in on a dimension-style
        # source layer (DIM_TEXT, or already named DIM_*) goes onto this
        # part's DIM_<NAME> layer; everything else (body outline) goes
        # onto PART_<NAME>. This is what actually separates the parts -
        # without it every part's geometry would sit on the source
        # files' shared "CycloneBody"/"DIM_TEXT" layer names and merge
        # together indistinguishably once imported into one document.
        for e in new_entities:
            src_layer = e.dxf.layer
            e.dxf.layer = dim_layer if src_layer.upper().startswith("DIM") else body_layer

        box = _bbox_of_entities(new_entities)
        if box is None:
            skipped.append(name)
            continue
        xmin, xmax, ymin, ymax = box
        width = xmax - xmin
        height = ymax - ymin

        # Wrap to a new row once the current row gets too wide.
        if cursor_x + width > max_row_width and cursor_x > 0:
            cursor_x = 0.0
            cursor_y += row_height + gap_y
            row_height = 0.0

        dx = cursor_x - xmin
        dy = cursor_y - ymin
        for e in new_entities:
            e.translate(dx, dy, 0)

        # Label above the part - own layer too, so labels can be hidden
        # as a group independently of any part's geometry.
        out_msp.add_text(
            name,
            dxfattribs={
                "layer": LABEL_LAYER,
                "height": label_height,
                "insert": (cursor_x, cursor_y + height + 20),
            },
        )

        cursor_x += width + gap_x
        row_height = max(row_height, height)
        placed += 1

    out_doc.saveas(out_path)
    print(
        f"[combine_cyclone_sheet] {placed} parts placed, each on its own "
        f"PART_*/DIM_* layer -> {out_path}"
        + (f" (skipped, no file: {skipped})" if skipped else "")
    )
    return out_path


if __name__ == "__main__":
    demo_paths = {
        "cyclone_front": "cyclone_2d.dxf",
        "cyclone_top": "cyclone_top.dxf",
        "cyclone_side": "cyclone_side.dxf",
        "barrel": "barrel.dxf",
        "cone": "cone.dxf",
        "air_out_pipe": "air_out_pipe.dxf",
        "dust_outlet_pipe": "dust_outlet_pipe.dxf",
        "inlet_duct": "inlet_duct.dxf",
    }
    combine_all_into_one_sheet(demo_paths, "cyclone_all_parts.dxf")