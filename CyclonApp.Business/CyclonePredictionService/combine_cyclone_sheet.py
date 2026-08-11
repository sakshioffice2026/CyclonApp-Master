"""
combine_cyclone_sheet.py
------------------------
Consolidate the ALREADY-GENERATED 2D DXFs into one engineering drawing.

This module is deliberately a COMPOSITION stage only.

SOURCE DATA IS SACRED
---------------------
The input DXFs are read as already-generated drawing documents. Their
geometry, dimensions, scale, layers, linetypes, text properties and other
entity properties are not regenerated or re-authored here. The only
coordinate operation applied to imported source entities is a translation
so the complete source drawing can be positioned at an intentional sheet
anchor. No source entity is scaled, rotated, mirrored, exploded, flattened,
redrawn or re-dimensioned.

The old implementation used a bounding-box cursor/grid algorithm. That is
not an engineering drawing layout, so it has been removed. The new layout
uses explicit drawing anchors for the three principal assembly views and
for the five fabrication details. A source drawing's bounding box is used
ONLY to determine its local visual centre/extent for alignment to its
predefined anchor; it is never used to choose the next position or a grid
cell.

The three assembly views are laid out as a related orthographic group:

    TOP / PLAN
        |
        | projected alignment
        v
    FRONT ELEVATION -------- RIGHT-SIDE ELEVATION

The five component DXFs are placed in a dedicated DETAIL area below the
principal views. They remain independent source drawings; only sheet-level
labels, border, projection symbol and title block are added.

Usage:
    combine_all_into_one_sheet(
        {**result["views"], **result["sections"]},
        out_path="/path/to/cyclone_all_parts.dxf",
    )
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable

import ezdxf
from ezdxf.addons import importer as ezdxf_importer


SHEET_LAYER = "SHEET"
SHEET_TEXT_LAYER = "SHEET_TEXT"
SHEET_BORDER_LAYER = "SHEET_BORDER"

# A drawing-sheet coordinate space. These are intentionally fixed layout
# coordinates, not calculated packing/grid cells. The size is generous
# enough for the existing TechDraw DXF page-relative coordinates while
# keeping the layout readable in CAD viewers.
SHEET_WIDTH = 3600.0
SHEET_HEIGHT = 2500.0
BORDER_MARGIN = 60.0
TITLE_BLOCK_W = 720.0
TITLE_BLOCK_H = 210.0
DETAIL_BAND_H = 600.0

# Explicit anchors. These are sheet coordinates chosen for the drawing,
# not destinations computed from the previous view's width/height.
# The anchor is the intended visual centre of each imported source drawing.
PRINCIPAL_ANCHORS = {
    "top": (1120.0, 1780.0),
    "front": (1120.0, 1110.0),
    "side": (2650.0, 1110.0),
}

DETAIL_ANCHORS = {
    "barrel": (420.0, 430.0),
    "cone": (1060.0, 430.0),
    "air_out_pipe": (1700.0, 430.0),
    "dust_outlet_pipe": (2340.0, 430.0),
    "inlet_duct": (2980.0, 430.0),
}

LABELS = {
    "top": "TOP / PLAN VIEW",
    "front": "FRONT ELEVATION",
    "side": "RIGHT-SIDE ELEVATION",
    "barrel": "DETAIL A - BARREL",
    "cone": "DETAIL B - CONE",
    "air_out_pipe": "DETAIL C - AIR-OUT PIPE / VORTEX FINDER",
    "dust_outlet_pipe": "DETAIL D - DUST-OUTLET PIPE",
    "inlet_duct": "DETAIL E - INLET DUCT",
}


@dataclass(frozen=True)
class SourceBounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)


# ---------------------------------------------------------------------------
# Source-DXF inspection
# ---------------------------------------------------------------------------

def _entity_points(entity) -> Iterable[tuple[float, float]]:
    """Yield representative XY points without changing the source entity."""
    t = entity.dxftype()
    d = entity.dxf

    if t == "LINE":
        yield d.start.x, d.start.y
        yield d.end.x, d.end.y
    elif t == "LWPOLYLINE":
        for p in entity.get_points():
            yield p[0], p[1]
    elif t == "POLYLINE":
        for v in entity.vertices:
            p = v.dxf.location
            yield p.x, p.y
    elif t in {"CIRCLE", "ARC"}:
        c = d.center
        r = d.radius
        # Full radial bounds are intentionally used only for determining
        # the source drawing's local centre. No geometry is altered.
        yield c.x - r, c.y - r
        yield c.x + r, c.y + r
    elif t == "ELLIPSE":
        c = d.center
        # Conservative source extent for alignment only.
        mx = abs(d.major_axis.x) + abs(d.major_axis.y) + abs(d.major_axis.z)
        my = mx * max(abs(d.ratio), 1e-9)
        yield c.x - mx, c.y - my
        yield c.x + mx, c.y + my
    elif t == "SPLINE":
        for p in entity.control_points:
            yield p[0], p[1]
    elif t in {"TEXT", "MTEXT", "INSERT"}:
        p = d.insert
        yield p.x, p.y


def _source_bounds(entities) -> SourceBounds | None:
    points = []
    for entity in entities:
        points.extend(_entity_points(entity))
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return SourceBounds(min(xs), max(xs), min(ys), max(ys))


# ---------------------------------------------------------------------------
# Sheet primitives
# ---------------------------------------------------------------------------

def _ensure_layer(doc, name: str, color: int = 7, linetype: str = "CONTINUOUS"):
    if name not in doc.layers:
        doc.layers.add(name, color=color, linetype=linetype)


def _add_text(msp, text: str, position: tuple[float, float], height: float,
              layer: str = SHEET_TEXT_LAYER, rotation: float = 0.0):
    return msp.add_text(
        text,
        dxfattribs={
            "layer": layer,
            "height": height,
            "rotation": rotation,
            "insert": position,
        },
    )


def _add_line(msp, p1, p2, layer=SHEET_BORDER_LAYER, color=None):
    attribs = {"layer": layer}
    if color is not None:
        attribs["color"] = color
    return msp.add_line(p1, p2, dxfattribs=attribs)


def _add_rect(msp, xmin, ymin, xmax, ymax, layer=SHEET_BORDER_LAYER, color=None):
    _add_line(msp, (xmin, ymin), (xmax, ymin), layer, color)
    _add_line(msp, (xmax, ymin), (xmax, ymax), layer, color)
    _add_line(msp, (xmax, ymax), (xmin, ymax), layer, color)
    _add_line(msp, (xmin, ymax), (xmin, ymin), layer, color)


def _add_center_marks(msp, center, size=35.0, layer=SHEET_BORDER_LAYER):
    x, y = center
    _add_line(msp, (x - size, y), (x + size, y), layer)
    _add_line(msp, (x, y - size), (x, y + size), layer)


def _draw_sheet_frame(doc, msp):
    _ensure_layer(doc, SHEET_BORDER_LAYER, color=7)
    _ensure_layer(doc, SHEET_TEXT_LAYER, color=7)
    _ensure_layer(doc, SHEET_LAYER, color=7)

    _add_rect(
        msp,
        BORDER_MARGIN,
        BORDER_MARGIN,
        SHEET_WIDTH - BORDER_MARGIN,
        SHEET_HEIGHT - BORDER_MARGIN,
    )
    _add_rect(
        msp,
        BORDER_MARGIN + 20,
        BORDER_MARGIN + 20,
        SHEET_WIDTH - BORDER_MARGIN - 20,
        SHEET_HEIGHT - BORDER_MARGIN - 20,
    )

    # Dedicated detail band separator.
    detail_top = BORDER_MARGIN + DETAIL_BAND_H
    _add_line(
        msp,
        (BORDER_MARGIN + 20, detail_top),
        (SHEET_WIDTH - BORDER_MARGIN - 20, detail_top),
    )
    _add_text(msp, "FABRICATION / COMPONENT DETAILS", (110.0, detail_top + 30.0), 28.0)

    # Title block in the lower-right corner.
    x0 = SHEET_WIDTH - BORDER_MARGIN - TITLE_BLOCK_W
    y0 = BORDER_MARGIN
    x1 = SHEET_WIDTH - BORDER_MARGIN
    y1 = BORDER_MARGIN + TITLE_BLOCK_H
    _add_rect(msp, x0, y0, x1, y1)
    _add_line(msp, (x0, y0 + 70), (x1, y0 + 70))
    _add_line(msp, (x0 + 360, y0), (x0 + 360, y0 + 70))
    _add_line(msp, (x0 + 510, y0), (x0 + 510, y0 + 70))
    _add_line(msp, (x0, y0 + 140), (x1, y0 + 140))

    _add_text(msp, "CYCLONE ASSEMBLY", (x0 + 20, y0 + 162), 34.0)
    _add_text(msp, "2D MECHANICAL ENGINEERING DRAWING", (x0 + 20, y0 + 115), 22.0)
    _add_text(msp, "UNITS: mm", (x0 + 20, y0 + 42), 20.0)
    _add_text(msp, "SHEET: 1 / 1", (x0 + 380, y0 + 42), 20.0)
    _add_text(msp, "REV: -", (x0 + 530, y0 + 42), 20.0)
    _add_text(msp, "ORTHOGRAPHIC VIEWS: 1:1 SOURCE SCALE", (x0 + 20, y0 + 88), 18.0)

    # Projection notation: simple third-angle symbol made from a cone-like
    # profile and circle. It is sheet annotation only and does not touch
    # imported source entities.
    px, py = 250.0, SHEET_HEIGHT - 145.0
    _add_text(msp, "THIRD-ANGLE PROJECTION", (px - 120.0, py + 80.0), 20.0)
    _add_line(msp, (px - 45, py - 35), (px + 5, py + 15), SHEET_LAYER)
    _add_line(msp, (px - 45, py + 35), (px + 5, py - 15), SHEET_LAYER)
    msp.add_circle((px + 55, py), 38, dxfattribs={"layer": SHEET_LAYER})


def _translate_entities(entities, dx: float, dy: float):
    """Translate source entities only; no scale/rotation/property changes."""
    for entity in entities:
        try:
            entity.translate(dx, dy, 0.0)
        except Exception:
            # ezdxf entities that do not implement translate are left in
            # their source form rather than being exploded/redrawn.
            # Normal TechDraw DXFs use LINE/LWPOLYLINE/TEXT and translate.
            raise RuntimeError(
                f"Entity type {entity.dxftype()} cannot be translated without "
                "re-authoring it; refusing to modify source geometry."
            )


def _import_source(doc, path: str):
    """Import a source DXF while retaining its original entity attributes."""
    src = ezdxf.readfile(path)
    out_msp = doc.modelspace()
    before = {e.dxf.handle for e in out_msp}

    importer = ezdxf_importer.Importer(src, doc)
    importer.import_modelspace()
    importer.finalize()

    imported = [e for e in out_msp if e.dxf.handle not in before]
    if not imported:
        raise ValueError(f"Source DXF contains no modelspace entities: {path}")
    return imported


def _place_source(doc, name: str, path: str, anchor: tuple[float, float]):
    if not path or not os.path.isfile(path):
        return False

    entities = _import_source(doc, path)
    bounds = _source_bounds(entities)
    if bounds is None:
        raise ValueError(f"Unable to determine placement anchor for {name}: {path}")

    cx, cy = bounds.center
    dx = anchor[0] - cx
    dy = anchor[1] - cy
    _translate_entities(entities, dx, dy)
    return True


def _add_view_labels(msp):
    for key, anchor in PRINCIPAL_ANCHORS.items():
        _add_text(msp, LABELS[key], (anchor[0] - 250.0, anchor[1] + 420.0), 28.0)

    for key, anchor in DETAIL_ANCHORS.items():
        _add_text(msp, LABELS[key], (anchor[0] - 250.0, anchor[1] + 250.0), 22.0)


def combine_all_into_one_sheet(view_paths: dict, out_path: str) -> str:
    """Merge existing 2D DXFs into one intentionally laid-out sheet.

    `view_paths` is expected to contain the existing assembly views and
    component detail DXFs returned by generate_cyclone_cad(). Missing files
    are skipped and reported; no source DXF is overwritten.

    IMPORTANT: source entities are imported with their original properties
    and are only translated to the explicit sheet anchors. No scale,
    rotation, layer, linetype, dimension, geometry or entity-property edits
    are performed on source entities.
    """
    out_doc = ezdxf.new("R2010")
    _draw_sheet_frame(out_doc, out_doc.modelspace())

    placed = []
    skipped = []
    failures = []

    # Principal orthographic group. The anchors are fixed drawing datums,
    # deliberately chosen to establish projection relationships.
    for key in ("top", "front", "side"):
        path = view_paths.get(key)
        if not path or not os.path.isfile(path):
            skipped.append(key)
            continue
        try:
            if _place_source(out_doc, key, path, PRINCIPAL_ANCHORS[key]):
                placed.append(key)
        except Exception as exc:
            failures.append(f"{key}: {exc}")

    # Fabrication details are a separate presentation zone. Their positions
    # are explicit anchors, not automatically packed based on source size.
    for key in ("barrel", "cone", "air_out_pipe", "dust_outlet_pipe", "inlet_duct"):
        path = view_paths.get(key)
        if not path or not os.path.isfile(path):
            skipped.append(key)
            continue
        try:
            if _place_source(out_doc, key, path, DETAIL_ANCHORS[key]):
                placed.append(key)
        except Exception as exc:
            failures.append(f"{key}: {exc}")

    msp = out_doc.modelspace()
    _add_view_labels(msp)

    # Centre marks are sheet-level registration aids. They do not alter the
    # source DXF geometry.
    for anchor in PRINCIPAL_ANCHORS.values():
        _add_center_marks(msp, anchor, size=25.0)

    out_doc.saveas(out_path)

    if failures:
        raise RuntimeError("DXF consolidation failed: " + "; ".join(failures))

    print(
        f"[combine_cyclone_sheet] consolidated {len(placed)} source DXFs -> {out_path}"
        + (f"; skipped={skipped}" if skipped else "")
    )
    return out_path


if __name__ == "__main__":
    demo_paths = {
        "front": "cyclone_front.dxf",
        "top": "cyclone_top.dxf",
        "side": "cyclone_side.dxf",
        "barrel": "barrel.dxf",
        "cone": "cone.dxf",
        "air_out_pipe": "air_out_pipe.dxf",
        "dust_outlet_pipe": "dust_outlet_pipe.dxf",
        "inlet_duct": "inlet_duct.dxf",
    }
    combine_all_into_one_sheet(demo_paths, "cyclone_all_parts.dxf")
