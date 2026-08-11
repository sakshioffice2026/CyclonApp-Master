"""
combine_cyclone_sheet.py

Consolidate the ALREADY-GENERATED 2D DXFs into one engineering drawing
sheet (border, title block, view labels, center marks) using a
process-flow / real-bounding-box layout instead of fixed anchors.

SOURCE DATA IS SACRED
---------------------
The input DXFs are read as already-generated drawing documents. Their
geometry, dimensions, scale, layers, linetypes, text properties and other
entity properties are not regenerated or re-authored here. The only
coordinate operation applied to imported source entities is a translation
so the complete source drawing can be positioned at its computed sheet
location. No source entity is scaled, rotated, mirrored, exploded,
flattened, redrawn or re-dimensioned.

Layout logic (from process-flow revision):
    Column 1: air_out_pipe above front
    Column 2: top -> inlet_duct -> barrel -> cone -> dust_outlet_pipe
    Column 3: side (standalone reference view)

Each column stacks top-to-bottom using each source's REAL bounding box,
so spacing adapts to the cyclone's actual parametric size. Columns are
placed left-to-right the same way. Sheet frame (border, title block,
detail-band label, projection symbol, view labels, center marks) is
drawn around this computed layout.

Expected source drawings:
    cyclone_front.dxf / cyclone_top.dxf / cyclone_side.dxf
    barrel.dxf, cone.dxf, air_out_pipe.dxf, dust_outlet_pipe.dxf, inlet_duct.dxf

Usage:
    combine_all_into_one_sheet(
        {**result["views"], **result["sections"]},
        out_path="/path/to/cyclone_all_parts.dxf",
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

import ezdxf
from ezdxf.addons import importer as ezdxf_importer


SHEET_LAYER = "SHEET"
SHEET_TEXT_LAYER = "SHEET_TEXT"
SHEET_BORDER_LAYER = "SHEET_BORDER"

# Fixed sheet-frame styling constants (from HEAD). The sheet extents
# themselves are now computed from the layout, not hardcoded, since the
# process-flow layout is bounding-box driven and must not clip content.
BORDER_MARGIN = 60.0
TITLE_BLOCK_W = 720.0
TITLE_BLOCK_H = 210.0
DETAIL_BAND_H = 600.0
SHEET_PADDING = 300.0  # extra room around the computed layout, inside the border

LABELS = {
    "front": "FRONT ELEVATION",
    "top": "TOP / PLAN VIEW",
    "side": "RIGHT-SIDE ELEVATION",
    "barrel": "DETAIL A - BARREL",
    "cone": "DETAIL B - CONE",
    "air_out_pipe": "DETAIL C - AIR-OUT PIPE / VORTEX FINDER",
    "dust_outlet_pipe": "DETAIL D - DUST-OUTLET PIPE",
    "inlet_duct": "DETAIL E - INLET DUCT",
}

# Process-flow column layout (from fa968fa revision).
COLUMN_LAYOUT: tuple[tuple[str, ...], ...] = (
    ("air_out_pipe", "front"),
    ("top", "inlet_duct", "barrel", "cone", "dust_outlet_pipe"),
    ("side",),
)

MARGIN_MM = 150.0

ALIASES: dict[str, tuple[str, ...]] = {
    "front": ("front", "cyclone_front"),
    "top": ("top", "cyclone_top"),
    "side": ("side", "cyclone_side"),
    "barrel": ("barrel",),
    "cone": ("cone",),
    "air_out_pipe": ("air_out_pipe",),
    "dust_outlet_pipe": ("dust_outlet_pipe",),
    "inlet_duct": ("inlet_duct",),
}

SOURCE_ORDER = (
    "front", "top", "side", "barrel", "cone",
    "air_out_pipe", "dust_outlet_pipe", "inlet_duct",
)

# Keys treated as "principal views" for center-mark annotation.
PRINCIPAL_KEYS = ("front", "top", "side")


@dataclass(frozen=True)
class Bounds:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin


# ---------------------------------------------------------------------------
# Source-DXF inspection (real bounding box, fa968fa)
# ---------------------------------------------------------------------------

def _entity_bounds(entity) -> Bounds | None:
    t = entity.dxftype()
    d = entity.dxf

    if t == "LINE":
        return Bounds(
            min(d.start.x, d.end.x), min(d.start.y, d.end.y),
            max(d.start.x, d.end.x), max(d.start.y, d.end.y),
        )
    if t == "LWPOLYLINE":
        points = [(p[0], p[1]) for p in entity.get_points()]
        if not points:
            return None
        xs = [p[0] for p in points]; ys = [p[1] for p in points]
        return Bounds(min(xs), min(ys), max(xs), max(ys))
    if t == "POLYLINE":
        points = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
        if not points:
            return None
        xs = [p[0] for p in points]; ys = [p[1] for p in points]
        return Bounds(min(xs), min(ys), max(xs), max(ys))
    if t in {"CIRCLE", "ARC"}:
        c = d.center; r = d.radius
        return Bounds(c.x - r, c.y - r, c.x + r, c.y + r)
    if t == "ELLIPSE":
        c = d.center
        major = d.major_axis
        ratio = max(abs(d.ratio), 1e-12)
        rx = (major.x * major.x + major.y * major.y) ** 0.5
        ry = rx * ratio
        return Bounds(c.x - rx, c.y - ry, c.x + rx, c.y + ry)
    if t == "SPLINE":
        points = list(entity.control_points)
        if not points:
            return None
        xs = [p[0] for p in points]; ys = [p[1] for p in points]
        return Bounds(min(xs), min(ys), max(xs), max(ys))
    if t in {"TEXT", "MTEXT", "INSERT"}:
        if not hasattr(d, "insert"):
            return None
        p = d.insert
        return Bounds(p.x, p.y, p.x, p.y)
    return None


def _modelspace_bounds(doc) -> Bounds | None:
    try:
        from ezdxf import bbox
        box = bbox.extents(doc.modelspace(), fast=False)
        if not box.has_data:
            return None
        return Bounds(box.extmin.x, box.extmin.y, box.extmax.x, box.extmax.y)
    except Exception:
        bounds = []
        for entity in doc.modelspace():
            item = _entity_bounds(entity)
            if item is not None:
                bounds.append(item)
        if not bounds:
            return None
        return Bounds(
            min(b.xmin for b in bounds), min(b.ymin for b in bounds),
            max(b.xmax for b in bounds), max(b.ymax for b in bounds),
        )


def _resolve_path(view_paths: Mapping[str, str], key: str) -> str | None:
    for alias in ALIASES[key]:
        value = view_paths.get(alias)
        if value:
            return os.fspath(value)
    return None


def _read_source_bounds(source_path: str) -> Bounds:
    source_doc = ezdxf.readfile(source_path)
    bounds = _modelspace_bounds(source_doc)
    if bounds is None:
        raise ValueError(f"Source DXF contains no measurable modelspace geometry: {source_path}")
    return bounds


def _compute_layout_centers(
    source_bounds: Mapping[str, Bounds],
) -> tuple[dict[str, tuple[float, float]], Bounds]:
    """Process-flow column layout, sized from each source's real bbox.
    Returns (centers, overall_layout_bounds) so the caller can size the sheet."""
    centers: dict[str, tuple[float, float]] = {}
    col_x_cursor = 0.0

    for column in COLUMN_LAYOUT:
        keys_present = [k for k in column if k in source_bounds]
        if not keys_present:
            continue

        col_width = max(source_bounds[k].width for k in keys_present)
        col_center_x = col_x_cursor + col_width / 2.0

        cursor_top_y = 0.0
        for k in keys_present:
            h = source_bounds[k].height
            centers[k] = (col_center_x, cursor_top_y - h / 2.0)
            cursor_top_y -= h + MARGIN_MM

        col_x_cursor += col_width + MARGIN_MM

    if not centers:
        raise ValueError("No placeable sources for layout.")

    min_x = min(centers[k][0] - source_bounds[k].width / 2.0 for k in centers)
    min_y = min(centers[k][1] - source_bounds[k].height / 2.0 for k in centers)
    shift_x = -min_x if min_x < 0 else 0.0
    shift_y = -min_y if min_y < 0 else 0.0
    if shift_x or shift_y:
        centers = {k: (x + shift_x, y + shift_y) for k, (x, y) in centers.items()}

    max_x = max(centers[k][0] + source_bounds[k].width / 2.0 for k in centers)
    max_y = max(centers[k][1] + source_bounds[k].height / 2.0 for k in centers)
    layout_bounds = Bounds(0.0, 0.0, max_x, max_y)

    return centers, layout_bounds


# ---------------------------------------------------------------------------
# Sheet primitives (HEAD)
# ---------------------------------------------------------------------------

def _ensure_layer(doc, name: str, color: int = 7, linetype: str = "CONTINUOUS"):
    if name not in doc.layers:
        doc.layers.add(name, color=color, linetype=linetype)


def _add_text(msp, text: str, position: tuple[float, float], height: float,
              layer: str = SHEET_TEXT_LAYER, rotation: float = 0.0):
    return msp.add_text(
        text,
        dxfattribs={"layer": layer, "height": height, "rotation": rotation, "insert": position},
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


def _draw_sheet_frame(doc, msp, sheet_width: float, sheet_height: float, detail_top: float):
    """Sheet border + title block + detail-band separator + projection symbol.
    Extents are computed from the actual layout (not hardcoded)."""
    _ensure_layer(doc, SHEET_BORDER_LAYER, color=7)
    _ensure_layer(doc, SHEET_TEXT_LAYER, color=7)
    _ensure_layer(doc, SHEET_LAYER, color=7)

    _add_rect(msp, BORDER_MARGIN, BORDER_MARGIN, sheet_width - BORDER_MARGIN, sheet_height - BORDER_MARGIN)
    _add_rect(msp, BORDER_MARGIN + 20, BORDER_MARGIN + 20, sheet_width - BORDER_MARGIN - 20, sheet_height - BORDER_MARGIN - 20)

    _add_line(msp, (BORDER_MARGIN + 20, detail_top), (sheet_width - BORDER_MARGIN - 20, detail_top))
    _add_text(msp, "FABRICATION / COMPONENT DETAILS", (110.0, detail_top + 30.0), 28.0)

    x0 = sheet_width - BORDER_MARGIN - TITLE_BLOCK_W
    y0 = BORDER_MARGIN
    x1 = sheet_width - BORDER_MARGIN
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

    px, py = 250.0, sheet_height - 145.0
    _add_text(msp, "THIRD-ANGLE PROJECTION", (px - 120.0, py + 80.0), 20.0)
    _add_line(msp, (px - 45, py - 35), (px + 5, py + 15), SHEET_LAYER)
    _add_line(msp, (px - 45, py + 35), (px + 5, py - 15), SHEET_LAYER)
    msp.add_circle((px + 55, py), 38, dxfattribs={"layer": SHEET_LAYER})


def _translate_entities(entities, dx: float, dy: float):
    for entity in entities:
        try:
            entity.translate(dx, dy, 0.0)
        except Exception as exc:
            raise RuntimeError(
                f"Entity type {entity.dxftype()} cannot be translated without "
                "re-authoring it; refusing to modify source geometry."
            ) from exc


def _import_source(doc, path: str):
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


def _place_source(doc, path: str, target_center: tuple[float, float], source_bounds: Bounds):
    entities = _import_source(doc, path)
    cx, cy = source_bounds.center
    dx = target_center[0] - cx
    dy = target_center[1] - cy
    _translate_entities(entities, dx, dy)


def _add_view_labels(msp, centers: dict[str, tuple[float, float]], source_bounds: dict[str, Bounds]):
    for key, (cx, cy) in centers.items():
        label_y = cy + source_bounds[key].height / 2.0 + 60.0
        _add_text(msp, LABELS[key], (cx - 250.0, label_y), 28.0 if key in PRINCIPAL_KEYS else 22.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def combine_all_into_one_sheet(view_paths: Mapping[str, str], out_path: str) -> str:
    """Merge existing 2D DXFs into one engineering sheet.

    Placement uses the process-flow, real-bounding-box column layout:
        air_out_pipe / front | top -> inlet_duct -> barrel -> cone -> dust_outlet_pipe | side
    The sheet frame (border, title block, detail-band label, projection
    symbol, view labels, center marks) is sized around this computed layout.
    No source entity is scaled, rotated, mirrored, exploded, flattened,
    redrawn or re-dimensioned - only translated into place.
    """
    if not out_path:
        raise ValueError("out_path is required.")
    out_path = os.fspath(out_path)

    resolved: dict[str, str] = {}
    skipped: list[str] = []
    for key in SOURCE_ORDER:
        source_path = _resolve_path(view_paths, key)
        if not source_path or not os.path.isfile(source_path):
            skipped.append(key)
            continue
        resolved[key] = source_path

    if not resolved:
        raise ValueError("No source DXFs were available for consolidation.")

    failures: list[str] = []
    source_bounds: dict[str, Bounds] = {}
    for key, source_path in resolved.items():
        try:
            source_bounds[key] = _read_source_bounds(source_path)
        except Exception as exc:
            failures.append(f"{key}: {exc}")

    if failures:
        raise RuntimeError("DXF consolidation failed. No output was saved.\n" + "\n".join(failures))

    centers, layout_bounds = _compute_layout_centers(source_bounds)

    # Sheet extents computed from the actual layout, not hardcoded.
    detail_top = BORDER_MARGIN + DETAIL_BAND_H
    sheet_width = max(layout_bounds.width, 0.0) + 2 * BORDER_MARGIN + 2 * SHEET_PADDING + TITLE_BLOCK_W
    sheet_height = max(layout_bounds.height, 0.0) + 2 * BORDER_MARGIN + 2 * SHEET_PADDING + detail_top

    # Shift the computed layout inward so it sits inside the border/padding.
    layout_origin_x = BORDER_MARGIN + SHEET_PADDING
    layout_origin_y = BORDER_MARGIN + SHEET_PADDING
    centers = {k: (x + layout_origin_x, y + layout_origin_y) for k, (x, y) in centers.items()}

    out_doc = ezdxf.new("R2010")
    msp = out_doc.modelspace()

    placed: list[str] = []
    for key in SOURCE_ORDER:
        if key not in resolved:
            continue
        try:
            _place_source(out_doc, resolved[key], centers[key], source_bounds[key])
            placed.append(key)
        except Exception as exc:
            failures.append(f"{key}: {exc}")

    if failures:
        raise RuntimeError("DXF consolidation failed. No output was saved.\n" + "\n".join(failures))

    _draw_sheet_frame(out_doc, msp, sheet_width, sheet_height, detail_top)
    _add_view_labels(msp, centers, source_bounds)

    for key in PRINCIPAL_KEYS:
        if key in centers:
            _add_center_marks(msp, centers[key], size=25.0)

    out_doc.saveas(out_path)

    print(
        f"[combine_cyclone_sheet] consolidated {len(placed)} source DXFs -> {out_path}"
        + (f"; skipped={skipped}" if skipped else "")
    )

    return out_path


__all__ = ["combine_all_into_one_sheet"]


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