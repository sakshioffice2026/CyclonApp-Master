"""
Intentional single-sheet mechanical drawing composer for the existing cyclone DXFs.

This module deliberately does NOT regenerate cyclone geometry.  It consumes the
existing flattened orthographic/detail DXFs produced by cad_generator.py and
lays them out on one A3 landscape sheet using an explicit mechanical-drawing
layout (third-angle projection), fixed scales, view alignment, centerlines,
hidden-line conventions, dimensions, notes, border, and title block.

The source DXFs remain independent fabrication/orthographic exports; this file
only performs the composition stage requested by the drawing workflow.
"""
from __future__ import annotations

import os
import math
from copy import copy
from typing import Iterable

import ezdxf
from ezdxf.entities import DXFGraphic
from ezdxf.math import Vec3


SHEET_W = 420.0  # A3 landscape, mm
SHEET_H = 297.0
BORDER = 10.0

ASSEMBLY_SCALE = 0.14  # 1:7.14; one common scale for the principal views
DETAIL_SCALE = 0.07    # 1:14.29; fabrication details

# Explicit, intentional view anchors.  These are not computed grid cells.
FRONT_ORIGIN = (95.0, 140.0)
TOP_ORIGIN = (95.0, 246.0)
SIDE_ORIGIN = (220.0, 140.0)

DETAIL_ORIGINS = {
    "barrel": (35.0, 31.0),
    "cone": (82.0, 31.0),
    "air_out_pipe": (129.0, 31.0),
    "dust_outlet_pipe": (176.0, 31.0),
    "inlet_duct": (218.0, 31.0),
}


# ---------------------------------------------------------------------------
# DXF import / placement
# ---------------------------------------------------------------------------
def _entity_bbox(entity: DXFGraphic):
    """Best-effort XY bbox for the already-flattened DXF entities."""
    pts = []
    t = entity.dxftype()
    try:
        if t == "LINE":
            pts.extend([entity.dxf.start, entity.dxf.end])
        elif t == "CIRCLE":
            c = Vec3(entity.dxf.center)
            r = float(entity.dxf.radius)
            pts.extend([Vec3(c.x-r, c.y-r), Vec3(c.x+r, c.y+r)])
        elif t == "ARC":
            c = Vec3(entity.dxf.center)
            r = float(entity.dxf.radius)
            # Conservative circular bbox; exact extrema are unnecessary for
            # sheet placement because TechDraw DXFs are predominantly lines.
            pts.extend([Vec3(c.x-r, c.y-r), Vec3(c.x+r, c.y+r)])
        elif t in {"LWPOLYLINE", "POLYLINE"}:
            for p in entity.get_points("xy") if t == "LWPOLYLINE" else entity.points():
                pts.append(Vec3(p[0], p[1]))
        elif t in {"ELLIPSE", "SPLINE"}:
            # ezdxf entities expose construction/flattening helpers inconsistently;
            # use the entity's virtual entities when available.
            try:
                for v in entity.virtual_entities():
                    b = _entity_bbox(v)
                    if b:
                        pts.extend([Vec3(b[0], b[1]), Vec3(b[2], b[3])])
            except Exception:
                pass
    except Exception:
        return None
    if not pts:
        return None
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _doc_bbox(doc: ezdxf.document.Drawing):
    boxes = []
    for entity in doc.modelspace():
        b = _entity_bbox(entity)
        if b:
            boxes.append(b)
    if not boxes:
        raise ValueError("Source DXF contains no drawable geometry")
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _ensure_layers(doc: ezdxf.document.Drawing):
    layers = {
        "OBJECT": 7,
        "CENTER": 7,
        "HIDDEN": 7,
        "DIMENSION": 7,
        "ANNOTATION": 7,
        "BORDER": 7,
        "TITLE": 7,
    }
    for name, color in layers.items():
        if name not in doc.layers:
            doc.layers.add(name, color=color)

    if "DASHED" not in doc.linetypes:
        doc.linetypes.add("DASHED", pattern=[0.5, 0.25, -0.25])
    if "CENTER" not in doc.linetypes:
        doc.linetypes.add("CENTER", pattern=[1.0, 0.5, -0.125, 0.125, -0.125])


def _copy_entity(target_msp, entity, scale: float, dx: float, dy: float, layer: str = "OBJECT"):
    """Copy a 2D source entity and apply uniform scale + translation."""
    try:
        new_entity = copy(entity)
        # Matrix transform is supported by DXF graphic entities and preserves
        # curves instead of exploding them into arbitrary polygons.
        from ezdxf.math import Matrix44
        new_entity.transform(Matrix44.chain(Matrix44.scale(scale, scale, 1), Matrix44.translate(dx, dy, 0)))
        new_entity.dxf.layer = layer
        target_msp.add_entity(new_entity)
    except Exception:
        # A few exotic TechDraw entities are not transformable.  Ignore them;
        # the core line/circle projection remains intact and fabrication source
        # DXFs are still preserved separately.
        return


def _import_view(sheet, path: str, origin: tuple[float, float], scale: float, layer: str = "OBJECT"):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Expected generated DXF was not found: {path}")
    src = ezdxf.readfile(path)
    # Explicit placement uses the source's own 0,0 coordinate as the drawing
    # origin; the source DXFs are generated directly from the same CAD model
    # coordinates. This keeps principal views intentionally aligned rather
    # than recentering each view independently.
    ox, oy = origin
    for entity in src.modelspace():
        _copy_entity(sheet.modelspace(), entity, scale, ox, oy, layer=layer)
    return _doc_bbox(src)


# ---------------------------------------------------------------------------
# Sheet graphics / annotation
# ---------------------------------------------------------------------------
def _add_text(msp, text: str, x: float, y: float, height: float = 3.0,
              layer: str = "ANNOTATION", align="LEFT"):
    t = msp.add_text(text, dxfattribs={"height": height, "layer": layer})
    if align == "CENTER":
        t.set_placement((x, y), align="MIDDLE_CENTER")
    elif align == "RIGHT":
        t.set_placement((x, y), align="MIDDLE_RIGHT")
    else:
        t.set_placement((x, y), align="LEFT")
    return t


def _line(msp, p1, p2, layer="OBJECT", linetype=None):
    attrs = {"layer": layer}
    if linetype:
        attrs["linetype"] = linetype
    return msp.add_line(p1, p2, dxfattribs=attrs)


def _centerline(msp, p1, p2):
    _line(msp, p1, p2, layer="CENTER", linetype="CENTER")


def _hidden_line(msp, p1, p2):
    _line(msp, p1, p2, layer="HIDDEN", linetype="DASHED")


def _dim_style(doc):
    name = "CYCLONE-SHEET-ISO"
    if name not in doc.dimstyles:
        ds = doc.dimstyles.new(name)
        ds.dxf.dimtxt = 3.0
        ds.dxf.dimasz = 2.2
        ds.dxf.dimexo = 1.2
        ds.dxf.dimexe = 1.2
        ds.dxf.dimtad = 1
        ds.dxf.dimdec = 0
        ds.dxf.dimclrd = 7
        ds.dxf.dimclre = 7
        ds.dxf.dimclrt = 7
    return name


def _hdim(msp, doc, x1, x2, y, text=None, text_height=3.0):
    dim = msp.add_linear_dim(
        base=((x1+x2)/2.0, y),
        p1=(x1, y), p2=(x2, y),
        angle=0, dimstyle=_dim_style(doc), text=text,
    )
    dim.render()
    # Keep text legible on a small A3 sheet.
    try:
        dim.dimension.dxf.text_height = text_height
    except Exception:
        pass


def _vdim(msp, doc, x, y1, y2, text=None):
    dim = msp.add_linear_dim(
        base=(x, (y1+y2)/2.0),
        p1=(x, y1), p2=(x, y2),
        angle=90, dimstyle=_dim_style(doc), text=text,
    )
    dim.render()


def _draw_view_title(msp, title, x, y):
    _add_text(msp, title, x, y, height=3.2, align="CENTER")


def _draw_border_and_title_block(doc, revision_id):
    msp = doc.modelspace()
    _line(msp, (BORDER, BORDER), (SHEET_W-BORDER, BORDER), layer="BORDER")
    _line(msp, (SHEET_W-BORDER, BORDER), (SHEET_W-BORDER, SHEET_H-BORDER), layer="BORDER")
    _line(msp, (SHEET_W-BORDER, SHEET_H-BORDER), (BORDER, SHEET_H-BORDER), layer="BORDER")
    _line(msp, (BORDER, SHEET_H-BORDER), (BORDER, BORDER), layer="BORDER")
    # Inner drawing-frame line.
    inset = 3.0
    _line(msp, (BORDER+inset, BORDER+inset), (SHEET_W-BORDER-inset, BORDER+inset), layer="BORDER")
    _line(msp, (SHEET_W-BORDER-inset, BORDER+inset), (SHEET_W-BORDER-inset, SHEET_H-BORDER-inset), layer="BORDER")
    _line(msp, (SHEET_W-BORDER-inset, SHEET_H-BORDER-inset), (BORDER+inset, SHEET_H-BORDER-inset), layer="BORDER")
    _line(msp, (BORDER+inset, SHEET_H-BORDER-inset), (BORDER+inset, BORDER+inset), layer="BORDER")

    # Title block: deliberately fixed to the lower-right corner, as on an A3
    # production drawing, rather than treated as another view cell.
    x0, y0, x1, y1 = 265.0, 10.0, 410.0, 45.0
    for x in (x0, x1):
        _line(msp, (x, y0), (x, y1), layer="TITLE")
    for y in (y0, y1):
        _line(msp, (x0, y), (x1, y), layer="TITLE")
    _line(msp, (x0, 24.0), (x1, 24.0), layer="TITLE")
    _line(msp, (330.0, y0), (330.0, 24.0), layer="TITLE")
    _line(msp, (380.0, y0), (380.0, 24.0), layer="TITLE")
    _add_text(msp, "CYCLONE ASSEMBLY", 337.5, 35.0, 4.5, layer="TITLE", align="CENTER")
    _add_text(msp, "MECHANICAL ENGINEERING DRAWING", 337.5, 29.0, 2.7, layer="TITLE", align="CENTER")
    _add_text(msp, "DRAWING NO.", 272.0, 20.0, 2.2, layer="TITLE")
    _add_text(msp, "CYCL-ASSY-001", 272.0, 14.5, 2.8, layer="TITLE")
    _add_text(msp, "REV", 336.0, 20.0, 2.2, layer="TITLE", align="CENTER")
    _add_text(msp, str(revision_id), 336.0, 14.5, 2.8, layer="TITLE", align="CENTER")
    _add_text(msp, "SHEET", 355.0, 20.0, 2.2, layer="TITLE", align="CENTER")
    _add_text(msp, "1 / 1", 355.0, 14.5, 2.8, layer="TITLE", align="CENTER")
    _add_text(msp, "UNITS", 395.0, 20.0, 2.2, layer="TITLE", align="CENTER")
    _add_text(msp, "mm", 395.0, 14.5, 2.8, layer="TITLE", align="CENTER")

    _add_text(msp, "THIRD ANGLE PROJECTION", 278.0, 51.0, 3.0, align="LEFT")
    _add_text(msp, "ALL DIMENSIONS IN mm", 278.0, 56.0, 3.0, align="LEFT")
    _add_text(msp, "PRINCIPAL VIEWS 1:7.14   |   DETAIL VIEWS 1:14.29", 278.0, 61.0, 3.0, align="LEFT")


def _draw_assembly_dimensions(doc, front_bbox, top_bbox, side_bbox, dims):
    msp = doc.modelspace()
    # The DXF projection is intentionally placed at its CAD origin.  Bboxes
    # are used only to place dimension lines outside the visible geometry.
    fx0, fy0, fx1, fy1 = front_bbox
    sx0, sy0, sx1, sy1 = side_bbox
    tx0, ty0, tx1, ty1 = top_bbox

    # Front: barrel diameter, barrel height, cone height, overall height,
    # exhaust diameter, dust outlet diameter and pipe/inlet heights.
    ox, oy = FRONT_ORIGIN
    sc = ASSEMBLY_SCALE
    bx0, bx1 = ox + fx0*sc, ox + fx1*sc
    by0, by1 = oy + fy0*sc, oy + fy1*sc
    _hdim(msp, doc, bx0, bx1, by0-10.0, text=f"{dims['BarrelDiameterMm']:.0f}")
    _vdim(msp, doc, bx1+10.0, oy + (-dims['ConeHeightMm']-dims.get('DustOutletPipeLengthMm',100))*sc, by1+8.0,
          text=f"{dims['ConeHeightMm'] + dims.get('DustOutletPipeLengthMm',100) + dims['BarrelHeightMm'] + 50:.0f}")
    _vdim(msp, doc, bx1+20.0, oy, oy+dims['BarrelHeightMm']*sc,
          text=f"{dims['BarrelHeightMm']:.0f}")
    _vdim(msp, doc, bx1+30.0, oy-dims['ConeHeightMm']*sc, oy,
          text=f"{dims['ConeHeightMm']:.0f}")
    _hdim(msp, doc, ox-dims['ExhaustDiaMm']*sc/2, ox+dims['ExhaustDiaMm']*sc/2,
          by1+10.0, text=f"{dims['ExhaustDiaMm']:.0f}")
    _hdim(msp, doc, ox-dims['BottomOutletMm']*sc/2, ox+dims['BottomOutletMm']*sc/2,
          by0-20.0, text=f"{dims['BottomOutletMm']:.0f}")

    # Top: inlet projection length and width, plus centerlines.
    tox, toy = TOP_ORIGIN
    _hdim(msp, doc, tox+tx0*sc-20*sc, tox+tx1*sc+20*sc, toy+ty0*sc-9.0,
          text=f"{dims['BarrelDiameterMm']:.0f}")
    _centerline(msp, (tox+tx0*sc-6, toy), (tox+tx1*sc+6, toy))
    _centerline(msp, (tox, toy+ty0*sc-5), (tox, toy+ty1*sc+5))

    # Side: overall height and inlet run; hidden vortex-finder wall line.
    sox, soy = SIDE_ORIGIN
    _vdim(msp, doc, sox+sx1*sc+10.0,
          soy-dims['ConeHeightMm']*sc-dims.get('DustOutletPipeLengthMm',100)*sc,
          soy+dims['BarrelHeightMm']*sc+50*sc,
          text=f"{dims['ConeHeightMm'] + dims.get('DustOutletPipeLengthMm',100) + dims['BarrelHeightMm'] + 50:.0f}")
    _hdim(msp, doc, sox+sx0*sc, sox+sx1*sc, soy-dims['ConeHeightMm']*sc-dims.get('DustOutletPipeLengthMm',100)*sc-10,
          text=f"{max(0.0, dims['BarrelDiameterMm']*0.75):.0f}")
    _centerline(msp, (sox, soy-dims['ConeHeightMm']*sc-10), (sox, soy+dims['BarrelHeightMm']*sc+10))
    # Vortex finder hidden line in the side elevation.
    vf_x = sox + (dims['ExhaustDiaMm']/2.0)*sc
    _hidden_line(msp, (vf_x, soy+dims['BarrelHeightMm']*sc-dims['ExhaustLengthMm']*sc),
                 (vf_x, soy+dims['BarrelHeightMm']*sc+50*sc))


def _draw_detail_dimensions(doc, name, bbox, dims):
    msp = doc.modelspace()
    ox, oy = DETAIL_ORIGINS[name]
    sc = DETAIL_SCALE
    x0, y0, x1, y1 = bbox
    ax0, ax1 = ox+x0*sc, ox+x1*sc
    ay0, ay1 = oy+y0*sc, oy+y1*sc
    gap = 4.0

    if name == "barrel":
        _hdim(msp, doc, ax0, ax1, ay0-gap, text=f"{dims['BarrelDiameterMm']:.0f}")
        _vdim(msp, doc, ax1+gap, ay0, ay1, text=f"{dims['BarrelHeightMm']:.0f}")
        _centerline(msp, (ox, ay0-2), (ox, ay1+2))
    elif name == "cone":
        _hdim(msp, doc, ax0, ax1, ay1+gap, text=f"{dims['BarrelDiameterMm']:.0f}")
        _hdim(msp, doc, ox-dims['BottomOutletMm']*sc/2, ox+dims['BottomOutletMm']*sc/2,
              ay0-gap, text=f"{dims['BottomOutletMm']:.0f}")
        _vdim(msp, doc, ax1+gap, ay0, ay1, text=f"{dims['ConeHeightMm']:.0f}")
        _centerline(msp, (ox, ay0-2), (ox, ay1+2))
    elif name == "air_out_pipe":
        _hdim(msp, doc, ax0, ax1, ay0-gap, text=f"{dims['ExhaustDiaMm']:.0f}")
        _vdim(msp, doc, ax1+gap, ay0, ay1, text=f"{dims['ExhaustLengthMm'] + 50:.0f}")
        _centerline(msp, (ox, ay0-2), (ox, ay1+2))
    elif name == "dust_outlet_pipe":
        _hdim(msp, doc, ax0, ax1, ay0-gap, text=f"{dims['BottomOutletMm']:.0f}")
        _vdim(msp, doc, ax1+gap, ay0, ay1, text=f"{dims.get('DustOutletPipeLengthMm',100):.0f}")
        _centerline(msp, (ox, ay0-2), (ox, ay1+2))
    elif name == "inlet_duct":
        _hdim(msp, doc, ax0, ax1, ay0-gap, text=f"{dims['InletWidthMm']:.0f}")
        _vdim(msp, doc, ax1+gap, ay0, ay1, text=f"{dims['InletHeightMm']:.0f}")
        _centerline(msp, (ax0-2, (ay0+ay1)/2), (ax1+2, (ay0+ay1)/2))


def compose_engineering_sheet(view_paths: dict, section_paths: dict, output_path: str,
                              dims: dict, revision_id: int | str) -> str:
    """Compose existing generated DXFs into one intentional A3 drawing sheet."""
    doc = ezdxf.new("R2018", setup=True)
    doc.header["$INSUNITS"] = 4  # millimetres
    _ensure_layers(doc)
    msp = doc.modelspace()

    # Principal views — third-angle: TOP above FRONT, RIGHT SIDE to the right.
    front_bbox = _import_view(doc, view_paths["front"], FRONT_ORIGIN, ASSEMBLY_SCALE)
    top_bbox = _import_view(doc, view_paths["top"], TOP_ORIGIN, ASSEMBLY_SCALE)
    side_bbox = _import_view(doc, view_paths["side"], SIDE_ORIGIN, ASSEMBLY_SCALE)

    _draw_view_title(msp, "FRONT ELEVATION", FRONT_ORIGIN[0], 33.0)
    _draw_view_title(msp, "TOP / PLAN", TOP_ORIGIN[0], 284.0)
    _draw_view_title(msp, "RIGHT SIDE ELEVATION", SIDE_ORIGIN[0], 33.0)

    # Detail/fabrication views are explicitly grouped in a detail band, not
    # intermixed with the principal orthographic arrangement.
    detail_titles = {
        "barrel": "DETAIL A — BARREL",
        "cone": "DETAIL B — CONE",
        "air_out_pipe": "DETAIL C — AIR-OUT PIPE",
        "dust_outlet_pipe": "DETAIL D — DUST-OUTLET PIPE",
        "inlet_duct": "DETAIL E — INLET DUCT",
    }
    detail_bboxes = {}
    for name in ("barrel", "cone", "air_out_pipe", "dust_outlet_pipe", "inlet_duct"):
        detail_bboxes[name] = _import_view(doc, section_paths[name], DETAIL_ORIGINS[name], DETAIL_SCALE)
        _draw_view_title(msp, detail_titles[name], DETAIL_ORIGINS[name][0], 5.0)
        _draw_detail_dimensions(doc, name, detail_bboxes[name], dims)

    # Principal-view center marks/centerlines and engineering dimensions.
    _centerline(msp, (FRONT_ORIGIN[0], 65.0), (FRONT_ORIGIN[0], 225.0))
    _centerline(msp, (SIDE_ORIGIN[0], 65.0), (SIDE_ORIGIN[0], 225.0))
    _draw_assembly_dimensions(doc, front_bbox, top_bbox, side_bbox, dims)

    # A conventional note identifying hidden-line and centerline conventions.
    _add_text(msp, "HIDDEN LINES — DASHED", 20.0, 274.0, 2.5)
    _add_text(msp, "CENTERLINES — CHAIN", 20.0, 268.0, 2.5)

    _draw_border_and_title_block(doc, revision_id)

    # Layer lineweights / visual hierarchy.
    doc.layers.get("OBJECT").dxf.lineweight = 35
    doc.layers.get("CENTER").dxf.lineweight = 13
    doc.layers.get("HIDDEN").dxf.lineweight = 13
    doc.layers.get("DIMENSION").dxf.lineweight = 13
    doc.layers.get("ANNOTATION").dxf.lineweight = 18
    doc.layers.get("BORDER").dxf.lineweight = 50
    doc.layers.get("TITLE").dxf.lineweight = 35

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc.saveas(output_path)
    return output_path
