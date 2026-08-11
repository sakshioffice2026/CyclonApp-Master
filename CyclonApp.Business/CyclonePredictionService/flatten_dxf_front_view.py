"""
flatten_dxf_front_view.py
--------------------------
Converts the raw 3D wireframe DXF (cad_generator.py's importDXF.export
output) into a genuinely FLAT 2D front-view drawing.

WHY THIS EXISTS: cyclone.dxf (result["dxf_path"]) is exported via
importDXF.export([shape_obj], dxf_path) — that is a straight dump of the
solid's 3D edges, so its ARC/LINE/CIRCLE/SPLINE entities carry real,
varying Z (and Y) coordinates matching the actual 3D model (confirmed by
inspection: Z values like -710, -600, 450 spread across the whole
model). It LOOKS like a front view when FreeCAD's camera happens to be
aimed along -Y, but the file itself is not 2D — every entity still has
full 3D coordinates.

This module projects every entity onto the XZ plane (drops Y — the same
"front view looking along -Y" convention add_dxf_dimensions.py already
assumes: model X -> DXF X, model Z -> DXF Y) and writes a brand-new DXF
containing only flat 2D geometry (LWPOLYLINE/LINE/TEXT, all at a single
plane). Curves (ARC/CIRCLE/SPLINE) are flattened to polyline
approximations via ezdxf's Path.flattening(), which works regardless of
each curve's original 3D orientation — simple axis-drop projection alone
is only exact for entities already parallel to the view plane, so
flattening-to-polyline is the robust, general-purpose way to handle
whatever cad_generator.py's solid produces.

Layers are preserved (by name) so the DIM_TEXT layer added afterward by
add_dxf_dimensions.py still lines up with existing layer conventions.

This module only projects/flattens geometry — it does not add any
dimensions itself. Call add_engineering_dimensions_2d (from
add_dxf_dimensions.py) on the OUTPUT of this module, not on the raw
wireframe file, since dimension coordinates are written assuming a flat
XZ-plane frame that only this flattened file actually has.
"""
from __future__ import annotations
import ezdxf
from ezdxf import path as ezdxf_path

BODY_LAYER = "CycloneBody"


def _add_synthetic_silhouette_lines(out_msp, dims_mm: dict, layer: str = BODY_LAYER) -> int:
    """ROOT CAUSE (found by checking cad_generator.py): barrel, cone,
    dust-outlet pipe and exhaust pipe are all Part.makeCylinder /
    Part.makeCone revolve solids. A revolve solid has only ONE straight
    seam edge in its B-Rep (at its single start angle) - it does NOT
    have two edges for "left wall" and "right wall". So
    importDXF.export's raw edge dump only ever contains ONE side of each
    of these round features; the opposite side has no 3D edge to export
    at all, which is why it renders as empty space in the flattened
    front view even though the dimension coordinates are correct.

    This adds the missing tangent/silhouette lines explicitly, computed
    straight from dims_mm (the same numbers cad_generator.py used to
    build the solids), so the front view shows a complete outline on
    both sides for every round feature. Purely additive: does not touch
    or alter any projected geometry already written by
    flatten_to_front_view_2d.
    """
    barrel_r = dims_mm["BarrelDiameterMm"] / 2.0
    barrel_h = dims_mm["BarrelHeightMm"]
    cone_h = dims_mm["ConeHeightMm"]
    bottom_outlet_r = dims_mm["BottomOutletMm"] / 2.0
    exhaust_r = dims_mm["ExhaustDiaMm"] / 2.0
    exhaust_l = dims_mm["ExhaustLengthMm"]
    dust_stub_len = dims_mm.get("DustOutletPipeLengthMm", 100.0)

    exhaust_bottom_z = barrel_h - exhaust_l
    exhaust_top_z = barrel_h + 50  # matches cad_generator.py's exhaust_top_z
    dust_pipe_bottom_z = -cone_h - dust_stub_len

    attribs = {"layer": layer}
    added = 0
    for side in (1, -1):
        # Barrel side wall
        out_msp.add_line((side * barrel_r, 0), (side * barrel_r, barrel_h), dxfattribs=attribs)
        # Cone slant wall
        out_msp.add_line((side * barrel_r, 0), (side * bottom_outlet_r, -cone_h), dxfattribs=attribs)
        # Dust outlet pipe wall
        out_msp.add_line((side * bottom_outlet_r, -cone_h), (side * bottom_outlet_r, dust_pipe_bottom_z), dxfattribs=attribs)
        # Exhaust pipe wall
        out_msp.add_line((side * exhaust_r, exhaust_bottom_z), (side * exhaust_r, exhaust_top_z), dxfattribs=attribs)
        added += 4
    return added


def flatten_to_front_view_2d(dxf_path: str, out_path: str | None = None, sagitta: float = 0.5, dims_mm: dict | None = None) -> str:
    """Reads dxf_path (raw 3D wireframe), projects every entity onto the
    XZ plane (model X -> DXF X, model Z -> DXF Y, model Y dropped), and
    writes the result to out_path (defaults to a NEW file next to
    dxf_path, so the original 3D wireframe is left untouched for any
    other consumer that still wants full 3D data).

    sagitta = max deviation (mm) allowed when approximating curves
    (ARC/CIRCLE/SPLINE) as straight polyline segments during flattening —
    smaller = more segments = smoother curve, at the cost of a larger
    file. 0.5mm is plenty tight for an engineering reference drawing at
    typical cyclone sizes (hundreds of mm).

    dims_mm = OPTIONAL. Same dims dict passed to cad_generator.py /
    add_engineering_dimensions_2d. When given, adds the barrel/cone/
    dust-pipe/exhaust-pipe silhouette lines that the raw wireframe is
    missing (see _add_synthetic_silhouette_lines). When omitted (None,
    the default), behavior is 100% unchanged from before.
    """
    src_doc = ezdxf.readfile(dxf_path)
    src_msp = src_doc.modelspace()

    out_doc = ezdxf.new(src_doc.dxfversion)
    out_msp = out_doc.modelspace()

    # Carry over layer names/colors so anything added later (e.g. the
    # DIM_TEXT layer from add_dxf_dimensions.py) composes cleanly with
    # whatever layers the solid's edges originally used.
    for layer in src_doc.layers:
        name = layer.dxf.name
        if name not in out_doc.layers:
            out_doc.layers.add(name, color=layer.color)

    converted = 0
    skipped = 0

    for e in src_msp:
        t = e.dxftype()
        layer_attribs = {"layer": e.dxf.layer}

        try:
            if t == "TEXT":
                ins = e.dxf.insert
                out_msp.add_text(
                    e.dxf.text,
                    dxfattribs={
                        **layer_attribs,
                        "height": e.dxf.height,
                        "insert": (ins.x, ins.z),
                        "rotation": e.dxf.get("rotation", 0),
                    },
                )
                converted += 1
                continue

            # ARC / CIRCLE / LINE / SPLINE / LWPOLYLINE / POLYLINE all
            # convert to a generic Path via make_path, then flatten to a
            # sequence of 3D points we can drop Y from — this works
            # uniformly regardless of each entity's original plane/normal,
            # unlike a plain "read start/end and drop Y" approach which
            # only holds up for entities already parallel to the view.
            p = ezdxf_path.make_path(e)
            points_3d = list(p.flattening(sagitta))
            if len(points_3d) < 2:
                skipped += 1
                continue

            points_2d = [(v.x, v.z) for v in points_3d]
            is_closed = (
                abs(points_2d[0][0] - points_2d[-1][0]) < 1e-6
                and abs(points_2d[0][1] - points_2d[-1][1]) < 1e-6
            )
            out_msp.add_lwpolyline(
                points_2d, close=is_closed, dxfattribs=layer_attribs
            )
            converted += 1

        except Exception:
            # A single unconvertible entity must not fail the whole
            # export — skip it and keep going, same defensive posture as
            # the rest of this service's CAD pipeline.
            skipped += 1
            continue

    silhouette_added = 0
    if dims_mm is not None:
        silhouette_added = _add_synthetic_silhouette_lines(out_msp, dims_mm)

    out_path = out_path or dxf_path
    out_doc.saveas(out_path)
    print(
        f"[flatten_dxf] Flattened {dxf_path} -> {out_path}: "
        f"{converted} entities converted, {skipped} skipped, "
        f"{silhouette_added} synthetic silhouette lines added."
    )
    return out_path