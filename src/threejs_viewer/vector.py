"""
Vector entities: exact DXF-style curves for :meth:`ViewerClient.add_vector_entities`.

Each function returns one entity as a plain dict in the wire and glTF
``extras`` schema (camelCase keys), validated so a bad value fails here
rather than as a skipped entity in the browser.

Repeated geometry goes in a block (a named list of entities) placed with
:func:`insert`, DXF's ``BLOCK``/``INSERT`` pair. The viewer keeps the exact
definition and re-tessellates it as the camera moves.

Coordinates are in the object's local frame (the viewer's Z-up scene unless
a parent or transform is given). Angles are radians. Planar entities take a
``normal`` (default +Z) and measure angles from the in-plane x-axis that
DXF's arbitrary axis algorithm derives from it, so the values mean what
they mean in a DXF file.

Every function accepts the same style keywords, each falling back to the
entity's layer when omitted (DXF ``BYLAYER``):

- ``layer``: layer name (default ``"0"``).
- ``color``: ``0xRRGGBB``.
- ``linetype``: a linetype name such as ``"DASHED"``.
- ``line_width``: width in screen pixels.
- ``linetype_scale``: multiplies the linetype pattern lengths.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "point",
    "points",
    "line",
    "polyline",
    "circle",
    "arc",
    "ellipse",
    "spline",
    "insert",
    "check_blocks",
]


def _vec3(name: str, v) -> list[float]:
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be 3 finite numbers")
    return a.tolist()


def _flat_points(name: str, v, minimum: int) -> list[float]:
    a = np.asarray(v, dtype=float)
    if a.ndim == 1 and a.size % 3 == 0:
        a = a.reshape(-1, 3)
    if a.ndim != 2 or a.shape[1] != 3 or a.shape[0] < minimum:
        raise ValueError(f"{name} must be an (N, 3) array with N >= {minimum}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a.reshape(-1).tolist()


def _finite(name: str, v) -> float:
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"{name} must be finite")
    return f


def _normal(entity: dict, normal) -> None:
    if normal is None:
        return
    n = _vec3("normal", normal)
    if not any(n):
        raise ValueError("normal must be non-zero")
    entity["normal"] = n


def _style(
    entity: dict,
    layer: Optional[str],
    color: Optional[int],
    linetype: Optional[str],
    line_width: Optional[float],
    linetype_scale: Optional[float],
) -> dict:
    if layer is not None:
        entity["layer"] = str(layer)
    if color is not None:
        entity["color"] = int(color)
    if linetype is not None:
        entity["linetype"] = str(linetype)
    if line_width is not None:
        if not _finite("line_width", line_width) > 0:
            raise ValueError("line_width must be positive")
        entity["lineWidth"] = float(line_width)
    if linetype_scale is not None:
        if not _finite("linetype_scale", linetype_scale) > 0:
            raise ValueError("linetype_scale must be positive")
        entity["linetypeScale"] = float(linetype_scale)
    return entity


def point(
    position,
    *,
    layer: Optional[str] = None,
    color: Optional[int] = None,
    line_width: Optional[float] = None,
) -> dict:
    """A single point (DXF ``POINT``). Drawn ``2 * line_width`` pixels wide, at least 4."""
    return _style(
        {"type": "point", "position": _vec3("position", position)},
        layer,
        color,
        None,
        line_width,
        None,
    )


def points(
    positions,
    *,
    layer: Optional[str] = None,
    color: Optional[int] = None,
    line_width: Optional[float] = None,
) -> dict:
    """Many points sharing one style; ``positions`` is ``(N, 3)``."""
    return _style(
        {"type": "points", "positions": _flat_points("positions", positions, 1)},
        layer,
        color,
        None,
        line_width,
        None,
    )


def line(
    start,
    end,
    *,
    layer: Optional[str] = None,
    color: Optional[int] = None,
    linetype: Optional[str] = None,
    line_width: Optional[float] = None,
    linetype_scale: Optional[float] = None,
) -> dict:
    """A straight segment (DXF ``LINE``)."""
    return _style(
        {"type": "line", "start": _vec3("start", start), "end": _vec3("end", end)},
        layer,
        color,
        linetype,
        line_width,
        linetype_scale,
    )


def polyline(
    vertices,
    *,
    bulges: Optional[Sequence[float]] = None,
    closed: bool = False,
    normal=None,
    layer: Optional[str] = None,
    color: Optional[int] = None,
    linetype: Optional[str] = None,
    line_width: Optional[float] = None,
    linetype_scale: Optional[float] = None,
) -> dict:
    """A polyline through ``vertices`` ``(N, 3)`` (DXF ``LWPOLYLINE``).

    ``bulges[i]`` turns the edge from vertex ``i`` to ``i + 1`` (to vertex 0
    for the closing edge) into an arc: ``tan(angle / 4)``, positive
    counter-clockwise about ``normal``. ``1`` is a half circle. Bulged
    polylines must lie in a plane perpendicular to ``normal``.
    """
    entity = {"type": "polyline", "vertices": _flat_points("vertices", vertices, 2)}
    n_vertices = len(entity["vertices"]) // 3
    if bulges is not None:
        b = [_finite("bulges", x) for x in np.asarray(bulges, dtype=float).reshape(-1)]
        edges = n_vertices if closed else n_vertices - 1
        if len(b) < edges:
            raise ValueError(f"bulges needs one value per edge ({edges})")
        entity["bulges"] = b[:edges]
    if closed:
        entity["closed"] = True
    _normal(entity, normal)
    return _style(entity, layer, color, linetype, line_width, linetype_scale)


def circle(
    center,
    radius: float,
    *,
    normal=None,
    layer: Optional[str] = None,
    color: Optional[int] = None,
    linetype: Optional[str] = None,
    line_width: Optional[float] = None,
    linetype_scale: Optional[float] = None,
) -> dict:
    """A full circle in the plane through ``center`` perpendicular to ``normal``."""
    if not _finite("radius", radius) > 0:
        raise ValueError("radius must be positive")
    entity = {
        "type": "circle",
        "center": _vec3("center", center),
        "radius": float(radius),
    }
    _normal(entity, normal)
    return _style(entity, layer, color, linetype, line_width, linetype_scale)


def arc(
    center,
    radius: float,
    start_angle: float,
    end_angle: float,
    *,
    normal=None,
    layer: Optional[str] = None,
    color: Optional[int] = None,
    linetype: Optional[str] = None,
    line_width: Optional[float] = None,
    linetype_scale: Optional[float] = None,
) -> dict:
    """A circular arc running counter-clockwise (about ``normal``) from
    ``start_angle`` to ``end_angle``, in radians (DXF ``ARC``)."""
    if not _finite("radius", radius) > 0:
        raise ValueError("radius must be positive")
    entity = {
        "type": "arc",
        "center": _vec3("center", center),
        "radius": float(radius),
        "startAngle": _finite("start_angle", start_angle),
        "endAngle": _finite("end_angle", end_angle),
    }
    _normal(entity, normal)
    return _style(entity, layer, color, linetype, line_width, linetype_scale)


def ellipse(
    center,
    major_axis,
    ratio: float,
    *,
    start_param: float = 0.0,
    end_param: float = 2 * math.pi,
    normal=None,
    layer: Optional[str] = None,
    color: Optional[int] = None,
    linetype: Optional[str] = None,
    line_width: Optional[float] = None,
    linetype_scale: Optional[float] = None,
) -> dict:
    """An ellipse or elliptical arc (DXF ``ELLIPSE``).

    ``major_axis`` is the vector from the centre to the end of the major
    axis; the minor axis is ``ratio`` times as long. The curve is
    ``center + major·cos t + minor·sin t`` for ``t`` from ``start_param`` to
    ``end_param``, with ``minor = ratio · (normal × major_axis)``.
    """
    r = _finite("ratio", ratio)
    if not 0 < r <= 1:
        raise ValueError("ratio must be in (0, 1]")
    entity = {
        "type": "ellipse",
        "center": _vec3("center", center),
        "majorAxis": _vec3("major_axis", major_axis),
        "ratio": r,
    }
    if not any(entity["majorAxis"]):
        raise ValueError("major_axis must be non-zero")
    if start_param != 0.0:
        entity["startParam"] = _finite("start_param", start_param)
    if end_param != 2 * math.pi:
        entity["endParam"] = _finite("end_param", end_param)
    _normal(entity, normal)
    if not np.any(np.cross(entity.get("normal", [0.0, 0.0, 1.0]), entity["majorAxis"])):
        raise ValueError("major_axis must not be parallel to the normal")
    return _style(entity, layer, color, linetype, line_width, linetype_scale)


def spline(
    control_points,
    *,
    degree: int = 3,
    knots: Optional[Sequence[float]] = None,
    weights: Optional[Sequence[float]] = None,
    closed: bool = False,
    layer: Optional[str] = None,
    color: Optional[int] = None,
    linetype: Optional[str] = None,
    line_width: Optional[float] = None,
    linetype_scale: Optional[float] = None,
) -> dict:
    """A NURBS curve (DXF ``SPLINE``).

    Without ``knots`` the knot vector is clamped and uniform, so the curve
    starts and ends on the first and last control points. ``closed=True``
    without ``knots`` builds a smooth periodic curve instead. Explicit
    ``knots`` need ``len(control_points) + degree + 1`` non-decreasing
    values. ``weights`` (positive, one per control point) make it rational.
    """
    if int(degree) != degree or degree < 1:
        raise ValueError("degree must be an integer >= 1")
    degree = int(degree)
    cps = _flat_points("control_points", control_points, degree + 1)
    n = len(cps) // 3
    entity: dict = {"type": "spline", "degree": degree, "controlPoints": cps}
    if knots is not None:
        k = np.asarray(knots, dtype=float).reshape(-1)
        if k.size != n + degree + 1:
            raise ValueError(
                f"knots needs len(control_points) + degree + 1 = {n + degree + 1} values"
            )
        if not np.all(np.isfinite(k)) or np.any(np.diff(k) < 0):
            raise ValueError("knots must be finite and non-decreasing")
        if not k[n] > k[degree]:
            raise ValueError("knots leave the curve with an empty domain")
        entity["knots"] = k.tolist()
    if weights is not None:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != n:
            raise ValueError("weights needs one value per control point")
        if not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise ValueError("weights must be finite and positive")
        entity["weights"] = w.tolist()
    if closed:
        entity["closed"] = True
    return _style(entity, layer, color, linetype, line_width, linetype_scale)


def insert(
    block: str,
    positions,
    *,
    rotation: float = 0.0,
    rotations: Optional[Sequence[float]] = None,
    scale=None,
    scales: Optional[Sequence[float]] = None,
    normal=None,
    rows: int = 1,
    columns: int = 1,
    row_spacing: float = 0.0,
    column_spacing: float = 0.0,
    layer: Optional[str] = None,
    color: Optional[int] = None,
    linetype: Optional[str] = None,
    line_width: Optional[float] = None,
    linetype_scale: Optional[float] = None,
) -> dict:
    """Place a block at one or more points (DXF ``INSERT`` / ``MINSERT``).

    The block's entities are defined once (the ``blocks`` argument of
    :meth:`ViewerClient.add_vector_entities`) and drawn at every placement:
    for each of ``positions`` (``(N, 3)``, or a single point), the block is
    scaled, rotated by ``rotation`` (or ``rotations[i]``) radians about
    ``normal`` (default +Z) with its XY plane mapped onto the plane of
    ``normal``, then moved to the position.

    ``scale`` is a number or ``[sx, sy, sz]`` applied to every placement;
    ``scales`` multiplies it per placement. ``rows`` x ``columns`` repeats
    each placement on a grid, ``column_spacing`` along the rotated block X
    axis and ``row_spacing`` along its Y axis.

    Block entities without their own layer (or on layer ``"0"``) take the
    insert's layer. Without their own colour, linetype or width they take
    the insert's when it sets one, otherwise their layer's. Blocks may
    insert other blocks.
    """
    if not isinstance(block, str) or not block:
        raise ValueError("block must be a non-empty name")
    pos = np.asarray(positions, dtype=float)
    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    entity: dict = {"type": "insert", "block": block}
    entity["positions"] = _flat_points("positions", pos, 1)
    n = pos.shape[0]
    if rotations is not None:
        r = np.asarray(rotations, dtype=float).reshape(-1)
        if r.size != n or not np.all(np.isfinite(r)):
            raise ValueError(f"rotations needs one finite value per position ({n})")
        entity["rotations"] = r.tolist()
    elif rotation:
        entity["rotation"] = _finite("rotation", rotation)
    if scale is not None:
        if np.ndim(scale) == 0:
            sc = _finite("scale", scale)
            if sc == 0:
                raise ValueError("scale must be non-zero")
            entity["scale"] = sc
        else:
            sv = _vec3("scale", scale)
            if 0 in sv:
                raise ValueError("scale must be non-zero")
            entity["scale"] = sv
    if scales is not None:
        k = np.asarray(scales, dtype=float).reshape(-1)
        if k.size != n or not np.all(np.isfinite(k)) or np.any(k == 0):
            raise ValueError(
                f"scales needs one finite non-zero value per position ({n})"
            )
        entity["scales"] = k.tolist()
    for name, value in (("rows", rows), ("columns", columns)):
        if int(value) != value or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    if rows != 1 or columns != 1:
        entity["rows"] = int(rows)
        entity["columns"] = int(columns)
        entity["rowSpacing"] = _finite("row_spacing", row_spacing)
        entity["columnSpacing"] = _finite("column_spacing", column_spacing)
    _normal(entity, normal)
    return _style(entity, layer, color, linetype, line_width, linetype_scale)


def check_blocks(entities: Sequence[dict], blocks: dict) -> None:
    """Raise ``ValueError`` when an insert names a block missing from
    ``blocks``, or when blocks insert each other in a cycle."""
    for name, ents in blocks.items():
        if not isinstance(name, str) or not name:
            raise ValueError("block names must be non-empty strings")
        if not isinstance(ents, (list, tuple)):
            raise ValueError(f"block '{name}' must be a list of entities")

    def inserts(ents):
        return [
            e["block"]
            for e in ents
            if isinstance(e, dict) and e.get("type") == "insert"
        ]

    for b in inserts(entities):
        if b not in blocks:
            raise ValueError(f"insert names unknown block '{b}'")
    state: dict = {}

    def visit(name, path):
        if state.get(name) == "done":
            return
        if state.get(name) == "active":
            cycle = " -> ".join([*path, name])
            raise ValueError(f"blocks insert each other in a cycle: {cycle}")
        state[name] = "active"
        for child in inserts(blocks[name]):
            if child not in blocks:
                raise ValueError(f"block '{name}' inserts unknown block '{child}'")
            visit(child, [*path, name])
        state[name] = "done"

    for name in blocks:
        visit(name, [])
