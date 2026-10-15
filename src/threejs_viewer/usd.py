"""
Turn USD BRep geometry into render-ready arrays for :meth:`ViewerClient.add_usda`.

The tessellation is done by the optional ``usd_brep`` package, imported on
first use so the viewer itself keeps working without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np

USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})


@dataclass
class UsdBody:
    """One tessellated body: an unwelded triangle soup plus its BRep edges."""

    positions: np.ndarray  # (N, 3) float32, three rows per triangle
    normals: np.ndarray  # (N, 3) float32
    indices: np.ndarray  # (N,) uint32
    edges: np.ndarray  # (2K, 3) float32 segment endpoints, empty for meshes


def _usd_brep():
    try:
        import usd_brep
    except ImportError as exc:
        raise ImportError(
            "add_usda needs the usd_brep package to read and tessellate USD "
            "BRep geometry; install it in this environment."
        ) from exc
    return usd_brep


def is_usd_path(source: Any) -> bool:
    """True for a str/Path naming a USD file (by suffix)."""
    return (
        isinstance(source, (str, Path)) and Path(source).suffix.lower() in USD_SUFFIXES
    )


def triangle_corners(faces) -> tuple[np.ndarray, np.ndarray]:
    """Fan-triangulate a ``[count, i0, i1, ..., count, ...]`` polygon stream.

    Returns ``(vertices, corners)``, three entries per triangle: the vertex
    index of each triangle corner, and the ordinal of the polygon corner it
    came from (for looking up per-corner attributes such as normals).
    """
    faces = np.asarray(faces, dtype=np.int64).reshape(-1)
    if faces.size % 4 == 0 and np.all(faces[::4] == 3):  # all triangles
        vertices = faces.reshape(-1, 4)[:, 1:].reshape(-1)
        return vertices, np.arange(vertices.size, dtype=np.int64)
    vertices: List[int] = []
    corners: List[int] = []
    cursor = ordinal = 0
    while cursor < faces.size:
        count = int(faces[cursor])
        polygon = faces[cursor + 1 : cursor + 1 + count]
        for k in range(1, count - 1):
            vertices += [polygon[0], polygon[k], polygon[k + 1]]
            corners += [ordinal, ordinal + k, ordinal + k + 1]
        cursor += count + 1
        ordinal += count
    return np.asarray(vertices, dtype=np.int64), np.asarray(corners, dtype=np.int64)


def _mesh_body(poly, edges: np.ndarray) -> UsdBody:
    # Tessellated points are shared across sharp BRep edges, so per-vertex
    # normals would shade a cube like a pillow. Corner normals come from the
    # source surfaces; unweld the mesh to carry one normal per corner.
    arrays = poly.to_numpy_arrays(face_varying_normals=True)
    vertices, corners = triangle_corners(arrays["faces"])
    return UsdBody(
        positions=np.asarray(arrays["points"], dtype=np.float32)[vertices],
        normals=np.asarray(arrays["normals"], dtype=np.float32)[corners],
        indices=np.arange(vertices.size, dtype=np.uint32),
        edges=edges,
    )


def edge_segments(points, counts) -> np.ndarray:
    """Per-edge polylines (flat points + per-edge counts) as segment endpoints."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    out = []
    start = 0
    for count in counts:
        if count >= 2:
            run = points[start : start + count]
            out.append(np.stack([run[:-1], run[1:]], axis=1).reshape(-1, 3))
        start += count
    if not out:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(out)


def _brep_body(usd_brep, brep, surface_angle, curve_angle, chord_height) -> UsdBody:
    poly = usd_brep.tessellate(
        brep,
        chord_height_tolerance=chord_height,
        curve_angle_tolerance_deg=curve_angle,
        surface_angle_tolerance_deg=surface_angle,
    )
    points, counts, _ = usd_brep.tessellate_boundaries(
        brep, angle_tolerance_deg=min(curve_angle, 5.0), allow_partial=True
    )
    return _mesh_body(poly, edge_segments(points, counts))


def load_bodies(
    source: Any,
    *,
    surface_angle: float = 10.0,
    curve_angle: float = 10.0,
    chord_height: float = 0.0,
) -> List[UsdBody]:
    """Tessellate ``source`` into bodies.

    ``source`` is a USD file path (every BrepArray prim, or the Mesh prims
    when the file holds no BReps), a ``usd_brep.Brep``, a ``usd_brep.PolyBrep``,
    or a list/tuple of those.
    """
    usd_brep = _usd_brep()
    no_edges = np.zeros((0, 3), dtype=np.float32)
    if isinstance(source, (list, tuple)):
        bodies: List[UsdBody] = []
        for item in source:
            bodies += load_bodies(
                item,
                surface_angle=surface_angle,
                curve_angle=curve_angle,
                chord_height=chord_height,
            )
        return bodies
    if isinstance(source, usd_brep.Brep):
        return [_brep_body(usd_brep, source, surface_angle, curve_angle, chord_height)]
    if isinstance(source, usd_brep.PolyBrep):
        return [_mesh_body(source, no_edges)]
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"USD file not found: {path}")
        try:
            breps = usd_brep.usd.import_breps(str(path))
        except RuntimeError:
            breps = []  # no BrepArray prims; try the meshes below
        if breps:
            return [
                _brep_body(usd_brep, b, surface_angle, curve_angle, chord_height)
                for b in breps
            ]
        try:
            meshes = usd_brep.usd.import_meshes(str(path))
        except RuntimeError:
            raise ValueError(
                f"No BrepArray or Mesh prims could be read from {path}"
            ) from None
        return [_mesh_body(m, no_edges) for m in meshes]
    raise TypeError(
        "source must be a USD file path, a usd_brep Brep/PolyBrep, or a list "
        f"of those, got {type(source).__name__}"
    )
