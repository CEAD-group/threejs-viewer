"""Regenerate materials.gltf / materials.bin: one test body per material.

The test body is an upright cylinder standing on the floor (axis +Z — the
file keeps Blender's Z-up frame, like the original export) with a 90° "pac-man"
wedge removed (0°..90°), so two flat radial cut faces are exposed. The closed
top rim is treated two ways over the remaining 270°: a quarter-round fillet on
the first half and a 45° chamfer on the second half, both sized 25% of the
height, so fillet and chamfer are compared on one material. The bottom rim is
sharp. Normals are smooth around the circumference and across the fillet
(tangent-continuous into wall and cap); the chamfer, the cut faces, and the
bottom rim are hard edges. The bodies sit at the same grid positions as the
original Blender export (``CENTRES``).

The material definitions are read from the existing materials.gltf, so this
script only replaces the geometry and layout:

    uv run python examples/materials/make_materials_gltf.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
GLTF = HERE / "materials.gltf"
BIN = HERE / "materials.bin"

RADIUS = 1.29  # footprint of the original cubes
HEIGHT = 2.0
EDGE = 0.25 * HEIGHT  # fillet radius / chamfer size on the top rim
SEGMENTS = 64  # columns per full circle
WEDGE_DEG = 90.0  # pac-man wedge removed, starting at 0°
RIM_STEPS = 12  # rings across the fillet arc (and the chamfer, for a uniform strip)

# Node name -> centre, taken from the original Blender export's bounding boxes.
CENTRES: dict[str, list[float]] = {
    "ANODIZED_BLACK": [0.0, 0.0, 0.0],
    "ANODIZED_CLEAR": [3.877000093460083, 0.0, 0.0],
    "ANODIZED_VIBRANT_BLUE": [7.7540998458862305, 0.0, 0.0],
    "ANODIZED_CHARCOAL_GREY": [11.631099700927734, 0.0, 0.0],
    "BEADBLASTED_STEEL": [27.139299392700195, 0.0, 0.0],
    "BEADBLASTED_ALUMINIUM": [31.016399383544922, 0.0, 0.0],
    "POWDERCOAT_WHITE": [3.877000093460083, 3.877000093460083, 0.0],
    "POWDERCOAT_BLACK": [7.7540998458862305, 3.877000093460083, 0.0],
    "SPRAYPAINTED_BLACK": [11.631099700927734, 3.877000093460083, 0.0],
    "SPRAYPAINTED_WHITE": [15.508199691772461, 3.877000093460083, 0.0],
    "HARDCHROMATIC": [19.38520050048828, 3.877000093460083, 0.0],
    "BLACK_OXIDE": [34.89339828491211, 0.0, 0.0],
    "COLORDYE_BLACK": [23.262300491333008, 3.877000093460083, 0.0],
    "POLISHED_STEEL": [7.7540998458862305, 7.7540998458862305, 0.0],
    "GOLD": [31.016399383544922, 3.877000093460083, 0.0],
    "COPPER": [15.508199691772461, 7.7540998458862305, 0.0],
    "TITANIUM": [19.38520050048828, 7.7540998458862305, 0.0],
    "BRASS": [23.262300491333008, 7.7540998458862305, 0.0],
    "STEEL": [15.508199691772461, 11.631099700927734, 0.0],
    "PLASTIC_BLUE": [34.89339828491211, 3.877000093460083, 0.0],
    "PA630GF": [0.0, 7.7540998458862305, 0.0],
    "PLASTIC_BLACK": [3.877000093460083, 7.7540998458862305, 0.0],
    "PC": [0.0, 11.631099700927734, 0.0],
    "PVC": [3.877000093460083, 11.631099700927734, 0.0],
    "HDPE": [7.7540998458862305, 11.631099700927734, 0.0],
    "POM": [11.631099700927734, 11.631099700927734, 0.0],
    "PA630GF.001": [0.0, 7.7540998458862305, 0.0],
    "POM.001": [11.631099700927734, 11.631099700927734, 0.0],
    "PAINT_GREY": [34.89339828491211, 7.7540998458862305, 0.0],
    "PAINT_LIGHT_GREY": [27.139299392700195, 7.7540998458862305, 0.0],
}


def pacman_cylinder():
    """Return (positions (N,3) f32, normals (N,3) f32, indices (M,) u32)."""
    top, bot = HEIGHT / 2, -HEIGHT / 2  # centre at z=0 like the original cubes
    start = np.radians(WEDGE_DEG)
    ncol = int(round(SEGMENTS * (360.0 - WEDGE_DEG) / 360.0)) + 1
    ang = np.linspace(start, 2 * np.pi, ncol)  # open sweep, both ends kept
    c, s_ = np.cos(ang), np.sin(ang)
    radial = np.column_stack([c, s_, np.zeros(ncol)])
    up = np.array([0.0, 0.0, 1.0])
    chamfer_col = ang >= (start + 2 * np.pi) / 2  # second half of the sweep

    pos, nrm, tris = [], [], []

    def add(p, n):
        pos.extend(np.atleast_2d(p))
        nrm.extend(np.atleast_2d(n))

    def ring(radius, z):
        return np.column_stack([radius * c, radius * s_, np.full(ncol, z)])

    def strip(rings):
        """Open quad strip between consecutive rings of ncol columns."""
        base = len(pos) - len(rings) * ncol
        for r in range(len(rings) - 1):
            a0, b0 = base + r * ncol, base + (r + 1) * ncol
            for i in range(ncol - 1):
                tris.extend(
                    [(a0 + i, a0 + i + 1, b0 + i), (b0 + i, a0 + i + 1, b0 + i + 1)]
                )

    # Wall.
    add(ring(RADIUS, bot), radial)
    add(ring(RADIUS, top - EDGE), radial)
    strip([0, 1])

    # Rim: fillet arc on the first half, straight 45° chamfer on the second.
    rim_rings = []
    for k in range(RIM_STEPS + 1):
        t = np.pi / 2 * k / RIM_STEPS
        n_fillet = np.cos(t) * radial + np.sin(t) * up
        p_fillet = ring(RADIUS - EDGE, top - EDGE) + EDGE * n_fillet
        f = k / RIM_STEPS
        p_chamfer = ring(RADIUS - EDGE * f, top - EDGE * (1 - f))
        n_chamfer = (radial + up) / np.sqrt(2.0)
        p = np.where(chamfer_col[:, None], p_chamfer, p_fillet)
        n = np.where(chamfer_col[:, None], n_chamfer, n_fillet)
        rim_rings.append(p)
        add(p, n)
    strip(rim_rings)

    # Top cap: own rim vertices (hard against the chamfer, seamless on the fillet).
    centre = len(pos)
    add([0.0, 0.0, top], up)
    add(ring(RADIUS - EDGE, top), np.tile(up, (ncol, 1)))
    for i in range(ncol - 1):
        tris.append((centre, centre + 1 + i, centre + 2 + i))

    # Bottom cap.
    centre = len(pos)
    add([0.0, 0.0, bot], -up)
    add(ring(RADIUS, bot), np.tile(-up, (ncol, 1)))
    for i in range(ncol - 1):
        tris.append((centre, centre + 2 + i, centre + 1 + i))

    # Two flat radial cut faces, each a fan from the bottom axis point over the
    # column's full profile (rim -> wall top -> rim strip -> top axis).
    for col in (0, ncol - 1):
        profile = [pos[col]] + [
            r[col] for r in rim_rings
        ]  # rim_rings[0] is the wall top
        profile += [[0.0, 0.0, top]]
        # Outward = away from the solid, which lies at increasing angle for the
        # first cut and decreasing angle for the last.
        tangent = np.array([-s_[col], c[col], 0.0])
        n = -tangent if col == 0 else tangent
        base = len(pos)
        add([0.0, 0.0, bot], n)
        add(np.asarray(profile), np.tile(n, (len(profile), 1)))
        for i in range(len(profile) - 1):
            tris.append((base, base + 1 + i, base + 2 + i))

    pos = np.asarray(pos, np.float32)
    nrm = np.asarray(nrm, np.float32)
    tris = np.asarray(tris, np.int64)
    # Orient every triangle to agree with its (per-surface constant) vertex normal.
    t = pos[tris]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    flip = (fn * nrm[tris].mean(1)).sum(1) < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return pos, nrm, tris.ravel().astype(np.uint32)


def main() -> None:
    old = json.loads(GLTF.read_text())
    materials = old["materials"]
    # Preserve node name -> material mapping from the current file.
    node_mats = [
        (n["name"], old["meshes"][n["mesh"]]["primitives"][0].get("material"))
        for n in old["nodes"]
    ]

    positions, normals, indices = pacman_cylinder()
    pos_b = positions.tobytes()
    nrm_b = normals.tobytes()
    idx_b = indices.tobytes()
    blob = pos_b + nrm_b + idx_b
    blob += b"\0" * (-len(blob) % 4)
    BIN.write_bytes(blob)

    buffer_views = [
        {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_b), "target": 34962},
        {
            "buffer": 0,
            "byteOffset": len(pos_b),
            "byteLength": len(nrm_b),
            "target": 34962,
        },
        {
            "buffer": 0,
            "byteOffset": len(pos_b) + len(nrm_b),
            "byteLength": len(idx_b),
            "target": 34963,
        },
    ]
    accessors = [
        {
            "bufferView": 0,
            "componentType": 5126,
            "count": len(positions),
            "type": "VEC3",
            "min": positions.min(0).tolist(),
            "max": positions.max(0).tolist(),
        },
        {"bufferView": 1, "componentType": 5126, "count": len(normals), "type": "VEC3"},
        {
            "bufferView": 2,
            "componentType": 5125,
            "count": len(indices),
            "type": "SCALAR",
        },
    ]

    meshes, nodes = [], []
    for i, (name, mat) in enumerate(node_mats):
        meshes.append(
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1},
                        "indices": 2,
                        "material": mat,
                    }
                ],
            }
        )
        nodes.append({"name": name, "mesh": i, "translation": CENTRES[name]})

    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "threejs-viewer examples/materials/make_materials_gltf.py",
            "copyright": old["asset"].get("copyright", ""),
        },
        "extensionsUsed": old.get("extensionsUsed", []),
        "scene": 0,
        "scenes": [{"name": "Scene", "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(blob), "uri": BIN.name}],
    }
    GLTF.write_text(json.dumps(gltf, indent=1) + "\n")
    print(
        f"{GLTF.name}: {len(nodes)} nodes, {len(positions)} verts, {len(indices) // 3} tris"
    )


if __name__ == "__main__":
    main()
