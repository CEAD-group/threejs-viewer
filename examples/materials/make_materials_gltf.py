"""Regenerate materials.gltf / materials.bin: one chamfered cylinder per material.

The test shape is a cylinder (axis +Y, glTF up) with a 45° chamfer on the top
rim and a sharp bottom rim, so every material is seen on a flat cap, a
conical chamfer band, and a curved wall in one silhouette. Normals are smooth
around the circumference and split (hard) between the four surfaces.

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

RADIUS = 1.0
HEIGHT = 2.0
CHAMFER = 0.3  # 45° chamfer size on the top rim
SEGMENTS = 64
COLUMNS = 6
SPACING = 3.0


def chamfered_cylinder():
    """Return (positions (N,3) f32, normals (N,3) f32, indices (M,) u32)."""
    ang = np.linspace(0.0, 2 * np.pi, SEGMENTS, endpoint=False)
    c, s = np.cos(ang), np.sin(ang)
    top = HEIGHT / 2
    bot = -HEIGHT / 2
    r_in = RADIUS - CHAMFER
    y_ch = top - CHAMFER

    pos, nrm, idx = [], [], []

    def ring(radius, y):
        return np.column_stack([radius * c, np.full(SEGMENTS, y), radius * s])

    def band(ring_a, ring_b, normal_a, normal_b):
        """Quad strip between two rings with per-ring normals; returns base index."""
        base = len(pos)
        pos.extend(ring_a)
        pos.extend(ring_b)
        nrm.extend(normal_a)
        nrm.extend(normal_b)
        for i in range(SEGMENTS):
            j = (i + 1) % SEGMENTS
            a0, a1 = base + i, base + j
            b0, b1 = base + SEGMENTS + i, base + SEGMENTS + j
            # CCW seen from outside
            idx.extend([a0, b0, a1, a1, b0, b1])
        return base

    radial = np.column_stack([c, np.zeros(SEGMENTS), s])

    # Wall: sharp bottom rim, up to the chamfer start.
    band(ring(RADIUS, bot), ring(RADIUS, y_ch), radial, radial)

    # Chamfer band: 45° between radial and +Y.
    ch_n = (radial + np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0)
    band(ring(RADIUS, y_ch), ring(r_in, top), ch_n, ch_n)

    def cap(radius, y, normal_y):
        base = len(pos)
        pos.append([0.0, y, 0.0])
        nrm.append([0.0, normal_y, 0.0])
        pos.extend(ring(radius, y))
        nrm.extend(np.tile([0.0, normal_y, 0.0], (SEGMENTS, 1)))
        for i in range(SEGMENTS):
            j = (i + 1) % SEGMENTS
            if normal_y > 0:
                idx.extend([base, base + 1 + i, base + 1 + j])
            else:
                idx.extend([base, base + 1 + j, base + 1 + i])

    cap(r_in, top, 1.0)
    cap(RADIUS, bot, -1.0)

    return (
        np.asarray(pos, np.float32),
        np.asarray(nrm, np.float32),
        np.asarray(idx, np.uint32),
    )


def main() -> None:
    old = json.loads(GLTF.read_text())
    materials = old["materials"]
    # Preserve node name -> material mapping from the current file.
    node_mats = [
        (n["name"], old["meshes"][n["mesh"]]["primitives"][0].get("material"))
        for n in old["nodes"]
    ]

    positions, normals, indices = chamfered_cylinder()
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
        col, row = i % COLUMNS, i // COLUMNS
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
        nodes.append(
            {
                "name": name,
                "mesh": i,
                "translation": [col * SPACING, 0.0, row * SPACING],
            }
        )

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
