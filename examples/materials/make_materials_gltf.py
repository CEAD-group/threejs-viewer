"""Regenerate materials.gltf / materials.bin: one chamfered cylinder per material.

The test shape is an upright cylinder standing on the floor (axis +Z — the
file keeps Blender's Z-up frame, like the original export) with a closed top
whose rim carries a quarter-round fillet of radius 25% of the height, and a
sharp bottom rim. Every material is thus seen on a flat cap, a doubly-curved
fillet, and a curved wall in one silhouette. Normals are smooth around the
circumference and across the fillet (tangent-continuous into wall and cap);
only the bottom rim is a hard edge. The cylinders sit
at the same grid positions as the original Blender export (``CENTRES``).

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
FILLET = 0.25 * HEIGHT  # quarter-round on the top rim
SEGMENTS = 64  # around the circumference
FILLET_STEPS = 12  # rings across the 90° fillet arc

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


def filleted_cylinder():
    """Return (positions (N,3) f32, normals (N,3) f32, indices (M,) u32)."""
    ang = np.linspace(0.0, 2 * np.pi, SEGMENTS, endpoint=False)
    c, s = np.cos(ang), np.sin(ang)
    top = HEIGHT / 2
    bot = -HEIGHT / 2  # centre at z=0 like the original cubes: floor at z=-1
    radial = np.column_stack([c, s, np.zeros(SEGMENTS)])
    up = np.array([0.0, 0.0, 1.0])

    def ring(radius, z):
        return np.column_stack([radius * c, radius * s, np.full(SEGMENTS, z)])

    # One smooth strip from the bottom rim up the wall and over the fillet to
    # the cap's inner rim: rings share vertices, so shading is continuous.
    rings, normals = [ring(RADIUS, bot), ring(RADIUS, top - FILLET)], [radial, radial]
    for k in range(1, FILLET_STEPS + 1):
        t = np.pi / 2 * k / FILLET_STEPS  # 0 = wall tangent, 90° = cap tangent
        n = np.cos(t) * radial + np.sin(t) * up
        centre_r, centre_z = RADIUS - FILLET, top - FILLET
        rings.append(ring(centre_r, centre_z) + FILLET * n)
        normals.append(n)

    pos = np.concatenate(rings)
    nrm = np.concatenate(normals)
    idx = []
    for r in range(len(rings) - 1):
        a0 = r * SEGMENTS
        b0 = a0 + SEGMENTS
        for i in range(SEGMENTS):
            j = (i + 1) % SEGMENTS
            idx.extend([a0 + i, a0 + j, b0 + i, b0 + i, a0 + j, b0 + j])  # CCW outside

    # Top cap fan continues the last fillet ring (already carrying the +Z normal).
    top_ring = len(rings) - 1
    centre = len(pos)
    pos = np.vstack([pos, [[0.0, 0.0, top]]])
    nrm = np.vstack([nrm, [[0.0, 0.0, 1.0]]])
    for i in range(SEGMENTS):
        j = (i + 1) % SEGMENTS
        idx.extend([centre, top_ring * SEGMENTS + i, top_ring * SEGMENTS + j])

    # Bottom cap: own rim vertices (hard edge) with -Z normals.
    base = len(pos)
    pos = np.vstack([pos, [[0.0, 0.0, bot]], ring(RADIUS, bot)])
    nrm = np.vstack([nrm, np.tile([0.0, 0.0, -1.0], (SEGMENTS + 1, 1))])
    for i in range(SEGMENTS):
        j = (i + 1) % SEGMENTS
        idx.extend([base, base + 1 + j, base + 1 + i])

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

    positions, normals, indices = filleted_cylinder()
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
