"""
Mesh Vertex Alpha

`add_mesh` with per-vertex RGBA colors: pass an (N, 4) float array in [0, 1]
to `colors`. Here the RGB forms a gradient across the plane and the alpha
falls from opaque at the centre to transparent at the edge, over orange
spheres and a floor grid so the transparency is visible.

Run: uv run python examples/33_mesh_vertex_alpha.py
"""

from itertools import product

import numpy as np

from threejs_viewer import viewer

NX = NY = 60
SIZE = 8.0

x, y = np.meshgrid(
    np.linspace(-SIZE / 2, SIZE / 2, NX), np.linspace(-SIZE / 2, SIZE / 2, NY)
)
x, y = x.ravel(), y.ravel()
positions = np.column_stack([x, y, np.zeros_like(x)]).astype(np.float32)

# Two triangles per grid cell, counter-clockwise so the normals point up (+Z)
cell = (np.arange(NY - 1)[:, None] * NX + np.arange(NX - 1)).ravel()
indices = np.stack(
    [cell, cell + 1, cell + NX, cell + 1, cell + NX + 1, cell + NX], axis=1
).astype(np.uint32)

# R along X, G along Y; alpha falls off radially
alpha = np.clip(1 - (np.hypot(x, y) / (SIZE / 2)) ** 1.5, 0, 1)
colors = np.column_stack(
    [
        0.15 + 0.5 * (x / SIZE + 0.5),
        0.15 + 0.5 * (y / SIZE + 0.5),
        np.full_like(x, 0.6),
        alpha,
    ]
).astype(np.float32)

v = viewer()
v.clear()
v.add_grid("floor", cell_size=1.0, extent=40.0, fade_start=0.4)
# Spheres under the plane: the ones near the edge show through, the centre one is hidden
for i, (sx, sy) in enumerate([(0, 0), (-3, -2.5), (3, 2.5), (3, -2.5), (-3, 2.5)]):
    v.add_sphere(f"ball{i}", radius=0.7, color=0xFF5500, position=[sx, sy, 0.7])
v.add_mesh(
    "plane",
    positions=positions,
    indices=indices.ravel(),
    normals=np.tile([0, 0, 1], (len(x), 1)).astype(np.float32),
    colors=colors,
    roughness=0.9,
    metalness=0.0,
    position=[0, 0, 1.6],
)

v.set_view("iso", animate=False)
v.frame_points(list(product([-4, 4], [-4, 4], [0, 1.7])), fill=0.9)
v.wait_for_assets()
