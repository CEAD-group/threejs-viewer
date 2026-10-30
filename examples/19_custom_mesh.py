"""
Custom mesh

`add_mesh` takes raw positions, indices, normals and per-vertex colours, with no
constraint on topology. This builds a procedural terrain heightmap (~320k
triangles), colours it by altitude, and reveals it row by row with a
draw_range animation.

Run: uv run python examples/19_custom_mesh.py
"""

import numpy as np

from threejs_viewer import Animation, viewer

N, SIZE = 400, 12.0  # grid vertices per side, world size

v = viewer()
v.clear()
v.add_grid("floor", cell_size=1.0, extent=40, center_color=0x888888)

# Layered trig "noise" height field
xg, yg = np.meshgrid(*[np.linspace(-SIZE / 2, SIZE / 2, N, dtype=np.float32)] * 2)
z = sum(
    a * np.sin(f * xg + 0.3) * np.cos(f * yg * 1.1 + 0.7)
    for f, a in [(0.5, 1.0), (0.9, 0.5), (2.3, 0.25), (4.7, 0.12)]
)
z = 2 * z + 0.4 * np.sin(0.7 * xg + 0.9 * yg) * np.cos(1.3 * xg - 0.6 * yg) + 3.0
positions = np.column_stack([xg.ravel(), yg.ravel(), z.ravel()]).astype(np.float32)

# Two triangles per quad, row-major so the draw range reveals row by row
v00 = (np.arange(N - 1)[:, None] * N + np.arange(N - 1)).ravel()
indices = np.column_stack([v00, v00 + N, v00 + 1, v00 + 1, v00 + N, v00 + N + 1])
indices = indices.astype(np.uint32).ravel()

# Vertex normals: accumulate face normals onto their three vertices
tri = positions[indices].reshape(-1, 3, 3)
face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
normals = np.zeros_like(positions)
np.add.at(normals, indices, np.repeat(face_n, 3, axis=0))
normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)

# Altitude ramp: green valleys, brown midlands, grey, white peaks
h = (z.ravel() - z.min()) / (z.max() - z.min())
stops = np.array([[0.15, 0.5, 0.1], [0.6, 0.3, 0.1], [0.5, 0.5, 0.4], [1, 1, 1]])
colors = np.column_stack(
    [np.interp(h, [0, 0.4, 0.7, 1], stops[:, c]) for c in range(3)]
)

v.add_mesh(
    "terrain",
    positions=positions,
    indices=indices,
    normals=normals,
    colors=colors.astype(np.float32),
    roughness=0.8,
    metalness=0.05,
)

# Reveal the buffer from the first row to the last over 8 s
frames = 300
anim = Animation(loop=True)
anim.set_frame_times(np.linspace(0, 8.0, frames))
anim.set_draw_range_data(["terrain"], np.linspace(0, 1, frames)[:, None])
v.load_animation(anim)

v.set_camera(position=[0, -24, 16], target=[0, 0, 1.5])  # view direction
v.frame_points(positions, fill=0.85)
v.wait_for_assets()
