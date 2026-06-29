"""
GPU Point Cloud — metrology heatmap + material-removal reveal

Demonstrates ``add_points``: a single ``THREE.Points`` draw call rendering a
dense point cloud, coloured per-point by a scalar field (here a signed
deviation heatmap) and revealed progressively with draw_range — a cheap
material-removal animation.

The cloud samples a machined "cut surface": a nominal flat top with a curved
pocket milled into it. Each point is coloured by signed deviation (actual
height − nominal), so the pocket reads as a depth heatmap. Points are ordered
along a serpentine toolpath, so the draw_range animation sweeps across the
surface like the tool removing material.

THREE.Points stays in one draw call, so this scales to multi-million-point
clouds (LiDAR scans, voxel fields) — bump ``side`` to feel it.

Run: uv run python examples/25_point_cloud.py
"""

import numpy as np

from threejs_viewer import Animation, viewer

v = viewer()
v.clear()
v.unload_animation()

# --- Serpentine raster over a square patch ---
side = 400  # points per axis -> side*side total
extent = 8.0
xs = np.linspace(-extent / 2, extent / 2, side, dtype=np.float32)
ys = np.linspace(-extent / 2, extent / 2, side, dtype=np.float32)

# Boustrophedon row order so draw_range sweeps back and forth, the way a
# milling raster actually runs.
rows = []
for j, y in enumerate(ys):
    row_x = xs if j % 2 == 0 else xs[::-1]
    rows.append(np.column_stack([row_x, np.full(side, y, dtype=np.float32)]))
xy = np.concatenate(rows, axis=0)
x = xy[:, 0]
y = xy[:, 1]

# --- Cut surface: flat nominal top (z=0) with a curved pocket milled in ---
r = np.sqrt(x**2 + y**2)
pocket = -1.6 * np.exp(-(r**2) / 2.0)  # smooth round pocket
pocket += (
    -0.25 * np.cos(4 * np.arctan2(y, x)) * np.exp(-(r**2) / 4.0)
)  # scalloped walls
ripple = 0.05 * np.sin(6 * x) * np.cos(6 * y)  # tool-mark texture
z = pocket + ripple

# Signed deviation from the nominal flat top: negative = material removed.
deviation = z

positions = np.column_stack([x, y, z]).astype(np.float32)

v.add_points(
    "cut_surface",
    positions,
    colors=deviation,
    colormap="turbo",
    size=3.0,
    size_attenuation=True,
)

# --- Material-removal animation: reveal points along the toolpath ---
n_frames = 240
frame_times = np.linspace(0, 6.0, n_frames, dtype=np.float32)
draw_fracs = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)

anim = Animation(loop=True)
anim.set_frame_times(frame_times)
anim.set_draw_range_data(["cut_surface"], draw_fracs.reshape(n_frames, 1))
v.load_animation(anim)

n = len(positions)
print(f"Point cloud: {n:,} points, coloured by signed deviation (turbo).")
print("draw_range sweeps the serpentine toolpath like a material-removal pass.")
print("Bump `side` to push toward millions of points — still one draw call.")
v.wait_for_assets()
