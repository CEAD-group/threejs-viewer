"""
Octree LOD point cloud with per-point lifetimes, streamed on demand.

``add_points(lod=True)`` builds an octree in Python; the browser fetches node payloads
as you zoom, under a ~1.5M-point budget, so the full cloud never has to fit in one draw
call. Each point has a [birth, removal) lifetime: a build front sweeps across the
wavy sheet and an erosion front follows it, while a bedrock band never erodes. The
animation slider scrubs this field time. Nodes are served by this process, so keep it
running (Ctrl+C to quit). The cloud is empty at t=0 and fills as the animation plays.

Run: uv run python examples/27_points_octree_lod.py [n_points]   (default 2M)
"""

import sys
import time

import numpy as np

from threejs_viewer import Animation, viewer

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000

v = viewer()
v.clear()
v.unload_animation()

# --- Abstract point field: a thick wavy sheet with density structure ---
# Centered on the origin so the default camera view frames it naturally.
rng = np.random.default_rng(0)
x = rng.uniform(-20.0, 20.0, N).astype(np.float32)
y = rng.uniform(-10.0, 10.0, N).astype(np.float32)
sheet = 1.8 * np.sin(x / 3.0) * np.cos(y / 2.5)  # wavy mid-surface
thickness = 0.8 + 0.5 * np.sin(x / 5.0 + y / 7.0)
z = (sheet + thickness * rng.normal(0.0, 0.6, N)).astype(np.float32)
positions = np.column_stack([x, y, z])

# Colour by height above the local mid-surface (reads as a heatmap of the
# sheet's grain at every LOD).
deviation = z - sheet

# --- Per-point lifetimes ---
# Build front: sweeps diagonally, points appear over ~16s of field time.
birth = ((x + 20.0) + 0.4 * (y + 10.0)) / 3.0 + rng.normal(0.0, 0.25, N)
# Erosion front: follows ~5s behind with some scatter...
removal = birth + 5.0 + rng.exponential(1.5, N)
# ...but the lowest points are bedrock and never erode (NaN = never removed).
removal[z < sheet - 0.5] = np.nan

t0 = time.perf_counter()
v.add_points(
    "field",
    positions,
    colors=deviation,
    colormap="turbo",
    size=0.045,  # world-ish size under size_attenuation
    size_attenuation=True,
    birth_times=birth,
    removal_times=removal,
    lod=True,  # or a dict: node_capacity / point_budget / refine_pixels / seed
)
build_s = time.perf_counter() - t0

# --- Drive the field time from the animation slider ---
# One point_times channel maps the playhead 1:1 onto the scrub time, so
# play/pause/drag on the standard controls scrubs points in and out.
t_end = float(np.nanmax(removal[np.isfinite(removal)]) + 1.0)
n_frames = 240
frame_times = np.linspace(0.0, t_end, n_frames, dtype=np.float32)
anim = Animation(loop=True)
anim.set_frame_times(frame_times)
anim.set_point_time_data(["field"], frame_times.reshape(n_frames, 1))
v.load_animation(anim)

v.set_camera(position=[0, -45, 25], target=[0, 0, 0])
print(
    f"{N:,} points, octree built in {build_s:.1f}s. Zoom in to stream detail; Ctrl+C to quit."
)
v.wait_for_assets(disconnect=False)
try:
    while True:
        time.sleep(1.0)
except KeyboardInterrupt:
    pass
finally:
    v.disconnect()
