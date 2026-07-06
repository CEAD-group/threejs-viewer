"""
Octree LOD point cloud + time-window scrubbing (issue #79)

Demonstrates ``add_points(lod=...)`` and per-point time windows together on
one abstract cloud, with the point count trivially tunable:

    uv run python examples/27_points_octree_lod.py            # 2M points
    uv run python examples/27_points_octree_lod.py 10000000   # 10M points

What happens:

- Python builds a Potree-style additive sampled octree over the cloud (a
  seeded, time-stratified pseudo-random sample per node) and the browser
  streams node payloads **on demand**: a ~1.5M-point budget of the
  biggest-on-screen nodes is drawn each frame, refined as you zoom in and
  coarsened (and LRU-evicted) as you zoom out. Zoom into the wave sheet to
  watch detail stream in; the full cloud never has to fit in one draw call.
- Every point carries a [birth, removal) lifetime: a build front sweeps
  diagonally across the sheet (points appear), an erosion front follows a
  few seconds behind (points disappear), and a band of "bedrock" points
  near the base is never removed (NaN removal). **Drag the animation
  slider** to scrub the field time — points appear and disappear out of
  buffer order, at every LOD, because the filter runs per point in the
  vertex shader and each octree node's sample is stratified over time.

Note the script stays running: LOD nodes stream from this Python process
on demand, so exiting it freezes refinement at whatever is already loaded
(the standard trade-off of the streamed tier). Ctrl+C to quit. At t=0
nothing has been built yet — the cloud appears as the animation plays
(it autoplays) or when you drag the slider forward.

Run: uv run python examples/27_points_octree_lod.py [n_points]
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

print(f"{N:,} points -> octree built + registered in {build_s:.1f}s.")
print("Zoom in/out: node detail streams on demand under a ~1.5M-point budget.")
print("Drag the time slider: the build front adds points, the erosion front")
print("removes them (out of buffer order), bedrock near the base persists.")
print("Press F in the viewer to frame the cloud.")
print("Tune the count: uv run python examples/27_points_octree_lod.py 10000000")
print()
print("Serving LOD nodes from this process — leave it running (Ctrl+C to quit).")
v.wait_for_assets(disconnect=False)
try:
    while True:
        time.sleep(1.0)
except KeyboardInterrupt:
    pass
