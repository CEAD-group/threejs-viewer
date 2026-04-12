"""
Parametric Tube — variable-cross-section bead visualization

Demonstrates ``parametric_tube``: a client-built extruded tube whose
cross-section width and height vary along the spine. The viewer receives
only the spine polyline plus per-point width/height/color arrays and
builds the triangles locally, so a 5000-point bead uploads in ~200 KB
instead of the ~1 MB a baked mesh would cost.

Highlights:
- Variable width/height along the path (neck narrows, belly swells).
- Draw-range animation reveals the bead one ring-pair at a time — the
  cut edge is snapped to ring boundaries so the wavefront stays clean.
- A capsule parented to an animated group acts as a nozzle-tip
  indicator, showing where material is currently being deposited.
- After 3 seconds, ``update_parametric_tube_colors`` swaps the bead's
  color map from layer-height gradient to width gradient without
  rebuilding any geometry.

Run: uv run python examples/18_parametric_tube.py
"""

import math
import time

import numpy as np

from threejs_viewer import Animation, viewer


def make_spiral(n=2500, turns=14, radius=1.8, height=4.0):
    t = np.linspace(0, 1, n)
    angle = t * turns * 2 * math.pi
    r = radius * (0.4 + 0.6 * (1 - (2 * t - 1) ** 2))
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    z = t * height
    return np.column_stack([x, y, z]).astype(np.float32), t


def turbo_rgb(values):
    """Cheap turbo-ish colormap: values in [0, 1] → uint32 0x00RRGGBB."""
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - abs(4 * v - 1), 0, 1)
    ri = (r * 255).astype(np.uint32)
    gi = (g * 255).astype(np.uint32)
    bi = (b * 255).astype(np.uint32)
    return (ri << 16) | (gi << 8) | bi


v = viewer()
v.clear()

# --- Build the spine and its variable cross-section parameters. ---
spine, t = make_spiral()
n = len(spine)

# Width modulates along the path so the effect is visible at a glance.
widths = 0.06 + 0.05 * np.sin(t * math.pi * 8) + 0.015 * np.cos(t * math.pi * 24)
heights = 0.03 + 0.02 * np.sin(t * math.pi * 12)
widths = np.maximum(widths, 1e-3).astype(np.float32)
heights = np.maximum(heights, 1e-3).astype(np.float32)

# Initial colormap: paint each ring by normalized layer height (z).
z_frac = (spine[:, 2] - spine[:, 2].min()) / (
    spine[:, 2].max() - spine[:, 2].min() + 1e-9
)
layer_colors = turbo_rgb(z_frac)

v.add_parametric_tube(
    "bead",
    spine=spine,
    widths=widths,
    heights=heights,
    colors=layer_colors,
)

# --- Nozzle indicator: a group that rides the animation, with a capsule
# child that visually represents the bead being deposited right now. ---
v.add_group("nozzle")
v.add_capsule(
    "nozzle_tip",
    radius=0.035,
    length=0.05,
    color=0xFF4400,
    parent="nozzle",
)

# --- Animation: reveal the tube with draw_range while the nozzle group
# follows the spine tip. ---
n_frames = 240
frame_times = np.linspace(0, 6.0, n_frames, dtype=np.float32)

spine_idx = np.linspace(0, n - 1, n_frames).astype(np.int64)
nozzle_transforms = np.tile(np.eye(4, dtype=np.float32).reshape(1, 16), (n_frames, 1))
nozzle_transforms[:, 12:15] = spine[spine_idx]

draw_fracs = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)

anim = Animation(loop=True)
anim.set_frame_times(frame_times)
anim.set_transform_data(["nozzle"], nozzle_transforms.reshape(n_frames, 1, 16))
anim.set_draw_range_data(["bead"], draw_fracs.reshape(n_frames, 1))
v.load_animation(anim)

print("Playing bead animation. Switching color mode in 4 s...")
time.sleep(4.0)

# --- Color swap: recolor by local width without touching geometry. ---
width_frac = (widths - widths.min()) / (widths.max() - widths.min() + 1e-9)
v.update_parametric_tube_colors("bead", turbo_rgb(width_frac))

print("Colors now encode bead width. Close the viewer window to exit.")
while True:
    time.sleep(1.0)
