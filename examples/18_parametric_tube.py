"""
Variable-cross-section bead with ``add_parametric_tube``.

The viewer receives only the spine plus per-point width, height and colour arrays and
builds the chamfered-hexagon tube itself, so a 2500-point bead uploads in a few KB.
A ``draw_ranges`` animation grows it with smooth frontier-ring morphing while a capsule
tracks the tip. After 4 s the colouring switches from height to width with
``update_parametric_tube_colors``, with no geometry rebuild.

Run: uv run python examples/18_parametric_tube.py   (Ctrl+C to quit)
"""

import math
import time

import numpy as np

from threejs_viewer import Animation, Toolpath, viewer


def make_spiral(n=2500, turns=14, radius=1.8, height=4.0):
    t = np.linspace(0, 1, n)
    angle = t * turns * 2 * math.pi
    r = radius * (0.4 + 0.6 * (1 - (2 * t - 1) ** 2))
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    z = t * height
    return np.column_stack([x, y, z]).astype(np.float32), t


v = viewer()
v.clear()

# --- Build the spine and its variable cross-section parameters. ---
spine, t = make_spiral()
n = len(spine)

# Width modulates along the path so the effect is visible at a glance.
widths = 0.26 + 0.25 * np.sin(t * math.pi * 8) + 0.015 * np.cos(t * math.pi * 24)
heights = 0.13 + 0.12 * np.sin(t * math.pi * 12)
widths = np.maximum(widths, 1e-3).astype(np.float32)
heights = np.maximum(heights, 1e-3).astype(np.float32)

# Colour rings by height; Toolpath.colorize maps scalars through a library colormap.
tp = Toolpath.from_points(spine, widths, heights)
layer_colors = tp.colorize(spine[:, 2], "turbo").packed_colors

v.add_parametric_tube(
    "bead",
    spine=spine,
    widths=widths,
    heights=heights,
    colors=layer_colors,
)


v.add_capsule("nozzle", radius=0.12, length=0.3, color=0xFFFFFF)

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

v.add_grid(
    "floor", cell_size=1.0, extent=30.0, center_color=0x888888, position=[0, 0, -0.2]
)
v.set_camera(position=[6, -7, 5], target=[0, 0, 2])
print("Switching color mode in 4 s. Ctrl+C to quit.")
try:
    time.sleep(4.0)
    v.update_parametric_tube_colors("bead", tp.colorize(widths, "turbo").packed_colors)
    while True:
        time.sleep(1.0)
except KeyboardInterrupt:
    pass
finally:
    v.disconnect()
