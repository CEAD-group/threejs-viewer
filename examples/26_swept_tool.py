"""
Swept 5-axis tool body with ``add_swept_tool``.

A ball-nose tool profile is lofted about the per-station tool axis (which is not the
path tangent) and swept along the contact path, showing the shank as the tool tilts.
Colour is the reorientation rate (tool-axis swing per unit travel, turbo: red is
fastest). A ``draw_ranges`` animation advances the tool along the path.

Run: uv run python examples/26_swept_tool.py
"""

import numpy as np

from threejs_viewer import Animation, Toolpath, viewer


v = viewer()
v.clear()

# --- A 5-axis contact path with a tilting tool axis ---
N = 240
t = np.linspace(0, 1, N)
x = np.linspace(-7, 7, N)
y = 1.5 * np.sin(t * 2 * np.pi * 1.5)
z = 0.4 * np.cos(t * 2 * np.pi * 2.0)
positions = np.column_stack([x, y, z]).astype(np.float32)

# Tool axis (tip -> holder) leans back and forth and rolls — uneven wrist speed.
lean = 0.7 * np.sin(t * 2 * np.pi * 2.0)
ax = np.sin(lean)
ay = 0.35 * np.sin(t * 2 * np.pi * 3.0)
az = np.cos(lean)
axes = np.column_stack([ax, ay, az]).astype(np.float32)

# --- Tool profile: ball nose (radius R) + shank (radius rs, length L) ---
R, rs, L = 0.6, 0.5, 6.0
ball_h = np.linspace(0.0, R, 7)
ball_r = np.sqrt(np.maximum(R**2 - (R - ball_h) ** 2, 0.0))
profile = np.array(
    [*zip(ball_h, ball_r), (R, rs), (R + L, rs)],
    dtype=np.float32,
)

# --- Colour by reorientation rate (deg of tool-axis swing per unit travel) ---
unit = axes / np.linalg.norm(axes, axis=1, keepdims=True)
dot = np.clip(np.sum(unit[1:] * unit[:-1], axis=1), -1.0, 1.0)
dist = np.linalg.norm(np.diff(positions, axis=0), axis=1)
rate = np.zeros(N, dtype=np.float32)
rate[1:] = np.degrees(np.arccos(dot)) / np.maximum(dist, 1e-6)
rate_n = rate / (np.percentile(rate, 98) + 1e-9)
colors = (
    Toolpath.from_points(positions, 0.1, 0.1)
    .colorize(np.clip(rate_n, 0, 1), "turbo")
    .packed_colors
)

v.add_swept_tool(
    "shank",
    positions,
    axes,
    profile,
    colors=colors,
    sections=20,
    opacity=0.9,
    metalness=0.4,
    roughness=0.5,
)

# A thin line marking the cutting-contact path for reference.
v.add_polyline("contact", positions, color=0x222222, line_width=2, fat=False)

# --- Reveal the swept body progressively along the path ---
n_frames = 200
anim = Animation(loop=True)
anim.set_frame_times(np.linspace(0, 6.0, n_frames, dtype=np.float32))
anim.set_draw_range_data(
    ["shank"], np.linspace(0, 1, n_frames, dtype=np.float32).reshape(n_frames, 1)
)
v.load_animation(anim)

v.add_grid("floor", cell_size=1.0, extent=40.0, center_color=0x888888)
v.set_view("iso", animate=False)
v.frame_points(
    np.array(np.meshgrid(*zip([-8, -3, -1], [8, 3, 7]))).reshape(3, -1).T, fill=0.8
)
v.wait_for_assets()
