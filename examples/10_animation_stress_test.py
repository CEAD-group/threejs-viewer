"""
Animation Stress Test

The torus knot of 07_stress_test.py with 600 followers, but pre-computed as
one Animation (600 objects x 1800 frames) with timeline scrubbing.
Binary channels: `transforms` (pose per follower) and `colors` (palette index
from height, uint8).

Run: uv run python examples/10_animation_stress_test.py
"""

import colorsys

import numpy as np

from threejs_viewer import Animation, viewer

N_POINTS, N_FOLLOWERS, FPS, DURATION = 1_000_000, 600, 30, 60


def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


# Torus knot centre line, wrapped by a helix of 500 turns (radius 0.4)
t = np.linspace(0, 2 * np.pi, N_POINTS)
r = 1 + 0.5 * np.cos(7 * t)
centre = 5 * np.stack([r * np.cos(3 * t), r * np.sin(3 * t), 0.5 * np.sin(7 * t)], 1)
tangent = unit(np.gradient(centre, axis=0))
normal = unit(np.cross([0, 0, 1], tangent))
binormal = np.cross(tangent, normal)
phase = (500 * t)[:, None]
helix = 0.4 * (np.cos(phase) * normal + np.sin(phase) * binormal)
points = (centre + helix + [0, 0, 4]).astype(np.float32)

v = viewer()
v.clear()
v.add_grid("floor", cell_size=1.0, extent=80.0, fade_start=0.4)
v.add_polyline(
    "knot", points, colors=t.astype(np.float32), colormap="turbo", line_width=2
)

rng = np.random.default_rng(0)
ids = [f"f{i}" for i in range(N_FOLLOWERS)]
for i, fid in enumerate(ids):  # grey: the colors channel repaints them every frame
    size = rng.uniform(0.05, 0.15)
    if i % 3 == 0:
        v.add_sphere(fid, radius=size, color=0x888888)
    elif i % 3 == 1:
        v.add_box(fid, width=size, height=2 * size, depth=size, color=0x888888)
    else:
        v.add_cylinder(
            fid, radius_top=0, radius_bottom=size / 2, height=2 * size, color=0x888888
        )

# Fully vectorized: (frames, followers) point indices -> poses and colours
n_frames = DURATION * FPS
offsets = rng.integers(0, N_POINTS, N_FOLLOWERS)
speeds = rng.exponential(10, N_FOLLOWERS) + 1
i = ((offsets + np.arange(n_frames)[:, None] * speeds) % N_POINTS).astype(int)
pos = points[i]
y = unit(points[(i + 100) % N_POINTS] - pos)
x = unit(np.cross(y, [0, 0, 1]))
# Followers grow from 0.4x to 1.2x along X
scale = 0.4 + 0.8 * (pos[..., :1] - pos[..., :1].min()) / np.ptp(pos[..., :1])
zero, one = np.zeros_like(scale), np.ones_like(scale)
transforms = np.concatenate(
    [x * scale, zero, y * scale, zero, np.cross(x, y) * scale, zero, pos, one], axis=-1
).astype(np.float32)

# Palette index from height, blue (low) to red (high)
z = pos[..., 2]
height = ((z - z.min()) / np.ptp(z) * 255).astype(np.uint8)
palette = [
    int.from_bytes(
        bytes(
            int(c * 255) for c in colorsys.hsv_to_rgb(0.65 * (1 - k / 255), 0.8, 0.95)
        )
    )
    for k in range(256)
]

animation = Animation(loop=True)
animation.set_frame_times(np.arange(n_frames) / FPS)
animation.set_transform_data(ids, transforms)
animation.add_channel(
    "colors", ids, height, dtype="uint8", metadata={"colormap": palette}
)
for k, name in enumerate(["Start", "Quarter", "Halfway", "Three quarters"]):
    animation.add_marker(k * DURATION / 4, name)

v.load_animation(animation)
v.frame_points(points[::1000], fill=1.0)
v.wait_for_assets()
