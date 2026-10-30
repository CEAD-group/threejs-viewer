"""
Stress Test

A million-point polyline wound into a torus knot, with 600 primitives racing
along it. Every frame all 600 poses go out in one batch_update at 60 fps.

Run: uv run python examples/07_stress_test.py   (Ctrl+C to stop)
"""

import colorsys
import time

import numpy as np

from threejs_viewer import viewer

N_POINTS, N_FOLLOWERS = 1_000_000, 600


def unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


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
for i, fid in enumerate(ids):
    size = rng.uniform(0.05, 0.15)
    rgb = colorsys.hsv_to_rgb(rng.random(), 0.9, 0.9)
    color = sum(int(c * 255) << s for c, s in zip(rgb, (16, 8, 0)))
    if i % 3 == 0:
        v.add_sphere(fid, radius=size, color=color)
    elif i % 3 == 1:
        v.add_box(fid, width=size, height=2 * size, depth=size, color=color)
    else:
        v.add_cylinder(
            fid, radius_top=0, radius_bottom=size / 2, height=2 * size, color=color
        )

v.frame_points(points[::1000], fill=0.85)
v.wait_for_assets(disconnect=False)

offsets = rng.integers(0, N_POINTS, N_FOLLOWERS)
speeds = rng.uniform(5, 30, N_FOLLOWERS)
zero, one = np.zeros((N_FOLLOWERS, 1)), np.ones((N_FOLLOWERS, 1))
frame, t0 = 0, time.time()
try:
    while True:
        i = ((offsets + frame * speeds) % N_POINTS).astype(int)
        pos = points[i]
        # Orientation with local +Y along the direction of travel, +Z kept upright
        y = unit(points[(i + 100) % N_POINTS] - pos)
        x = unit(np.cross(y, [0, 0, 1]))
        m = np.hstack([x, zero, y, zero, np.cross(x, y), zero, pos, one])
        v.batch_update({fid: {"matrix": row.tolist()} for fid, row in zip(ids, m)})
        frame += 1
        if frame % 60 == 0:
            rate = frame / (time.time() - t0)
            print(f"{N_FOLLOWERS} objects, {rate:.1f} updates/s", end="\r")
        time.sleep(1 / 60)
except KeyboardInterrupt:
    v.disconnect()
