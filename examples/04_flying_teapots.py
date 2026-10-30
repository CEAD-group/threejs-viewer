"""
Teapot carousel

A ring of OBJ teapots orbiting a golden sphere. The ring breathes, bobs and
tilts while each teapot tumbles. All frequencies are whole multiples of the
loop, so the pre-computed animation repeats seamlessly.

Run: uv run python examples/04_flying_teapots.py
"""

from itertools import product
from pathlib import Path

import numpy as np

from threejs_viewer import Animation, viewer

TEAPOT = Path(__file__).parent / "teapot.obj"
N, SCALE, DURATION, FPS = 12, 0.35, 16.0, 20

v = viewer()
v.clear()
v.add_grid("floor", cell_size=1.0, extent=40, center_color=0x888888)
v.add_sphere(
    "orb", radius=0.8, color=0xFFCC22, roughness=0.2, metalness=0.9, position=[0, 0, 2]
)
ids = [f"t{i}" for i in range(N)]
for id in ids:
    v.add_model_binary(id, TEAPOT, format="obj")

# Per-frame ring motion (F, 1) and per-teapot phase (1, N)
t = (np.arange(int(DURATION * FPS) + 1) / FPS)[:, None]
w = 2 * np.pi / DURATION  # one loop per DURATION
phase = np.arange(N)[None, :] * 2 * np.pi / N
angle = w * t + phase
radius = 5 + 1.5 * np.sin(w * t)
z = 2 + 1.2 * np.sin(w * t) + 0.25 * radius * np.sin(w * t) * np.sin(angle)

# Euler XYZ tumble, packed as (F, N, 16) column-major 4x4 matrices
rx, ry, rz = 2 * w * t + phase, w * t + phase / 2, w * t + phase / 4
cx, cy, cz = np.cos(rx), np.cos(ry), np.cos(rz)
sx, sy, sz = np.sin(rx), np.sin(ry), np.sin(rz)
zero = np.zeros_like(cx)
data = np.stack(
    [
        cy * cz, cy * sz, -sy, zero,
        sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy, zero,
        cx * sy * cz + sx * sz, cx * sy * sz - sx * cz, cx * cy, zero,
        radius * np.cos(angle), radius * np.sin(angle), z, zero + 1,
    ],
    axis=-1,
)  # fmt: skip
data[..., :12] *= SCALE  # uniform scale on the rotation columns
data = data.astype(np.float32)

animation = Animation(loop=True)
animation.set_frame_times(t[:, 0])
animation.set_transform_data(ids, data)
animation.add_marker(0.0, "Start", color=0x00FF00)
animation.add_marker(DURATION / 2, "Half-cycle", color=0xFFFF00)

v.load_animation(animation)
v.set_camera(position=[0, -21, 15], target=[0, 0, 2])  # view direction
v.frame_points(list(product([-6.8, 6.8], [-6.8, 6.8], [0, 4.5])), fill=0.85)
v.wait_for_assets()
