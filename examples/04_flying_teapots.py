"""
Model Loading Demo — Teapot Carousel

A synchronized ring of teapots orbiting a golden sphere.
The ring breathes (expands/contracts), bobs, and tilts as it rotates.

Demonstrates loading OBJ models and pre-computed animation.

Run: uv run python examples/04_flying_teapots.py
"""

import math
from pathlib import Path

from threejs_viewer import Animation, viewer

TEAPOT_PATH = Path(__file__).parent / "teapot.obj"
N_TEAPOTS = 8
SCALE = 0.35


def make_matrix(pos, rx, ry, rz, scale=1.0):
    """Column-major 4x4 transform: Euler XYZ rotation + translation + scale."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    r00 = cy * cz
    r01 = cy * sz
    r02 = -sy
    r10 = sx * sy * cz - cx * sz
    r11 = sx * sy * sz + cx * cz
    r12 = sx * cy
    r20 = cx * sy * cz + sx * sz
    r21 = cx * sy * sz - sx * cz
    r22 = cx * cy
    s = scale
    return [
        s * r00,
        s * r01,
        s * r02,
        0,
        s * r10,
        s * r11,
        s * r12,
        0,
        s * r20,
        s * r21,
        s * r22,
        0,
        pos[0],
        pos[1],
        pos[2],
        1,
    ]


v = viewer()
v.clear()

v.add_box(
    "ground",
    width=24,
    height=24,
    depth=0.05,
    color=0x1A1A2E,
    position=[0, 0, -0.025],
    roughness=0.9,
    metalness=0.0,
)
v.add_sphere(
    "orb",
    radius=0.8,
    color=0xFFCC22,
    roughness=0.2,
    metalness=0.9,
    position=[0, 0, 2.0],
)

print(f"Loading {N_TEAPOTS} teapots...")
for i in range(N_TEAPOTS):
    v.add_model_binary(f"t{i}", TEAPOT_PATH, format="obj")

duration = 16.0
fps = 30
n_frames = int(duration * fps)
animation = Animation(loop=True)

print(f"Computing {n_frames} frames...")
for fi in range(n_frames):
    t = fi / fps
    transforms = {}

    # Animate ring as a whole
    spin = t * 0.4
    radius = 4.5 + 1.5 * math.sin(t * 0.5)
    height = 2.0 + 1.2 * math.sin(t * 0.35)
    tilt = 0.25 * math.sin(t * 0.28)

    for i in range(N_TEAPOTS):
        phase = i * 2 * math.pi / N_TEAPOTS
        angle = spin + phase
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = height + tilt * radius * math.sin(angle)
        rx = t * 0.8 + phase
        ry = t * 0.5 + i * 0.4
        rz = t * 0.3
        transforms[f"t{i}"] = make_matrix([x, y, z], rx, ry, rz, SCALE)

    animation.add_frame(time=t, transforms=transforms)

animation.add_marker(0.0, "Start", color=0x00FF00)
animation.add_marker(duration * 0.5, "Half-cycle", color=0xFFFF00)

v.load_animation(animation)
print(f"Carousel: {N_TEAPOTS} teapots, {animation.n_frames} frames")
v.wait_for_assets()
