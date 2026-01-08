"""
Flying Utah Teapots Demo

A flock of iconic Utah teapots flying through the scene in graceful patterns.
Demonstrates loading 3D models and animating multiple objects.

The Utah teapot is a classic 3D test model created by Martin Newell in 1975.

Run: uv run python examples/04_flying_teapots.py
"""

import math
from pathlib import Path

from threejs_viewer import Animation, viewer

TEAPOT_PATH = Path(__file__).parent / "teapot.obj"


def make_transform_matrix(position, rotation=(0, 0, 0), scale=1.0):
    """Create a 4x4 transform matrix (column-major) with full rotation."""
    rx, ry, rz = rotation

    # Rotation matrices
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    # Combined rotation matrix R = Rz * Ry * Rx (in column-major order)
    r00 = cy * cz
    r01 = cy * sz
    r02 = -sy
    r10 = sx * sy * cz - cx * sz
    r11 = sx * sy * sz + cx * cz
    r12 = sx * cy
    r20 = cx * sy * cz + sx * sz
    r21 = cx * sy * sz - sx * cz
    r22 = cx * cy

    # Column-major 4x4 matrix
    return [
        scale * r00,
        scale * r01,
        scale * r02,
        0,
        scale * r10,
        scale * r11,
        scale * r12,
        0,
        scale * r20,
        scale * r21,
        scale * r22,
        0,
        position[0],
        position[1],
        position[2],
        1,
    ]


v = viewer()
v.clear()

# Configuration
N_TEAPOTS = 12
TEAPOT_SCALE = 0.4  # Scale for the Three.js teapot model

# Add ground plane
v.add_box("ground", width=20, height=20, depth=0.05, color=0x2A2A2A, position=[0, 0, -0.025])

# Add some reference pillars
for i in range(4):
    angle = i * math.pi / 2 + math.pi / 4
    x, y = 8 * math.cos(angle), 8 * math.sin(angle)
    v.add_cylinder(f"pillar_{i}", radius_top=0.3, radius_bottom=0.4, height=6, color=0x666666, position=[x, y, 3])

# Load teapots via websocket
print(f"Loading {N_TEAPOTS} teapots...")
for i in range(N_TEAPOTS):
    v.add_model_binary(f"teapot_{i}", TEAPOT_PATH, format="obj")


# Define flight paths - each teapot follows a unique path
def flight_path(t, teapot_idx):
    """Compute position and rotation for a teapot at time t."""
    freq_x = 1 + (teapot_idx % 3) * 0.3
    freq_y = 1.5 + (teapot_idx % 4) * 0.2
    freq_z = 0.5 + (teapot_idx % 5) * 0.15

    phase = teapot_idx * 2 * math.pi / N_TEAPOTS

    # Position on a 3D curve
    x = 6 * math.sin(freq_x * t + phase)
    y = 6 * math.cos(freq_y * t + phase * 1.3)
    z = 3 + 2 * math.sin(freq_z * t + phase * 0.7)

    # Rotation - teapots tumble as they fly
    rx = t * 0.5 + phase
    ry = t * 0.3 + teapot_idx * 0.5
    rz = math.sin(t * 0.8 + phase) * 0.5

    return (x, y, z), (rx, ry, rz)


# Create animation
duration = 20.0
fps = 30
n_frames = int(duration * fps)

print("Computing flight paths...")
animation = Animation(loop=True)

for i in range(n_frames):
    t = i / fps
    transforms = {}

    for teapot_idx in range(N_TEAPOTS):
        pos, rot = flight_path(t, teapot_idx)
        transforms[f"teapot_{teapot_idx}"] = make_transform_matrix(pos, rot, TEAPOT_SCALE)

    animation.add_frame(time=t, transforms=transforms)

# Add some markers
animation.add_marker(0.0, "Teapots take flight!", color=0x00FF00)
animation.add_marker(5.0, "Graceful dance", color=0x00FFFF)
animation.add_marker(10.0, "Mid-flight", color=0xFFFF00)
animation.add_marker(15.0, "Coming around", color=0xFF00FF)

v.load_animation(animation)

print(f"Animation loaded: {N_TEAPOTS} teapots, {animation.n_frames} frames")
print("Press Ctrl+C to exit.")

try:
    while True:
        pass
except KeyboardInterrupt:
    v.disconnect()
