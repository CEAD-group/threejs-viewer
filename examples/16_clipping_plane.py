"""
Clipping Plane Demo

Demonstrates the interactive clipping plane feature, including slab mode,
and a toolpath printed on a 45-degree inclined plane with clipping defaults.

- Press C in the viewer to toggle the clipping panel
- Use the panel to adjust axis, position, mode (Single/Slab)
- Arrow keys nudge position; Up/Down adjust thickness in slab mode
- This script also shows programmatic control via set_clipping_plane() and set_clipping_slab()

Run: uv run python examples/16_clipping_plane.py
"""

import math
import time

import numpy as np

from threejs_viewer import Toolpath, viewer

v = viewer()
v.clear()
v.unload_animation()

# Build a dense scene to clip through

# Ground
v.add_box(
    "ground",
    width=10,
    height=10,
    depth=0.05,
    color=0x444444,
    position=[0, 0, -0.025],
    roughness=0.9,
    metalness=0.0,
)

# Nested spheres (like a Russian doll)
radii = [2.0, 1.5, 1.0, 0.5]
colors = [0x2266CC, 0xCC4422, 0x22CC44, 0xCCCC22]
for i, (r, c) in enumerate(zip(radii, colors)):
    v.add_sphere(
        f"shell_{i}",
        radius=r,
        color=c,
        position=[0, 0, 2.0],
        roughness=0.4,
        metalness=0.2,
    )

# Ring of cylinders
for i in range(8):
    angle = i * math.pi / 4
    x = 4 * math.cos(angle)
    y = 4 * math.sin(angle)
    v.add_cylinder(
        f"pillar_{i}",
        radius_top=0.2,
        radius_bottom=0.3,
        height=3.0,
        color=0xB87333,
        position=[x, y, 1.5],
        rotation=[math.pi / 2, 0, 0],
        roughness=0.3,
        metalness=0.7,
    )

# === Toolpath on a 45-degree inclined plane ===
# Rotate local frame -45° about X. The local Z (layer stacking) direction
# maps to world (0, sin45, cos45) via R_x(-45°).
tilt = math.radians(45)
plane_normal = np.array([0, math.sin(tilt), math.cos(tilt)], dtype=np.float32)

# Generate a spiral toolpath in the tilted plane's local coordinates,
# then rotate into world space.
n_points = 50000
n_turns = 30
t = np.linspace(0, 1, n_points)
angle = t * n_turns * 2 * math.pi
r = 0.3 + 1.2 * t  # expanding spiral
local_x = r * np.cos(angle)
local_y = r * np.sin(angle)
local_z = t * n_turns * 0.08  # layer height in local frame

# Rotation matrix: rotate local Z to plane_normal (rotate -45° about X)
cos_t, sin_t = math.cos(-tilt), math.sin(-tilt)
world_x = local_x
world_y = local_y * cos_t - local_z * sin_t
world_z = local_y * sin_t + local_z * cos_t

# Offset to place it next to the spheres
points = np.column_stack([world_x - 4.0, world_y, world_z + 2.0]).astype(np.float32)

tp = Toolpath.from_points(points, bead_width=0.15, bead_height=0.04)
tp.colorize("plasma")
v.add_toolpath(
    "tilted_toolpath",
    tp,
    up_vector=plane_normal.tolist(),
    roughness=0.4,
    metalness=0.15,
)

# Clipping plane along the toolpath's slice plane normal
clip_distance = float(np.dot(points.mean(axis=0), plane_normal))
v.set_clipping_defaults(normal=plane_normal.tolist(), distance=clip_distance)

time.sleep(0.5)  # Let objects load
v.set_clipping_plane(normal=plane_normal.tolist(), distance=clip_distance)

print(
    f"Clipping plane enabled along toolpath slice normal, distance={clip_distance:.2f}"
)
print("Switching to slab mode in 3 seconds...")
time.sleep(3)

# Switch to slab mode — thin slice through the toolpath
v.set_clipping_slab(normal=plane_normal.tolist(), center=clip_distance, thickness=1.0)

print("Slab mode enabled! Showing a 1.0-thick slice through the toolpath")
print()
print("Controls:")
print("  C: toggle clipping panel")
print("  ←→: nudge position")
print("  ↑↓: adjust thickness (slab mode)")
print("  H: toggle helper visibility")
print()
print("Press Ctrl+C to exit.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    v.disconnect()
