"""
Clipping Plane Demo

Demonstrates the interactive clipping plane feature, including slab mode.
A clipping plane slices through geometry, revealing internal structure.
Slab mode uses two parallel planes to show only a thin slice.

- Press C in the viewer to toggle the clipping panel
- Use the panel to adjust axis, position, mode (Single/Slab), and options
- Arrow keys nudge position; Up/Down adjust thickness in slab mode
- This script also shows programmatic control via set_clipping_plane() and set_clipping_slab()

Run: uv run python examples/16_clipping_plane.py
"""

import math
import time

from threejs_viewer import viewer

v = viewer()
v.clear()
v.stop_animation()

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

# Enable clipping plane from Python — slice through Z at height 2.0
time.sleep(0.5)  # Let objects load
v.set_clipping_plane(normal=[0, 0, -1], distance=2.0)

print("Single clipping plane enabled at Z=2.0")
print("Switching to slab mode in 3 seconds...")
time.sleep(3)

# Switch to slab mode — show a 1.0-thick slice centered at Z=2.0
v.set_clipping_slab(normal=[0, 0, 1], center=2.0, thickness=1.0)

print("Slab mode enabled! Showing a 1.0-thick slice around Z=2.0")
print()
print("Controls:")
print("  C: toggle clipping panel")
print("  ←→: nudge position | Shift+←→: nudge faster")
print("  ↑↓: adjust thickness (slab mode)")
print("  T/R: switch move/rotate gizmo")
print("  F: flip normal | H: toggle helper")
print()
print("Press Ctrl+C to exit.")

try:
    while True:
        pass
except KeyboardInterrupt:
    v.disconnect()
