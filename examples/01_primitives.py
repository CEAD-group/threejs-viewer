"""
Basic Primitives Demo

Demonstrates adding boxes, spheres, and cylinders with different colors
and positions. A simple "hello world" for threejs-viewer.

Run: uv run python examples/01_primitives.py
"""

from threejs_viewer import viewer

# Connect to viewer (starts WebSocket server, waits for browser)
v = viewer()
v.clear()
v.stop_animation()

# Add a ground plane (flat box)
v.add_box("ground", width=10, height=10, depth=0.05, color=0x444444, position=[0, 0, -0.025])

# Add colored spheres in a row
colors = [0xFF0000, 0xFF7F00, 0xFFFF00, 0x00FF00, 0x0000FF, 0x8B00FF]
for i, color in enumerate(colors):
    x = (i - 2.5) * 1.5
    v.add_sphere(f"sphere_{i}", radius=0.4, color=color, position=[x, 0, 0.4])

# Add some boxes with rotation
import math

for i in range(4):
    angle = i * math.pi / 4
    x = 3 * math.cos(angle)
    y = 3 * math.sin(angle)
    v.add_box(
        f"box_{i}",
        width=0.6,
        height=0.6,
        depth=1.2,
        color=0x4A90D9,
        position=[x, y, 0.6],
        rotation=[0, 0, angle],
    )

# Add cylinders as "pillars"
pillar_positions = [(-4, -4), (-4, 4), (4, -4), (4, 4)]
for i, (x, y) in enumerate(pillar_positions):
    v.add_cylinder(
        f"pillar_{i}",
        radius_top=0.3,
        radius_bottom=0.4,
        height=2.0,
        color=0xB87333,  # Copper color
        position=[x, y, 1.0],
    )

print("Scene created! Press Ctrl+C to exit.")

# Keep the server running
try:
    while True:
        pass
except KeyboardInterrupt:
    v.disconnect()
