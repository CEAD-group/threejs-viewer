"""
Primitives

Spheres, boxes and cylinders with colours, PBR roughness/metalness and a parent
group, standing on a shader floor grid.

Run: uv run python examples/01_primitives.py
"""

import math
from itertools import product

from threejs_viewer import viewer

v = viewer()
v.clear()
v.unload_animation()

v.add_grid("floor", cell_size=1.0, extent=30, center_color=0x888888)

# A rainbow of spheres in a row
rainbow = [0xFF0000, 0xFF7F00, 0xFFFF00, 0x00FF00, 0x0000FF, 0x8B00FF]
for i, color in enumerate(rainbow):
    v.add_sphere(
        f"sphere_{i}",
        radius=0.4,
        color=color,
        position=[(i - 2.5) * 1.5, 0, 0.4],
        roughness=0.3,
        metalness=0.2,
    )

# Boxes on an arc, each turned to face the centre
for i in range(4):
    angle = i * math.pi / 4
    v.add_box(
        f"box_{i}",
        width=0.6,
        height=0.6,
        depth=1.2,
        color=0x4A90D9,
        position=[3 * math.cos(angle), 3 * math.sin(angle), 0.6],
        rotation=[0, 0, angle],
        roughness=0.5,
        metalness=0.1,
    )

# Pillars share a parent group: v.delete("pillars") would remove them all
v.add_group("pillars")
for i, (x, y) in enumerate([(-4, -4), (-4, 4), (4, -4), (4, 4)]):
    v.add_cylinder(
        f"pillar_{i}",
        radius_top=0.3,
        radius_bottom=0.4,
        height=2.0,
        color=0xB87333,
        position=[x, y, 1.0],
        rotation=[math.pi / 2, 0, 0],  # cylinders are built along Y
        roughness=0.3,
        metalness=0.7,
        parent="pillars",
    )

v.set_camera(position=[7, -9, 6], target=[0, 0, 0.5])  # view direction
v.frame_points(list(product([-4.4, 4.4], [-4.4, 4.4], [0, 2])), fill=0.85)
v.wait_for_assets()
