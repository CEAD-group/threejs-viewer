"""
Material properties

A grid of spheres varying roughness (columns, left to right) and metalness
(rows, front to back), plus a polished metal box and a matte cylinder.

Run: uv run python examples/13_material_properties.py
"""

import math

from threejs_viewer import viewer

v = viewer()
v.clear()
v.unload_animation()
v.add_grid("floor", cell_size=1.0, extent=40, center_color=0x888888)

# 5x5 grid of spheres: roughness varies across columns, metalness across rows
n = 5
spacing = 1.5
base_color = 0xD4956A  # warm copper tone shows PBR well

for row in range(n):
    metalness = row / (n - 1)
    for col in range(n):
        roughness = col / (n - 1)
        v.add_sphere(
            f"sphere_{row}_{col}",
            radius=0.5,
            color=base_color,
            roughness=roughness,
            metalness=metalness,
            position=[col * spacing, row * spacing, 0.5],
        )

# Metallic box
v.add_box(
    "metal_box",
    width=1.2,
    height=1.2,
    depth=1.2,
    color=0xC0C0C0,
    roughness=0.1,
    metalness=1.0,
    position=[-2.5, 1.5, 0.6],
)

# Rough cylinder
v.add_cylinder(
    "rough_cylinder",
    radius_top=0.5,
    radius_bottom=0.5,
    height=1.5,
    color=0x8B4513,
    roughness=1.0,
    metalness=0.0,
    position=[-2.5, 4.5, 0.75],
    rotation=[math.pi / 2, 0, 0],
)

v.set_camera(position=[3, -9, 12], target=[1, 3, 0])
v.wait_for_assets()
