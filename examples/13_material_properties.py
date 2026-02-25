"""
Example 13: Material Properties (Roughness & Metalness)

Grid of spheres varying roughness (columns) and metalness (rows)
to demonstrate PBR material control on primitives. Also includes
a metallic box and a rough cylinder.
"""

import time

from threejs_viewer import viewer

v = viewer()
v.clear()

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
            position=[col * spacing, row * spacing, 0],
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
    position=[-2.5, 1.5, 0],
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
    position=[-2.5, 4.5, 0],
)

print("Material properties demo loaded.")
print(f"Sphere grid: roughness 0→1 (left→right), metalness 0→1 (bottom→top)")
print("Left column: metallic box (shiny) and rough cylinder (matte)")

time.sleep(0.5)
v.disconnect()
