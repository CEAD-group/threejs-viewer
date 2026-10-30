"""
A CAD part from exact BRep geometry: an engine connecting rod on a floor grid.

The rod is modelled with usd_brep booleans (tapered I-beam shank, crank and
piston-pin ends, bolt lugs, bores) and handed straight to the viewer with
`add_usda`, which tessellates it and draws shaded faces plus the BRep edges.
Dimensions are in millimetres; the grid cells are 10 mm.

Requires the usd_brep package in this environment.

Usage:
    uv run python examples/38_usd_connecting_rod.py
"""

import math

import usd_brep
from usd_brep import booleans, primitives

from threejs_viewer import viewer

LENGTH = 150.0  # crank bore centre to piston-pin bore centre
THICKNESS = 22.0
# The shank narrows from 32 mm wide at the crank end to 18 mm at the pin end.
TAPER_DEG = math.degrees(math.atan((16.0 - 9.0) / 115.0))


def tapered_bar(x0, x1, half_width, z0, z1):
    """Bar along X, `half_width` wide at x=25 and narrowing toward the pin end."""
    bar = primitives.create_box((x0, -40.0, z0), x1 - x0, 80.0, z1 - z0)
    for side in (1.0, -1.0):
        y0 = half_width if side > 0 else -half_width - 60.0
        cutter = primitives.create_box(
            (-20.0, y0, z0 - 5.0), 220.0, 60.0, z1 - z0 + 10.0
        )
        pivot = (25.0, side * half_width, 0.0)
        usd_brep.rotate(cutter, pivot, (0.0, 0.0, 1.0), -side * TAPER_DEG)
        bar = booleans.boolean_difference(bar, cutter)
    return bar


def cylinder_along_x(x0, y, z, radius, length):
    """Cylinders are created along Z; lay one down along X."""
    cylinder = primitives.create_cylinder((0.0, 0.0, 0.0), radius, length)
    usd_brep.rotate(cylinder, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 90.0)
    return usd_brep.translate(cylinder, (x0, y, z))


def build_connecting_rod():
    # I-beam shank: a tapered bar with a rounded pocket sunk into each face.
    rod = tapered_bar(0.0, LENGTH, 16.0, 3.0, 19.0)
    for z0, z1 in ((14.0, 25.0), (-3.0, 8.0)):
        pocket = tapered_bar(42.0, 122.0, 11.5, z0, z1)
        upright = [
            edge
            for edge in pocket.edges()
            if abs(edge.tangent_at(sum(edge.parameter_range()) / 2)[2]) > 0.99
        ]
        usd_brep.fillet_edges(pocket, upright, radius=3.0)
        rod = booleans.boolean_difference(rod, pocket)

    # Crank (big) end with its two bolt lugs, and the piston-pin (small) end.
    crank_end = primitives.create_cylinder((0.0, 0.0, 0.0), 32.0, THICKNESS)
    rod = booleans.boolean_union(rod, crank_end)
    pin_end = primitives.create_cylinder((LENGTH, 0.0, 1.0), 17.0, THICKNESS - 2.0)
    rod = booleans.boolean_union(rod, pin_end)
    for y0 in (26.0, -44.0):
        lug = primitives.create_box((-12.0, y0, 0.0), 24.0, 18.0, THICKNESS)
        rod = booleans.boolean_union(rod, lug)

    # Bores, bolt holes through the lugs, and the oil hole into the pin bore.
    crank_bore = primitives.create_cylinder((0.0, 0.0, -5.0), 24.0, THICKNESS + 10.0)
    rod = booleans.boolean_difference(rod, crank_bore)
    pin_bore = primitives.create_cylinder((LENGTH, 0.0, -5.0), 10.0, THICKNESS + 10.0)
    rod = booleans.boolean_difference(rod, pin_bore)
    for y in (37.0, -37.0):
        bolt_hole = cylinder_along_x(-20.0, y, THICKNESS / 2, 4.0, 40.0)
        rod = booleans.boolean_difference(rod, bolt_hole)
    oil_hole = cylinder_along_x(LENGTH + 5.0, 0.0, THICKNESS / 2, 2.0, 20.0)
    return booleans.boolean_difference(rod, oil_hole)


rod = build_connecting_rod()
info = rod.info()
print(
    f"Connecting rod: {info['faces']} faces, {info['edges']} edges, "
    f"volume {info['volume'] / 1000:.1f} cm^3, manifold: {info['manifold']}"
)

v = viewer()
v.clear()
v.add_grid("floor", cell_size=10.0, extent=600.0, center_color=0x888888)
v.add_usda(rod, id="rod", color=0x9AA3AD, metalness=0.6, roughness=0.4)
v.set_camera(position=[240.0, -190.0, 150.0], target=[70.0, 0.0, 5.0])

print("Waiting for browser...")
v.wait_for_assets()
print("Done.")
