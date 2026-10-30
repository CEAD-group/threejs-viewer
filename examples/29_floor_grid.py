"""
Shader floor grid with ``add_grid``.

The grid is an anti-aliased plane with a screen-space line width, a distinct colour on
the origin axes and a radial fade at the edge. Here a fine 1 m grid with a faint fill
is layered over a coarse 10 m major grid. Grids are tracked objects but are ignored by
framing and picking. Orbit to a grazing angle to see cells thin out instead of
collapsing into a solid sheet.

Run: uv run python examples/29_floor_grid.py
"""

from threejs_viewer import viewer

v = viewer()
v.clear()

# The main floor: 1 m cells over a 40 m plane, brighter axis lines,
# a faint dark fill between the lines.
v.add_grid(
    "floor",
    cell_size=1.0,
    extent=40.0,
    line_width=1.5,
    color=0x555555,
    center_color=0x8899AA,
    background_color=0x2A2A2A,
    background_opacity=0.35,
    fade_start=0.35,
)

# A second, much larger 10 m "major" grid, nudged slightly below the
# fine one so the two transparent planes never z-fight.
v.add_grid(
    "floor_major",
    cell_size=10.0,
    extent=400.0,
    line_width=2.0,
    color=0x777777,
    background_opacity=0,
    fade_start=0.15,
    position=[0, 0, -0.001],
)

# Content to frame against; the grid is excluded from framing bounds.
for i, color in enumerate([0xE24A4A, 0x4AE24A, 0x4A90D9]):
    v.add_box(
        f"box_{i}",
        width=1.2,
        height=1.2,
        depth=1.2,
        color=color,
        position=[(i - 1) * 3.0, 0, 0.6],
        roughness=0.5,
    )
v.add_sphere("ball", radius=0.8, color=0xD9C24A, position=[0, 4, 0.8])

v.set_camera(position=[9, -14, 7], target=[0, 1, 0.5])
v.wait_for_assets()
