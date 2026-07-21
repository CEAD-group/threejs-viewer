"""
Shader Floor Grid Demo (issue #126)

Demonstrates `add_grid`: a first-class, anti-aliased, distance-faded shader
floor grid. Unlike `show_grid` (a fixed THREE.GridHelper toggle) the grid is
a tracked object — it has an id, can be deleted/moved/parented, and several
grids can coexist. Line width is screen-space stable (pixels, not world
units), the two axis lines through the origin get a distinct colour, and a
radial alpha fade dissolves the plane edge so it reads as an infinite floor.
The grid never counts toward camera framing and is never a pick target.

Orbit to a grazing angle to see the anti-aliasing + cell-density thinning:
where cells shrink below a few pixels the grid fades out instead of
collapsing into a solid sheet.

Run: uv run python examples/29_floor_grid.py
"""

from threejs_viewer import viewer

v = viewer()
v.clear()
v.unload_animation()

# The main floor: 1 m cells over a 400 m plane, brighter axis lines,
# a faint dark fill between the lines.
v.add_grid(
    "floor",
    cell_size=1.0,
    extent=400.0,
    line_width=1.5,
    color=0x555555,
    center_color=0x8899AA,
    background_color=0x2A2A2A,
    background_opacity=0.35,
    fade_start=0.35,
)

# A second, coarser grid slightly above: 10 m "major" lines only.
v.add_grid(
    "floor_major",
    cell_size=10.0,
    extent=400.0,
    line_width=2.0,
    color=0x777777,
    fade_start=0.35,
    position=[0, 0, 0.001],
)

# Some content to frame against — double-click / F frames these, not the grid.
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

print("Shader floor grid up. Orbit to a grazing angle to see the AA + fade.")
print("Double-click a box: framing ignores the grid (excluded from bounds).")
