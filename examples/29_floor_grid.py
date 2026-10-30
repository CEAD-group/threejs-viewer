"""
Shader floor grid with ``add_grid``.

The grid is an anti-aliased plane with a screen-space line width, a distinct colour on
the origin axes and a radial fade at the edge. Here a 1 m minor grid and a 10 m
major grid share one floor (every 10th minor line is a major line). Grids are tracked objects but are ignored by
framing and picking. Orbit to a grazing angle to see cells thin out instead of
collapsing into a solid sheet.

Run: uv run python examples/29_floor_grid.py
"""

import numpy as np

from threejs_viewer import viewer

v = viewer()
v.clear()

# One floor: 1 m minor lines that fade out within ~15 m, and 10 m major lines on
# every 10th minor line that reach further. The major plane sits a hair below.
v.add_grid(
    "floor",
    cell_size=1.0,
    extent=40.0,
    line_width=1.2,
    color=0x4A4A4A,
    center_color=0x8899AA,
    fade_start=0.15,
)
v.add_grid(
    "floor_major",
    cell_size=10.0,
    extent=40.0,
    line_width=2.0,
    color=0x707070,
    fade_start=0.5,
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

v.set_view("iso", animate=False)
v.frame_points(
    np.array(np.meshgrid(*zip([-4, -1, 0], [4, 4.8, 1.6]))).reshape(3, -1).T, fill=0.8
)
v.wait_for_assets()
