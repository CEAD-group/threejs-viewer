"""
CAD-style inspection: click a part to select it, click empty space to clear.

A small shaft assembly (base plate, two bearing blocks, shaft, pulley) built
from primitives in millimetres. Python owns the selection: each click comes
back through `on_object_click`, and the callback outlines the part with
`set_highlight`, frames it and dimensions its bounding box.

Usage:
    uv run python examples/39_cad_inspection.py
"""

import math
import time

from threejs_viewer import viewer

STEEL = dict(metalness=0.6, roughness=0.35)
QUARTER = math.pi / 2

# id -> (primitive, kwargs, AABB min, AABB max).
# Cylinders are built along Y; a quarter turn about Z lays them along X.
PARTS = {
    "base": (
        "box",
        dict(width=180, height=100, depth=12, color=0x8A939C, position=[0, 0, 6]),
        (-90, -50, 0),
        (90, 50, 12),
    ),
    "block_left": (
        "box",
        dict(width=30, height=44, depth=48, color=0x3D6FA8, position=[-55, 0, 36]),
        (-70, -22, 12),
        (-40, 22, 60),
    ),
    "block_right": (
        "box",
        dict(width=30, height=44, depth=48, color=0x3D6FA8, position=[55, 0, 36]),
        (40, -22, 12),
        (70, 22, 60),
    ),
    "shaft": (
        "cylinder",
        dict(
            radius_top=10,
            radius_bottom=10,
            height=200,
            color=0xC9CED3,
            position=[0, 0, 45],
            rotation=[0, 0, QUARTER],
        ),
        (-100, -10, 35),
        (100, 10, 55),
    ),
    "pulley": (
        "cylinder",
        dict(
            radius_top=32,
            radius_bottom=32,
            height=16,
            color=0xC77B30,
            position=[0, 0, 45],
            rotation=[0, 0, QUARTER],
        ),
        (-8, -32, 13),
        (8, 32, 77),
    ),
}

v = viewer()
v.clear()
v.clear_dimensions()
v.add_grid("floor", cell_size=10, extent=600, center_color=0x888888)
for pid, (kind, kwargs, *_rest) in PARTS.items():
    getattr(v, f"add_{kind}")(pid, **kwargs, **STEEL)
v.set_dimension_format(decimals=0, unit="mm", text_height=6)
v.set_view("home", animate=False)

selected = None


def select(pid):
    """Outline, frame and dimension one part; None clears the selection."""
    global selected
    if selected is not None:
        v.set_highlight(selected, False)
    v.clear_dimensions()
    selected = pid
    if pid is None:
        v.set_view("home")
        return
    kind, _kwargs, lo, hi = PARTS[pid]
    gap = 12  # offset of the dimension lines from the part, in mm
    mid = [(a + b) / 2 for a, b in zip(lo, hi)]
    v.set_highlight(pid, True)
    if kind == "box":
        # Width and depth along the bottom edges, height up the front corner.
        y0, z0 = lo[1], lo[2]
        dims = {
            "X": (lo, (hi[0], y0, z0), (mid[0], y0 - gap, z0), None),
            "Y": ((hi[0], y0, z0), (hi[0], hi[1], z0), (hi[0] + gap, mid[1], z0), None),
            "Z": ((hi[0], y0, z0), (hi[0], y0, hi[2]), (hi[0], y0 - gap, mid[2]), None),
        }
    else:
        # A cylinder along X: its length over the top, its diameter on the end face.
        top, end = (lo[0], 0, hi[2]), (hi[0], 0, hi[2])
        dims = {
            "X": (top, end, (mid[0], 0, hi[2] + gap), None),
            "Z": (
                (hi[0], 0, lo[2]),
                end,
                (hi[0], lo[1] - gap, mid[2]),
                f"Ø{hi[2] - lo[2]} mm",
            ),
        }
    for axis, (p1, p2, origin, label) in dims.items():
        v.add_dimension(
            f"{pid}_{axis}",
            p1=p1,
            p2=p2,
            draw_origin=origin,
            direction=axis,
            color=0xFFAA00,
            label=label,
        )
    pad = gap + 8  # room for the dimension labels
    corners = [
        [x, y, z]
        for x in (lo[0], hi[0] + pad)
        for y in (lo[1] - pad, hi[1])
        for z in (lo[2], hi[2] + pad)
    ]
    v.frame_points(corners, fill=0.8)


def on_click(click):
    # The floor grid is never hit, so a click on it arrives as empty space (id None).
    if click["button"] == 0 and click["id"] != selected:
        select(click["id"])


v.on_object_click(on_click)
print(__doc__)

try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nDone.")
finally:
    v.disconnect()
