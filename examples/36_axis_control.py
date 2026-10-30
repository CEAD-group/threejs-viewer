"""
Axis-control range widgets that drive the boxes they are attached to.

One box per ``kind``: rotary (arc 0-180 degrees about X), rotary_unlimited (full
circle about Z, keeps turning past +-180), linear (-1..3 along Y) and linear_unlimited
(a 2 m window along Z that re-centres as you drag). Drag a handle: its value comes back
to Python, which writes it onto the box. Each control targets a static mount group and
the box is a child of it, because the widget re-anchors on its target every frame and
would double-count updates if attached to the box it drives.

Run: uv run python examples/36_axis_control.py   (Ctrl+C to quit)
"""

import math
import time

from threejs_viewer import viewer

RED, GREEN, BLUE = (0xAE4346, 0xC74E52), (0x5D9C74, 0x6DB587), (0x4369A2, 0x4D7ABD)
rad = math.radians

# name, axis, kind, start value, (color, hover color), extra kwargs
SPECS = [
    ("rotary", "x", "rotary", rad(45), RED, dict(min=0.0, max=rad(180), radius=1.0)),
    ("round", "z", "rotary_unlimited", rad(45), BLUE, dict(radius=1.0)),
    ("slide", "y", "linear", 0.0, GREEN, dict(min=-1.0, max=3.0)),
    ("window", "z", "linear_unlimited", 1.0, BLUE, {}),
]

v = viewer()
v.clear()
v.add_grid("floor", cell_size=1.0, extent=30.0, center_color=0x888888)


def drive(name, axis, kind, value):
    """Write a control's value onto its box's local rotation or position."""
    vec = [0.0, 0.0, 0.0]
    vec["xyz".index(axis)] = value
    key = "rotation" if "rotary" in kind else "position"
    v.batch_update({f"box_{name}": {key: vec}})


for i, (name, axis, kind, start, (color, hover), extra) in enumerate(SPECS):
    v.add_group(f"mount_{name}", position=[(i - 1.5) * 3.0, 0, 0.5])
    v.add_box(
        f"box_{name}", 1, 1, 1, color=0xE0E0E0, roughness=0.5, parent=f"mount_{name}"
    )
    v.add_axis_control(
        f"ctl_{name}",
        target_id=f"mount_{name}",
        axis=axis,
        kind=kind,
        value=start,
        color=color,
        hover_color=hover,
        **extra,
    )
    drive(name, axis, kind, start)

by_id = {f"ctl_{s[0]}": s for s in SPECS}


def on_change(m):
    name, axis, kind, *_ = by_id[m["id"]]
    drive(name, axis, kind, m["value"])
    if m["phase"] == "end":
        print(f"{m['id']}: {m['value']:.3f}")


v.on_axis_control_change(on_change)
v.set_camera(position=[0, -10, 6], target=[0, 0.5, 1])

print("Drag a handle. Ctrl+C to quit.")
try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
finally:
    v.disconnect()
