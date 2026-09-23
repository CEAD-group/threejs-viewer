"""
Rotary and linear axis-control range widgets: a line showing an axis's range
of motion with a short, thicker handle segment marking the current value —
and, in this example, actually driving the box it's attached to.

Four boxes, one per `kind`:

  * **box_1 — rotary**: an arc from 0 to 180 degrees, radius 1. Rotates
    box_1 about X; starts at 45 degrees.
  * **box_2 — rotary_unlimited**: a full circle, radius 1. Rotates box_2
    about Z; drag past +-180 degrees and it keeps turning instead of
    snapping back.
  * **box_3 — linear**: a segment from -1 to 3 metres along Y. Translates
    box_3 along Y.
  * **box_4 — linear_unlimited**: a 2 m window centred on the current value,
    along Z — the window re-centres as you drag, always keeping +-1 m of
    travel visible in either direction. Translates box_4 along Z; starts
    1 m up.

Each control is attached to a static "mount" group rather than to the box
itself, and the box is a *child* of that mount: dragging reports a value
back to Python, which writes it straight onto the box's local
position/rotation on the control's axis. The mount never moves, so the
control's own frame stays put too — attaching a control directly to the
object it also drives would double-count every update, since the widget
re-anchors on its target's live pose every frame. Only the handle's hitbox
is draggable (and highlights on hover); the line is a guide.

Run: uv run python examples/36_axis_control.py
(then drag a handle in the browser; Ctrl+C to quit)
"""

import logging
import math
import time

from threejs_viewer import viewer

log = logging.getLogger("axis-control")
log.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)-5s [%(name)s] %(message)s", "%H:%M:%S")
)
log.addHandler(_handler)
log.propagate = False


v = viewer()
v.clear()
v.unload_animation()
v.show_grid(True)

RED = 0xAE4346
RED_HOVER = 0xC74E52
GREEN = 0x5D9C74
GREEN_HOVER = 0x6DB587
BLUE = 0x4369A2
BLUE_HOVER = 0x4D7ABD

# id -> which axis component of the child box's local position/rotation an
# axis-control value drives.
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
CONTROLS = {
    "axis_control_1": {"box": "box_1", "axis": "x", "rotary": True},
    "axis_control_2": {"box": "box_2", "axis": "z", "rotary": True},
    "axis_control_3": {"box": "box_3", "axis": "y", "rotary": False},
    "axis_control_4": {"box": "box_4", "axis": "z", "rotary": False},
}

for i in range(4):
    mount_id = f"mount_{i + 1}"
    v.add_group(mount_id, position=[(i - 1.5) * 2.5, 0, 0.5])
    v.add_box(
        f"box_{i + 1}",
        width=1.0,
        height=1.0,
        depth=1.0,
        color=0xE0E0E0,
        roughness=0.5,
        parent=mount_id,
    )

v.add_axis_control(
    "axis_control_1",
    target_id="mount_1",
    axis="x",
    kind="rotary",
    value=math.radians(45),
    min=math.radians(0),
    max=math.radians(180),
    color=RED,
    hover_color=RED_HOVER,
    radius=1.0,
)
v.add_axis_control(
    "axis_control_2",
    target_id="mount_2",
    axis="z",
    kind="rotary_unlimited",
    value=math.radians(45),
    color=BLUE,
    hover_color=BLUE_HOVER,
    radius=1.0,
)
v.add_axis_control(
    "axis_control_3",
    target_id="mount_3",
    axis="y",
    kind="linear",
    value=0.0,
    min=-1.0,
    max=3.0,
    color=GREEN,
    hover_color=GREEN_HOVER,
)
v.add_axis_control(
    "axis_control_4",
    target_id="mount_4",
    axis="z",
    kind="linear_unlimited",
    value=1.0,
    color=BLUE,
    hover_color=BLUE_HOVER,
)


def apply_axis_value(control_id: str, value: float) -> None:
    """Write a control's value onto its box's local rotation (rotary) or
    position (linear), on the control's own axis component only."""
    spec = CONTROLS[control_id]
    idx = AXIS_INDEX[spec["axis"]]
    vec = [0.0, 0.0, 0.0]
    vec[idx] = value
    key = "rotation" if spec["rotary"] else "position"
    v.batch_update({spec["box"]: {key: vec}})


# Pose each box to match its control's starting value.
for control_id in CONTROLS:
    initial_value = {
        "axis_control_1": math.radians(45),
        "axis_control_2": math.radians(45),
        "axis_control_3": 0.0,
        "axis_control_4": 1.0,
    }[control_id]
    apply_axis_value(control_id, initial_value)


def on_change(m):
    """Drive the box transform from a dragged axis control's value, and log
    it (browser -> Python)."""
    apply_axis_value(m["id"], m["value"])
    log.info("%-16s %-4s value=%.3f", m["id"], m["phase"], m["value"])


v.on_axis_control_change(on_change)

print(__doc__)

try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nDone.")
finally:
    v.disconnect()
