"""Robot cell driven by a kinematic chain.

A 6-axis arm on a linear rail moves parts between two tables. The chain description
lives in Python; the viewer runs the forward kinematics, so the animation is just 7 axis
values per keyframe (`set_joint_data`). Python evaluates the same FK with
`kinematics.compose` to draw the tool-centre-point path.

Run: uv run python examples/14_robot_cell.py
"""

import math

import numpy as np

from threejs_viewer import Animation, kinematics, viewer

ROBOT, DARK, STEEL, PART = 0xE8731A, 0x2E3338, 0xB8BEC6, 0x3FA7D6
X, Y, Z = [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]
AXES = ["E1", "A1", "A2", "A3", "A4", "A5", "A6"]


def link(name, parent, xyz, axis=None, drive=None, prismatic=False):
    """A link at offset `xyz` on its parent, optionally moved by one drive axis."""
    origin = np.eye(4)
    origin[:3, 3] = xyz
    joint = None
    if drive:
        joint = {"type": "prismatic" if prismatic else "revolute", "axis": axis}
        joint["value"] = {"offset": 0.0, "terms": {drive: 1.0}}
        if not prismatic:
            joint["point"] = [0.0, 0.0, 0.0]
    return {"name": name, "parent": parent, "origin": origin.tolist(), "joint": joint}


CELL = {
    "format": kinematics.FORMAT,
    "version": kinematics.VERSION,
    "axes": [{"name": n, "kind": "linear" if n == "E1" else "rotary"} for n in AXES],
    "links": [
        link("carriage", None, [0, 0, 0.2], X, "E1", prismatic=True),
        link("base", "carriage", [0, 0, 0.1], Z, "A1"),
        link("upper_arm", "base", [0.15, 0, 0.45], Y, "A2"),
        link("elbow", "upper_arm", [0, 0, 0.7], Y, "A3"),
        link("forearm", "elbow", [0.2, 0, 0], X, "A4"),
        link("wrist", "forearm", [0.45, 0, 0], Y, "A5"),
        link("flange", "wrist", [0.1, 0, 0], X, "A6"),
        link("tcp", "flange", [0.22, 0, 0]),
    ],
}

v = viewer()
v.clear()
v.unload_animation()
v.add_grid("floor", cell_size=0.5, extent=14, color=0x4A4F55, center_color=0x6A7078)

# Static cell: rail and two tables; the robot moves a part from table A to table B.
v.add_box("rail", width=4.4, height=0.5, depth=0.2, color=DARK, position=[0, 0, 0.1])
for name, x, y in [("table_a", -1.3, -1.0), ("table_b", 1.3, 1.0)]:
    v.add_box(name, width=0.8, height=0.6, depth=0.5, color=0x5B6770,
              position=[x, y, 0.25], roughness=0.8)  # fmt: skip
    v.add_box(f"part_{name[-1]}", width=0.2, height=0.2, depth=0.12, color=PART,
              position=[x, y, 0.56])  # fmt: skip

# One group per link; its primitives sit in the link frame (origin at the joint).
cyl_x, cyl_z = [0, 0, math.pi / 2], [math.pi / 2, 0, 0]  # cylinders default to +Y
bodies = {
    "carriage": [("box", dict(width=0.7, height=0.6, depth=0.1, color=DARK), [0, 0, 0.05], None)],
    "base": [("cylinder", dict(radius_top=0.2, radius_bottom=0.26, height=0.42, color=ROBOT), [0.05, 0, 0.21], cyl_z)],
    "upper_arm": [
        ("cylinder", dict(radius_top=0.15, radius_bottom=0.15, height=0.34, color=DARK), [0, 0, 0], None),
        ("box", dict(width=0.2, height=0.22, depth=0.72, color=ROBOT), [0, 0, 0.35], None),
    ],
    "elbow": [
        ("cylinder", dict(radius_top=0.13, radius_bottom=0.13, height=0.3, color=DARK), [0, 0, 0], None),
        ("box", dict(width=0.3, height=0.2, depth=0.2, color=ROBOT), [0.08, 0, 0], None),
    ],
    "forearm": [("cylinder", dict(radius_top=0.07, radius_bottom=0.09, height=0.42, color=ROBOT), [0.21, 0, 0], cyl_x)],
    "wrist": [("sphere", dict(radius=0.08, color=DARK), [0, 0, 0], None)],
    "flange": [
        ("cylinder", dict(radius_top=0.06, radius_bottom=0.06, height=0.03, color=STEEL, metalness=0.8), [0.015, 0, 0], cyl_x),
        ("cylinder", dict(radius_top=0.03, radius_bottom=0.05, height=0.16, color=STEEL, metalness=0.8), [0.11, 0, 0], cyl_x),
        ("cylinder", dict(radius_top=0.003, radius_bottom=0.025, height=0.04, color=0xF2C230), [0.2, 0, 0], cyl_x),
    ],
}  # fmt: skip
for name, prims in bodies.items():
    v.add_group(name)
    for i, (kind, kw, pos, rot) in enumerate(prims):
        getattr(v, f"add_{kind}")(
            f"{name}_{i}", position=pos, rotation=rot, parent=name, **kw
        )
v.set_kinematic_chain("cell", CELL, object_ids={n: n for n in bodies})
# A second copy of the part rides the gripper (flange +x points out of the tool).
v.add_box("part_held", width=0.12, height=0.2, depth=0.2, color=PART,
          position=[0.28, 0, 0], parent="flange", visible=False)  # fmt: skip

# Joint-space waypoints [E1 m, A1..A6 deg]: above A, down to A, up, travel, down to B, up.
above_a = [-0.9, -111.8, 25.0, -10.5, 0, 75.5, 0]
at_a = [-0.9, -111.8, 34.9, 1.2, 0, 53.9, 0]
above_b = [0.9, 68.2, 25.0, -10.5, 0, 75.5, 0]
at_b = [0.9, 68.2, 34.9, 1.2, 0, 53.9, 0]
home = [0.0, -30.0, -10.0, 20.0, 0, 80.0, 0]
waypoints = np.array([home, above_a, at_a, above_a, above_b, at_b, above_b, home])
seconds = np.array([0.0, 1.6, 2.3, 3.0, 5.4, 6.1, 6.8, 8.4])

# Eased motion between waypoints, keyed at 10 Hz; the viewer lerps the joints between keys.
keys = np.arange(0.0, seconds[-1] + 1e-9, 0.1)
seg = np.clip(np.searchsorted(seconds, keys, side="right") - 1, 0, len(seconds) - 2)
u = (keys - seconds[seg]) / (seconds[seg + 1] - seconds[seg])
ease = (0.5 - 0.5 * np.cos(np.pi * u))[:, None]
q = waypoints[seg] + ease * (waypoints[seg + 1] - waypoints[seg])

anim = Animation(loop=True)
anim.set_frame_times(keys)
anim.set_joint_data({"cell": q})
# Gripping at A (2.3 s) and releasing at B (6.1 s) swaps which copy of the part shows.
held = (keys >= 2.3) & (keys < 6.1)
vis = np.column_stack([keys < 2.3, held, keys >= 6.1])
anim.add_channel("visibility", ["part_a", "part_held", "part_b"], vis, dtype="uint8")
v.load_animation(anim)

# Same FK in Python: the TCP path at 60 Hz with the viewer's joint-space lerp.
t = np.linspace(0.0, keys[-1], int(keys[-1] * 60) + 1)
dense = np.column_stack([np.interp(t, keys, q[:, j]) for j in range(len(AXES))])
tcp = np.array([kinematics.compose(CELL, row)["tcp"][:3, 3] for row in dense])
v.add_polyline("tcp_path", tcp, colors=t, colormap="plasma", line_width=3)

v.set_camera(position=[3.1, -3.9, 3.0], target=[0, 0.1, 0.45])
print(
    f"{len(keys)} joint keyframes; TCP path of {len(tcp)} points from kinematics.compose"
)
v.wait_for_assets()
