"""
Object Grouping Demo

Demonstrates parent-child grouping with a simple robot arm.
Each joint is a group — animating a joint automatically moves all
downstream links and the end-effector. This means you only need to
set local joint angles, not world-space transforms for every piece.

Run: uv run python examples/14_grouping.py
"""

import math

from threejs_viewer import Animation, viewer


def mat_rz(tx, ty, tz, angle):
    """4x4 column-major matrix: translate to (tx,ty,tz) + rotate around Z."""
    c, s = math.cos(angle), math.sin(angle)
    return [c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, tx, ty, tz, 1]


def mat_ry(tx, ty, tz, angle):
    """4x4 column-major matrix: translate to (tx,ty,tz) + rotate around Y."""
    c, s = math.cos(angle), math.sin(angle)
    return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, tx, ty, tz, 1]


v = viewer()
v.clear()
v.unload_animation()

# Ground plane
v.add_box(
    "ground",
    width=12,
    height=12,
    depth=0.05,
    color=0x333333,
    position=[0, 0, -0.025],
    roughness=0.9,
    metalness=0.0,
)

# === Build robot arm using nested groups ===
#
# Hierarchy:
#   base (group at origin, yaw around Z)
#     base_mesh (cylinder)
#     shoulder (group at top of base, pitch around Y)
#       upper_arm (box, extends upward in Z)
#       shoulder_joint (sphere)
#       elbow (group at top of upper arm, pitch around Y)
#         lower_arm (box, extends upward in Z)
#         elbow_joint (sphere)
#         wrist (group at top of lower arm, pitch around Y)
#           hand (sphere)

# Base — rotates around Z (turntable yaw)
v.add_group("base")
v.add_cylinder(
    "base_mesh",
    radius_top=0.6,
    radius_bottom=0.8,
    height=0.4,
    color=0x555555,
    position=[0, 0, 0.2],
    rotation=[math.pi / 2, 0, 0],
    parent="base",
    roughness=0.6,
    metalness=0.3,
)

# Shoulder — at top of base, bends around Y
v.add_group("shoulder", parent="base", position=[0, 0, 0.4])
v.add_box(
    "upper_arm",
    width=0.3,
    height=0.3,
    depth=2.0,
    color=0xDD6633,
    position=[0, 0, 1.0],
    parent="shoulder",
    roughness=0.4,
    metalness=0.2,
)
v.add_sphere(
    "shoulder_joint",
    radius=0.2,
    color=0x888888,
    parent="shoulder",
    roughness=0.3,
    metalness=0.5,
)

# Elbow — at top of upper arm, bends around Y
v.add_group("elbow", parent="shoulder", position=[0, 0, 2.0])
v.add_box(
    "lower_arm",
    width=0.25,
    height=0.25,
    depth=1.5,
    color=0x3366DD,
    position=[0, 0, 0.75],
    parent="elbow",
    roughness=0.4,
    metalness=0.2,
)
v.add_sphere(
    "elbow_joint",
    radius=0.17,
    color=0x888888,
    parent="elbow",
    roughness=0.3,
    metalness=0.5,
)

# Wrist — at top of lower arm, bends around Y
v.add_group("wrist", parent="elbow", position=[0, 0, 1.5])
v.add_sphere(
    "hand",
    radius=0.2,
    color=0x33DD66,
    parent="wrist",
    roughness=0.4,
    metalness=0.3,
)

# === Animate: only set local joint transforms ===
duration = 8.0
fps = 30
n_frames = int(duration * fps)
animation = Animation(loop=True)

for i in range(n_frames):
    t = i / fps
    transforms = {}

    # Base: slow yaw around Z
    base_angle = math.sin(t * 0.8) * math.pi / 3
    transforms["base"] = mat_rz(0, 0, 0, base_angle)

    # Shoulder: pitch around Y — arm swings forward/back
    shoulder_angle = math.sin(t * 1.2) * math.pi / 4 + math.pi / 6
    transforms["shoulder"] = mat_ry(0, 0, 0.4, shoulder_angle)

    # Elbow: pitch around Y — lower arm bends
    elbow_angle = math.sin(t * 1.8 + 1.0) * math.pi / 3
    transforms["elbow"] = mat_ry(0, 0, 2.0, elbow_angle)

    # Wrist: fast wiggle around Y
    wrist_angle = math.sin(t * 3.0) * math.pi / 6
    transforms["wrist"] = mat_ry(0, 0, 1.5, wrist_angle)

    animation.add_frame(time=t, transforms=transforms)

animation.add_marker(0.0, "Start", color=0x00FF00)
animation.add_marker(4.0, "Mid-cycle", color=0xFFFF00)

v.load_animation(animation)

print(f"Robot arm demo: {animation.n_frames} frames, {animation.duration:.1f}s")
print("Only 4 joints are animated — 5 visual meshes follow automatically via grouping.")
v.wait_for_assets()
