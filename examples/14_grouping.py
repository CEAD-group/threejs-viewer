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


def make_transform_matrix(position, rotation_z=0, scale=1.0):
    """Create a 4x4 transform matrix (column-major for Three.js)."""
    c, s = math.cos(rotation_z), math.sin(rotation_z)
    return [
        scale * c,
        scale * s,
        0,
        0,
        -scale * s,
        scale * c,
        0,
        0,
        0,
        0,
        scale,
        0,
        position[0],
        position[1],
        position[2],
        1,
    ]


v = viewer()
v.clear()
v.stop_animation()

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
#   base (group at origin)
#     base_mesh (cylinder)
#     shoulder (group, offset Z up)
#       upper_arm (box)
#       elbow (group, offset along arm)
#         lower_arm (box)
#         wrist (group, offset along arm)
#           hand (sphere)

# Base — rotates around Z
v.add_group("base")
v.add_cylinder(
    "base_mesh",
    radius_top=0.6,
    radius_bottom=0.8,
    height=0.4,
    color=0x555555,
    position=[0, 0, 0.2],
    parent="base",
    roughness=0.6,
    metalness=0.3,
)

# Shoulder joint — positioned on top of base, rotates around Z in XY plane
v.add_group("shoulder", parent="base", position=[0, 0, 0.4])
v.add_box(
    "upper_arm",
    width=0.3,
    height=2.0,
    depth=0.3,
    color=0xDD6633,
    position=[0, 1.0, 0],
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

# Elbow joint — at the end of upper arm
v.add_group("elbow", parent="shoulder", position=[0, 2.0, 0])
v.add_box(
    "lower_arm",
    width=0.25,
    height=1.5,
    depth=0.25,
    color=0x3366DD,
    position=[0, 0.75, 0],
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

# Wrist joint — at the end of lower arm
v.add_group("wrist", parent="elbow", position=[0, 1.5, 0])
v.add_sphere(
    "hand",
    radius=0.2,
    color=0x33DD66,
    parent="wrist",
    roughness=0.4,
    metalness=0.3,
)

# === Animate: only set local joint rotations ===
duration = 8.0
fps = 30
n_frames = int(duration * fps)
animation = Animation(loop=True)

for i in range(n_frames):
    t = i / fps
    transforms = {}

    # Base rotates slowly around Z
    base_angle = math.sin(t * 0.8) * math.pi / 3
    transforms["base"] = make_transform_matrix([0, 0, 0], rotation_z=base_angle)

    # Shoulder oscillates
    shoulder_angle = math.sin(t * 1.2) * math.pi / 4 + math.pi / 6
    transforms["shoulder"] = make_transform_matrix(
        [0, 0, 0.4], rotation_z=shoulder_angle
    )

    # Elbow oscillates opposite to shoulder
    elbow_angle = math.sin(t * 1.8 + 1.0) * math.pi / 3
    transforms["elbow"] = make_transform_matrix([0, 2.0, 0], rotation_z=elbow_angle)

    # Wrist wiggles fast
    wrist_angle = math.sin(t * 3.0) * math.pi / 6
    transforms["wrist"] = make_transform_matrix([0, 1.5, 0], rotation_z=wrist_angle)

    animation.add_frame(time=t, transforms=transforms)

animation.add_marker(0.0, "Start", color=0x00FF00)
animation.add_marker(4.0, "Mid-cycle", color=0xFFFF00)

v.load_animation(animation)

print(f"Robot arm demo: {animation.n_frames} frames, {animation.duration:.1f}s")
print("Only 4 joints are animated — 5 visual meshes follow automatically via grouping.")
print("Press Ctrl+C to exit.")

try:
    while True:
        pass
except KeyboardInterrupt:
    v.disconnect()
