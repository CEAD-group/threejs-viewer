"""
Toolpath Visualization — draw_range + Tube demo

Generates a spiral vase toolpath and shows it two ways side by side:
- Left: polyline with animated draw_range (path grows over time)
- Right: pill-shaped tube with animated draw_range + nozzle following the tip

Run: uv run python examples/11_toolpath.py
"""

import math

import numpy as np

from threejs_viewer import Animation, viewer


def make_identity():
    """Identity 4x4 matrix (column-major)."""
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def make_nozzle_matrix(x, y, z):
    """4x4 matrix: rotate Y-up cylinder so wide end is up, narrow tip down + translate.

    Three.js cylinders point along +Y. Rotate +90deg around X: Y -> -Z.
    Rot(+90° about X): [[1,0,0],[0,0,-1],[0,1,0]]
    """
    # Column-major: col0=[1,0,0,0], col1=[0,0,1,0], col2=[0,-1,0,0], col3=[x,y,z,1]
    return [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, x, y, z, 1]


def spiral_vase(n_points=12000, n_turns=80, base_radius=2.0, height=5.0, wobble=0.15):
    """Generate a spiral vase toolpath — helix with varying radius."""
    t = np.linspace(0, 1, n_points)
    angle = t * n_turns * 2 * math.pi

    # Vase profile: radius varies along height
    profile = base_radius * (0.6 + 0.4 * np.cos(t * math.pi * 2 - math.pi))
    profile += wobble * np.sin(angle * 3)

    x = profile * np.cos(angle)
    y = profile * np.sin(angle)
    z = t * height

    return np.column_stack([x, y, z]).astype(np.float32)


v = viewer()
v.clear()

# Ground plane
v.add_box(
    "ground", width=16, height=8, depth=0.02, color=0x333333, position=[0, 0, -0.01]
)

# Generate one set of points, offset for each display
spacing = 5.0
points = spiral_vase()
n_points = len(points)

points_line = points.copy()
points_line[:, 0] -= spacing / 2

points_tube = points.copy()
points_tube[:, 0] += spacing / 2

# Left: polyline (N points → N-1 segments)
v.add_polyline(
    "path_line",
    points_line,
    colors=np.linspace(0, 1, n_points),
    colormap="turbo",
    line_width=2,
)

# Right: pill-shaped tube (use N-1 tubular segments to match polyline)
v.add_tube(
    "path_tube",
    points_tube,
    width=0.3,
    height=0.08,
    tubular_segments=n_points - 1,
    color=0x4A90D9,
    metalness=0.1,
    roughness=0.8,
)

# Nozzles: tapered cylinders hovering above each path tip
nozzle_height = 0.8
nozzle_gap = 0.05  # gap between nozzle bottom and print surface
for nozzle_id in ("nozzle_line", "nozzle_tube"):
    v.add_cylinder(
        nozzle_id,
        radius_top=0.25,
        radius_bottom=0.08,
        height=nozzle_height,
        color=0xAAAAAA,
    )

# Animate draw_range + nozzle position
duration = 10.0
fps = 60
n_frames = int(duration * fps)

animation = Animation(loop=True)

for i in range(n_frames):
    t = i / fps
    frac = max(0.005, t / duration)

    # Nozzles follow the tip of each path
    pt_idx = min(int(frac * (n_points - 1)), n_points - 1)
    tip_line = points_line[pt_idx]
    tip_tube = points_tube[pt_idx]
    nz_line = float(tip_line[2]) + nozzle_height / 2 + nozzle_gap
    nz_tube = float(tip_tube[2]) + nozzle_height / 2 + nozzle_gap

    animation.add_frame(
        time=t,
        transforms={
            "path_line": make_identity(),
            "path_tube": make_identity(),
            "nozzle_line": make_nozzle_matrix(
                float(tip_line[0]), float(tip_line[1]), nz_line
            ),
            "nozzle_tube": make_nozzle_matrix(
                float(tip_tube[0]), float(tip_tube[1]), nz_tube
            ),
        },
        draw_ranges={
            "path_line": frac,
            "path_tube": frac,
        },
    )

animation.add_marker(0.0, "Start", color=0x00FF00)
animation.add_marker(duration / 2, "50%", color=0xFFFF00)
animation.add_marker(duration * 0.99, "Done", color=0xFF0000)

v.load_animation(animation)

print(f"Toolpath: {n_points} points, {animation.n_frames} frames at {fps} fps")
print(
    "Left: polyline | Right: pill tube + nozzle — both grow via draw_range animation."
)
print("Press Ctrl+C to exit.")

try:
    while True:
        pass
except KeyboardInterrupt:
    v.disconnect()
