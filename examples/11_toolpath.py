"""
Toolpath Visualization — draw_range + Bead demo

Generates a spiral vase toolpath and shows it two ways side by side:
- Left: polyline with animated draw_range (path grows over time)
- Right: bead mesh with animated draw_range + nozzle following the tip

Run: uv run python examples/11_toolpath.py
"""

import math

import numpy as np

from threejs_viewer import Animation, viewer


def spiral_vase(
    n_points=800000, n_turns=80, radius=2.0, height=5.0,
    lumps=18, bump=0.35, steep=3.5, twist=1.2,
):
    """Generate a lumpy asymmetric spiral vase toolpath.

    The silhouette has a narrow neck, steep rise, and wide belly driven by
    a power-law envelope. Angular lobes at different frequencies break
    rotational symmetry, and a slow helical twist rotates the bump pattern
    as it rises — giving the bead layers a dynamic texture.
    """
    TAU = 2 * math.pi
    t = np.linspace(0, 1, n_points)
    angle = t * n_turns * TAU

    # Power-law envelope: pinched neck, wide belly
    u = t * 2 - 1  # -1..+1
    envelope = 1.0 - np.abs(u) ** steep * 0.85
    r = radius * (0.25 + 0.75 * envelope)

    # Angular lump deformation — two overlapping lobe patterns
    lump_angle = angle + t * twist * TAU
    lobe1 = np.sin(lump_angle * lumps * 0.7 + 0.9)
    lobe2 = np.sin(lump_angle * lumps * 1.3 + 2.3)
    r += r * bump * (lobe1 * 0.6 + lobe2 * 0.4)

    # Slight ellipse squash so it's not perfectly circular
    x = r * 1.10 * np.cos(angle)
    y = r * 0.92 * np.sin(angle)
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

# Per-point layer colors: alternating every turn
n_turns = 80
layer_index = (np.linspace(0, 1, n_points) * n_turns).astype(int)
color_a = np.array([0.48, 0.72, 0.80])  # light blue
color_b = np.array([0.85, 0.55, 0.25])  # warm orange
bead_colors = np.where((layer_index % 2 == 0)[:, None], color_a, color_b).astype(np.float32)

# Right: bead (extruded bevelled rectangle cross-section)
v.add_bead(
    "path_tube",
    points_tube,
    width=0.3,
    height=0.08,
    colors=bead_colors,
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

# Animate draw_range + nozzle position (vectorized)
duration = 3600.0
fps = 60
n_frames = int(duration * fps)

print(f"Pre-computing {n_frames} frames...")

# Frame times and draw_range fractions
frame_times = np.arange(n_frames) / fps
fracs = np.clip(frame_times / duration, 0.005, 1.0)

# Path indices for each frame's nozzle position
pt_indices = np.clip((fracs * (n_points - 1)).astype(int), 0, n_points - 1)
tips_line = points_line[pt_indices]  # (n_frames, 3)
tips_tube = points_tube[pt_indices]  # (n_frames, 3)

# Object order: path_line, path_tube, nozzle_line, nozzle_tube
object_ids = ["path_line", "path_tube", "nozzle_line", "nozzle_tube"]
transforms = np.zeros((n_frames, 4, 16), dtype=np.float32)

# path_line and path_tube: identity matrices
transforms[:, 0, [0, 5, 10, 15]] = 1.0
transforms[:, 1, [0, 5, 10, 15]] = 1.0

# Nozzle rotation matrix (constant): Rot(+90° about X) in column-major
# col0=[1,0,0,0], col1=[0,0,1,0], col2=[0,-1,0,0], col3=[x,y,z,1]
nz_z_line = tips_line[:, 2] + nozzle_height / 2 + nozzle_gap
nz_z_tube = tips_tube[:, 2] + nozzle_height / 2 + nozzle_gap

for ni, (tips, nz_z) in enumerate([(tips_line, nz_z_line), (tips_tube, nz_z_tube)]):
    m = transforms[:, 2 + ni]
    m[:, 0] = 1.0   # col0.x
    m[:, 5] = 0.0   # col1.y (cos90=0)
    m[:, 6] = 1.0   # col1.z (sin90=1)
    m[:, 9] = -1.0  # col2.y (-sin90=-1)
    m[:, 10] = 0.0  # col2.z (cos90=0)
    m[:, 12] = tips[:, 0]
    m[:, 13] = tips[:, 1]
    m[:, 14] = nz_z
    m[:, 15] = 1.0

# Build animation — fully binary, no Python loop
animation = Animation(loop=True)
animation.set_frame_times(frame_times)
animation.set_transform_data(object_ids, transforms)
animation.set_draw_range_data(
    ["path_line", "path_tube"],
    np.column_stack([fracs, fracs]),
)

animation.add_marker(0.0, "Start", color=0x00FF00)
animation.add_marker(duration / 2, "50%", color=0xFFFF00)
animation.add_marker(duration * 0.99, "Done", color=0xFF0000)

v.load_animation(animation)

print(f"Toolpath: {n_points} points, {animation.n_frames} frames at {fps} fps")
print(
    "Left: polyline | Right: bead + nozzle — both grow via draw_range animation."
)
print("Press Ctrl+C to exit.")

try:
    while True:
        pass
except KeyboardInterrupt:
    v.disconnect()
