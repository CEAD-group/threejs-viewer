"""
Toolpath Visualization — draw_range + Bead demo

Generates a spiral vase toolpath with animated draw_range and a nozzle
following the tip.

Run: uv run python examples/11_toolpath.py
"""

import math

import numpy as np

from threejs_viewer import Animation, Toolpath, viewer


def spiral_vase(
    n_points=800000,
    n_turns=80,
    radius=2.0,
    height=5.0,
    lumps=18,
    bump=0.35,
    steep=3.5,
    twist=1.2,
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
    "ground", width=8, height=8, depth=0.02, color=0x333333, position=[0, 0, -0.01]
)

# Generate toolpath and wrap in Toolpath for animation + coloring
duration = 3600.0
fps = 60
n_frames = int(duration * fps)

tp = Toolpath.from_points(
    spiral_vase(), bead_width=0.3, bead_height=0.08, duration=duration
)

# Bead (extruded bevelled rectangle cross-section)
v.add_bead(
    "path_tube",
    tp.points,
    width=tp.widths,
    height=tp.heights,
    colors=tp.gradient_colors("viridis"),
    roughness=0.4,
    metalness=0.15,
)

# Nozzle: tapered cylinder hovering above the path tip
nozzle_height = 0.8
nozzle_gap = 0.05  # gap between nozzle bottom and print surface
v.add_cylinder(
    "nozzle",
    radius_top=0.25,
    radius_bottom=0.08,
    height=nozzle_height,
    color=0xCD7F32,
    roughness=0.3,
    metalness=0.8,
)

# Animate draw_range + nozzle position (vectorized)
print(f"Pre-computing {n_frames} frames...")

frame_times, draw_fracs = tp.frame_times(n_frames)

# Path indices for each frame's nozzle position
pt_indices = np.clip((draw_fracs * (len(tp) - 1)).astype(int), 0, len(tp) - 1)
tips = tp.points[pt_indices]  # (n_frames, 3)

# Object order: path_tube, nozzle
transforms = np.zeros((n_frames, 2, 16), dtype=np.float32)

# path_tube: identity matrix
transforms[:, 0, [0, 5, 10, 15]] = 1.0

# Nozzle rotation matrix (constant): Rot(+90° about X) in column-major
# col0=[1,0,0,0], col1=[0,0,1,0], col2=[0,-1,0,0], col3=[x,y,z,1]
nz_z = tips[:, 2] + nozzle_height / 2 + nozzle_gap
m = transforms[:, 1]
m[:, 0] = 1.0  # col0.x
m[:, 5] = 0.0  # col1.y (cos90=0)
m[:, 6] = 1.0  # col1.z (sin90=1)
m[:, 9] = -1.0  # col2.y (-sin90=-1)
m[:, 10] = 0.0  # col2.z (cos90=0)
m[:, 12] = tips[:, 0]
m[:, 13] = tips[:, 1]
m[:, 14] = nz_z
m[:, 15] = 1.0

# Build animation — fully binary, no Python loop
animation = Animation(loop=True)
animation.set_frame_times(frame_times)
animation.set_transform_data(["path_tube", "nozzle"], transforms)
animation.set_draw_range_data(["path_tube"], draw_fracs[:, None])

animation.add_marker(0.0, "Start", color=0x00FF00)
animation.add_marker(duration / 2, "50%", color=0xFFFF00)
animation.add_marker(duration * 0.99, "Done", color=0xFF0000)

v.load_animation(animation)

print(f"Toolpath: {len(tp)} points, {animation.n_frames} frames at {fps} fps")
print("Bead + nozzle — grows via draw_range animation.")
print("Press Ctrl+C to exit.")

try:
    while True:
        pass
except KeyboardInterrupt:
    v.disconnect()
