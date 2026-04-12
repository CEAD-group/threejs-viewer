"""
Toolpath Visualization — draw_range + Bead demo

Generates a spiral vase toolpath with animated draw_range and a nozzle
following the tip.  Uses ``add_toolpath`` which renders the bead as a
parametric tube (client-side geometry, smooth frontier morphing).

The animation uses one keyframe per spine point with linear interpolation,
so even a 10k-point toolpath plays back smoothly at 60 fps without
pre-computing hundreds of thousands of frames.

Run: uv run python examples/11_toolpath.py
"""

import math

import numpy as np

from threejs_viewer import Animation, Toolpath, viewer


def spiral_vase(
    n_points=30000,
    n_turns=180,
    radius=4.0,
    height=9.0,
    lumps=7,
    bump=0.05,
    steep=3.5,
    twist=0.2,
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

# Generate toolpath
duration = 600.0

tp = Toolpath.from_points(
    spiral_vase(), bead_width=0.3, bead_height=0.08, duration=duration
)

# Bead (parametric tube — chamfered hex cross-section, built client-side)
tp.colorize("viridis")
v.add_toolpath("path_tube", tp, roughness=0.4, metalness=0.15)

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

# One keyframe per spine point — linear interpolation handles 60 fps smoothly
n_frames = len(tp)
frame_times = tp.times
draw_fracs = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)

# Nozzle transforms: Rot(+90 about X) so Y-up cylinder stands vertically
tips = tp.points
nz_z = tips[:, 2] + nozzle_height / 2 + nozzle_gap
rx90 = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1], dtype=np.float32)

transforms = np.zeros((n_frames, 2, 16), dtype=np.float32)
transforms[:, 0, [0, 5, 10, 15]] = 1.0  # path_tube: identity
transforms[:, 1] = rx90
transforms[:, 1, 12] = tips[:, 0]
transforms[:, 1, 13] = tips[:, 1]
transforms[:, 1, 14] = nz_z

# Build animation — fully binary, linear interpolation fills in 60 fps
animation = Animation(loop=True, camera_follow="nozzle")
animation.set_frame_times(frame_times)
animation.set_transform_data(["path_tube", "nozzle"], transforms)
animation.set_draw_range_data(["path_tube"], draw_fracs[:, None])

animation.add_marker(0.0, "Start", color=0x00FF00)
animation.add_marker(duration / 2, "50%", color=0xFFFF00)
animation.add_marker(duration * 0.99, "Done", color=0xFF0000)

v.load_animation(animation)

print(f"Toolpath: {len(tp)} points/keyframes, {duration:.0f}s duration")
print("Bead + nozzle — linear interpolation gives smooth 60 fps playback.")
print("Waiting for browser to finish loading assets...")
v.wait_for_assets()
print("Assets loaded — server closed.")
