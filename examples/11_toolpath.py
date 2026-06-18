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


N_POINTS = 1_000_000
N_TURNS = 300
HEIGHT = 9.0
WAVES_PER_ROTATION = 100.5  # cross-bead sine cycles per layer (.5 → brick stagger)


def spiral_vase(
    n_points=N_POINTS,
    n_turns=N_TURNS,
    height=HEIGHT,
):
    """Generate a curvy vase toolpath with interesting geometry.

    The silhouette follows a classic vase profile: wide base, narrow waist,
    flared rim.  Angular ripples at multiple frequencies create organic
    surface texture that tests LOD at various scales.
    """
    TAU = 2 * math.pi
    t = np.linspace(0, 1, n_points)
    angle = t * n_turns * TAU

    # Vase silhouette: base → waist → belly → neck → flared rim
    # Piecewise smooth profile using sine blends
    r_base = 3.0
    r_waist = 1.8
    r_belly = 4.2
    r_neck = 2.0
    r_rim = 3.5

    # Smooth interpolation through control radii
    r = np.where(
        t < 0.15,
        r_base + (r_waist - r_base) * np.sin(t / 0.15 * math.pi / 2) ** 2,
        np.where(
            t < 0.45,
            r_waist
            + (r_belly - r_waist) * np.sin((t - 0.15) / 0.30 * math.pi / 2) ** 2,
            np.where(
                t < 0.75,
                r_belly
                + (r_neck - r_belly) * np.sin((t - 0.45) / 0.30 * math.pi / 2) ** 2,
                r_neck
                + (r_rim - r_neck) * np.sin((t - 0.75) / 0.25 * math.pi / 2) ** 2,
            ),
        ),
    )

    # Multi-frequency surface ripples — organic texture
    twist_rate = 0.3  # slow rotation of the pattern as it rises
    ripple_angle = angle + t * twist_rate * TAU
    ripple = (
        0.08 * np.sin(ripple_angle * 5 + 1.0)  # broad lobes
        + 0.04 * np.sin(ripple_angle * 11 + 2.7)  # medium detail
        + 0.02 * np.sin(ripple_angle * 23 + 0.3)  # fine texture
    )
    # Modulate ripple amplitude by height — stronger on belly, subtle at rim
    ripple_strength = 1.0 - 0.6 * np.abs(t - 0.45) ** 0.8
    r += r * ripple * ripple_strength

    # Cross-bead modulation: a sine wiggling the spine in the cross-bead
    # (radial) direction.  Amplitude = a fraction of bead width, ramped from 0
    # on the first layer to full strength on the top layer.  The wave is driven
    # by a fixed number of cycles per rotation; a non-integer count (e.g. 100.5)
    # advances the phase by an extra half-wave each turn, so consecutive layers
    # land staggered by half a wavelength (brick pattern).
    bead_w = HEIGHT / N_TURNS * 4
    amplitude = 0.2 * bead_w
    phase = WAVES_PER_ROTATION * angle
    r = r + amplitude * t * np.sin(phase)  # ramp: 0 at base → max at top

    # Slight ellipse squash for asymmetry
    x = r * 1.08 * np.cos(angle)
    y = r * 0.93 * np.sin(angle)
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
    spiral_vase(),
    bead_width=HEIGHT / N_TURNS * 4,
    bead_height=HEIGHT / N_TURNS,
    duration=duration,
)

# Bead (parametric tube — chamfered hex cross-section, built client-side)
tp.colorize("viridis")
v.add_toolpath("path_tube", tp, roughness=0.55, metalness=0.75)

# Nozzle: tapered cylinder hovering above the path tip
bead_width = HEIGHT / N_TURNS * 4
nozzle_height = bead_width * 3
nozzle_gap = bead_width / 2  # gap between nozzle bottom and print surface
v.add_cylinder(
    "nozzle",
    radius_top=bead_width * 0.6,
    radius_bottom=bead_width * 0.25,
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
animation = Animation(loop=True, camera_follow=None)
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
