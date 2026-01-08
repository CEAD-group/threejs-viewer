"""
Lissajous Curves Explorer

Visualizes beautiful 3D Lissajous curves with animated parameters.
A sphere traces the curve in real-time while the full path is shown.

Run: uv run python examples/05_lissajous_curves.py
"""

import math

import numpy as np

from threejs_viewer import Animation, viewer


def make_transform_matrix(position, scale=1.0):
    """Create a simple translation matrix."""
    return [
        scale,
        0,
        0,
        0,
        0,
        scale,
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


def lissajous_3d(t, a, b, c, delta_x=0, delta_y=0, scale=3):
    """Compute 3D Lissajous curve point."""
    x = scale * math.sin(a * t + delta_x)
    y = scale * math.sin(b * t + delta_y)
    z = scale * math.sin(c * t) + scale + 0.5  # Offset above ground
    return x, y, z


v = viewer()
v.clear()

# Ground plane
v.add_box(
    "ground", width=10, height=10, depth=0.02, color=0x333333, position=[0, 0, -0.01]
)

# Curve parameters (interesting ratios create beautiful patterns)
A, B, C = 3, 4, 5  # Frequency ratios
DELTA_X = math.pi / 2
DELTA_Y = 0
SCALE = 3

# Generate the full curve for static display
n_curve_points = 2000
curve_points = np.zeros((n_curve_points, 3))
t_values = np.linspace(0, 2 * math.pi, n_curve_points)

for i, t in enumerate(t_values):
    curve_points[i] = lissajous_3d(t, A, B, C, DELTA_X, DELTA_Y, SCALE)

# Add the curve with gradient coloring
v.add_polyline(
    "lissajous_curve",
    curve_points,
    colors=np.linspace(0, 1, n_curve_points),
    colormap="viridis",
    line_width=2,
)

# Add a tracer sphere
v.add_sphere("tracer", radius=0.15, color=0xFF4444)

# Add a "ghost" trail of smaller spheres
N_TRAIL = 8
for i in range(N_TRAIL):
    alpha = 1.0 - (i / N_TRAIL)  # Fade out
    v.add_sphere(f"trail_{i}", radius=0.08 * alpha, color=0xFF8888)

# Create animation - sphere traces the curve
duration = 10.0
fps = 60  # Smooth tracing
n_frames = int(duration * fps)

animation = Animation(loop=True)

for frame_idx in range(n_frames):
    t_anim = frame_idx / fps
    t_param = (t_anim / duration) * 2 * math.pi  # Map to curve parameter

    transforms = {}

    # Main tracer position
    pos = lissajous_3d(t_param, A, B, C, DELTA_X, DELTA_Y, SCALE)
    transforms["tracer"] = make_transform_matrix(pos)

    # Trail positions (previous positions)
    for i in range(N_TRAIL):
        trail_offset = (i + 1) * 0.05  # Time offset for each trail sphere
        t_trail = t_param - trail_offset
        trail_pos = lissajous_3d(t_trail, A, B, C, DELTA_X, DELTA_Y, SCALE)
        transforms[f"trail_{i}"] = make_transform_matrix(trail_pos)

    animation.add_frame(time=t_anim, transforms=transforms)

# Markers at interesting points
animation.add_marker(0.0, "Start", color=0x00FF00)
animation.add_marker(duration / 4, "1/4 cycle", color=0x0088FF)
animation.add_marker(duration / 2, "1/2 cycle", color=0xFFFF00)
animation.add_marker(3 * duration / 4, "3/4 cycle", color=0xFF8800)

v.load_animation(animation)

print(f"Lissajous curve ({A}:{B}:{C}), {animation.n_frames} frames at {fps} fps")
print("Press Ctrl+C to exit.")

try:
    while True:
        pass
except KeyboardInterrupt:
    v.disconnect()
