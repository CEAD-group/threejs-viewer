"""
Lissajous Curves Explorer

Visualizes beautiful 3D Lissajous curves with animated parameters.
A sphere traces the curve in real-time with a fading ghost trail.

Demonstrates binary animation channels:
- transforms: position for tracer + trail spheres (float32, stride 16)
- opacity: trail fades from fully opaque to invisible (float32)

Run: uv run python examples/05_lissajous_curves.py
"""

import math

import numpy as np

from threejs_viewer import Animation, viewer


def lissajous_3d(t, a, b, c, delta_x=0, delta_y=0, scale=3):
    """Compute 3D Lissajous curve points (vectorized)."""
    x = scale * np.sin(a * t + delta_x)
    y = scale * np.sin(b * t + delta_y)
    z = scale * np.sin(c * t) + scale + 0.5
    return np.column_stack([x, y, z])


v = viewer()
v.clear()

# Ground plane
v.add_box(
    "ground", width=10, height=10, depth=0.02, color=0x333333, position=[0, 0, -0.01]
)

# Curve parameters (interesting ratios create beautiful patterns)
A, B, C = 3, 4, 5
DELTA_X = math.pi / 2
DELTA_Y = 0
SCALE = 3

# Generate the full curve for static display
n_curve_points = 2000
t_curve = np.linspace(0, 2 * math.pi, n_curve_points)
curve_points = lissajous_3d(t_curve, A, B, C, DELTA_X, DELTA_Y, SCALE)

v.add_polyline(
    "lissajous_curve",
    curve_points,
    colors=np.linspace(0, 1, n_curve_points),
    colormap="viridis",
    line_width=2,
)

# Add tracer + trail spheres (all same size — opacity does the fading)
N_TRAIL = 8
v.add_sphere("tracer", radius=0.15, color=0xFF4444)
for i in range(N_TRAIL):
    v.add_sphere(f"trail_{i}", radius=0.12, color=0xFF8888)

# Pre-compute animation with binary channels
duration = 10.0
fps = 60
n_frames = int(duration * fps)
n_objects = 1 + N_TRAIL  # tracer + trail spheres
object_ids = ["tracer"] + [f"trail_{i}" for i in range(N_TRAIL)]

# Compute all curve parameter values for tracer and each trail offset
trail_offsets = np.array([0.0] + [(i + 1) * 0.05 for i in range(N_TRAIL)])
frame_times = np.arange(n_frames) / fps
t_params = (frame_times / duration) * 2 * math.pi  # (n_frames,)

# (n_frames, n_objects) curve parameter for each object at each frame
t_all = t_params[:, None] - trail_offsets[None, :]  # broadcast

# Vectorized position computation for all objects and frames at once
positions = lissajous_3d(t_all.ravel(), A, B, C, DELTA_X, DELTA_Y, SCALE).reshape(
    n_frames, n_objects, 3
)

# Build transform matrices: identity with translation
all_transforms = np.zeros((n_frames, n_objects, 16), dtype=np.float32)
all_transforms[:, :, 0] = 1.0  # scale x
all_transforms[:, :, 5] = 1.0  # scale y
all_transforms[:, :, 10] = 1.0  # scale z
all_transforms[:, :, 15] = 1.0  # w
all_transforms[:, :, 12] = positions[:, :, 0]  # tx
all_transforms[:, :, 13] = positions[:, :, 1]  # ty
all_transforms[:, :, 14] = positions[:, :, 2]  # tz

# Opacity: tracer=1.0, trail fades linearly to 0
trail_opacity = np.linspace(
    1.0, 0.0, N_TRAIL + 1, endpoint=False
)  # [1.0, 0.875, ..., 0.125]
# Same opacity every frame — broadcast to (n_frames, n_objects)
all_opacity = np.broadcast_to(trail_opacity, (n_frames, n_objects)).astype(np.float32)

# Build animation
animation = Animation(loop=True)
animation.set_frame_times(frame_times)
animation.set_transform_data(object_ids, all_transforms)
animation.add_channel("opacity", object_ids, all_opacity, dtype="float32")

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
