"""
Animation Stress Test

Same torus knot tube and followers as 07_stress_test.py, but pre-computed
as an Animation with timeline scrubbing. Tests how the viewer handles
large animation payloads (500+ objects x 2500 frames).

Demonstrates binary animation channels:
- transforms: position/rotation for all followers (float32, stride 16)
- colors: dynamic color based on Z-height via indexed colormap (uint8)

Run: uv run python examples/10_animation_stress_test.py
"""

import random
import time
from pathlib import Path

import numpy as np

from threejs_viewer import Animation, viewer


def torus_knot(t: np.ndarray, p: int = 3, q: int = 7, scale: float = 5.0):
    """Torus knot parametric curve with analytical tangent."""
    r = 0.5
    x = scale * (np.cos(p * t) * (1 + r * np.cos(q * t)))
    y = scale * (np.sin(p * t) * (1 + r * np.cos(q * t)))
    z = scale * (r * np.sin(q * t))

    dx = scale * (
        -p * np.sin(p * t) * (1 + r * np.cos(q * t))
        - r * q * np.cos(p * t) * np.sin(q * t)
    )
    dy = scale * (
        p * np.cos(p * t) * (1 + r * np.cos(q * t))
        - r * q * np.sin(p * t) * np.sin(q * t)
    )
    dz = scale * (r * q * np.cos(q * t))

    return (x, y, z), (dx, dy, dz)


def compute_frame(dx, dy, dz):
    """Compute stable frame from tangent vectors."""
    t_len = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
    tx, ty, tz = dx / t_len, dy / t_len, dz / t_len

    up_x, up_y, up_z = 0.0, 0.0, 1.0
    nx = up_y * tz - up_z * ty
    ny = up_z * tx - up_x * tz
    nz = up_x * ty - up_y * tx

    n_len = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
    nx, ny, nz = nx / n_len, ny / n_len, nz / n_len

    bx = ty * nz - tz * ny
    by = tz * nx - tx * nz
    bz = tx * ny - ty * nx

    return (tx, ty, tz), (nx, ny, nz), (bx, by, bz)


def create_tube(t, pos, tangent, tube_radius=0.3, windings=500, z_offset=5.0):
    """Create a tube around a curve by sweeping a circle along it."""
    x, y, z = pos
    dx, dy, dz = tangent

    _, N, B = compute_frame(dx, dy, dz)
    nx, ny, nz = N
    bx, by, bz = B

    theta = windings * t
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    px = x + tube_radius * (cos_theta * nx + sin_theta * bx)
    py = y + tube_radius * (cos_theta * ny + sin_theta * by)
    pz = z + tube_radius * (cos_theta * nz + sin_theta * bz) + z_offset

    return np.column_stack([px, py, pz]).astype(np.float32)


def quaternions_from_directions(directions):
    """Compute quaternions to rotate Y-axis to given directions. Vectorized (N, 3) -> (N, 4)."""
    norms = np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8
    d = directions / norms
    y_axis = np.array([0, 1, 0])
    dots = d @ y_axis  # (N,)

    axes = np.cross(y_axis, d)  # (N, 3)
    axis_norms = np.linalg.norm(axes, axis=1, keepdims=True) + 1e-8
    axes = axes / axis_norms

    angles = np.arccos(np.clip(dots, -1, 1))
    half = angles / 2
    s = np.sin(half)

    quats = np.empty((len(directions), 4))
    quats[:, 0] = axes[:, 0] * s
    quats[:, 1] = axes[:, 1] * s
    quats[:, 2] = axes[:, 2] * s
    quats[:, 3] = np.cos(half)

    # Handle degenerate cases
    aligned = dots > 0.9999
    anti = dots < -0.9999
    quats[aligned] = [0, 0, 0, 1]
    quats[anti] = [1, 0, 0, 0]

    return quats


# Configuration
NUM_POINTS = 1_000_000
TUBE_RADIUS = 0.4
WINDINGS = 500
NUM_FOLLOWERS = 500
DURATION = 83.3
FPS = 30

print(f"Generating torus knot with {NUM_POINTS:,} points...")
start = time.time()

t = np.linspace(0, 2 * np.pi, NUM_POINTS, dtype=np.float64)
pos, tangent = torus_knot(t)
points = create_tube(t, pos, tangent, tube_radius=TUBE_RADIUS, windings=WINDINGS)

print(f"Generated in {time.time() - start:.2f}s ({points.nbytes / 1024 / 1024:.1f} MB)")

# Connect and send
v = viewer()
v.clear()

print("Sending polyline to viewer...")
start = time.time()
v.add_polyline(
    "tube", points, colors=t.astype(np.float32), colormap="turbo", line_width=2
)
print(f"Sent in {time.time() - start:.2f}s")

# Create followers with different primitives
print(f"Creating {NUM_FOLLOWERS} followers...")

TEAPOT_PATH = Path(__file__).parent / "teapot.obj"
NUM_TEAPOTS = 20
PRIMITIVES = ["sphere", "box", "cylinder", "capsule", "cone"]

followers = []
for i in range(NUM_FOLLOWERS):
    size = random.uniform(0.1, 0.3)

    fid = f"f{i}"
    ptype = PRIMITIVES[i % len(PRIMITIVES)]

    # Start grey — the binary color channel will override each frame
    if ptype == "sphere":
        v.add_sphere(fid, radius=size, color=0x888888)
    elif ptype == "box":
        v.add_box(fid, width=size, height=size, depth=size * 2, color=0x888888)
    elif ptype == "cylinder":
        v.add_cylinder(
            fid,
            radius_top=size * 0.4,
            radius_bottom=size * 0.4,
            height=size * 2,
            color=0x888888,
        )
    elif ptype == "capsule":
        v.add_capsule(fid, radius=size * 0.4, length=size, color=0x888888)
    elif ptype == "cone":
        v.add_cylinder(
            fid, radius_top=0, radius_bottom=size * 0.5, height=size * 2, color=0x888888
        )

    followers.append(
        {
            "id": fid,
            "speed": random.expovariate(1 / 10) + 1,
            "offset": random.randint(0, NUM_POINTS - 1),
        }
    )

# Add teapots
print(f"Adding {NUM_TEAPOTS} teapots...")
for i in range(NUM_TEAPOTS):
    fid = f"teapot{i}"
    v.add_model_binary(fid, TEAPOT_PATH, format="obj")
    followers.append(
        {
            "id": fid,
            "speed": random.expovariate(1 / 8) + 1,
            "offset": random.randint(0, NUM_POINTS - 1),
            "scale": 0.2,
        }
    )

# Pre-compute animation (fully vectorized — no Python lists)
n_frames = int(DURATION * FPS)
n_followers = len(followers)
print(f"Pre-computing {n_frames} frames for {n_followers} objects...")
start = time.time()

# Gather follower params into arrays
f_ids = [f["id"] for f in followers]
f_offsets = np.array([f["offset"] for f in followers])
f_speeds = np.array([f["speed"] for f in followers])
f_scales = np.array([f.get("scale", 1.0) for f in followers])

# Pre-allocate the full transform array: (n_frames, n_followers, 16)
all_transforms = np.zeros((n_frames, n_followers, 16), dtype=np.float32)
# Color index per follower per frame (uint8, indexes into a colormap)
all_colors = np.zeros((n_frames, n_followers), dtype=np.uint8)

# Ranges for normalizing position to colormap / scale
z_min, z_max = points[:, 2].min(), points[:, 2].max()
x_min, x_max = points[:, 0].min(), points[:, 0].max()

# Build a smooth 256-entry colormap by linearly interpolating key stops
COLOR_STOPS = np.array(
    [
        [0x33, 0x66, 0xCC],  # deep blue (bottom)
        [0x33, 0xAA, 0xCC],  # teal
        [0x33, 0xCC, 0x66],  # green
        [0xAA, 0xCC, 0x33],  # lime
        [0xFF, 0xCC, 0x00],  # yellow
        [0xFF, 0x88, 0x00],  # orange
        [0xFF, 0x33, 0x33],  # red (top)
    ],
    dtype=np.float32,
)
n_stops = len(COLOR_STOPS)
t_palette = np.linspace(0, 1, 256)
t_stops = np.linspace(0, 1, n_stops)
# Interpolate each RGB component
palette_rgb = np.column_stack(
    [np.interp(t_palette, t_stops, COLOR_STOPS[:, c]) for c in range(3)]
).astype(np.uint8)
COLOR_PALETTE = [(int(r) << 16) | (int(g) << 8) | int(b) for r, g, b in palette_rgb]
N_COLORS = 256

frame_indices = np.arange(n_frames)

for frame_idx in frame_indices:
    # Vectorized index computation for all followers
    indices = ((f_offsets + frame_idx * f_speeds) % NUM_POINTS).astype(int)
    indices_next = (indices + 100) % NUM_POINTS

    pos_all = points[indices]  # (N, 3)
    tan_all = points[indices_next] - pos_all  # (N, 3)
    quats = quaternions_from_directions(tan_all)  # (N, 4)

    # Dynamic scale: 0.5x to 2x based on X position
    x_norm = (pos_all[:, 0] - x_min) / (x_max - x_min + 1e-8)
    scale = f_scales * (0.5 + 1.5 * x_norm)

    # Vectorized quaternion -> rotation matrix, written directly into array
    qx, qy, qz, qw = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    m = all_transforms[frame_idx]
    m[:, 0] = scale * (1 - 2 * (qy * qy + qz * qz))
    m[:, 1] = scale * (2 * (qx * qy + qz * qw))
    m[:, 2] = scale * (2 * (qx * qz - qy * qw))
    m[:, 4] = scale * (2 * (qx * qy - qz * qw))
    m[:, 5] = scale * (1 - 2 * (qx * qx + qz * qz))
    m[:, 6] = scale * (2 * (qy * qz + qx * qw))
    m[:, 8] = scale * (2 * (qx * qz + qy * qw))
    m[:, 9] = scale * (2 * (qy * qz - qx * qw))
    m[:, 10] = scale * (1 - 2 * (qx * qx + qy * qy))
    m[:, 12] = pos_all[:, 0]
    m[:, 13] = pos_all[:, 1]
    m[:, 14] = pos_all[:, 2]
    m[:, 15] = 1.0

    # Color index from Z-height: normalize to 0..255
    z_norm = (pos_all[:, 2] - z_min) / (z_max - z_min + 1e-8)
    all_colors[frame_idx] = (np.clip(z_norm, 0, 1) * 255).astype(np.uint8)

# Build animation with binary channels (no Frame objects needed)
animation = Animation(loop=True)
animation.set_frame_times(np.arange(n_frames) / FPS)
animation.set_transform_data(f_ids, all_transforms)
animation.add_channel(
    "colors",
    f_ids,
    all_colors,
    dtype="uint8",
    metadata={"colormap": COLOR_PALETTE},
)

compute_time = time.time() - start
print(f"Computed in {compute_time:.1f}s")

animation.add_marker(0.0, "Start")
animation.add_marker(DURATION / 4, "Quarter")
animation.add_marker(DURATION / 2, "Halfway")
animation.add_marker(3 * DURATION / 4, "Three quarters")

print("Sending animation to viewer...")
start = time.time()
v.load_animation(animation)
send_time = time.time() - start
print(f"Sent in {send_time:.1f}s ({len(followers)} objects x {n_frames} frames)")
v.wait_for_assets()
