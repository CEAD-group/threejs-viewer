"""
Stress Test Demo

A torus knot tube with millions of points and hundreds of followers
racing along the path. Tests polyline rendering and batch transform updates.

Run: uv run python examples/07_stress_test.py
"""

import colorsys
import random
import time
from pathlib import Path

import numpy as np

from threejs_viewer import viewer


def torus_knot(t: np.ndarray, p: int = 3, q: int = 7, scale: float = 5.0):
    """Torus knot parametric curve with analytical tangent."""
    r = 0.5
    x = scale * (np.cos(p * t) * (1 + r * np.cos(q * t)))
    y = scale * (np.sin(p * t) * (1 + r * np.cos(q * t)))
    z = scale * (r * np.sin(q * t))

    dx = scale * (-p * np.sin(p * t) * (1 + r * np.cos(q * t)) - r * q * np.cos(p * t) * np.sin(q * t))
    dy = scale * (p * np.cos(p * t) * (1 + r * np.cos(q * t)) - r * q * np.sin(p * t) * np.sin(q * t))
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


# Configuration
NUM_POINTS = 1_000_000
TUBE_RADIUS = 0.4
WINDINGS = 500
NUM_FOLLOWERS = 500

print(f"Generating torus knot with {NUM_POINTS:,} points...")
start = time.time()

t = np.linspace(0, 2 * np.pi, NUM_POINTS, dtype=np.float64)
pos, tangent = torus_knot(t)
points = create_tube(t, pos, tangent, tube_radius=TUBE_RADIUS, windings=WINDINGS)

print(f"Generated in {time.time() - start:.2f}s ({points.nbytes / 1024 / 1024:.1f} MB)")

# Connect and send
v = viewer()
v.clear()
v.stop_animation()

print("Sending polyline to viewer...")
start = time.time()
v.add_polyline("tube", points, colors=t.astype(np.float32), colormap="turbo", line_width=2)
print(f"Sent in {time.time() - start:.2f}s")

# Create followers with different primitives
print(f"Creating {NUM_FOLLOWERS} followers...")

TEAPOT_PATH = Path(__file__).parent / "teapot.obj"
NUM_TEAPOTS = 20
PRIMITIVES = ["sphere", "box", "cylinder", "capsule", "cone"]

followers = []
for i in range(NUM_FOLLOWERS):
    hue = random.uniform(0, 1)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
    color = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
    size = random.uniform(0.1, 0.3)

    fid = f"f{i}"
    ptype = PRIMITIVES[i % len(PRIMITIVES)]

    if ptype == "sphere":
        v.add_sphere(fid, radius=size, color=color)
    elif ptype == "box":
        v.add_box(fid, width=size, height=size, depth=size * 2, color=color)
    elif ptype == "cylinder":
        v.add_cylinder(fid, radius_top=size * 0.4, radius_bottom=size * 0.4, height=size * 2, color=color)
    elif ptype == "capsule":
        v.add_capsule(fid, radius=size * 0.4, length=size, color=color)
    elif ptype == "cone":
        v.add_cylinder(fid, radius_top=0, radius_bottom=size * 0.5, height=size * 2, color=color)

    followers.append({
        "id": fid,
        "speed": random.uniform(5, 30),
        "offset": random.randint(0, NUM_POINTS - 1),
    })

# Add teapots
print(f"Adding {NUM_TEAPOTS} teapots...")
for i in range(NUM_TEAPOTS):
    fid = f"teapot{i}"
    v.add_model_binary(fid, TEAPOT_PATH, format="obj")
    followers.append({
        "id": fid,
        "speed": random.uniform(5, 20),
        "offset": random.randint(0, NUM_POINTS - 1),
        "scale": 0.2,  # 20% scale
    })


def quaternion_from_direction(direction):
    """Compute quaternion to rotate Y-axis to given direction."""
    d = direction / (np.linalg.norm(direction) + 1e-8)
    y_axis = np.array([0, 1, 0])
    dot = np.dot(y_axis, d)

    if dot > 0.9999:
        return [0, 0, 0, 1]
    if dot < -0.9999:
        return [1, 0, 0, 0]

    axis = np.cross(y_axis, d)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    angle = np.arccos(np.clip(dot, -1, 1))
    half = angle / 2
    s = np.sin(half)
    return [axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half)]


print(f"Animating {len(followers)} objects. Press Ctrl+C to stop.")

frame = 0
t_start = time.time()

try:
    while True:
        loop_start = time.time()

        transforms = {}
        for f in followers:
            idx = int((f["offset"] + frame * f["speed"]) % NUM_POINTS)
            idx_next = (idx + 100) % NUM_POINTS
            pos = points[idx]
            tangent = points[idx_next] - pos
            quat = quaternion_from_direction(tangent)

            scale = f.get("scale", 1.0)
            transforms[f["id"]] = {
                "position": pos.tolist(),
                "quaternion": quat,
                "scale": [scale, scale, scale],
            }

        v.batch_update(transforms)
        frame += 1

        if frame % 60 == 0:
            elapsed = time.time() - t_start
            fps = frame / elapsed
            print(f"  {len(followers)} objects @ {fps:.1f} fps", end="\r")

        sleep_time = 1 / 60 - (time.time() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    elapsed = time.time() - t_start
    print(f"\nStopped: {frame} frames, {frame/elapsed:.1f} fps avg")
    v.disconnect()
