"""
Polylines with Colormaps Demo

Demonstrates gradient-colored polylines using different colormaps.
Shows helices with colors mapped to height/velocity/curvature.

Run: uv run python examples/02_polylines.py
"""

import numpy as np

from threejs_viewer import viewer

v = viewer()
v.clear()
v.stop_animation()

# Add a subtle ground reference
v.add_box("ground", width=12, height=12, depth=0.02, color=0x333333, position=[0, 0, -0.01])


# === Helix with height-based coloring (viridis) ===
def make_helix(radius, height, turns, n_points):
    t = np.linspace(0, turns * 2 * np.pi, n_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.linspace(0, height, n_points)
    return np.column_stack([x, y, z])


helix1 = make_helix(radius=1.5, height=4, turns=5, n_points=500)
z_values = helix1[:, 2]  # Color by height
v.add_polyline("helix_viridis", helix1, colors=z_values, colormap="viridis", line_width=4)


# === Spiral with velocity coloring (plasma) ===
def make_spiral(max_radius, height, turns, n_points):
    t = np.linspace(0, turns * 2 * np.pi, n_points)
    r = np.linspace(0.1, max_radius, n_points)
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = np.linspace(0, height, n_points)
    return np.column_stack([x, y, z])


spiral = make_spiral(max_radius=2, height=3, turns=4, n_points=400)
# Offset to the right
spiral[:, 0] += 5

# Color by "velocity" (derivative magnitude)
velocity = np.sqrt(np.sum(np.diff(spiral, axis=0) ** 2, axis=1))
velocity = np.append(velocity, velocity[-1])  # Match length
v.add_polyline("spiral_plasma", spiral, colors=velocity, colormap="plasma", line_width=3)


# === Lissajous curve with parameter coloring (turbo) ===
def make_lissajous_3d(a, b, c, delta, n_points):
    t = np.linspace(0, 2 * np.pi, n_points)
    x = np.sin(a * t + delta)
    y = np.sin(b * t)
    z = np.sin(c * t)
    return np.column_stack([x, y, z])


lissajous = make_lissajous_3d(a=3, b=4, c=5, delta=np.pi / 2, n_points=1000)
lissajous *= 1.5  # Scale up
lissajous[:, 0] -= 5  # Offset to the left
lissajous[:, 2] += 2  # Raise above ground

# Color by parameter t
t_param = np.linspace(0, 1, 1000)
v.add_polyline("lissajous_turbo", lissajous, colors=t_param, colormap="turbo", line_width=3)


# === Simple white polyline (no colormap) ===
circle = np.zeros((100, 3))
t = np.linspace(0, 2 * np.pi, 100)
circle[:, 0] = 3 * np.cos(t)
circle[:, 1] = 3 * np.sin(t)
circle[:, 2] = 5  # At the top

v.add_polyline("circle_white", circle, color=0xFFFFFF, line_width=2)

print("Polylines with colormaps: viridis (helix), plasma (spiral), turbo (lissajous)")
print("Press Ctrl+C to exit.")

try:
    while True:
        pass
except KeyboardInterrupt:
    v.disconnect()
