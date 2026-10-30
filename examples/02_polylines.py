"""
Polylines with colormaps

Gradient-coloured polylines: a helix coloured by height (viridis), a spiral by
speed (plasma), a 3D Lissajous curve by curve parameter (turbo) and a plain
white circle.

Run: uv run python examples/02_polylines.py
"""

import numpy as np

from threejs_viewer import viewer

v = viewer()
v.clear()
v.unload_animation()

v.add_grid("floor", cell_size=1.0, extent=30, center_color=0x888888)

# Helix coloured by height
t = np.linspace(0, 5 * 2 * np.pi, 500)
helix = np.column_stack([1.5 * np.cos(t), 1.5 * np.sin(t), np.linspace(0, 4, 500)])
v.add_polyline(
    "helix_viridis", helix, colors=helix[:, 2], colormap="viridis", line_width=4
)

# Spiral coloured by speed (segment length)
t = np.linspace(0, 4 * 2 * np.pi, 400)
r = np.linspace(0.1, 2, 400)
spiral = np.column_stack([r * np.cos(t) + 5, r * np.sin(t), np.linspace(0, 3, 400)])
speed = np.linalg.norm(np.diff(spiral, axis=0, prepend=spiral[:1]), axis=1)
v.add_polyline("spiral_plasma", spiral, colors=speed, colormap="plasma", line_width=3)

# 3D Lissajous curve coloured by its parameter
t = np.linspace(0, 2 * np.pi, 1000)
lissajous = 1.5 * np.column_stack(
    [np.sin(3 * t + np.pi / 2), np.sin(4 * t), np.sin(5 * t)]
) + [-5, 0, 2]
v.add_polyline("lissajous_turbo", lissajous, colors=t, colormap="turbo", line_width=3)

# Plain colour, no colormap
t = np.linspace(0, 2 * np.pi, 100)
circle = np.column_stack([3 * np.cos(t), 3 * np.sin(t), np.full_like(t, 5)])
v.add_polyline("circle_white", circle, color=0xFFFFFF, line_width=2)

v.set_camera(position=[3, -14, 9], target=[0, 0, 2.5])
v.wait_for_assets()
