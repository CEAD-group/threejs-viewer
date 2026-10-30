"""
Animation basics

Looping, pre-computed animation: a sun with four planets and a moon. The moon
is a child of the earth group, so it only animates its local orbit offset.
Orbit periods divide the duration, so the loop is seamless.

Run: uv run python examples/03_animation_basics.py
"""

from itertools import product

import numpy as np

from threejs_viewer import Animation, viewer

DURATION, FPS = 14.0, 20
# id -> (orbit radius, period); the moon orbits inside the earth group
ORBITS = {
    "mercury": (2.0, 2.0),
    "venus": (3.0, 3.5),
    "earth_system": (4.5, 7.0),
    "mars": (6.0, 14.0),
    "moon": (0.6, 0.7),
}

v = viewer()
v.clear()
v.add_grid("floor", cell_size=1.0, extent=30, center_color=0x888888)
v.add_sphere("sun", radius=1.0, color=0xFFDD00, roughness=0.3, metalness=0.8)
v.add_sphere("mercury", radius=0.15, color=0x888888, roughness=0.6, metalness=0.4)
v.add_sphere("venus", radius=0.25, color=0xFFAA55, roughness=0.8, metalness=0.1)
v.add_sphere("mars", radius=0.2, color=0xFF4422, roughness=0.9, metalness=0.05)
v.add_group("earth_system")
v.add_sphere("earth", radius=0.3, color=0x4488FF, parent="earth_system")
v.add_sphere("moon", radius=0.08, color=0xCCCCCC, parent="earth_system")

# Circular orbits as (frames, objects, 16) column-major translation matrices
ids = list(ORBITS)
times = np.arange(int(DURATION * FPS) + 1) / FPS
radius, period = np.array(list(ORBITS.values())).T
angle = 2 * np.pi * times[:, None] / period
data = np.zeros((len(times), len(ids), 16), dtype=np.float32)
data[..., [0, 5, 10, 15]] = 1
data[..., 12] = radius * np.cos(angle)
data[..., 13] = radius * np.sin(angle)

animation = Animation(loop=True)
animation.set_frame_times(times)
animation.set_transform_data(ids, data)
animation.add_marker(0.0, "Start", color=0x00FF00)
animation.add_marker(DURATION / 2, "Halfway", color=0xFFFF00)

v.load_animation(animation)
v.set_camera(position=[0, -5, 14], target=[0, 0, 0])  # view direction
v.frame_points(list(product([-6.2, 6.2], [-6.2, 6.2], [0])), fill=0.85)
v.wait_for_assets()
