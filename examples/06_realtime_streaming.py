"""
Real-time streaming

Streaming mode (as opposed to pre-computed animation): a grid of bouncing
spheres and a spinning box are updated from Python at ~60 fps with
`batch_update`, and the box colour cycles through the hue circle.
Press Ctrl+C to stop.

Run: uv run python examples/06_realtime_streaming.py
"""

import colorsys
import math
import time
from itertools import product

from threejs_viewer import viewer

GRID, SPACING, FPS = 8, 1.6, 60

v = viewer()
v.clear()
v.unload_animation()
v.add_grid("floor", cell_size=1.0, extent=40, center_color=0x888888)

# Sphere colour follows grid position; phase and frequency make a wave pattern
spheres = {}
for i in range(GRID):
    for j in range(GRID):
        x, y = (i - GRID / 2 + 0.5) * SPACING, (j - GRID / 2 + 0.5) * SPACING
        color = (int(255 * i / GRID) << 16) | (int(255 * j / GRID) << 8) | 128
        v.add_sphere(
            f"sphere_{i}_{j}",
            radius=0.3,
            color=color,
            position=[x, y, 0.3],
            roughness=0.35,
            metalness=0.5,
        )
        spheres[f"sphere_{i}_{j}"] = (x, y, (i + j) * 0.3, 1.5 + 0.1 * (i + j))

v.add_box("center_box", width=1, height=1, depth=1, position=[0, 0, 2], roughness=0.15)
v.set_camera(position=[0, -16, 12], target=[0, 0, 0.8])  # view direction
v.frame_points(list(product([-6.4, 6.4], [-6.4, 6.4], [0, 3])), fill=0.85)

print(f"Streaming {len(spheres)} spheres at ~{FPS} fps. Ctrl+C to stop.")
start = time.time()
frames = 0
try:
    while True:
        t = time.time() - start
        updates = {
            id: {"position": [x, y, 0.3 + 0.5 * abs(math.sin(freq * t + phase))]}
            for id, (x, y, phase, freq) in spheres.items()
        }
        updates["center_box"] = {
            "position": [0, 0, 2 + 0.5 * math.sin(t * 2)],
            "rotation": [t * 0.3, t * 0.8, t * 0.2],
        }
        v.batch_update(updates)
        r, g, b = colorsys.hsv_to_rgb((t * 0.1) % 1.0, 1.0, 1.0)
        v.set_color(
            "center_box", (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
        )

        frames += 1
        time.sleep(
            max(0.0, frames / FPS - (time.time() - start))
        )  # hold the frame rate
except KeyboardInterrupt:
    print(f"\nStopped after {frames} frames")
    v.disconnect()
