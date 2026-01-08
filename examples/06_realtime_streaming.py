"""
Real-time Streaming Demo

Demonstrates the streaming mode (as opposed to pre-computed animations).
Objects are updated in real-time from Python, useful for live simulations,
sensor data, or interactive applications.

Run: uv run python examples/06_realtime_streaming.py
"""

import math
import time

from threejs_viewer import viewer

v = viewer()
v.clear()
v.stop_animation()  # Clear any previous animation UI

# Ground
v.add_box(
    "ground", width=15, height=15, depth=0.02, color=0x333333, position=[0, 0, -0.01]
)

# Create a grid of bouncing spheres
GRID_SIZE = 5
SPACING = 2.0

spheres = []
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        sphere_id = f"sphere_{i}_{j}"
        x = (i - GRID_SIZE / 2 + 0.5) * SPACING
        y = (j - GRID_SIZE / 2 + 0.5) * SPACING

        # Color based on position
        r = int(255 * i / GRID_SIZE)
        g = int(255 * j / GRID_SIZE)
        b = 128
        color = (r << 16) | (g << 8) | b

        v.add_sphere(sphere_id, radius=0.3, color=color, position=[x, y, 0.3])
        spheres.append(
            {
                "id": sphere_id,
                "x": x,
                "y": y,
                "phase": (i + j) * 0.3,  # Phase offset for wave effect
                "freq": 1.5 + 0.1 * (i + j),  # Slightly different frequencies
            }
        )

# Also add a rotating box in the center
v.add_box("center_box", width=1, height=1, depth=1, color=0xFFAA00, position=[0, 0, 2])

print(f"Streaming {len(spheres)} spheres at ~60 fps. Press Ctrl+C to exit.")

# Real-time animation loop
TARGET_FPS = 60
FRAME_TIME = 1.0 / TARGET_FPS
start_time = time.time()
frame_count = 0

try:
    while True:
        t = time.time() - start_time

        # Batch update all sphere positions
        transforms = {}

        for sphere in spheres:
            # Bouncing motion with wave propagation
            z = 0.3 + 0.5 * abs(math.sin(sphere["freq"] * t + sphere["phase"]))
            transforms[sphere["id"]] = {"position": [sphere["x"], sphere["y"], z]}

        # Rotating center box
        angle = t * 0.8
        transforms["center_box"] = {
            "position": [0, 0, 2 + 0.5 * math.sin(t * 2)],
            "rotation": [t * 0.3, angle, t * 0.2],
        }

        # Send batch update
        v.batch_update(transforms)

        # Also update colors periodically (every 30 frames)
        frame_count += 1
        if frame_count % 30 == 0:
            # Pulse the center box color
            intensity = int(128 + 127 * math.sin(t * 3))
            v.set_color("center_box", (intensity << 16) | (intensity << 8) | 0)

        # Frame rate control
        elapsed = time.time() - start_time - t
        sleep_time = FRAME_TIME - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        # Print FPS every second
        if frame_count % TARGET_FPS == 0:
            actual_fps = frame_count / (time.time() - start_time)
            print(f"Running: {actual_fps:.1f} fps, t={t:.1f}s", end="\r")

except KeyboardInterrupt:
    print(f"\nStopped after {frame_count} frames")
    v.disconnect()
