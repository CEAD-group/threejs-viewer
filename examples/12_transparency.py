"""
Transparency Demo

Loads a GLB model and pulses its opacity using the binary opacity animation
channel. Also shows primitives with initial opacity.

Demonstrates:
- set_opacity() for one-shot opacity changes
- opacity parameter on primitives (add_sphere, add_box)
- Binary opacity animation channel for smooth pulsing

Run: uv run python examples/12_transparency.py
"""

import math
import urllib.request
from pathlib import Path

import numpy as np

from threejs_viewer import Animation, viewer

CACHE_DIR = Path(__file__).parent / "tmp"
MODEL_URL = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/DamagedHelmet/glTF-Binary/DamagedHelmet.glb"


def download_model() -> Path:
    """Download the DamagedHelmet GLB if not already cached."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / "helmet.glb"
    if path.exists():
        print("  helmet: cached")
        return path
    print("  helmet: downloading...")
    urllib.request.urlretrieve(MODEL_URL, path)
    print(f"  helmet: {path.stat().st_size / 1024:.0f} KB")
    return path


print("Downloading GLB model...")
helmet_path = download_model()

v = viewer()
v.clear()

# Ground plane
v.add_box(
    "ground", width=10, height=10, depth=0.02, color=0x333333, position=[0, 0, -0.01]
)

# Load GLB model — rotate 90° about X so it faces up (Z-up convention)
v.add_model_binary(
    "helmet",
    helmet_path,
    format="glb",
    matrix=[-1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1.5, 1],
)

# Primitives with initial opacity
v.add_sphere(
    "ghost_sphere", radius=0.5, color=0x4488FF, opacity=0.4, position=[-2, 0, 0.5]
)
v.add_box(
    "ghost_box",
    width=0.8,
    height=0.8,
    depth=0.8,
    color=0xFF4444,
    opacity=0.4,
    position=[2, 0, 0.5],
)
v.add_cylinder(
    "ghost_cyl",
    radius_top=0.3,
    radius_bottom=0.3,
    height=1,
    color=0x44FF44,
    opacity=0.4,
    position=[0, -2, 0.5],
)

# Animate: pulse helmet opacity + keep primitives at varying opacity
duration = 6.0
fps = 60
n_frames = int(duration * fps)
frame_times = np.arange(n_frames) / fps

# Helmet: smooth pulse between 0.2 and 1.0
t_norm = frame_times / duration
helmet_opacity = 0.6 + 0.4 * np.cos(2 * math.pi * t_norm)

# Primitives: staggered sine waves
sphere_opacity = 0.3 + 0.7 * np.abs(np.sin(2 * math.pi * t_norm))
box_opacity = 0.3 + 0.7 * np.abs(np.sin(2 * math.pi * t_norm + math.pi / 3))
cyl_opacity = 0.3 + 0.7 * np.abs(np.sin(2 * math.pi * t_norm + 2 * math.pi / 3))

object_ids = ["helmet", "ghost_sphere", "ghost_box", "ghost_cyl"]
all_opacity = np.column_stack(
    [
        helmet_opacity,
        sphere_opacity,
        box_opacity,
        cyl_opacity,
    ]
).astype(np.float32)

# Opacity-only animation — no transforms needed, objects stay where they were placed
animation = Animation(loop=True)
animation.set_frame_times(frame_times)
animation.add_channel("opacity", object_ids, all_opacity, dtype="float32")

animation.add_marker(0.0, "Opaque", color=0x00FF00)
animation.add_marker(duration / 2, "Translucent", color=0x0088FF)

v.load_animation(animation)

print(f"Pulsing opacity: {n_frames} frames at {fps} fps")
print("Press Ctrl+C to exit.")

try:
    while True:
        pass
except KeyboardInterrupt:
    v.disconnect()
