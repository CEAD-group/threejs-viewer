"""
Animated GLB with Embedded Clip Times

Loads the AnimatedMorphCube (which has an embedded morph target animation)
and an Avocado that orbits around it. The animation timeline drives both
the cube's embedded clip time and the avocado's orbit transform in sync.

Run: uv run python examples/09_animated_glb.py
"""

import math
import urllib.request
from pathlib import Path

from threejs_viewer import Animation, viewer

CACHE_DIR = Path(__file__).parent / "tmp"

MODELS = {
    "morphcube": {
        "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/AnimatedMorphCube/glTF-Binary/AnimatedMorphCube.glb",
    },
    "avocado": {
        "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Avocado/glTF-Binary/Avocado.glb",
    },
}


def download_model(name: str, url: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{name}.glb"
    if path.exists():
        print(f"  {name}: cached")
        return path
    print(f"  {name}: downloading...")
    urllib.request.urlretrieve(url, path)
    print(f"  {name}: {path.stat().st_size / 1024:.0f} KB")
    return path


print("Downloading models...")
model_paths = {name: download_model(name, info["url"]) for name, info in MODELS.items()}

v = viewer()
v.clear()

# Load models
print("Loading models...")
v.add_model_binary("morphcube", model_paths["morphcube"], format="glb", y_up=True)
v.add_model_binary("avocado", model_paths["avocado"], format="glb", y_up=True)

# Place morphcube at origin, raised slightly
CUBE_SCALE = 0.5
# fmt: off
v.set_matrix("morphcube", [
    CUBE_SCALE, 0,          0,          0,
    0,          CUBE_SCALE, 0,          0,
    0,          0,          CUBE_SCALE, 0,
    0,          0,          1,          1,
])
# fmt: on

# The AnimatedMorphCube clip is ~3 seconds long
CLIP_DURATION = 3.0
ORBIT_RADIUS = 3.0
AVOCADO_SCALE = 20.0

# Animation: one full orbit = one full clip playthrough
duration = CLIP_DURATION
fps = 30
n_frames = int(duration * fps)

print("Building animation...")
animation = Animation(loop=True)

for i in range(n_frames):
    t = i / fps
    progress = t / duration  # 0 to 1

    # Avocado orbits in XY plane around the cube
    angle = progress * 2 * math.pi
    ax = ORBIT_RADIUS * math.cos(angle)
    ay = ORBIT_RADIUS * math.sin(angle)
    az = 0.5
    s = AVOCADO_SCALE

    animation.add_frame(
        time=t,
        transforms={
            # Keep morphcube stationary
            "morphcube": [
                CUBE_SCALE,
                0,
                0,
                0,
                0,
                CUBE_SCALE,
                0,
                0,
                0,
                0,
                CUBE_SCALE,
                0,
                0,
                0,
                1,
                1,
            ],
            # Orbit avocado, facing forward along its path
            "avocado": [
                s * math.cos(angle),
                s * math.sin(angle),
                0,
                0,
                s * -math.sin(angle),
                s * math.cos(angle),
                0,
                0,
                0,
                0,
                s,
                0,
                ax,
                ay,
                az,
                1,
            ],
        },
        # Sine wave: smoothly 0 -> CLIP_DURATION -> 0
        clip_times={
            "morphcube": CLIP_DURATION * 0.5 * (1 - math.cos(2 * math.pi * progress))
        },
    )

v.load_animation(animation)

print(f"Animation: {n_frames} frames, {duration:.1f}s (loop).")
print("Morphcube plays its embedded morph animation while avocado orbits.")
v.wait_for_assets()
