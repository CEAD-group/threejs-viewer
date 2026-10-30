"""
Animated GLB with Embedded Clip Times

An AnimatedMorphCube plays its embedded morph clip while an avocado orbits it.
One animation timeline drives the cube's clip time (clip_times channel) and
the avocado's orbit (transforms channel); scrub it with the timeline.

Run: uv run python examples/09_animated_glb.py
"""

import urllib.request
from pathlib import Path

from itertools import product

import numpy as np

from threejs_viewer import Animation, viewer

KHRONOS = (
    "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models"
)


def download(name):
    path = Path.home() / ".cache" / "threejs-viewer" / f"{name}.glb"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{KHRONOS}/{name}/glTF-Binary/{name}.glb", path)
    return path


CLIP_DURATION = 3.0  # length of the cube's embedded clip, in seconds
ORBIT_RADIUS, AVOCADO_SCALE = 3.0, 20.0

v = viewer()
v.clear()
v.add_grid("floor", cell_size=0.5, extent=30.0, fade_start=0.3)
v.add_model_binary(
    "morphcube",
    download("AnimatedMorphCube"),
    "glb",
    position=[0, 0, 1],
    scale=[1.0] * 3,
    y_up=True,
)
v.add_model_binary("avocado", download("Avocado"), "glb", y_up=True)

# One orbit per clip playthrough; the clip time follows a cosine so it runs 0 -> end -> 0
times = np.linspace(0, CLIP_DURATION, 91)
angle = 2 * np.pi * times / CLIP_DURATION
c, s, k, R = np.cos(angle), np.sin(angle), AVOCADO_SCALE, ORBIT_RADIUS
# Column-major 4x4: scaled rotation about Z (facing along the path) plus translation
orbit = np.zeros((len(times), 1, 16))
orbit[:, 0, [0, 1, 4, 5, 10]] = np.stack([k * c, k * s, -k * s, k * c, k + 0 * c], 1)
orbit[:, 0, 12:] = np.stack([R * c, R * s, 0.5 + 0 * c, 1 + 0 * c], 1)

animation = Animation(loop=True)
animation.set_frame_times(times)
animation.set_transform_data(["avocado"], orbit)
animation.set_clip_time_data(
    ["morphcube"], (CLIP_DURATION * 0.5 * (1 - np.cos(angle)))[:, None]
)
v.load_animation(animation)

v.wait_for_assets(disconnect=False)
v.set_view("iso", animate=False)
v.frame_points(list(product([-3.5, 3.5], [-3.5, 3.5], [0, 2.5])), fill=0.85)
v.disconnect()
