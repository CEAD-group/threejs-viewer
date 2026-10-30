"""
Transparency Demo

Pulses the opacity of a GLB model and three translucent primitives with the
binary opacity animation channel. Primitives take an initial `opacity=`,
`set_opacity()` changes it one-shot.

Run: uv run python examples/12_transparency.py
"""

import math
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


v = viewer()
v.clear()
v.add_grid("floor", cell_size=0.5, extent=30.0, fade_start=0.3)

v.add_model_binary(
    "helmet", download("DamagedHelmet"), "glb", position=[0, 0, 1], y_up=True
)
v.add_sphere("sphere", radius=0.5, color=0x4488FF, opacity=0.4, position=[-2, 0, 0.5])
v.add_box(
    "box",
    width=0.8,
    height=0.8,
    depth=0.8,
    color=0xFF4444,
    opacity=0.4,
    position=[2, 0, 0.4],
)
v.add_cylinder(
    "cylinder", radius_top=0.3, radius_bottom=0.3, height=1, color=0x44FF44, opacity=0.4,
    position=[0, 2, 0.5], rotation=[math.pi / 2, 0, 0],
)  # fmt: skip
v.set_opacity("helmet", 0.5)  # one-shot; the animation takes over once it plays

# One column per object: the helmet pulses, the primitives are phase-shifted
duration, fps = 6.0, 60
times = np.arange(int(duration * fps)) / fps
phase = 2 * np.pi * times / duration
opacity = np.column_stack(
    [0.6 + 0.4 * np.cos(phase)]
    + [0.3 + 0.7 * np.abs(np.sin(phase + k * np.pi / 3)) for k in range(3)]
).astype(np.float32)

animation = Animation(loop=True)
animation.set_frame_times(times)
animation.add_channel(
    "opacity", ["helmet", "sphere", "box", "cylinder"], opacity, dtype="float32"
)
animation.add_marker(0.0, "Opaque", color=0x00FF00)
animation.add_marker(duration / 2, "Translucent", color=0x0088FF)
v.load_animation(animation)

v.set_view("iso", animate=False)
v.frame_points(list(product([-2.5, 2.5], [-1.5, 2.3], [-0.3, 1.8])), fill=1.0)
v.wait_for_assets()
