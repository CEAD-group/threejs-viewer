"""
GLB Models with PBR Materials

Downloads two sample models from the Khronos glTF-Sample-Assets repository
(metalness, roughness, normal maps, emissive) and places them on a floor grid.

Run: uv run python examples/08_glb_models.py
"""

import urllib.request
from pathlib import Path

from itertools import product

from threejs_viewer import viewer

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

# y_up=True converts glTF's Y-up to the viewer's Z-up; the avocado is ~7 cm, so scale it up
v.add_model_binary(
    "helmet", download("DamagedHelmet"), "glb", position=[-1.5, 0, 1], y_up=True
)
v.add_model_binary(
    "avocado",
    download("Avocado"),
    "glb",
    position=[1.5, 0, 0],
    scale=[30] * 3,
    y_up=True,
)

v.set_view("iso", animate=False)
v.frame_points(list(product([-2.4, 2.0], [-0.5, 0.5], [0, 2.0])), fill=1.0)
v.wait_for_assets()
