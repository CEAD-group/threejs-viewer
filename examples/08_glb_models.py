"""
GLB Models with PBR Materials

Downloads sample GLB models from the Khronos glTF-Sample-Assets repository
to showcase PBR material rendering (metalness, roughness, normal maps, emissive).

Run: uv run python examples/08_glb_models.py
"""

import urllib.request
from pathlib import Path

from threejs_viewer import viewer

CACHE_DIR = Path(__file__).parent / "tmp"

MODELS = {
    "helmet": {
        "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/DamagedHelmet/glTF-Binary/DamagedHelmet.glb",
        "scale": 1.0,
        "position": [-2, 0, 1],
    },
    "avocado": {
        "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Avocado/glTF-Binary/Avocado.glb",
        "scale": 30.0,
        "position": [2, 0, 0],
    },
}


def download_model(name: str, url: str) -> Path:
    """Download a GLB file if not already cached."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{name}.glb"
    if path.exists():
        print(f"  {name}: cached")
        return path
    print(f"  {name}: downloading...")
    urllib.request.urlretrieve(url, path)
    print(f"  {name}: {path.stat().st_size / 1024:.0f} KB")
    return path


print("Downloading GLB models...")
model_paths = {name: download_model(name, info["url"]) for name, info in MODELS.items()}

v = viewer()
v.clear()

for name, info in MODELS.items():
    s = info["scale"]
    x, y, z = info["position"]
    v.add_model_binary(name, model_paths[name], format="glb", y_up=True)
    # Column-major 4x4 identity with scale and translation
    # fmt: off
    v.set_matrix(name, [
        s, 0, 0, 0,
        0, s, 0, 0,
        0, 0, s, 0,
        x, y, z, 1,
    ])
    # fmt: on

print("Models loaded.")
v.wait_for_assets()
