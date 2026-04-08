#!/usr/bin/env python3
# /// script
# dependencies = ["Pillow"]
# ///
"""Convert Z-up environment PNGs to Three.js Y-up cubemap JPGs.

Source: env/*.png (Z-up, Y-forward convention)
Output: viewer/static/*.jpg (Three.js Y-up cubemap)

Run from repo root:
    uv run src/threejs_viewer/env/convert_to_cubemap.py
"""

from pathlib import Path
from PIL import Image

ENV_DIR = Path(__file__).parent
STATIC_DIR = ENV_DIR.parent / "viewer" / "static"
SIZE = 64


def convert(size: int = SIZE):
    orig = {}
    for f in ["px", "nx", "py", "ny", "pz", "nz"]:
        orig[f] = (
            Image.open(ENV_DIR / f"{f}.png")
            .convert("RGB")
            .resize((size, size), Image.LANCZOS)
        )

    # Z-up (Y-forward) to Three.js Y-up cubemap face remapping + rotation:
    out = {}
    out["px"] = orig["nx"]
    out["nx"] = orig["px"]
    out["py"] = orig["nz"]
    out["ny"] = orig["pz"]
    out["pz"] = orig["py"].rotate(-90, expand=False)
    out["nz"] = orig["ny"]

    for f in ["px", "nx", "py", "ny", "pz", "nz"]:
        out[f].save(STATIC_DIR / f"{f}.jpg", quality=95)

    print(f"Converted {size}x{size} cubemap JPGs to {STATIC_DIR}")


if __name__ == "__main__":
    convert()
