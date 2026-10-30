"""
Watch glTF files and live-reload them in the viewer.

On each save of a source .gltf, convert it to .glb with gltf-transform
(global install, else `npx @gltf-transform/cli`), stage-load the new revision
under a fresh id, wait for it to land, then delete the older revision so the
scene never goes blank.

Each argument is a .gltf file, or a directory `name` meaning `name/name.gltf`
(one folder per asset, as Blender exports). Several can be watched at once.

Run: uv run python examples/31_watch_gltf.py examples/tools/materials/materials.gltf
     (Ctrl+C to stop; --scale, --position and --y-up place the models)
"""

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from threejs_viewer import viewer


def convert(src: Path, dst: Path) -> None:
    exe = shutil.which("gltf-transform")
    cmd = (
        [exe] if exe else [shutil.which("npx") or "npx", "--yes", "@gltf-transform/cli"]
    )
    result = subprocess.run(
        [*cmd, "copy", str(src), str(dst)],
        capture_output=True,
        text=True,
        cwd=src.parent,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())


def resolve(arg: Path) -> Path:
    p = arg.expanduser().resolve()
    return p / f"{p.name}.gltf" if p.is_dir() else p


parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("gltf", type=Path, nargs="+", help=".gltf file or directory NAME")
parser.add_argument("--poll", type=float, default=0.5, help="poll interval in seconds")
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument(
    "--position", type=float, nargs=3, default=[0, 0, 0], metavar=("X", "Y", "Z")
)
parser.add_argument("--y-up", action="store_true", help="convert glTF Y-up to Z-up")
args = parser.parse_args()

sources = [resolve(a) for a in args.gltf]
for src in sources:
    if src.suffix.lower() != ".gltf" or not src.exists():
        parser.error(f"not a .gltf file: {src}")

out_dir = Path(tempfile.mkdtemp(prefix="watch_gltf_"))
v = viewer()
v.clear()
v.add_grid("floor", cell_size=1.0, extent=60.0, fade_start=0.4)

# Per source: last seen mtime, revision counter and the id currently shown
state = {src: {"mtime": None, "rev": 0, "live": None} for src in sources}
print("Watching", ", ".join(str(s) for s in sources), "(Ctrl+C to stop)")

try:
    while True:
        for src, st in state.items():
            mtime = src.stat().st_mtime_ns if src.exists() else None
            if mtime is None or mtime == st["mtime"]:
                continue
            st["mtime"] = mtime
            try:
                glb = out_dir / f"{src.stem}.glb"
                convert(src, glb)
                st["rev"] += 1
                new_id = f"{src.stem}__rev_{st['rev']:06d}"
                v.add_model_binary(
                    new_id, glb, "glb", position=args.position,
                    scale=[args.scale] * 3, y_up=args.y_up,
                )  # fmt: skip
                v.wait_for_assets(disconnect=False)
                if st["live"]:
                    v.delete(st["live"])
                if st["live"] is None:
                    v.frame_object(new_id)
                st["live"] = new_id
                print(f"Reloaded {new_id} at {time.strftime('%H:%M:%S')}")
            except Exception as exc:
                st["mtime"] = (
                    None  # retry next poll (half-written file, npx cold start)
                )
                print(f"{src.name}: reload failed: {exc}")
        time.sleep(args.poll)
except KeyboardInterrupt:
    v.disconnect()
    shutil.rmtree(out_dir, ignore_errors=True)
