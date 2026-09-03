"""
Watch a GLTF file and live-reload it in the viewer.

On each save of the source .gltf file:
1) Convert to .glb using gltf-transform.
2) Write the .glb into examples/tmp/.
3) Stage-load a new model revision, then remove older revisions.

Usage:
    uv run python examples/29_watch_gltf.py path/to/model.gltf

Optional setup (if gltf-transform is not already available):
    npm install -g @gltf-transform/cli
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

from threejs_viewer import viewer

TMP_DIR = Path(__file__).parent / "tmp"


def _matrix_from_scale_and_position(scale: float, position: list[float]) -> list[float]:
    x, y, z = position
    # Column-major 4x4 matrix.
    return [
        scale,
        0,
        0,
        0,
        0,
        scale,
        0,
        0,
        0,
        0,
        scale,
        0,
        x,
        y,
        z,
        1,
    ]


def _convert_gltf_to_glb(src_gltf: Path, dst_glb: Path) -> None:
    commands: list[list[str]] = []
    gltf_transform_exe = shutil.which("gltf-transform") or shutil.which(
        "gltf-transform.cmd"
    )
    npx_exe = shutil.which("npx") or shutil.which("npx.cmd")

    if gltf_transform_exe:
        commands.append([gltf_transform_exe, "copy", str(src_gltf), str(dst_glb)])
    if npx_exe:
        commands.append(
            [
                npx_exe,
                "--yes",
                "@gltf-transform/cli",
                "copy",
                str(src_gltf),
                str(dst_glb),
            ]
        )

    if not commands:
        raise RuntimeError(
            "Could not find gltf-transform or npx. Install @gltf-transform/cli first."
        )

    last_error: str | None = None
    for cmd in commands:
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=str(src_gltf.parent),
            )
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        except OSError as exc:
            last_error = str(exc)

    raise RuntimeError(f"gltf-transform conversion failed: {last_error}")


def _stage_model_revision(
    v,
    model_id: str,
    revision: int,
    glb_path: Path,
    *,
    scale: float,
    position: list[float],
    y_up: bool,
) -> str:
    rev_id = f"{model_id}__rev_{revision:06d}"
    v.add_model_binary(rev_id, glb_path, format="glb", y_up=y_up)
    v.set_matrix(rev_id, _matrix_from_scale_and_position(scale, position))
    return rev_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Watch a .gltf file, convert it to .glb on save, and reload it in threejs-viewer."
        )
    )
    parser.add_argument("gltf", type=Path, help="Path to source .gltf file")
    parser.add_argument(
        "--id",
        dest="model_id",
        default="watched_model",
        help="Viewer object id to replace on each save",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=0.5,
        help="Polling interval in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Uniform model scale (default: 1.0)",
    )
    parser.add_argument(
        "--position",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Model position (default: 0 0 0)",
    )
    parser.add_argument(
        "--y-up",
        action="store_true",
        help="Apply glTF Y-up to viewer Z-up correction",
    )
    args = parser.parse_args()

    src_gltf = args.gltf.expanduser().resolve()
    if src_gltf.suffix.lower() != ".gltf":
        raise ValueError(f"Expected a .gltf file, got: {src_gltf}")

    if not src_gltf.exists():
        raise FileNotFoundError(f"Source GLTF not found: {src_gltf}")

    TMP_DIR.mkdir(exist_ok=True)
    dst_glb = TMP_DIR / f"{src_gltf.stem}.glb"

    v = viewer()

    seen_mtime_ns: int | None = None
    revision = 0

    print(f"Watching: {src_gltf}")
    print(f"Output:   {dst_glb}")
    print("Press Ctrl+C to stop.")

    while True:
        try:
            mtime_ns = src_gltf.stat().st_mtime_ns
        except FileNotFoundError:
            print("Source file missing. Waiting for it to reappear...")
            time.sleep(args.poll)
            continue

        if mtime_ns != seen_mtime_ns:
            seen_mtime_ns = mtime_ns
            try:
                _convert_gltf_to_glb(src_gltf, dst_glb)
                revision += 1
                new_id = _stage_model_revision(
                    v,
                    args.model_id,
                    revision,
                    dst_glb,
                    scale=args.scale,
                    position=args.position,
                    y_up=args.y_up,
                )

                # Ensure the staged model has finished loading before we
                # remove older revisions, so the scene never goes blank.
                v.wait_for_assets(disconnect=False)

                scene = v.query_scene()
                object_ids = list(scene.get("objects", {}).keys())
                for object_id in object_ids:
                    if object_id != new_id:
                        v.delete(object_id)
                print(
                    f"Reloaded {new_id} at {time.strftime('%H:%M:%S')} "
                    f"({dst_glb.stat().st_size / 1024:.0f} KB)"
                )
            except Exception as exc:
                print(f"Reload failed: {exc}")

        time.sleep(args.poll)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")