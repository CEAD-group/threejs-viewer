"""
Watch a GLTF file and live-reload it in the viewer.

On each save of the source .gltf file:
1) Convert to .glb using gltf-transform.
2) Write the .glb into examples/tmp/.
3) Stage-load a new model revision, then remove older revisions.

Usage:
    uv run python examples/31_watch_gltf.py path/to/model.gltf
    uv run python examples/31_watch_gltf.py name name2   # -> name/name.gltf, name2/name2.gltf

Each argument is either a .gltf file or a directory ``name`` holding
``name/name.gltf`` (the Blender export convention of one folder per asset).
Several can be watched at once; each gets its own viewer id (the file stem).

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


def _resolve_gltf(arg: Path) -> Path:
    """Accept ``foo.gltf`` or a directory ``foo`` meaning ``foo/foo.gltf``."""
    p = arg.expanduser().resolve()
    if p.is_dir():
        p = p / f"{p.name}.gltf"
    if p.suffix.lower() != ".gltf":
        raise ValueError(f"Expected a .gltf file or a directory, got: {arg}")
    if not p.exists():
        raise FileNotFoundError(f"Source GLTF not found: {p}")
    return p


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Watch a .gltf file, convert it to .glb on save, and reload it in threejs-viewer."
        )
    )
    parser.add_argument(
        "gltf",
        type=Path,
        nargs="+",
        help="Source .gltf file(s), or directory NAME meaning NAME/NAME.gltf",
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

    sources = [_resolve_gltf(a) for a in args.gltf]
    TMP_DIR.mkdir(exist_ok=True)

    v = viewer()

    # Per-source state: output .glb, last seen mtime, revision counter, live id.
    state: dict[Path, dict] = {
        src: {"glb": TMP_DIR / f"{src.stem}.glb", "mtime": None, "rev": 0, "live": None}
        for src in sources
    }

    for src, st in state.items():
        print(f"Watching: {src}  ->  {st['glb']}")
    print("Press Ctrl+C to stop.")

    while True:
        for src, st in state.items():
            try:
                mtime_ns = src.stat().st_mtime_ns
            except FileNotFoundError:
                if st["mtime"] is not None:
                    print(
                        f"{src.name}: source file missing. Waiting for it to reappear..."
                    )
                    st["mtime"] = None
                continue
            if mtime_ns == st["mtime"]:
                continue
            st["mtime"] = mtime_ns
            try:
                _convert_gltf_to_glb(src, st["glb"])
                st["rev"] += 1
                new_id = _stage_model_revision(
                    v,
                    src.stem,
                    st["rev"],
                    st["glb"],
                    scale=args.scale,
                    position=args.position,
                    y_up=args.y_up,
                )

                # Ensure the staged model has finished loading before we
                # remove the previous revision, so the scene never goes blank.
                v.wait_for_assets(disconnect=False)
                if st["live"] is not None:
                    v.delete(st["live"])
                st["live"] = new_id
                print(
                    f"Reloaded {new_id} at {time.strftime('%H:%M:%S')} "
                    f"({st['glb'].stat().st_size / 1024:.0f} KB)"
                )
            except Exception as exc:
                print(f"{src.name}: reload failed: {exc}")

        time.sleep(args.poll)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
