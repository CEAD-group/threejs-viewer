"""Re-record the README gallery images in docs/media from the examples.

Usage:
    uv run --with pillow python tools/record_readme_media.py [NAME ...] [--out DIR] [--jobs N]

Each shot runs one example in a subprocess, with the viewer in headless
Chromium (Metal ANGLE, so the GPU is used), and writes a PNG or an animated
GIF. Without names every shot in SHOTS is recorded. Animated shots of an
example with a loaded animation are seeked frame by frame (deterministic
loops); live examples are captured at a fixed interval. Needs the
browser-test group (`uv sync --group browser-test`).

Shots that use `at=` or `live=`, and the octree-LOD shot, wait on wall-clock
time (node streaming and live updates run on their own clocks), so their
output can differ with machine speed. Seeked shots only differ by GPU.

PNGs are palette-quantized after capture: with pngquant when it is on PATH,
otherwise with Pillow. Both are deterministic for a given tool version.

A shot whose example needs a missing optional package (`usd_connecting_rod`
needs `usd_brep`) is skipped with a message; any other failure exits non-zero
as soon as the example fails.

The script depends on a few viewer internals, each marked where it is used.
"""

import argparse
import inspect
import os
import runpy
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# name: example, optional args, and one of
#   at=<seconds after connect> (PNG),
#   seek=(start, end, frames) as fractions of the animation duration (GIF,
#   or a PNG for one frame),
#   live=(start, frames, interval) in seconds (GIF).
# js runs in the page before the first capture, click=<object id> clicks the
# centre of that object first, keys=<list> presses keys before the capture.
# view=dict(dir=, box=None, fill=0.9) switches to perspective, looks along -dir
# and frames box=(min, max), or all visible content without one; the examples
# open in an orthographic iso view, which reads flat in a still. size is the
# page size; gif_width the GIF width.
FRAME_ALL = "window.threejsViewer.frameAll();"
# Exit code of a child whose example needs a package that is not installed.
SKIPPED = 4

SHOTS = {
    "robot_cell": dict(
        example="14_robot_cell.py", seek=(0.0, 1.0, 60), js=FRAME_ALL, frame_ms=140
    ),
    "section_caps": dict(
        example="16_section_caps.py",
        at=10,
        # Hides the left tool rail that set_clipping_slab brings up.
        js="['clip', 'lighting'].forEach(id => window.threejsViewer.getMenu(id).setHidden(true));"
        + FRAME_ALL,
    ),
    "cad_inspection": dict(example="39_cad_inspection.py", click="block_right", at=5),
    "million_point_toolpath": dict(
        example="40_million_point_toolpath.py", seek=(0.02, 1.0, 40)
    ),
    "stress_test": dict(example="07_stress_test.py", live=(4, 30, 0.1), gif_width=640),
    "animation_stress": dict(
        example="10_animation_stress_test.py", seek=(0.0, 0.5, 30), gif_width=640
    ),
    "flying_teapots": dict(
        example="04_flying_teapots.py", seek=(0.0, 1.0, 48), js=FRAME_ALL
    ),
    "toolpath_vase": dict(example="11_toolpath.py", seek=(0.03, 1.0, 30), js=FRAME_ALL),
    "toolpath_interrupted": dict(example="15_toolpath_interrupted.py", at=5),
    "tube_breaks": dict(example="28_tube_breaks.py", at=5),
    "swept_tool": dict(example="26_swept_tool.py", seek=(0.03, 1.0, 30), js=FRAME_ALL),
    "bead_texture_print": dict(
        example="37_bead_texture.py", seek=(0.03, 1.0, 36), js=FRAME_ALL, gif_width=960
    ),
    "bead_texture_corner": dict(
        example="37_bead_texture.py",
        seek=(1.0, 1.0, 1),
        view=dict(dir=(-1, -1.1, 0.8), box=((-5, -5, 0), (80, 75, 42)), fill=1.0),
    ),
    "bead_texture_hero": dict(
        example="37_bead_texture.py",
        seek=(1.0, 1.0, 1),
        view=dict(dir=(-0.45, -1, 0.75), box=((0, 0, 0), (292, 200, 42)), fill=0.7),
    ),
    "point_cloud": dict(example="25_point_cloud.py", seek=(1.0, 1.0, 1)),
    "point_cloud_sweep": dict(example="25_point_cloud.py", seek=(0.03, 1.0, 36)),
    "points_octree_lod": dict(
        example="27_points_octree_lod.py",
        seek=(0.0, 1.0, 32),
        # The box skips the sheet's thin z tails, which the example frames.
        view=dict(dir=(0.4, -1, 0.8), box=((-20, -10, -1), (20, 10, 1)), fill=0.9),
    ),
    "live_points_stream": dict(example="30_live_points_stream.py", live=(2, 24, 0.25)),
    "depth_cues": dict(
        example="20_line_depth_cues.py",
        at=5,
        view=dict(dir=(0.25, -1, 0.25), box=((-4, -4, -4.4), (4, 4, 4)), fill=0.95),
    ),
    "polyline_picking": dict(example="22_polyline_picking.py", at=5, js=FRAME_ALL),
    "glb_models": dict(example="08_glb_models.py", at=6),
    "materials_sampler": dict(
        example="31_watch_gltf.py",
        args=["examples/tools/materials/materials.gltf"],
        at=25,
        view=dict(dir=(0, -1, 1.4)),
    ),
    "billboards": dict(example="34_billboard_modes.py", at=5),
    "gizmos": dict(example="24_gizmo_dimensions.py", at=5),
    "axis_controls": dict(example="36_axis_control.py", at=5),
    "floor_grid": dict(example="29_floor_grid.py", at=5),
    "client_menu": dict(
        example="35_client_menu.py",
        at=5,
        js="window.threejsViewer.getMenu('demo').open();",
    ),
    "lighting_panel": dict(example="01_primitives.py", at=5, keys=["e"]),
    # Needs the optional usd_brep package.
    "usd_connecting_rod": dict(
        example="38_usd_connecting_rod.py", at=8, size=(1600, 1000)
    ),
}


def _object_centre_js(obj_id):
    # Centroid of the canvas pixels where pick() hits the object, on a 10 px grid.
    return f"""(() => {{
        const v = window.threejsViewer, r = v.getCanvas().getBoundingClientRect();
        let sx = 0, sy = 0, n = 0;
        for (let y = r.top + 5; y < r.bottom; y += 10)
            for (let x = r.left + 5; x < r.right; x += 10)
                if (v.pick(x, y)?.objectId === {obj_id!r}) {{ sx += x; sy += y; n++; }}
        return n ? [sx / n, sy / n] : null;
    }})()"""


def _focus(page):
    # Viewer shortcuts listen on the container, which has a tabindex.
    page.evaluate("window.threejsViewer.container.focus()")


def _setup(page, shot):
    """Apply the shot's camera view, then its js."""
    view = shot.get("view")
    if view:
        if page.evaluate("window.threejsViewer.getCameraPose().fov === null"):
            _focus(page)
            page.keyboard.press("o")  # the only public switch to perspective
        box = view.get("box")
        page.evaluate(
            """([d, box, fill]) => {
                const v = window.threejsViewer, t = v.getCameraPose().target;
                v.setCameraPose({position: [t.x + d[0], t.y + d[1], t.z + d[2]],
                                 up: [0, 0, 1]});
                if (box) v.frameBox(box[0], box[1], {fill});
                else v.frameAll();
            }""",
            [list(view["dir"]), [list(c) for c in box] if box else None,
             view.get("fill", 0.9)],
        )  # fmt: skip
    if shot.get("js"):
        page.evaluate(shot["js"])


def _record(client, connected, captured, shot, out):
    from playwright.sync_api import sync_playwright

    width, height = shot.get("size", (1280, 800))
    frames = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=["--use-angle=metal", "--ignore-gpu-blocklist"]
        )
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(client.viewer_url)
        if not connected.wait(timeout=120):
            raise TimeoutError("viewer did not connect")
        t0 = time.monotonic()

        def wait_until(t):
            time.sleep(max(0.0, t0 + t - time.monotonic()))

        def settle():
            page.evaluate(
                "new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
            )

        if "seek" in shot:
            page.wait_for_function(
                "window.threejsViewer?.getAnimationState() && window.threejsViewer.getLoadState().pending === 0",
                timeout=120_000,
            )
            wait_until(shot.get("warmup", 3))
            _setup(page, shot)
            duration = page.evaluate(
                "(() => { const v = window.threejsViewer; v.setAnimationPlaying(false);"
                " return v.getAnimationState().duration; })()"
            )
            a, b, n = shot["seek"]
            for i in range(n):
                t = duration * (a + (b - a) * i / max(n - 1, 1))
                page.evaluate(f"window.threejsViewer.seekAnimationTime({t})")
                settle()
                time.sleep(shot.get("frame_wait", 0.15))
                frames.append(page.screenshot())
        elif "live" in shot:
            start, n, interval = shot["live"]
            wait_until(start)
            _setup(page, shot)
            for i in range(n):
                wait_until(start + i * interval)
                frames.append(page.screenshot())
        else:
            if shot.get("click"):
                wait_until(shot["at"] / 2)
                hit = page.evaluate(_object_centre_js(shot["click"]))
                if hit is None:
                    raise RuntimeError(f"{shot['click']} is not on screen")
                page.mouse.click(*hit)
            if shot.get("js") or shot.get("view"):
                # After the example's own framing and loads.
                wait_until(shot["at"] * 0.6)
                page.wait_for_function(
                    "window.threejsViewer.getLoadState().pending === 0"
                )
                _setup(page, shot)
            if shot.get("keys"):
                _focus(page)
                for key in shot["keys"]:
                    page.keyboard.press(key)
            wait_until(shot["at"])
            frames.append(page.screenshot())
        # A live example fails its next send once the browser closes.
        captured.set()
        browser.close()
    if len(frames) == 1:
        _write_png(frames[0], out.with_suffix(".png"))
    elif frames:
        _write_gif(frames, out.with_suffix(".gif"), shot)


def _write_png(data, path):
    """Write a screenshot as a palette PNG, which is a third to a tenth the size."""
    path.write_bytes(data)
    pngquant = shutil.which("pngquant")
    if pngquant:
        # 98/99: quality floor not met or no gain; the RGB file stays.
        r = subprocess.run(
            [pngquant, "--force", "--skip-if-larger", "--strip", "--speed", "1",
             "--quality", "80-100", "--output", str(path), str(path)]
        )  # fmt: skip
        if r.returncode in (0, 98, 99):
            return
    import io

    from PIL import Image

    im = Image.open(io.BytesIO(data)).convert("RGB")
    im = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    if buf.tell() < len(data):
        path.write_bytes(buf.getvalue())


def _write_gif(frames, path, shot):
    import io

    from PIL import Image

    gif_width = shot.get("gif_width", 800)
    images = []
    for data in frames:
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im = im.resize(
            (gif_width, round(im.height * gif_width / im.width)), Image.LANCZOS
        )
        images.append(im)
    # One palette for every frame keeps the background from flickering.
    palette = images[len(images) // 2].quantize(
        colors=255, method=Image.Quantize.MEDIANCUT
    )
    quantized = [
        im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images
    ]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=shot.get("frame_ms", 100),
        loop=0,
        optimize=True,
    )


def _exit(code, message=None):
    if message:
        print(message, file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _child(name, out_dir):
    from threejs_viewer import ViewerClient

    shot = SHOTS[name]
    out = Path(out_dir).resolve() / name
    started, connected, captured = (threading.Event() for _ in range(3))
    orig_init, orig_connect = ViewerClient.__init__, ViewerClient.connect
    signature = inspect.signature(orig_init)

    # The examples build their own client: patch in a free port and no system
    # browser, and start the capture from connect().
    def init(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.arguments.update(port=0, open_browser=False)
        orig_init(*bound.args, **bound.kwargs)

    def connect(self, timeout=30.0):
        if not started.is_set():
            started.set()
            threading.Thread(target=run, args=(self,), daemon=True).start()
        result = orig_connect(self, timeout=120)
        connected.set()
        return result

    def run(client):
        # port is 0 until connect() has bound the servers.
        deadline = time.monotonic() + 60
        while client.port == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        try:
            _record(client, connected, captured, shot, out)
        except Exception:  # noqa: BLE001
            _exit(1, f"{name}: {traceback.format_exc()}")
        _exit(0)

    ViewerClient.__init__ = init
    ViewerClient.connect = connect
    watchdog = threading.Timer(300, _exit, args=(3, f"{name}: timed out"))
    watchdog.daemon = True
    watchdog.start()
    example = ROOT / "examples" / shot["example"]
    sys.argv = [str(example), *shot.get("args", [])]
    sys.path.insert(0, str(example.parent))
    try:
        runpy.run_path(str(example), run_name="__main__")
    except SystemExit as e:
        if e.code not in (None, 0) and not captured.is_set():
            _exit(1, f"{name}: example exited with {e.code!r}")
    except ModuleNotFoundError as e:
        if not started.is_set():
            _exit(SKIPPED, f"{name}: skipped, needs the {e.name} package")
        _exit(1, f"{name}: {traceback.format_exc()}")
    except BaseException:  # noqa: BLE001
        if not captured.is_set():
            _exit(1, f"{name}: {traceback.format_exc()}")
    if not started.is_set():
        _exit(1, f"{name}: the example never connected a viewer")
    # The example returned before the capture finished; the capture thread exits.
    threading.Event().wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--out", default=str(ROOT / "docs" / "media"))
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        _child(args.child, args.out)
        return
    names = args.names or list(SHOTS)
    unknown = set(names) - set(SHOTS)
    if unknown:
        sys.exit(f"unknown shots: {', '.join(sorted(unknown))}")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    def run(name):
        cmd = [sys.executable, __file__, "--child", name, "--out", args.out]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode == SKIPPED:
            print(r.stderr.strip().splitlines()[-1])
            return 0
        print(f"{name}: {'ok' if r.returncode == 0 else 'failed'}")
        if r.returncode:
            print(r.stderr[-2000:], file=sys.stderr)
        return r.returncode

    with ThreadPoolExecutor(args.jobs) as pool:
        codes = list(pool.map(run, names))
    sys.exit(1 if any(codes) else 0)


if __name__ == "__main__":
    main()
