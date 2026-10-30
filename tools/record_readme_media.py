"""Re-record the README gallery images in docs/media from the examples.

Usage:
    uv run --with pillow python tools/record_readme_media.py [NAME ...] [--out DIR] [--jobs N]

Each shot runs one example in a subprocess, with the viewer in headless
Chromium (Metal ANGLE, so the GPU is used), and writes a PNG or an animated
GIF. Without names every shot in SHOTS is recorded. Animated shots of an
example with a loaded animation are seeked frame by frame (deterministic
loops); live examples are captured at a fixed interval. Needs the
browser-test group (`uv sync --group browser-test`).
"""

import argparse
import os
import runpy
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# name: example, optional args, and one of
#   at=<seconds after connect> (PNG),
#   seek=(start, end, frames) as fractions of the animation duration (GIF,
#   or a PNG for one frame),
#   live=(start, frames, interval) in seconds (GIF).
# js runs in the page before the first capture, click=<object id> clicks the
# centre of that object first. size is the page size; gif_width the GIF width.
FRAME_ALL = "window.tjsv.viewer.frameAll();"

SHOTS = {
    "robot_cell": dict(
        example="14_robot_cell.py", seek=(0.0, 1.0, 60), js=FRAME_ALL, frame_ms=140
    ),
    "section_caps": dict(
        example="16_section_caps.py",
        at=10,
        js="window.tjsv.viewer._clipPanelEl.style.display = 'none';" + FRAME_ALL,
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
    "bead_texture_corner": dict(example="37_bead_texture.py", seek=(1.0, 1.0, 1)),
    "bead_texture_hero": dict(
        example="37_bead_texture.py", seek=(1.0, 1.0, 1), js=FRAME_ALL
    ),
    "point_cloud": dict(example="25_point_cloud.py", seek=(1.0, 1.0, 1)),
    "point_cloud_sweep": dict(example="25_point_cloud.py", seek=(0.03, 1.0, 36)),
    "points_octree_lod": dict(example="27_points_octree_lod.py", seek=(0.0, 1.0, 32)),
    "live_points_stream": dict(example="30_live_points_stream.py", live=(2, 24, 0.25)),
    "depth_cues": dict(example="20_line_depth_cues.py", at=5),
    "polyline_picking": dict(example="22_polyline_picking.py", at=5, js=FRAME_ALL),
    "glb_models": dict(example="08_glb_models.py", at=6),
    "materials_sampler": dict(
        example="31_watch_gltf.py",
        args=["examples/tools/materials/materials.gltf"],
        at=25,
        js=FRAME_ALL,
    ),
    "billboards": dict(example="34_billboard_modes.py", at=5),
    "gizmos": dict(example="24_gizmo_dimensions.py", at=5),
    "axis_controls": dict(example="36_axis_control.py", at=5),
    "floor_grid": dict(example="29_floor_grid.py", at=5),
    "client_menu": dict(
        example="35_client_menu.py",
        at=5,
        js="window.tjsv.viewer.getMenu('demo').open();",
    ),
    # Needs the optional usd_brep package.
    "usd_connecting_rod": dict(
        example="38_usd_connecting_rod.py", at=8, size=(1600, 1000)
    ),
}


def _object_centre_js(obj_id):
    return f"""(() => {{
        const t = window.tjsv, v = t.viewer, o = v.getObject({obj_id!r});
        const c = new t.THREE.Box3().setFromObject(o).getCenter(new t.THREE.Vector3());
        c.project(v._camera);
        const r = v.getCanvas().getBoundingClientRect();
        return [r.left + (c.x + 1) / 2 * r.width, r.top + (1 - c.y) / 2 * r.height];
    }})()"""


def _record(client, shot, out):
    from playwright.sync_api import sync_playwright

    width, height = shot.get("size", (1280, 800))
    frames = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=["--use-angle=metal", "--ignore-gpu-blocklist"]
        )
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(client.viewer_url)
        client._connected_event.wait(timeout=120)
        t0 = time.monotonic()

        def wait_until(t):
            time.sleep(max(0.0, t0 + t - time.monotonic()))

        def settle():
            page.evaluate(
                "new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
            )

        if "seek" in shot:
            page.wait_for_function(
                "window.tjsv?.viewer.getAnimationState() && window.tjsv.viewer.getLoadState().pending === 0",
                timeout=120_000,
            )
            wait_until(shot.get("warmup", 3))
            if shot.get("js"):
                page.evaluate(shot["js"])
            duration = page.evaluate(
                "(() => { const v = window.tjsv.viewer; v.setAnimationPlaying(false);"
                " return v.getAnimationState().duration; })()"
            )
            a, b, n = shot["seek"]
            for i in range(n):
                t = duration * (a + (b - a) * i / max(n - 1, 1))
                page.evaluate(f"window.tjsv.viewer.seekAnimationTime({t})")
                settle()
                time.sleep(shot.get("frame_wait", 0.15))
                frames.append(page.screenshot())
        elif "live" in shot:
            start, n, interval = shot["live"]
            wait_until(start)
            if shot.get("js"):
                page.evaluate(shot["js"])
            for i in range(n):
                wait_until(start + i * interval)
                frames.append(page.screenshot())
        else:
            if shot.get("click"):
                wait_until(shot["at"] / 2)
                x, y = page.evaluate(_object_centre_js(shot["click"]))
                page.mouse.click(x, y)
            if shot.get("js"):
                # After the example's own framing and loads.
                wait_until(shot["at"] * 0.6)
                page.wait_for_function(
                    "window.tjsv.viewer.getLoadState().pending === 0"
                )
                page.evaluate(shot["js"])
            wait_until(shot["at"])
            page.screenshot(path=str(out.with_suffix(".png")))
        browser.close()
    if len(frames) == 1:
        out.with_suffix(".png").write_bytes(frames[0])
    elif frames:
        _write_gif(frames, out.with_suffix(".gif"), shot)


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


def _child(name, out_dir):
    import threejs_viewer.client as tc

    shot = SHOTS[name]
    out = Path(out_dir).resolve() / name
    orig_init, orig_start = tc.ViewerClient.__init__, tc.ViewerClient._start_servers
    started = []

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        self.port = 0
        self.open_browser = False

    def start(self):
        orig_start(self)
        if not started:
            started.append(self)

            def run():
                try:
                    _record(self, shot, out)
                    code = 0
                except Exception as e:  # noqa: BLE001
                    print(f"{name}: {e}", file=sys.stderr)
                    code = 1
                sys.stdout.flush()
                os._exit(code)

            threading.Thread(target=run, daemon=True).start()

    tc.ViewerClient.__init__ = init
    tc.ViewerClient._start_servers = start
    threading.Timer(300, lambda: os._exit(3)).start()
    example = ROOT / "examples" / shot["example"]
    sys.argv = [str(example), *shot.get("args", [])]
    sys.path.insert(0, str(example.parent))
    try:
        runpy.run_path(str(example), run_name="__main__")
    except SystemExit:
        pass
    # The example returned before the capture finished; the thread exits.
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
        print(f"{name}: {'ok' if r.returncode == 0 else 'failed'}")
        if r.returncode:
            print(r.stderr[-2000:], file=sys.stderr)
        return r.returncode

    with ThreadPoolExecutor(args.jobs) as pool:
        codes = list(pool.map(run, names))
    sys.exit(1 if any(codes) else 0)


if __name__ == "__main__":
    main()
