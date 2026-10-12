"""Scale bench for round tube joins on a million-point toolpath.

Builds a serpentine infill toolpath (hairpin every ``--run`` points, a wavy
perimeter every layer) with ``--points`` spine points, adds it once with
``joins="round"`` and once with ``joins="miter"`` under the default LOD, and
measures for each:

- the wall time from the add message to the mesh being registered (the
  main-thread initial LOD reduction and build),
- the LOD worker rebuild after a camera move: wall time from the move to
  the geometry swap, the time from the worker dispatch to its result (which
  includes the 500 ms throttle and the message transfer), the worker's own
  time and its geometry build, and the main-thread apply,
- joints, vertices and triangles of the current LOD build.

Run with ``uv run python bench/tube_lab/join_scale.py --headed`` (needs the
browser-test extras; headless Chromium renders in software, which stalls the
worker result behind slow frames). Numbers are printed as a table and written to
``bench/tube_lab/out/join_scale.json``. Dev tooling, not run in CI.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from threejs_viewer import ViewerClient  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def serpentine(points: int, run: int, pitch: float = 1.0, step: float = 0.5):
    """Layered serpentine fill with a sharp hairpin every ``run`` points."""
    n_rows = points // run
    rows_per_layer = 40
    x = np.empty(points, np.float64)
    y = np.empty(points, np.float64)
    z = np.empty(points, np.float64)
    t = np.arange(run) * step
    for r in range(n_rows):
        s = slice(r * run, (r + 1) * run)
        layer, row = divmod(r, rows_per_layer)
        xs = t if row % 2 == 0 else t[::-1]
        # A gentle wave so the straight runs still carry curvature for LOD.
        x[s] = xs + 0.15 * np.sin(t * 0.7)
        y[s] = row * pitch + 0.05 * np.cos(t * 1.3)
        z[s] = layer * 0.4
    rest = points - n_rows * run
    if rest:
        x[-rest:] = x[n_rows * run - 1]
        y[-rest:] = y[n_rows * run - 1] + np.arange(1, rest + 1) * 0.01
        z[-rest:] = z[n_rows * run - 1]
    return np.column_stack([x, y, z]).astype(np.float32)


STATS = """() => {
    const v = window.threejsViewer;
    const o = v._objects.get('t');
    if (!o) return null;
    const ud = o.userData;
    const g = o.geometry;
    return {
        joints: ud.tubeJointVerts ? ud.tubeJointVerts.length : 0,
        verts: g.getAttribute('position').count,
        tris: g.index.count / 3,
        nSpine: ud.tubeNumSpinePoints,
        lod: !!ud.tubeLOD,
        timing: ud.tubeLOD ? ud.tubeLOD.lastTiming : null,
    };
}"""


def measure(page, client, spine, joins, camera):
    n = len(spine)
    print(f"[{joins}] adding {n} points", flush=True)
    t0 = time.perf_counter()
    client.add_parametric_tube(
        "t",
        spine,
        np.full(n, 1.2, np.float32),
        np.full(n, 0.4, np.float32),
        joins=joins,
        color=0x7AB8CC,
    )
    page.wait_for_function(
        "() => window.threejsViewer && window.threejsViewer._objects.has('t')",
        timeout=120000,
    )
    add_s = time.perf_counter() - t0
    print(f"[{joins}] added in {add_s:.2f} s", flush=True)
    page.wait_for_timeout(300)
    first = page.evaluate(STATS)
    # Camera move: time until the LOD worker swaps in a new build.
    client.set_camera(position=camera["near"], target=camera["target"])
    t1 = time.perf_counter()
    deadline = t1 + (120 if first["lod"] else 0)
    rebuilt = None
    while time.perf_counter() < deadline:
        page.wait_for_timeout(20)
        cur = page.evaluate(STATS)
        if cur["nSpine"] != first["nSpine"]:
            rebuilt = time.perf_counter() - t1
            break
    print(f"[{joins}] rebuilt in {rebuilt}", flush=True)
    page.wait_for_timeout(1500)
    near = page.evaluate(STATS)
    client.delete("t")
    page.wait_for_timeout(300)
    return {
        "joins": joins,
        "add_s": round(add_s, 2),
        "initial": first,
        "rebuild_s": None if rebuilt is None else round(rebuilt, 2),
        "near": near,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=1_000_000)
    ap.add_argument("--run", type=int, default=250, help="points per straight run")
    ap.add_argument(
        "--headed",
        action="store_true",
        help="open a browser window: headless Chromium renders in software, which"
        " stalls the worker result behind slow frames",
    )
    a = ap.parse_args()
    spine = serpentine(a.points, a.run)
    centre = spine.mean(axis=0).tolist()
    span = float(np.ptp(spine[:, 0]))
    camera = {
        "target": centre,
        "far": [centre[0], centre[1] - span * 2, centre[2] + span * 2],
        "near": [
            centre[0] + span * 0.2,
            centre[1] - span * 0.05,
            centre[2] + span * 0.08,
        ],
    }
    results = []
    # A fresh browser process per mode: a second million-point tube in the
    # same page paid tens of seconds for the first one's teardown.
    with sync_playwright() as p:
        for joins in ("round", "miter"):
            browser = p.chromium.launch(headless=not a.headed)
            client = ViewerClient(port=free_port(), open_browser=False)
            client._start_servers(http_port=0)
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.goto(
                f"file://{ROOT / 'src/threejs_viewer/viewer.html'}?ws_port={client.port}"
            )
            assert client._connected_event.wait(timeout=10), "viewer did not connect"
            client.set_camera(position=camera["far"], target=camera["target"])
            page.wait_for_timeout(300)
            results.append(measure(page, client, spine, joins, camera))
            page.close()
            client.disconnect()
            browser.close()
    OUT.mkdir(exist_ok=True)
    (OUT / "join_scale.json").write_text(
        json.dumps({"points": a.points, "run": a.run, "results": results}, indent=1)
    )
    print(f"{a.points} spine points, hairpin every {a.run} points")
    print(
        f"{'joins':>6} {'add s':>6} {'rebuild s':>9} {'wall s':>6} {'worker s':>8} {'build s':>7} {'apply s':>7} "
        f"{'joints':>7} {'verts':>9} {'tris':>9}"
    )
    for r in results:
        near = r["near"]
        tm = near.get("timing") or {}
        worker = f"{tm['totalMs'] / 1000:.2f}" if tm else "-"
        build = f"{tm['buildMs'] / 1000:.2f}" if tm else "-"
        wall = f"{tm['wallMs'] / 1000:.2f}" if tm.get("wallMs") is not None else "-"
        apply = f"{tm['applyMs'] / 1000:.2f}" if tm.get("applyMs") is not None else "-"
        print(
            f"{r['joins']:>6} {r['add_s']:>6} {str(r['rebuild_s']):>9} {wall:>6} {worker:>8} {build:>7} {apply:>7} {near['joints']:>7} "
            f"{near['verts']:>9} {int(near['tris']):>9}"
        )


if __name__ == "__main__":
    main()
