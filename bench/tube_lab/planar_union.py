"""Capture rounded-sweep comparisons and measure triangles / prefix latency.

Run: uv run python bench/tube_lab/planar_union.py
Writes out/planar_union_*.png and out/planar_union_metrics.json.
The analytical correctness oracle lives in tests/test_planar_union.py; the
old ring mesh is a visual comparison, not the expected swept silhouette.
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from threejs_viewer import ViewerClient  # noqa: E402

OUT = Path(__file__).parent / "out"


def camera(page, centre, distance, az=-60, el=35):
    page.evaluate(
        """([t, d, az, el]) => {
            const v = window.threejsViewer, a = az*Math.PI/180, e = el*Math.PI/180;
            v._camera.position.set(t[0]+d*Math.cos(e)*Math.cos(a),
                t[1]+d*Math.cos(e)*Math.sin(a), t[2]+d*Math.sin(e));
            v._controls.target.set(...t); v._camera.up.set(0,0,1);
            v._camera.lookAt(v._controls.target); v._controls.update();
        }""",
        [list(map(float, centre)), distance, az, el],
    )


def run():
    OUT.mkdir(exist_ok=True)
    metrics = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        client = ViewerClient(port=0, open_browser=False)
        client._start_servers(http_port=0)
        page = browser.new_page(viewport={"width": 1000, "height": 660})
        page.goto(f"{client.viewer_path.resolve().as_uri()}?ws_port={client.port}")
        page.wait_for_function("() => !!window.threejsViewer")
        assert client._connected_event.wait(10)
        for issue, width, height in [(277, 11.83, 2), (278, 9.46, 2.5)]:
            spine = np.asarray(
                json.loads((ROOT / f"tests/data/tube_issue_{issue}.json").read_text())
            )
            for mode in ("rings", "planar_union"):
                id = f"{issue}_{mode}"
                client.add_parametric_tube(
                    id,
                    spine,
                    np.full(len(spine), width),
                    np.full(len(spine), height),
                    surface=mode,
                    strand_collapse=issue == 277,
                    lod=False,
                )
                page.wait_for_function(
                    "id => window.threejsViewer._objects.has(id)", arg=id
                )
                if mode == "rings" and issue == 277:
                    page.wait_for_function(
                        "id => !!window.threejsViewer._objects.get(id).userData.collapsedPositions",
                        arg=id,
                    )
                camera(page, spine[10:20].mean(axis=0), 38 if issue == 277 else 45)
                metrics[id] = page.evaluate(
                    """id => {
                    const o = window.threejsViewer._objects.get(id);
                    return {triangles: o.geometry.index.count / 3,
                        workerMs: o.userData.unionWorkerMs, meshingMs: o.userData.unionMeshingMs};
                }""",
                    id,
                )
                for wire in (False, True):
                    page.evaluate(
                        "wire => {const s = window.threejsViewer._shading; s.wireframeMode = wire ? 2 : 0; s.applyWireframe();}",
                        wire,
                    )
                    page.wait_for_timeout(200)
                    page.screenshot(
                        path=str(
                            OUT / f"planar_union_{id}_{'wire' if wire else 'solid'}.png"
                        )
                    )
                if mode == "planar_union":
                    timings = page.evaluate(
                        """async id => {
                        const v = window.threejsViewer, o = v._objects.get(id), results = [];
                        for (const value of [0.45, 0.48, 0.52, 0.57, 0.60, 0.8, 1, 0.2]) {
                            const start = performance.now(); v._setDrawRange(id, value);
                            while (o.userData.unionRenderedValue !== value) await new Promise(r => setTimeout(r, 1));
                            results.push({value, ms: performance.now()-start,
                                triangles: o.geometry.index.count/3, workerMs: o.userData.unionWorkerMs,
                                meshingMs: o.userData.unionMeshingMs});
                        }
                        return results;
                    }""",
                        id,
                    )
                    metrics[id]["prefixes"] = timings
                client.set_visible(id, False)
                page.wait_for_function(
                    "id => !window.threejsViewer._objects.get(id).visible", arg=id
                )
        # Triangle count must follow outline complexity, not station density.
        for count in (2, 1001, 10001):
            spine = np.column_stack(
                [np.linspace(0, 100, count), np.zeros(count), np.zeros(count)]
            )
            id = f"straight_{count}"
            client.add_parametric_tube(
                id, spine, np.full(count, 10), np.full(count, 2), surface="planar_union"
            )
            page.wait_for_function(
                "id => window.threejsViewer._objects.has(id)", arg=id
            )
            metrics[id] = page.evaluate(
                """id => {
                const o = window.threejsViewer._objects.get(id);
                o.visible = false;
                return {triangles: o.geometry.index.count/3, workerMs: o.userData.unionWorkerMs,
                    meshingMs: o.userData.unionMeshingMs, segments: o.userData.unionSegmentCount};
            }""",
                id,
            )
        for issue in (277, 278):
            cards = []
            for style in ("solid", "wire"):
                for mode, label in [
                    ("rings", "Current ring mesh"),
                    ("planar_union", "Rounded swept union"),
                ]:
                    raw = (
                        OUT / f"planar_union_{issue}_{mode}_{style}.png"
                    ).read_bytes()
                    cards.append(
                        f'<div><h3>{label} · {style}</h3><img src="data:image/png;base64,{base64.b64encode(raw).decode()}"></div>'
                    )
            report = browser.new_page(viewport={"width": 1400, "height": 1010})
            report.set_content(
                "<style>body{margin:0;background:#222;color:white;font:16px system-ui}main{display:grid;grid-template-columns:1fr 1fr}h3{margin:12px}img{width:100%;display:block}</style><main>"
                + "".join(cards)
                + "</main>"
            )
            report.screenshot(
                path=str(OUT / f"planar_union_{issue}_comparison.png"), full_page=True
            )
            report.close()
        page.close()
        client.disconnect()
        browser.close()
    (OUT / "planar_union_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run()
