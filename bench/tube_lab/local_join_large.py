"""Measure mapped million-point paths: buffers, CPU playback, and GPU timer queries.

Run: .venv/bin/python bench/tube_lab/local_join_large.py --points 1000001
One tube/context at a time. LOD, shadows and viewport are controlled explicitly.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]
from bead_texture_maker import texture_set  # noqa: E402
from threejs_viewer import ViewerClient  # noqa: E402

PROBE = """async ({samples,seeks}) => {
    const v=window.threejsViewer,o=v._objects.get('large'),u=o.userData;
    const r=v._renderer,gl=r.getContext(),ext=gl.getExtension('EXT_disjoint_timer_query_webgl2');
    const debug=gl.getExtension('WEBGL_debug_renderer_info');
    const gpu=[],cpu=[],submit=[],queries=[];
    const next=()=>new Promise(resolve=>requestAnimationFrame(resolve));
    let triangles=0,calls=0,disjointSeen=false;
    for(let i=-8;i<samples;i++) {
        await next();
        if(ext&&gl.getParameter(ext.GPU_DISJOINT_EXT))disjointSeen=true;
        const f=i<0?.6:.2+.75*i/Math.max(1,samples-1);
        const t=performance.now();u.parametricTube.setDrawRange(f);const update=performance.now()-t;
        const q=ext?gl.createQuery():null;
        if(q)gl.beginQuery(ext.TIME_ELAPSED_EXT,q);
        r.info.reset();const started=performance.now();r.render(v._scene,v._camera);
        const elapsed=performance.now()-started;
        if(q){gl.endQuery(ext.TIME_ELAPSED_EXT);queries.push({q,record:i>=0});}
        if(i>=0){cpu.push(update);submit.push(elapsed);triangles=r.info.render.triangles;calls=r.info.render.calls;}
    }
    // Queries run asynchronously. Poll without gl.finish or busy waiting.
    const deadline=performance.now()+15000;
    while(queries.length && performance.now()<deadline) {
        const disjoint=ext&&gl.getParameter(ext.GPU_DISJOINT_EXT);
        if(disjoint)disjointSeen=true;
        for(let i=queries.length-1;i>=0;i--) {
            const {q,record}=queries[i];
            if(gl.getQueryParameter(q,gl.QUERY_RESULT_AVAILABLE)) {
                if(record&&!disjoint)gpu.push(gl.getQueryParameter(q,gl.QUERY_RESULT)/1e6);
                gl.deleteQuery(q);queries.splice(i,1);
            } else if(disjoint) {gl.deleteQuery(q);queries.splice(i,1);}
        }
        if(queries.length)await next();
    }
    for(const {q} of queries)gl.deleteQuery(q);
    const seekResults=[];
    if(seeks)for(const fraction of [1,0,.5,.99,.001,1]) {
        await next();
        const started=performance.now();u.parametricTube.setDrawRange(fraction);
        const rangeCpuMs=performance.now()-started;
        const q=ext?gl.createQuery():null;
        if(q)gl.beginQuery(ext.TIME_ELAPSED_EXT,q);
        const renderStarted=performance.now();r.render(v._scene,v._camera);
        const renderCpuMs=performance.now()-renderStarted;
        let gpuMs=null;
        if(q) {
            gl.endQuery(ext.TIME_ELAPSED_EXT);
            const deadline=performance.now()+15000;
            while(!gl.getQueryParameter(q,gl.QUERY_RESULT_AVAILABLE)&&performance.now()<deadline)await next();
            if(gl.getParameter(ext.GPU_DISJOINT_EXT))disjointSeen=true;
            if(gl.getQueryParameter(q,gl.QUERY_RESULT_AVAILABLE))gpuMs=gl.getQueryParameter(q,gl.QUERY_RESULT)/1e6;
            gl.deleteQuery(q);
        }
        seekResults.push({fraction,rangeCpuMs,renderCpuMs,gpuMs});
    }
    u.parametricTube.setDrawRange(1);
    const buffers=new Set();let geometryBytes=0;
    for(const g of [o.geometry,u.localJoinCompleted?.geometry].filter(Boolean)) {
        for(const a of [g.index,...Object.values(g.attributes)]) {
            if(!buffers.has(a.array.buffer)){buffers.add(a.array.buffer);geometryBytes+=a.array.byteLength;}
        }
    }
    const summarize=a=>{if(!a.length)return null;a.sort((a,b)=>a-b);return {median:a[Math.floor(a.length/2)],p95:a[Math.min(a.length-1,Math.floor(a.length*.95))],samples:a.length};};
    return {renderer:debug?gl.getParameter(debug.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER),
        reducedStations:u.tubeNumSpinePoints,geometryBytes,stats:u.localJoinStats||null,
        gpuMs:summarize(gpu),rangeUpdateCpuMs:summarize(cpu),renderSubmitCpuMs:summarize(submit),
        submittedTrianglesAt95Percent:triangles,drawCallsAt95Percent:calls,
        seekResults,disjointSeen,queryTimeouts:queries.length,webglError:gl.getError(),contextLost:gl.isContextLost()};
}"""


def path(n, kind):
    t = np.arange(n) * (2 * np.pi / 2500)
    radius = np.full(n, 1000.0)
    if kind == "dense":
        # Repeated radial excursions: 20,000 possible bend sites per million
        # stations, embedded throughout the print, not only in a short lead-in.
        phase = np.arange(n) % 100
        radius[(phase >= 40) & (phase <= 43)] += 6
    spine = np.column_stack([radius * np.cos(t), radius * np.sin(t), t * 0.12])
    if kind == "cluster":
        # Repeat the real #277 short-jog reproduction around the helix, retaining
        # its local sampling. This exercises capsule-envelope clusters, not
        # only the isolated arc joins in the radial-excursion stress case.
        motif = np.array(
            json.loads((ROOT / "tests/data/tube_issue_277.json").read_text())
        )
        motif = (motif - motif[len(motif) // 2]) * (3 / 11.83)
        block = np.arange(n) // len(motif) * len(motif)
        phase = np.arange(n) % len(motif)
        angle = t[block]
        local = motif[phase]
        spine[:, 0] = (
            1000 * np.cos(angle)
            - local[:, 0] * np.sin(angle)
            + local[:, 1] * np.cos(angle)
        )
        spine[:, 1] = (
            1000 * np.sin(angle)
            + local[:, 0] * np.cos(angle)
            + local[:, 1] * np.sin(angle)
        )
        spine[:, 2] = t * 0.12 + local[:, 2]
    if kind == "sparse":
        spine = np.vstack(
            [
                [
                    [1000, -40, 0],
                    [1000, -30, 0],
                    [990, -30, 0],
                    [990, -20, 0],
                    [1000, -20, 0],
                    [1000, -10, 0],
                ],
                spine,
            ]
        )
    return spine.astype(np.float32)


def run(points, samples, output, selected, repeats, angle, seeks):
    maps = texture_set(color="#698d9e", smoothness=0.6, fibers=True)
    rows = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=[f"--use-angle={angle}"] if angle else [])
        for kind, lod in [
            ("smooth", False),
            ("sparse", False),
            ("dense", False),
            ("dense", True),
            ("cluster", False),
        ]:
            key = f"{kind}_{'lod' if lod else 'full'}"
            if selected and key not in selected:
                continue
            spine = path(points, kind)
            for trial, surface in [
                (trial, surface)
                for trial in range(repeats)
                for surface in (
                    ["rings", "local_join"]
                    if trial % 2 == 0
                    else ["local_join", "rings"]
                )
            ]:
                context = browser.new_context(
                    viewport={"width": 1280, "height": 800}, device_scale_factor=1
                )
                page = context.new_page()
                errors = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                client = ViewerClient(port=0, open_browser=False)
                client._start_servers(http_port=0)
                page.goto(
                    f"{client.viewer_path.resolve().as_uri()}?ws_port={client.port}"
                )
                page.wait_for_function("() => !!window.threejsViewer")
                assert client._connected_event.wait(10)
                page.evaluate("""() => {
                    const v=window.threejsViewer;cancelAnimationFrame(v._animationFrameId);
                    v._renderer.setPixelRatio(1);v._renderer.shadowMap.enabled=false;
                    v._camera.position.set(3500,-3500,2500);v._camera.up.set(0,0,1);
                    v._controls.target.set(0,0,150);v._camera.lookAt(v._controls.target);
                    v._camera.near=1;v._camera.far=100000;v._camera.updateProjectionMatrix();
                }""")
                start = time.perf_counter()
                client.add_parametric_tube(
                    "large",
                    spine,
                    np.full(len(spine), 3),
                    np.full(len(spine), 2 * 3 / 11.83 if kind == "cluster" else 0.8),
                    surface=surface,
                    lod={} if lod else False,
                    texture=maps["color"],
                    normal_map=maps["normal"],
                    roughness_map=maps["roughness"],
                    pickable=False,
                )
                page.wait_for_function(
                    "() => window.threejsViewer._objects.has('large')", timeout=180000
                )
                ready_ms = (time.perf_counter() - start) * 1000
                result = page.evaluate(PROBE, {"samples": samples, "seeks": seeks})
                assert not errors, errors
                assert not result["contextLost"] and result["webglError"] == 0, result
                assert result["submittedTrianglesAt95Percent"] > 0, result
                assert not result["disjointSeen"] and not result["queryTimeouts"], (
                    result
                )
                if result["gpuMs"]:
                    assert result["gpuMs"]["samples"] == samples, result
                    assert all(x["gpuMs"] is not None for x in result["seekResults"]), (
                        result
                    )
                row = dict(
                    case=key,
                    trial=trial,
                    surface=surface,
                    stations=len(spine),
                    mapped=True,
                    viewport=[1280, 800],
                    shadows=False,
                    browserVersion=browser.version,
                    readyMs=ready_ms,
                    **result,
                )
                rows.append(row)
                output.write_text(json.dumps(rows, indent=2) + "\n")
                print(json.dumps(row), flush=True)
                client.disconnect()
                context.close()
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=1000001)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--seeks", action="store_true", help="Also time large forward/backward jumps"
    )
    parser.add_argument(
        "--angle",
        default="metal" if sys.platform == "darwin" else "",
        help="ANGLE backend; Metal avoids software fallback on macOS",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[
            "smooth_full",
            "sparse_full",
            "dense_full",
            "dense_lod",
            "cluster_full",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("local_join_large_results.json"),
    )
    args = parser.parse_args()
    run(
        args.points,
        args.samples,
        args.output,
        args.case,
        args.repeats,
        args.angle,
        args.seeks,
    )
