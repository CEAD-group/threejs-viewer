"""Capture every local-join trial and build an interactive, standalone demo.

Run: .venv/bin/python bench/tube_lab/local_joins.py
"""

import argparse
import base64
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]
from bead_texture_maker import texture_set  # noqa: E402
from threejs_viewer import ViewerClient  # noqa: E402

OUT = Path(__file__).parent / "out/local_joins"


def scenes():
    # Exact generators used by the PR #279/#280 visual benchmark.
    from zr_metric import SCENES

    for name, build in SCENES:
        old = build()
        yield dict(
            name=f"prior_{name}",
            title=f"Earlier benchmark · {name}",
            spine=old["sp"],
            w=old["w"],
            h=old["ht"],
            lo=old["lo"],
            hi=min(old["hi"], len(old["sp"]) - 1),
            up=[0, 0, 1],
        )
    # PR #281, examples/39_planar_swept_union.py at 05e85ef.
    loop = np.array(
        [
            [0, 0, 0],
            [20, 0, 0],
            [21, 1, 0],
            [20, 2, 0],
            [22, 3, 0],
            [40, 3, 0],
            [39, 3, 0],
            [40, 3, 0],
            [50, 3, 0],
            [50, 30, 0],
            [20, 30, 0],
            [20, 10, 0],
            [50, 10, 0],
        ],
        dtype=np.float32,
    )
    yield dict(
        name="prior_union_demo",
        title="PR #281 · jog, retrace and loop",
        spine=loop,
        w=10,
        h=2,
        lo=0,
        hi=len(loop) - 1,
        up=[0, 0, 1],
    )
    for issue, w, h, lo, hi in [(277, 11.83, 2, 9, 21), (278, 9.46, 2.5, 10, 20)]:
        spine = np.array(
            json.loads((ROOT / f"tests/data/tube_issue_{issue}.json").read_text())
        )
        spine -= spine[len(spine) // 2]
        yield dict(
            name=f"issue_{issue}",
            title=f"Reported #{issue} · planar",
            spine=spine,
            w=w,
            h=h,
            lo=lo,
            hi=hi,
            up=[0, 0, 1],
        )
    corner = np.array(
        [
            [-30, 0, 0],
            [-20, 0, 0],
            [-10, 0, 0],
            [0, 0, 0],
            [0, 10, 0],
            [0, 20, 0],
            [0, 30, 0],
        ],
        dtype=float,
    )
    yield dict(
        name="planar_corner",
        title="90° corner · planar",
        spine=corner,
        w=10,
        h=2,
        lo=2,
        hi=4,
        up=[0, 0, 1],
    )
    axis = np.array([1, 2, 3], dtype=float)
    axis /= np.linalg.norm(axis)
    u = np.cross(axis, [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    tilted = corner[:, 0, None] * u + corner[:, 1, None] * v
    yield dict(
        name="tilted_corner",
        title="90° corner · tilted plane",
        spine=tilted,
        w=10,
        h=2,
        lo=2,
        hi=4,
        up=axis.tolist(),
    )
    elbow = corner.copy()
    elbow[4:, 2] = elbow[4:, 1]
    elbow[4:, 1] = 0
    yield dict(
        name="vertical_elbow",
        title="Horizontal → vertical elbow",
        spine=elbow,
        w=10,
        h=2,
        lo=2,
        hi=4,
        up=[0, 0, 1],
    )
    spatial = np.array(
        [
            [-30, 0, 0],
            [-20, 0, 0],
            [-10, 0, 0],
            [0, 0, 0],
            [2, 2, 2],
            [0, 4, 4],
            [3, 7, 6],
            [13, 7, 6],
            [23, 7, 6],
            [33, 7, 6],
        ],
        dtype=float,
    )
    yield dict(
        name="spatial_jog",
        title="Spatial jog · changing elevation and direction",
        spine=spatial,
        w=10,
        h=2,
        lo=2,
        hi=7,
        up=[0, 0, 1],
    )
    slope = np.array(
        [
            [-30, 0, 0],
            [-20, 0, 0],
            [-10, 0, 0],
            [0, 0, 0],
            [4, 4, 5],
            [6, 8, 3],
            [2, 12, 7],
            [8, 16, 10],
            [18, 16, 10],
            [28, 16, 10],
            [38, 16, 10],
        ],
        dtype=float,
    )
    yield dict(
        name="spatial_zigzag",
        title="3D zigzag · alternating slopes",
        spine=slope,
        w=8,
        h=2.5,
        lo=2,
        hi=8,
        up=[0, 0, 1],
    )

    theta = np.linspace(0, 200 * np.pi, 10001)
    radius = 55 + 5 * np.sin(theta / 50)
    vase = np.column_stack(
        [radius * np.cos(theta), radius * np.sin(theta), theta * 0.12]
    )
    yield dict(
        name="spiral_vase",
        title="Spiral vase · 10,001 stations, continuous rise",
        spine=vase,
        w=3,
        h=0.8,
        lo=0,
        hi=len(vase) - 1,
        up=[0, 0, 1],
    )
    spatial_corner = corner.copy()
    spatial_corner[:, 2] = [0, 0.1, 0.2, 0.3, 1, 3, 5]
    yield dict(
        name="rising_corner",
        title="Nearly planar corner · drifting elevation",
        spine=spatial_corner,
        w=10,
        h=2,
        lo=2,
        hi=4,
        up=[0, 0, 1],
    )

    for name, title, widths in [
        (
            "width_gentle",
            "Gentle taper through a corner · 8 → 9.2 mm",
            np.linspace(8, 9.2, len(corner)),
        ),
        (
            "width_abrupt",
            "Strong width changes at a corner · 4–14 mm",
            [4, 5, 7, 12, 14, 10, 6],
        ),
    ]:
        yield dict(
            name=name,
            title=title,
            spine=corner.copy(),
            w=np.asarray(widths),
            h=2,
            lo=0,
            hi=len(corner) - 1,
            up=[0, 0, 1],
        )

    stepped = np.array(
        [
            [-30, 0, 0],
            [-20, 0, 0],
            [-10, 0, 0],
            [0, 0, 0],
            [0, 0, 5],
            [0, 0, 10],
            [-10, 0, 10],
            [-20, 0, 10],
            [-30, 0, 10],
        ],
        dtype=float,
    )
    yield dict(
        name="uturn_step",
        title="Out-of-plane U-turn · forward, up 10 mm, back",
        spine=stepped,
        w=8,
        h=2,
        lo=0,
        hi=len(stepped) - 1,
        up=[0, 0, 1],
    )
    angles = np.linspace(0, np.pi, 7)
    arc = np.column_stack(
        [6 * np.sin(angles), np.zeros(len(angles)), 6 * (1 - np.cos(angles))]
    )
    rounded = np.vstack(
        [
            [[-30, 0, 0], [-20, 0, 0], [-10, 0, 0]],
            arc,
            [[-10, 0, 12], [-20, 0, 12], [-30, 0, 12]],
        ]
    )
    yield dict(
        name="uturn_round",
        title="Out-of-plane U-turn · rounded rise of 12 mm",
        spine=rounded,
        w=8,
        h=2,
        lo=0,
        hi=len(rounded) - 1,
        up=[0, 0, 1],
    )
    yield dict(
        name="uturn_taper",
        title="Out-of-plane U-turn · widening from 6 to 12 mm",
        spine=rounded.copy(),
        w=np.linspace(6, 12, len(rounded)),
        h=2,
        lo=0,
        hi=len(rounded) - 1,
        up=[0, 0, 1],
    )


def section_values(scene, key):
    """Allow either a constant section dimension or one value per station."""
    return np.broadcast_to(
        np.asarray(scene[key], dtype=float), (len(scene["spine"]),)
    ).copy()


def demo(maps):
    client = ViewerClient(port=0, open_browser=False)
    client._http_port = 0
    messages = []
    client._send = messages.append
    cases = list(scenes())
    for s in cases:
        for surface in ["rings", "local_join"]:
            client.add_parametric_tube(
                s["name"] + "_" + surface,
                s["spine"],
                section_values(s, "w"),
                section_values(s, "h"),
                surface=surface,
                up_vector=s["up"],
                lod=False,
                texture=maps["color"],
                normal_map=maps["normal"],
                roughness_map=maps["roughness"],
                texture_length=25,
                roughness=1,
                metalness=0,
                visible=False,
            )

    def embed(value):
        if isinstance(value, dict):
            return {k: embed(v) for k, v in value.items()}
        if isinstance(value, list):
            return [embed(v) for v in value]
        if isinstance(value, str) and value.startswith("http"):
            blob = client._blob_store.get(urlparse(value).path)
            if blob is not None:
                return (
                    "data:application/octet-stream;base64,"
                    + base64.b64encode(blob).decode()
                )
        return value

    html = client.viewer_path.read_text().replace(
        "window.threejsViewer = new ThreeJSViewer(container, {",
        "window.threejsViewer = new ThreeJSViewer(container, {\n autoConnect:false,",
    )
    overview = html.replace(
        "    </script>\n</body>",
        (
            Path(__file__)
            .with_name("local_joins_overview.js")
            .read_text()
            .replace(
                "/* CASES */",
                json.dumps([dict(name=s["name"], title=s["title"]) for s in cases]),
            )
            .replace("/* MESSAGES */", json.dumps(embed(messages)))
        )
        + "\n    </script>\n</body>",
    )
    (OUT / "overview.html").write_text(overview)
    panel = (
        '<style>#lab{position:absolute;z-index:100;top:55px;left:15px;background:#18232def;color:#eef;padding:15px;border-radius:8px;font:14px system-ui;width:340px}#lab label{display:block;margin:8px 0}#lab small{display:block;margin-top:10px}#lab input[type=range]{width:100%}</style><aside id="lab"><b>Bounded local joins</b><label>Case <select id="case">'
        + "".join(f'<option value="{s["name"]}">{s["title"]}</option>' for s in cases)
        + '</select></label><label><input id="original" type="checkbox"> Original strip</label><label><input id="maps" type="checkbox" checked> Color, normal and roughness maps</label><label><input id="wire" type="checkbox"> Show triangles</label><label><input id="play" type="checkbox" checked> Animate</label><input id="progress" type="range" min="0" max="1" step=".001" value="0"><small id="metrics"></small><small>Tight jogs and reversals can fall back. Drag to orbit; scroll to zoom.</small></aside>'
    )
    script = """
const v=window.threejsViewer, messages=MESSAGES;
for(const m of messages){await v.handleMessage(m);if(m.id&&v._inflightLoads.has(m.id))await v._inflightLoads.get(m.id).promise;}
const $=id=>document.getElementById(id);
for(const o of v._objects.values()){o.userData.labMaps=o.material;o.userData.labPlain=new THREE.MeshStandardMaterial({color:0x7ab8cc,roughness:.8,side:THREE.DoubleSide});}
let selected=null,t=0,last=performance.now();
function choose(){for(const o of v._objects.values())o.visible=false;selected=v._objects.get($('case').value+($('original').checked?'_rings':'_local_join'));selected.visible=true;
 v._sceneBoundsDirty=true;v.frameAll();t=0;
 const st=selected.userData.localJoinStats;$('metrics').textContent=st?`${st.accepted} repaired joins · ${st.addedTriangles} added triangles · fallbacks ${JSON.stringify(st.fallback)}`:'Original strip (collapse off)';style();}
function style(){if(!selected)return;selected.material=$('maps').checked?selected.userData.labMaps:selected.userData.labPlain;if(selected.userData.localJoinCompleted)selected.userData.localJoinCompleted.material=selected.material;
 v._shading.wireframeMode=$('wire').checked?2:0;v._shading.applyWireframe();}
$('case').onchange=choose;$('original').onchange=choose;$('maps').onchange=style;$('wire').onchange=style;
$('progress').oninput=()=>{t=Number($('progress').value)*10;$('play').checked=false;};
function tick(now){if($('play').checked)t=(t+(now-last)/1000)%12;last=now;const f=Math.min(1,t/10);$('progress').value=f;selected.userData.parametricTube.setDrawRange(f);v._shadowDirty=true;requestAnimationFrame(tick);}
const requestedLabCase=new URLSearchParams(location.search).get('case');
$('case').value=[...$('case').options].some(o=>o.value===requestedLabCase)?requestedLabCase:'planar_corner';choose();window.demoReady=true;requestAnimationFrame(tick);
""".replace("MESSAGES", json.dumps(embed(messages)))
    html = html.replace(
        "    </script>\n</body>", script + "\n    </script>\n</body>"
    ).replace("</body>", panel + "\n</body>")
    (OUT / "demo.html").write_text(html)


def run(selected=None):
    OUT.mkdir(parents=True, exist_ok=True)
    maps = texture_set(color="#698d9e", smoothness=0.6, fibers=True)
    results = (
        json.loads((OUT / "metrics.json").read_text())
        if selected and (OUT / "metrics.json").exists()
        else {}
    )
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        client = ViewerClient(port=0, open_browser=False)
        client._start_servers(http_port=0)
        page = browser.new_page(viewport={"width": 850, "height": 610})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{client.viewer_path.resolve().as_uri()}?ws_port={client.port}")
        page.wait_for_function("() => !!window.threejsViewer")
        assert client._connected_event.wait(10)
        for s in scenes():
            if selected and s["name"] not in selected:
                continue
            name = s["name"]
            spine = s["spine"]
            for mode in ["rings", "local_join"]:
                client.add_parametric_tube(
                    name + mode,
                    spine,
                    section_values(s, "w"),
                    section_values(s, "h"),
                    up_vector=s["up"],
                    lod=False,
                    surface=mode,
                    texture=maps["color"],
                    normal_map=maps["normal"],
                    roughness_map=maps["roughness"],
                    texture_length=25,
                    roughness=1,
                    metalness=0,
                    visible=False,
                )
                page.wait_for_function(
                    "id=>window.threejsViewer._objects.has(id)", arg=name + mode
                )
            stats = page.evaluate(
                "id=>window.threejsViewer._objects.get(id).userData.localJoinStats",
                name + "local_join",
            )
            results[name] = stats
            print(name, json.dumps(stats), flush=True)
            roi = spine[s["lo"] : s["hi"] + 1]
            centre = roi.mean(axis=0)
            extent = float(np.ptp(roi, axis=0).max() + float(np.max(s["w"])) * 3)
            page.evaluate(
                """([t,d])=>{const v=window.threejsViewer;v._camera.position.set(t[0]+d*.75,t[1]-d,t[2]+d*.9);
             v._camera.up.set(0,0,1);v._controls.target.set(...t);v._camera.lookAt(v._controls.target);v._controls.update();
             v._applyAmbientIntensity(.35);v._scene.environmentIntensity=.6;}""",
                [centre.tolist(), extent],
            )
            cards = []
            for style in ["solid", "wire", "maps"]:
                for mode in ["rings", "local_join"]:
                    page.evaluate(
                        """async ({id,other,style})=>{const THREE=await import('three'),v=window.threejsViewer,o=v._objects.get(id),ud=o.userData;
                     v._objects.get(other).visible=false;o.visible=true;if(!ud.labMaps)ud.labMaps=o.material;
                     if(!ud.labPlain)ud.labPlain=new THREE.MeshStandardMaterial({color:0x7ab8cc,roughness:.8,side:THREE.DoubleSide});
                     o.material=style==='maps'?ud.labMaps:ud.labPlain;if(ud.localJoinCompleted)ud.localJoinCompleted.material=o.material;
                     v._shading.wireframeMode=style==='wire'?2:0;v._shading.applyWireframe();v._shadowDirty=true;}""",
                        dict(
                            id=name + mode,
                            other=name
                            + ("rings" if mode == "local_join" else "local_join"),
                            style=style,
                        ),
                    )
                    page.wait_for_timeout(150)
                    file = f"{name}_{style}_{mode}.png"
                    page.screenshot(path=str(OUT / file))
                    cards.append(
                        f'<article><b>{"Original strip" if mode == "rings" else "Local joins"} · {style}</b><img src="{file}"></article>'
                    )
            body = f"<h1>{s['title']}</h1><p>{stats['accepted']} repaired joins · {stats['addedTriangles']:+d} triangles · {stats['buildMs']:.1f} ms join build · fallback {stats['fallback']}</p><main>{''.join(cards)}</main>"
            html = (
                '<meta charset="utf-8"><style>body{margin:0;background:#202328;color:#eef;font:15px system-ui}h1,p{margin:16px}main{display:grid;grid-template-columns:1fr 1fr}b{display:block;margin:8px}img{display:block;width:100%}</style>'
                + body
            )
            (OUT / f"{name}.html").write_text(html)
            page.evaluate(
                "id=>window.threejsViewer._objects.get(id).visible=false",
                name + "local_join",
            )
            report = browser.new_page(viewport={"width": 1360, "height": 1600})
            report.goto((OUT / f"{name}.html").resolve().as_uri())
            report.screenshot(
                path=str(OUT / f"{name}_comparison.png"), full_page=True, timeout=60000
            )
            report.close()
            page.evaluate(
                "id=>window.threejsViewer._objects.get(id).visible=false",
                name + "local_join",
            )
        assert not errors, errors
        client.disconnect()
        browser.close()
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    (OUT / "index.html").write_text(
        '<meta charset="utf-8"><style>body{background:#202328;color:#eef;font:16px system-ui;margin:25px}a{color:#8dd}img{width:100%;max-width:1360px}</style><h1>Local join comparisons</h1><p><a href="overview.html">All examples: synchronized animated comparisons</a> · <a href="demo.html">Single-example demo</a></p><p>Left: original strip, collapse off. Right: bounded local joins. Solid, wire, and all three material maps. Rejected corners retain the original strip.</p>'
        + "".join(
            f'<h2>{s["title"]}</h2><a href="{s["name"]}.html"><img src="{s["name"]}_comparison.png"></a>'
            for s in scenes()
        )
    )
    demo(maps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Build both interactive views without recapturing screenshots",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[s["name"] for s in scenes()],
        help="Capture selected cases; retain existing gallery results",
    )
    args = parser.parse_args()
    if args.demo_only:
        OUT.mkdir(parents=True, exist_ok=True)
        demo(texture_set(color="#698d9e", smoothness=0.6, fibers=True))
    else:
        run(args.case)
