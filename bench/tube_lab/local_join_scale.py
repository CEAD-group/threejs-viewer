"""CPU/buffer comparison on growing spiral paths; not a GPU frame-rate benchmark."""

import json
import sys
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]
from bead_texture_maker import texture_set  # noqa: E402
from threejs_viewer import ViewerClient  # noqa: E402


def run():
    maps = texture_set(color="#698d9e", smoothness=0.6, fibers=True)
    rows = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        client = ViewerClient(port=0, open_browser=False)
        client._start_servers(http_port=0)
        page.goto(f"{client.viewer_path.resolve().as_uri()}?ws_port={client.port}")
        page.wait_for_function("() => !!window.threejsViewer")
        assert client._connected_event.wait(10)
        for n in [1001, 10001, 100001]:
            t = np.linspace(0, n / 250 * 2 * np.pi, n)
            spine = np.column_stack([100 * np.cos(t), 100 * np.sin(t), t * 0.12])
            for defect in [False, True] if n == 100001 else [False]:
                path = (
                    np.vstack(
                        [
                            [
                                [100, -40, 0],
                                [100, -30, 0],
                                [90, -30, 0],
                                [90, -20, 0],
                                [100, -20, 0],
                                [100, -10, 0],
                            ],
                            spine,
                        ]
                    )
                    if defect
                    else spine
                )
                for mapped in [False, True]:
                    for surface in ["rings", "local_join"]:
                        client.add_parametric_tube(
                            "scale",
                            path,
                            np.full(len(path), 3),
                            np.full(len(path), 0.8),
                            lod=False,
                            surface=surface,
                            texture=maps["color"] if mapped else None,
                            normal_map=maps["normal"] if mapped else None,
                            roughness_map=maps["roughness"] if mapped else None,
                        )
                        page.wait_for_function(
                            "() => window.threejsViewer._objects.has('scale')"
                        )
                        result = page.evaluate("""() => {
                          const o=window.threejsViewer._objects.get('scale'),g=o.geometry,c=o.userData.parametricTube;
                          for(let i=0;i<10;i++)c.setDrawRange(.9+i/100);
                          const started=performance.now();for(let i=0;i<500;i++)c.setDrawRange(.5+(i%100)/200);
                          const ms=(performance.now()-started)/500;
                          c.setDrawRange(1);
                          const st=o.userData.localJoinStats,f=o.userData.localJoinCompleted;
                          const buffers=new Set();let geometryBytes=0;
                          for(const geometry of [g,f?.geometry].filter(Boolean))
                            for(const a of [geometry.index,...Object.values(geometry.attributes)])
                              if(!buffers.has(a.array.buffer)){buffers.add(a.array.buffer);geometryBytes+=a.array.byteLength;}
                      return {triangles:st?st.originalTriangles+st.addedTriangles:g.index.count/3,vertices:g.attributes.position.count+(f?.geometry.attributes.position.count??0),
                        repairs:st?.accepted??0,
                            geometryBytes,
                            rangeUpdateCpuMs:ms,joinBuildMs:o.userData.localJoinStats?.buildMs??null,
                            addedTriangles:o.userData.localJoinStats?.addedTriangles??0};
                        }""")
                        row = dict(
                            stations=len(path),
                            defect=defect,
                            mapped=mapped,
                            surface=surface,
                            **result,
                        )
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                        client.delete("scale")
                        page.wait_for_function(
                            "() => !window.threejsViewer._objects.has('scale')"
                        )
        client.disconnect()
        browser.close()
    (Path(__file__).parent / "local_join_scale_results.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    run()
