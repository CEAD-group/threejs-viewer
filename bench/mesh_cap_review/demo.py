"""Serve or capture the PR #331 review cases. See README.md beside this file."""

import argparse
import base64
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
# Reuse the exact regression fixtures and frame barriers. This is a development
# bench, not a shipped example; the normal dev/test environment is required.
sys.path.insert(0, str(ROOT / "tests"))
from conftest import frames  # noqa: E402
from test_clip_caps import _Glb, _box_arrays, _png, _three_box_glb  # noqa: E402

CASES = {
    "overlap": (
        "Intersecting solids",
        "Union area 6.5; no overlapping cap triangles",
        6.5,
    ),
    "collinear": (
        "Collinear boundaries",
        "Union area 6; shared edges remain watertight",
        6,
    ),
    "hollow": (
        "A solid crossing a hollow shell",
        "Area 13.75; uncovered hole and color gradient survive",
        13.75,
    ),
    "replacement": (
        "Replacement position buffer",
        "Width doubles without a version bump; cap area becomes 8",
        8,
    ),
    "texture": (
        "Texture × vertex color",
        "COLOR_0 = 0.5; texture is multiplied once (also fixed by #329)",
        4,
    ),
    "groups": (
        "Changed material assignment",
        "Same group count; cap changes from brown to gold",
        4,
    ),
    "uv1": (
        "Selected texture coordinates",
        "TEXCOORD_1 samples the light half; compatibility check",
        4,
    ),
    "layouts": (
        "Three GLB buffer layouts",
        "Mirrored node · interleaved floats · normalized integers",
        12,
    ),
}


def make_glb(pos, idx, colors=None, texture=False, uv1=False):
    """Build one closed mesh; textures and colors are inside the GLB itself."""
    g = _Glb()
    attrs = {
        "POSITION": g.accessor(
            g.view(pos.astype("<f4").tobytes()),
            5126,
            len(pos),
            "VEC3",
            lo=pos.min(axis=0).tolist(),
            hi=pos.max(axis=0).tolist(),
        )
    }
    if colors is not None:
        attrs["COLOR_0"] = g.accessor(
            g.view(colors.astype("<f4").tobytes()), 5126, len(pos), "VEC3"
        )
    material = {
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.3, 0.65, 0.8, 1],
            "roughnessFactor": 0.8,
        }
    }
    if colors is not None or texture:
        material["pbrMetallicRoughness"]["baseColorFactor"] = [1, 1, 1, 1]
    if texture:
        for channel in range(2 if uv1 else 1):
            uv = np.tile(np.float32([0.75 if channel else 0.25, 0.5]), (len(pos), 1))
            attrs[f"TEXCOORD_{channel}"] = g.accessor(
                g.view(uv.tobytes()), 5126, len(pos), "VEC2"
            )
        png = (
            _png(16, 16, (32, 32, 32), (224, 208, 192))
            if uv1
            else _png(8, 8, (190, 120, 65))
        )
        g.gltf["images"] = [{"bufferView": g.view(png), "mimeType": "image/png"}]
        g.gltf["textures"] = [{"source": 0}]
        material["pbrMetallicRoughness"]["baseColorTexture"] = {
            "index": 0,
            "texCoord": int(uv1),
        }
    g.gltf["materials"] = [material]
    g.gltf["meshes"] = [
        {
            "primitives": [
                {
                    "attributes": attrs,
                    "indices": g.accessor(
                        g.view(idx.astype("<u4").tobytes()), 5125, len(idx), "SCALAR"
                    ),
                    "material": 0,
                }
            ]
        }
    ]
    return g.bytes([g.node(mesh=0)])


def generate():
    """Regenerate small, deterministic GLBs from the regression geometry."""
    pos, _, idx = _box_arrays()
    for name in CASES:
        p, i, c = pos, idx, None
        if name in ("overlap", "collinear"):
            p = np.concatenate([pos, pos + [1, 0.5 if name == "overlap" else 0, 0]])
            i = np.concatenate([idx, idx + len(pos)])
            c = np.repeat([[1, 0.3, 0.12], [0.12, 0.5, 1]], len(pos), axis=0)
        elif name == "hollow":
            p = np.concatenate([pos * 2, pos, pos + [1, 0.25, 0]])
            i = np.concatenate(
                [
                    idx,
                    idx.reshape(-1, 3)[:, ::-1].ravel() + len(pos),
                    idx + 2 * len(pos),
                ]
            )
            c = np.column_stack(
                [(p[:, 0] + 2) / 4, np.full(len(p), 0.5), np.zeros(len(p))]
            )
        elif name == "texture":
            c = np.full_like(pos, 0.5)
        payload = (
            _three_box_glb()
            if name == "layouts"
            else make_glb(
                p, i, c, texture=name in ("texture", "groups", "uv1"), uv1=name == "uv1"
            )
        )
        (HERE / "fixtures" / f"{name}.glb").write_bytes(payload)


def serve(args):
    cases = dict(CASES)
    files = {name: HERE / "fixtures" / f"{name}.glb" for name in CASES}
    for index, path in enumerate(args.glb):
        path = path.resolve(strict=True)
        if path.suffix.lower() != ".glb":
            raise ValueError(f"Expected a GLB file: {path}")
        name = f"model-{index + 1}"
        files[name] = path
        cases[name] = (
            path.stem,
            "Local GLB · fitted to view · C for clipping controls",
            None,
        )
    htmls = {"after": args.viewer_html.read_text()}
    if args.before:
        htmls["before"] = args.before.read_text()
    script = (HERE / "demo.js").read_text()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            req = urlsplit(self.path)
            if req.path.startswith("/fixtures/"):
                name = req.path.removeprefix("/fixtures/").removesuffix(".glb")
                if name not in files or req.path != f"/fixtures/{name}.glb":
                    self.send_error(404)
                    return
                data, mime = (
                    files[name].read_bytes(),
                    "model/gltf-binary",
                )
            elif req.path == "/":
                query = parse_qs(req.query)
                name = query.get("case", ["overlap"])[0]
                variant = query.get("variant", ["after"])[0]
                if name not in cases or variant not in htmls:
                    self.send_error(404)
                    return
                config = json.dumps(
                    dict(
                        case=name,
                        cases=cases,
                        variant=variant,
                        before=bool(args.before),
                    )
                ).replace("<", "\\u003c")
                html = htmls[variant].replace(
                    "htmlTemplate: HTML_TEMPLATE,",
                    "autoConnect: false, htmlTemplate: HTML_TEMPLATE,",
                )
                html = html.replace(
                    "</body>",
                    f"<script>const DEMO = {config};\n{script}</script></body>",
                )
                data, mime = html.encode(), "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("localhost", args.port), Handler)
    return server, f"http://localhost:{server.server_port}/"


def capture(args, url):
    from playwright.sync_api import sync_playwright

    args.capture.mkdir(parents=True, exist_ok=True)
    results = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": 1200, "height": 760}, device_scale_factor=1
        )
        for variant in ["before", "after"] if args.before else ["after"]:
            for name in CASES:
                page.goto(
                    f"{url}?case={name}&variant={variant}&capture=1&worker={int(args.worker)}"
                )
                page.wait_for_function(
                    "() => window.demoReady || window.demoError", timeout=60000
                )
                error = page.evaluate("window.demoError")
                if error:
                    raise RuntimeError(error)
                frames(page, 3)
                result = page.evaluate("window.demoStats")
                results[f"{variant}/{name}"] = result
                if variant == "after" and not result["pass"]:
                    raise AssertionError(f"{name}: {result}")
                page.screenshot(path=str(args.capture / f"{variant}-{name}.png"))
                print(f"{variant}/{name}: {result}", flush=True)
        if args.before:
            comparison_sheets(page, args.capture)
        browser.close()
    (args.capture / "results.json").write_text(json.dumps(results, indent=2) + "\n")


def comparison_sheets(page, output):
    """Pair unchanged browser captures: baseline left, branch right."""
    names = list(CASES)
    page.set_viewport_size({"width": 1600, "height": 1014})
    for start in range(0, len(names), 2):
        images = []
        for name in names[start : start + 2]:
            for variant in ("before", "after"):
                data = base64.b64encode(
                    (output / f"{variant}-{name}.png").read_bytes()
                ).decode()
                images.append(f'<img src="data:image/png;base64,{data}">')
        page.set_content(
            "<style>body{margin:0;display:grid;grid-template-columns:1fr 1fr;background:#141a22}img{width:800px;height:auto}</style>"
            + "".join(images)
        )
        page.wait_for_function(
            "() => [...document.images].every(i => i.complete && i.naturalWidth)"
        )
        page.screenshot(path=str(output / f"comparison-{start // 2 + 1}.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--glb",
        type=Path,
        action="append",
        default=[],
        help="add a local model to the interactive demo; repeat for multiple files",
    )
    parser.add_argument(
        "--viewer-html", type=Path, default=ROOT / "src/threejs_viewer/viewer.html"
    )
    parser.add_argument("--before", type=Path, help="baseline standalone viewer.html")
    parser.add_argument(
        "--generate", action="store_true", help="regenerate committed GLBs and exit"
    )
    parser.add_argument(
        "--capture",
        type=Path,
        help="capture every case and exit; requires Playwright Chromium",
    )
    parser.add_argument(
        "--worker", action="store_true", help="force cap worker during capture"
    )
    args = parser.parse_args()
    if args.generate:
        generate()
        return
    server, url = serve(args)
    print(f"Mesh cap review demo: {url}", flush=True)
    if args.capture:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            capture(args, url)
        finally:
            server.shutdown()
            server.server_close()
    else:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
