#!/usr/bin/env python3
"""Build the self-contained viewer.html by inlining all viewer assets.

Source files (edit these):
  - viewer.js      — ES module class for embedding
  - viewer.css     — styles for the viewer UI
  - template.html  — HTML template for UI controls
  - static/cubemaps/<name>/ — cubemap face sets (px..nz as .hdr, .png or .jpg)
  - static/draco/  — vendored Draco glTF decoder (wasm + emscripten wrapper)

Generated file (do not edit):
  - ../viewer.html — standalone file for file:// usage with everything inlined
  - static/cubemaps/index.json — cubemap set list for embedders serving viewer/

Run from repo root:
    uv run python src/threejs_viewer/viewer/build.py
"""

import base64
import gzip
import io
import json
import re
from pathlib import Path

VIEWER_DIR = Path(__file__).parent
OUTPUT_HTML = VIEWER_DIR.parent / "viewer.html"
STATIC_DIR = VIEWER_DIR / "static"
DRACO_DIR = STATIC_DIR / "draco"

THREE_VERSION = "0.183.2"
CUBEMAP_FACES = ["px", "nx", "py", "ny", "pz", "nz"]


def _gzip_b64(raw: bytes, previous: str | None = None) -> str:
    """gzip + base64 a blob (mtime=0 so rebuilds are stable).

    ``previous`` is the encoding already committed in viewer.html for this
    slot; it is returned unchanged when it inflates to ``raw``. The compressed
    bytes differ between zlib implementations (madler zlib vs the zlib-ng some
    python-build-standalone builds ship), so recompressing an unchanged blob
    would rewrite 1.4 MB of base64 per PR and fail CI's byte-exact freshness
    check on the other implementation. A changed blob always recompresses.
    """
    if previous:
        try:
            if gzip.decompress(base64.b64decode(previous, validate=True)) == raw:
                return previous
        except (ValueError, OSError, EOFError):
            pass
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9, mtime=0) as gz:
        gz.write(raw)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _previous_blobs() -> dict:
    """The gzip+base64 blobs inlined in the committed viewer.html, keyed by
    slot: ``draco.wasm``, ``draco.wrapper`` and ``cubemap.<name>.<face>``."""
    try:
        html = OUTPUT_HTML.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    blobs = {}
    for slot, field in (("draco.wasm", "wasmGzB64"), ("draco.wrapper", "wrapperGzB64")):
        m = re.search(rf"{field}: '([^']*)'", html)
        if m:
            blobs[slot] = m.group(1)
    for m in re.finditer(
        r"^    '([^']+)': \{ format: '\w+', faces: \{ ([^}]*) \} \}", html, re.M
    ):
        for face in re.finditer(r"(\w+): '([^']*)'", m.group(2)):
            blobs[f"cubemap.{m.group(1)}.{face.group(1)}"] = face.group(2)
    return blobs


CUBEMAPS_DIR = STATIC_DIR / "cubemaps"
DEFAULT_CUBEMAP = "paul-lobe-haus"
CUBEMAP_FORMATS = ("hdr", "png", "jpg")


def _read_cubemaps_b64(previous: dict) -> dict:
    """Read every face set under static/cubemaps/<name>/ and gzip+base64 each
    face (inflated in the browser; gzip roughly halves the HDR bytes, and keeps
    one decode path for the LDR sets). Returns {name: {format, faces}} with
    DEFAULT_CUBEMAP first."""
    sets = {}
    for d in sorted(p for p in CUBEMAPS_DIR.iterdir() if p.is_dir()):
        fmt = next((f for f in CUBEMAP_FORMATS if (d / f"px.{f}").is_file()), None)
        if fmt is None:
            continue
        sets[d.name] = {
            "format": fmt,
            "faces": {
                face: _gzip_b64(
                    (d / f"{face}.{fmt}").read_bytes(),
                    previous.get(f"cubemap.{d.name}.{face}"),
                )
                for face in CUBEMAP_FACES
            },
        }
    if DEFAULT_CUBEMAP not in sets:
        raise SystemExit(
            f"default cubemap {DEFAULT_CUBEMAP!r} missing in {CUBEMAPS_DIR}"
        )
    return {DEFAULT_CUBEMAP: sets.pop(DEFAULT_CUBEMAP), **sets}


def _read_draco_b64(previous: dict) -> dict:
    """Read the vendored Draco glTF decoder and return gzip+base64 payloads.

    Raw is ~251 KB (wasm 188 KB + wrapper 57 KB); gzip+base64 is ~100 KB, so
    the decoder costs ~15% of viewer.html rather than ~49%. The viewer inflates
    both with DecompressionStream('gzip') before handing them to DRACOLoader.
    """
    return {
        "wasm": _gzip_b64(
            (DRACO_DIR / "draco_decoder.wasm").read_bytes(), previous.get("draco.wasm")
        ),
        "wrapper": _gzip_b64(
            (DRACO_DIR / "draco_wasm_wrapper.js").read_bytes(),
            previous.get("draco.wrapper"),
        ),
    }


def build():
    js_content = (VIEWER_DIR / "viewer.js").read_text(encoding="utf-8")
    controls_content = (VIEWER_DIR / "controls.js").read_text(encoding="utf-8")
    css_content = (VIEWER_DIR / "viewer.css").read_text(encoding="utf-8")
    html_template = (VIEWER_DIR / "template.html").read_text(encoding="utf-8")
    previous = _previous_blobs()
    cubemaps = _read_cubemaps_b64(previous)
    draco_data = _read_draco_b64(previous)

    # Strip the local controls.js import from viewer.js (we inline it instead).
    js_content = re.sub(
        r"^\s*import\s+\{\s*ViewerControls\s*\}\s+from\s+['\"]\./controls\.js['\"];?\s*$\n?",
        "",
        js_content,
        count=1,
        flags=re.MULTILINE,
    )

    # Strip the `export { ... }` line from controls.js so it inlines as plain code.
    controls_inlined = re.sub(
        r"^\s*export\s*\{[^}]*\};?\s*$\n?",
        "",
        controls_content,
        flags=re.MULTILINE,
    )
    # Also drop its `import * as THREE from 'three';` — the outer module already imports THREE.
    controls_inlined = re.sub(
        r"^\s*import\s+\*\s+as\s+THREE\s+from\s+['\"]three['\"];?\s*$\n?",
        "",
        controls_inlined,
        count=1,
        flags=re.MULTILINE,
    )

    # Remove the 'export' keyword from 'export class ThreeJSViewer'
    js_inlined = re.sub(
        r"^export class ", "class ", js_content, count=1, flags=re.MULTILINE
    )

    # Prepend controls.js source so ViewerControls is in scope when ThreeJSViewer runs.
    js_inlined = controls_inlined + "\n" + js_inlined

    # Cubemap sets as a JS object literal: {name: {format, faces: {px: ...}}}
    cubemap_js_entries = []
    for name, cm in cubemaps.items():
        faces = ", ".join(f"{face}: '{cm['faces'][face]}'" for face in CUBEMAP_FACES)
        cubemap_js_entries.append(
            f"    '{name}': {{ format: '{cm['format']}', faces: {{ {faces} }} }}"
        )
    cubemap_js = "const CUBEMAPS = {\n" + ",\n".join(cubemap_js_entries) + "\n};"

    # Draco glTF decoder, gzip+base64 (inflated in the browser before use).
    draco_js = (
        "const DRACO_DECODER_DATA = {\n"
        f"    wasmGzB64: '{draco_data['wasm']}',\n"
        f"    wrapperGzB64: '{draco_data['wrapper']}'\n"
        "};"
    )

    # Escape backticks and ${} in template for JS template literal
    html_escaped = (
        html_template.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    )
    template_js = f"const HTML_TEMPLATE = `{html_escaped}`;"

    # Indent CSS for the <style> block
    css_indented = "\n".join(
        "        " + line if line.strip() else "" for line in css_content.splitlines()
    )

    html = f"""\
<!-- DO NOT EDIT — this file is generated by viewer/build.py from the source files in viewer/ -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Three.js Viewer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html, body {{ width: 100%; height: 100%; overflow: hidden; }}
        #viewer-container {{ width: 100%; height: 100%; }}
{css_indented}
    </style>
    <script type="importmap">
    {{
        "imports": {{
            "three": "https://unpkg.com/three@{THREE_VERSION}/build/three.module.js",
            "three/addons/": "https://unpkg.com/three@{THREE_VERSION}/examples/jsm/"
        }}
    }}
    </script>
</head>
<body>
    <div id="viewer-container"></div>
    <script type="module">
{cubemap_js}

{draco_js}

{template_js}

{js_inlined}

// Standalone instantiation
const container = document.getElementById('viewer-container');
window.threejsViewer = new ThreeJSViewer(container, {{
    htmlTemplate: HTML_TEMPLATE,
    cubemapData: CUBEMAPS['{DEFAULT_CUBEMAP}'].faces,
    cubemaps: CUBEMAPS,
    dracoDecoder: DRACO_DECODER_DATA,
    // Tests keep a short cutoff with ?handshake_timeout_ms=5000 (issue #273).
    handshakeTimeoutMs: new URLSearchParams(location.search).get('handshake_timeout_ms') ?? undefined
}});
    </script>
</body>
</html>
"""
    OUTPUT_HTML.write_text(html, encoding="utf-8", newline="\n")
    # Manifest for embedders serving the viewer/ source dir (ribweaver): the
    # viewer reads it to offer the same sets, fetching faces as static files.
    manifest = [{"name": n, "format": cm["format"]} for n, cm in cubemaps.items()]
    (CUBEMAPS_DIR / "index.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Built {OUTPUT_HTML} ({len(html)} bytes)")


if __name__ == "__main__":
    build()
