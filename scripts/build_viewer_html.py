#!/usr/bin/env python3
"""Build the self-contained viewer.html by inlining viewer.js.

viewer.js is the modular source of truth (ES module for embedding).
viewer.html is the standalone file for file:// usage with everything inlined.

Run from repo root:
    python scripts/build_viewer_html.py
"""

import re
from pathlib import Path

VIEWER_DIR = Path(__file__).parent.parent / "src" / "threejs_viewer"
VIEWER_JS = VIEWER_DIR / "viewer.js"
VIEWER_HTML = VIEWER_DIR / "viewer.html"

HTML_TEMPLATE = """\
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
    </style>
    <script type="importmap">
    {{
        "imports": {{
            "three": "https://unpkg.com/three@0.183.2/build/three.module.js",
            "three/addons/": "https://unpkg.com/three@0.183.2/examples/jsm/"
        }}
    }}
    </script>
</head>
<body>
    <div id="viewer-container"></div>
    <script type="module">
{inlined_js}

// Standalone instantiation
const container = document.getElementById('viewer-container');
new ThreeJSViewer(container);
    </script>
</body>
</html>
"""


def build():
    js_content = VIEWER_JS.read_text()

    # Remove the 'export' keyword from 'export class ThreeJSViewer'
    inlined_js = re.sub(
        r"^export class ", "class ", js_content, count=1, flags=re.MULTILINE
    )

    html = HTML_TEMPLATE.format(inlined_js=inlined_js)
    VIEWER_HTML.write_text(html)
    print(f"Built {VIEWER_HTML} ({len(html)} bytes)")


if __name__ == "__main__":
    build()
