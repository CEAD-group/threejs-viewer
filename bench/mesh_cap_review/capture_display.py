"""Capture separate-owner cap priority and M overlays on baseline and branch.

uv run python bench/mesh_cap_review/capture_display.py --before /tmp/main.html --output /tmp/cap-display
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

from threejs_viewer import ViewerClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from conftest import frames, settle  # noqa: E402
from test_clip_caps import _box_arrays  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for variant, html in [
            ("before", args.before),
            ("after", ROOT / "src/threejs_viewer/viewer.html"),
        ]:
            client = ViewerClient(port=0, open_browser=False)
            client._start_servers(http_port=0)
            page = browser.new_page(viewport={"width": 1100, "height": 800})
            try:
                page.goto(f"{html.resolve().as_uri()}?ws_port={client.port}")
                page.wait_for_function("window.threejsViewer?._ws?.readyState === 1")
                pos, _, idx = _box_arrays()
                client.add_mesh("first", pos, idx, color=0xEA743E)
                other = pos.copy()
                angle = 0.37
                other[:, 0] = (
                    pos[:, 0] * np.cos(angle) - pos[:, 1] * np.sin(angle) + 0.5
                )
                other[:, 1] = pos[:, 0] * np.sin(angle) + pos[:, 1] * np.cos(angle)
                client.add_mesh("last", other, idx, color=0x3589DC)
                client.set_sun(enabled=False)
                client.set_background("#20252b")
                client.show_grid(False)
                client.set_clipping_caps(True)
                client.set_clipping_slab(
                    normal=[0, 0, 1], center=0, thickness=0.4, show_helper=False
                )
                client.set_camera(
                    position=[0.01, -3, 8], target=[0.25, 0, 0], up=[0, 1, 0], fov=28
                )
                settle(client)
                frames(page)
                page.wait_for_function("window.threejsViewer._clipCaps.settled()")
                page.evaluate("""() => {
                    document.querySelectorAll('.tjsv-rail').forEach(n=>n.style.display='none');
                    const el=document.createElement('div');el.id='capture-title';
                    el.style.cssText='position:fixed;top:0;left:0;right:0;padding:24px;background:#141a22;color:white;font:22px system-ui;z-index:100';
                    document.body.appendChild(el);
                }""")
                for mode, label in [
                    (0, "Separate mesh caps · stable overlap priority"),
                    (2, "M mode · solid + cap triangle edges"),
                ]:
                    page.evaluate(
                        """([mode,title]) => {
                        window.threejsViewer._shading.setWireframeMode(mode);
                        document.getElementById('capture-title').textContent=title;
                    }""",
                        [mode, f"{variant.title()} · {label}"],
                    )
                    frames(page, 3)
                    page.screenshot(
                        path=str(
                            args.output
                            / f"{variant}-{'wire' if mode else 'priority'}.png"
                        )
                    )
            finally:
                page.close()
                client.disconnect()
        browser.close()


if __name__ == "__main__":
    main()
