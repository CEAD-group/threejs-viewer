"""Screenshot the strand_collapse repro bead in headless Chromium.

Usage:
    uv run python tools/capture_strand_collapse.py OUTPUT.png

Loads the 100-spine-point bead from repro_min.py, frames the camera around
it, switches to wireframe (M key cycle) so internal triangulation artifacts
are obvious, and writes a PNG to OUTPUT.
"""

import base64
import socket
import sys
import threading
from http.server import HTTPServer
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from threejs_viewer import ViewerClient  # noqa: E402
from threejs_viewer.client import _BlobHandler  # noqa: E402

# Same data as repro_min.py / tests/test_parametric_tube.py.
_SPINE_W_H_B64 = (
    "IAa3PYAKIb7ALya7oHu2PQApIL7O2SO7IC+2PYCrH74+CBq74OG1PSAuH75m5gK7AJa1PUCwHr7U"
    "9My6AEi1PTAzHr5IInG6wEazPcDaGr4gmvs5QFCxPXCCF76g2fM5gGetPRDNEL6gzvQ5QJClPUhp"
    "A77Ap/Q5AOqVPbAt0b2gpvQ5gEttPUCgS72ghfQ5QOpqPYBoRr0Ag/Q5QMNlPaD1Q71AlPQ5AEwj"
    "O8BAA71gnPQ5ACLHvADjxbzgHvQ5YHMavcCxpbwgnfY5oOQ1vUCFlbwgue85gJ1DvUB+jbzA/v05"
    "YFxRvUCUhbxAnvO5gDBTvYCLhLy4uHi64ANVvcB5g7xwfLC6wNZWvQBogrw8ete6IKxYvUBjgbyc"
    "5ue6IHhbvYAdf7ywr+m6QEJevYBOe7ywuem6QKdjvYD/cbyMsem6AH1ovYCQY7yQxem6wEBrvYBC"
    "TrwYuum6QLBrvQBiOrzkyum6oJ9rvYBlMLxAium64DNrvQB+Jrw4Sd+6oOxqvYD8HrxASMK6oKRq"
    "vYBxF7y0GJq6oDZqvYATELxodUC6wMJpvQCrCLzAire5wI9kvQD6e7sA7f05oDxfvQDgKzrgse05"
    "YGRUvYCBHTwAAvc5IAM/vQAR4jzA8vQ5wHIUvVBCgj2AB/U5IKERvbCGhj3gnvU5AN0HvfAYiz2g"
    "mfQ5wKT9vACajD0AdvQ5AKSsPBCeqz2gpPQ5gNFAPSAruz1gsfQ5gPl1PXD/wj3AqfQ5IE6IPQDh"
    "xj3ArvQ54PGOPTDXyD1gq/Q5QKaVPZCZyj0ArfQ5YJuWPTDLyj1ArPQ54JGXPZD5yj3grPQ5YIiY"
    "PZAlyz2grPQ5wIGZPWA4yz3ArPQ54LmaPTBayz3ArPQ5IPCbPXBeyz3ArPQ5AF+ePTAeyz3ArPQ5"
    "ALigPeAWyj3ArPQ5QKmhPWAxyT3ArPQ5AD2iPTAPyD3ArPQ5IJaiPeAhxz3ArPQ5IKqiPRAtxj3A"
    "rPQ5QLuiPYAxxT3ArPQ5QKeiPQA4xD3ArPQ54EKgPaCiuj3ArPQ5QHadPaAfsT3ArPQ5QOmXPVAf"
    "nj3ArPQ5gMSMPYApcD3ArPQ5wPNsPUDcrzzArPQ5gIRqPYBwpTzArPQ5AE9lPQCboDzArPQ5AGgw"
    "OwDrgjvArPQ5wCPEvADOebvArPQ5oJEYvQDJ/LvArPQ5oNgzvYBnHrzArPQ5IHhBvQBXLrzArPQ5"
    "IDVPvQDVPLzArPQ5gCdRvQCRPrzArPQ5gB5TvQD7P7zArPQ5QBZVvQBPQbzArPQ5gA5XvYCgQrzA"
    "rPQ5ILdZvYBwRLzArPQ5IGZcvYDDRbzArPQ5AL5hvYCaSLzArPQ5YORnvYCzTbzArPQ5ABptvYAf"
    "W7zArPQ54ORvvQC/a7zArPQ5oPZxvQDwfbzArPQ5YKZyvYC9grzArPQ5gGpzvUBohrzArPQ54AF0"
    "vQBCirygrPQ5AKN0vcACjrzArPQ5oKp4vUAfqbyArPQ5IMV8vcDlw7xArfQ50D+CvUBw+rwArPQ5"
    "4FGGvaAMGL2ArfQ5IBeLvQByMb3AsfQ5UO6MvcBiM72gsfQ5IJSPveB4Mr0AmvQ5kAKSvaAfK72A"
    "lvQ5XPyCPHJTgjzIgH087S9xPBLVYTwFQ1I8f2o8PH9qPDx/ajw8f2o8PH9qPDx/ajw8f2o8PH9q"
    "PDx/ajw8f2o8PH9qPDx/ajw8f2o8POS0ZTyNUXk8fGaFPODAizxNgI486sqOPOrKjjzqyo486sqO"
    "POrKjjzqyo486sqOPPgLjTzhYog8slyBPF7mcTxkR2A8f2o8PH9qPDx/ajw8f2o8PH9qPDx/ajw8"
    "f2o8PH9qPDx/ajw8f2o8PH9qPDx/ajw8f2o8PJIyXDzRcXA8WfiBPJKdiTxu54086sqOPOrKjjzq"
    "yo486sqOPOrKjjz6uY48uNaLPGpdhTxEnng8VjdkPH9qPDx/ajw8f2o8PH9qPDx/ajw8f2o8PH9q"
    "PDx/ajw8f2o8PH9qPDx/ajw8f2o8PJO+WzxodXA8MC6CPCDbiTyxCY486sqOPOrKjjzqyo486sqO"
    "POrKjjzqyo48WnyNPGhaiTw9iYI8JD10PIb3YTx/ajw8f2o8PH9qPDx/ajw8f2o8PH9qPDx/ajw8"
    "f2o8PKabRDsmX0Y7A+1PO61iYDsZ6HQ7ptuEO7x0kzuMZ5M7vHSTO2x0kztadJM7SnKTOyBykzs0"
    "c5M7t3OTO+Brkzu8dJM7giWTO7x0kzu3QGg7tCVKO0L2LzvFlhw7MFcUO7x0Ezu8dBM7vHQTOxpx"
    "Ezu8dBM7cW4TO7x0EztFrxg7TOYmO4ZHOzvTv1U7Tn1wO7x0kzsPBZM7vHSTO7x0kzu8dJM7vHST"
    "O4pzkztQcZM7O3STO7x0kzuNdJM7vHSTO6V0kzvXrnY7LfVXO25fOjsyKyM7JicWO7x0Ezu8dBM7"
    "vHQTO7x0Ezu8dBM7IqgTO9FrHDvJETA72Y1LO7yDaju8dJM7vHSTO7x0kzu8dJM7vHSTO7x0kzu8"
    "dJM7vHSTO7x0kzu8dJM7vHSTO7x0kzveXnc7uu9XOwW8OTtgcCI7Kr8VO7x0Ezu8dBM7vHQTO7x0"
    "Ezu8dBM7vHQTOyRsFzsM9yM7sKc4OywzUjuJ7W07t3STO7x0kzuxdJM7vHSTO7x0kzu8dJM7knOT"
    "O1hzkzs="
)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _decode_bead(scale=100.0):
    n = 100
    buf = np.frombuffer(base64.b64decode(_SPINE_W_H_B64), dtype=np.float32)
    spine = buf[: n * 3].reshape(n, 3).copy() * scale
    widths = buf[n * 3 : n * 3 + n].copy() * scale
    heights = buf[n * 3 + n :].copy() * scale
    return spine, widths, heights


def _spine_corner_lookat(
    spine: np.ndarray, widths: np.ndarray, heights: np.ndarray, idx: int, half: int = 4
):
    """Lookat target and per-axis half-extents from the *input spine*.

    The geometry's ring centroid matches spine[idx] up to the height-anchor
    offset, but indexing the position buffer requires assumptions about cap
    layout that get fragile if the pipeline changes. Reading straight from
    the input is unambiguous: spine[idx] is the corner, and
    spine[idx-half..idx+half] is the local arc.

    Half-extents in XY are floored to ~1.5× the bead width and in Z to
    ~4× the bead height — at a bulb apex the spine is dense and the
    arc-only extent collapses to a fraction of the cross-section, which
    would frame the camera *inside* the bead body.
    """
    n = len(spine)
    i0 = max(0, idx - half)
    i1 = min(n - 1, idx + half)
    arc = spine[i0 : i1 + 1]
    target = spine[idx].copy()
    arc_extent = np.abs(arc - target).max(axis=0)
    w = float(widths[idx])
    h = float(heights[idx])
    floor = np.array([max(w * 1.5, 1.5), max(w * 1.5, 1.5), max(h * 4.0, 0.5)])
    half_extents = np.maximum(arc_extent, floor)
    return target, half_extents


def capture(
    out_path: Path,
    mode: str = "overview",
    ring: int = 28,
    cam_dir: str = "a",
    zoom: float = 1.5,
):
    port = _free_port()
    client = ViewerClient(port=port, open_browser=False)

    client._http_port = port + 1
    http_server = HTTPServer((client.host, client._http_port), _BlobHandler)
    http_server.blob_store = client._blob_store
    client._http_server = http_server
    threading.Thread(target=http_server.serve_forever, daemon=True).start()
    threading.Thread(target=client._run_server, daemon=True).start()

    spine, widths, heights = _decode_bead()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1200})
        url = f"file://{client.viewer_path.resolve()}?ws_port={client.port}"
        page.goto(url)

        if not client._connected_event.wait(timeout=10):
            raise RuntimeError("viewer never connected")

        client.add_parametric_tube(
            "bead",
            spine=spine,
            widths=widths,
            heights=heights,
            color=0x7AB8CC,
            opacity=1.0,
            metalness=0.15,
            roughness=0.4,
            anchor="top",
            lod=False,
            strand_collapse=True,
        )

        # Wait for the mesh to appear.
        page.wait_for_function(
            "() => window.threejsViewer && window.threejsViewer._objects.has('bead')",
            timeout=10000,
        )

        # Frame + wireframe. M pressed twice = combined mode (solid + black
        # wireframe overlay): the surface gives shape context, the overlay
        # exposes triangulation. Used for both overview and corner shots.
        page.evaluate(
            """() => {
                const v = window.threejsViewer;
                v.frameAll();
                v._shading.wireframeMode = 0;
                v._shading.cycleWireframe();  // → 1
                v._shading.cycleWireframe();  // → 2 = combined
            }"""
        )
        # Let LOD / framing settle.
        page.wait_for_timeout(800)

        # Two modes:
        #   "overview" — oblique view, shows the rectangle-loop diamond fan
        #     (PR 49) clearly. Bulb-corner cube (PR 50) is subtle at this zoom.
        #   "corner"   — close zoom on a corner detected from the *input
        #     spine* (not from the geometry's position buffer). Two adjacent
        #     bulb apexes can share an XY footprint at different Z layers;
        #     buffer-indexed corner extents see them as overlapping, so the
        #     lookat must come from the input.
        if mode == "corner":
            radius_steps = max(3, round(zoom * 2))
            target, half_extents = _spine_corner_lookat(
                spine, widths, heights, ring, half=radius_steps
            )
            tx, ty, tz = float(target[0]), float(target[1]), float(target[2])
            hx, hy, hz = float(half_extents[0]), float(half_extents[1]), float(half_extents[2])
        else:
            tx = ty = tz = hx = hy = hz = 0.0  # unused in overview mode
        page.evaluate(
            f"""() => {{
                const v = window.threejsViewer;
                const obj = v._objects.get('bead');
                obj.geometry.computeBoundingBox();
                const bb = obj.geometry.boundingBox;
                const cx = (bb.min.x + bb.max.x) / 2;
                const cy = (bb.min.y + bb.max.y) / 2;
                const cz = (bb.min.z + bb.max.z) / 2;
                const sx = bb.max.x - bb.min.x;
                const sy = bb.max.y - bb.min.y;
                const sz = bb.max.z - bb.min.z;
                const s = Math.max(sx, sy, sz);
                const mode = "{mode}";
                if (mode === "corner") {{
                    const crx = {tx}, cry = {ty}, crz = {tz};
                    const halfX = {hx}, halfY = {hy}, halfZ = {hz};
                    // FOV-aware fit. Vertical FOV is the camera setting;
                    // horizontal half-FOV = atan(tan(vFOV/2) * aspect).
                    const vFov = (v._perspCamera.fov || 75) * Math.PI / 180 / 2;
                    const aspect = v._perspCamera.aspect || (1600 / 1200);
                    const hFov = Math.atan(Math.tan(vFov) * aspect);
                    const distV = Math.max(halfY, halfZ) / Math.tan(vFov);
                    const distH = Math.max(halfX, halfY) / Math.tan(hFov);
                    const dist = Math.max(distV, distH) * 1.4;
                    const dirCode = "{cam_dir}";
                    let dx, dy, dz;
                    if (dirCode === "b")      {{ dx =  0.3; dy =  0.3; dz =  1.0; }}
                    else if (dirCode === "c") {{ dx =  0.15; dy = -0.15; dz =  1.0; }}
                    else                       {{ dx =  0.3; dy = -0.3; dz =  1.0; }}
                    const dLen = Math.hypot(dx, dy, dz);
                    dx /= dLen; dy /= dLen; dz /= dLen;
                    v._camera.position.set(crx + dx * dist, cry + dy * dist, crz + dz * dist);
                    v._controls.target.set(crx, cry, crz);
                }} else {{
                    v._camera.position.set(cx + 0.8 * s, cy - 0.6 * s, cz + 0.55 * s);
                    v._controls.target.set(cx, cy, cz);
                }}
                v._camera.up.set(0, 0, 1);
                v._camera.lookAt(v._controls.target);
                v._controls.update();
            }}"""
        )
        page.wait_for_timeout(400)
        page.screenshot(path=str(out_path), full_page=False)
        browser.close()

    client.disconnect()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 6:
        print(__doc__)
        sys.exit(1)
    out = Path(sys.argv[1]).resolve()
    mode = sys.argv[2] if len(sys.argv) >= 3 else "overview"
    ring = int(sys.argv[3]) if len(sys.argv) >= 4 else 28
    cam_dir = sys.argv[4] if len(sys.argv) >= 5 else "a"
    zoom = float(sys.argv[5]) if len(sys.argv) == 6 else 1.5
    capture(out, mode=mode, ring=ring, cam_dir=cam_dir, zoom=zoom)
