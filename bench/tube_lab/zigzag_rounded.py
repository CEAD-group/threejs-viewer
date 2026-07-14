"""Middle-ground repro: long straight -> ROUNDED (larger-radius) +120/-120
zig-zag -> long straight. Finely + irregularly sampled with only a WHISPER of
(smoothed) jitter, and a smooth sinusoidal variation in the bead width so the
mesh has something to show. Top views only, MESH (combined wireframe) VISIBLE.

Panels emitted (same camera framing so they overlay):
  zr_R<..>_input_top.png  just the input data as a polyline (the spine)
  zr_R<..>_off_top.png    collapse OFF, wireframe overlay
  zr_R<..>_0.5_top.png    collapse default 0.5, wireframe overlay
  zr_R<..>_1.0_top.png    collapse 1.0, wireframe overlay
"""

import socket
import sys
import threading
import time
import math
from http.server import HTTPServer
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
from playwright.sync_api import sync_playwright  # noqa: E402
from threejs_viewer import ViewerClient  # noqa: E402
from threejs_viewer.client import _BlobHandler  # noqa: E402

W0, H0 = 8.0, 3.0
W_VAR = 0.18  # smooth width swing, fraction of W0 (peak-to-mean)


def free():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def smooth(pts, k=5):
    """Light moving-average on interior points (Hann window) to take the
    edge off the jitter without flattening the corner."""
    pts = np.asarray(pts, float)
    if len(pts) < 2 * k + 1:
        return pts
    w = np.hanning(2 * k + 1)
    w /= w.sum()
    out = pts.copy()
    for c in range(pts.shape[1]):
        out[k:-k, c] = np.convolve(pts[:, c], w, mode="valid")
    return out


def build(R=6.0, seed=7, base_step=0.15, jitter=0.012):
    """Heading-integrated path: straight in, +120 rounded turn (radius R),
    short leg, -120 rounded turn, straight out. Dense in the turns, with a
    whisper of perpendicular jitter that is then smoothed."""
    rng = np.random.default_rng(seed)
    pos = np.array([-120.0, 0.0, 0.0])
    heading = 0.0
    pts = [pos.copy()]
    marks = {}

    def advance(length, total_turn_deg, dense):
        nonlocal pos, heading
        arc = max(length, 1e-6)
        rate = math.radians(total_turn_deg) / arc  # rad per mm
        s = 0.0
        while s < length - 1e-9:
            step = (base_step * rng.uniform(0.6, 1.5)) if dense else 12.0
            step = min(step, length - s)
            s += step
            heading += rate * step
            d = np.array([math.cos(heading), math.sin(heading), 0.0])
            perp = np.array([-math.sin(heading), math.cos(heading), 0.0])
            j = rng.normal(0.0, jitter) if dense else 0.0
            pos = pos + d * step + perp * j
            pts.append(pos.copy())

    advance(120.0, 0.0, dense=False)  # long straight in
    marks["lo"] = len(pts)
    turn_arc = math.radians(120.0) * R  # arc length of a 120 deg turn at radius R
    advance(turn_arc, 120.0, dense=True)  # rounded +120
    advance(4.0, 0.0, dense=True)  # short leg
    advance(turn_arc, -120.0, dense=True)  # rounded -120
    marks["hi"] = len(pts)
    advance(120.0, 0.0, dense=False)  # long straight out
    sp = np.array(pts, np.float32)
    # smooth only the dense turn span; leave the sparse straights untouched
    lo, hi = marks["lo"], marks["hi"]
    sp[lo:hi] = smooth(sp[lo:hi], k=4)
    # pseudo-randomly drop ~80% of the dense-turn points, leaving UNEVEN gaps
    # (keep the two span endpoints so the straights still connect cleanly)
    rng2 = np.random.default_rng(seed + 1)
    dense = sp[lo:hi]
    keep = rng2.random(len(dense)) < 0.20
    keep[0] = keep[-1] = True
    dense_dec = dense[keep]
    sp = np.vstack([sp[:lo], dense_dec, sp[hi:]]).astype(np.float32)
    new_lo, new_hi = lo, lo + len(dense_dec)
    return sp, new_lo, new_hi


def widths_for(sp, lo, hi):
    """Smooth sinusoidal bead-width variation along arc length so the mesh
    surface undulates gently, plus a +40% bead-width boost in the dense
    turn region (rings lo..hi)."""
    seg = np.linalg.norm(np.diff(sp, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = max(arc[-1], 1e-6)
    phase = 2 * math.pi * 3.0 * arc / total  # 3 gentle swells over the path
    w = W0 * (1.0 + W_VAR * np.sin(phase))
    h = H0 * (1.0 + 0.5 * W_VAR * np.sin(phase + 0.9))
    # +40% bead-width bump, but as a smooth raised-cosine envelope confined to
    # the turn: 1.0 at the span ends, peaks 1.4 mid-turn, so there is no hard
    # step (a step would flare the tube where it meets the straights).
    boost = np.ones(len(sp), np.float32)
    m = hi - lo
    if m > 1:
        t = np.linspace(0.0, 1.0, m)  # 0..1 across the dense span
        env = 0.5 - 0.5 * np.cos(2 * math.pi * t)  # Hann: 0 at ends, 1 mid
        boost[lo:hi] = 1.0 + 0.40 * env
    w = w * boost
    return w.astype(np.float32), h.astype(np.float32)


PROBE = """(arg)=>{const [lo,hi]=arg;const o=window.threejsViewer._objects.get('t');const u=o.userData;
  if(!u.collapsedPositions||!u.uncollapsedPositions)return null;
  const a=u.collapsedPositions,b=u.uncollapsedPositions;const nCs=6;
  let moved=0,mx=0,straight=0,strMax=0;const nR=Math.floor(Math.min(a.length,b.length)/3/nCs);
  for(let r=0;r<nR;r++){let rm=0;for(let j=0;j<nCs;j++){const i=(r*nCs+j)*3;
    const d=Math.hypot(a[i]-b[i],a[i+1]-b[i+1],a[i+2]-b[i+2]);if(d>rm)rm=d;}
    if(rm>1e-3){moved++;if(rm>mx)mx=rm;if(r<lo-1||r>hi+1){straight++;if(rm>strMax)strMax=rm;}}}
  return {moved,mx,straight,strMax};}"""
JS_CAM = """([px,py,pz,tx,ty,tz,ux,uy,uz])=>{const v=window.threejsViewer;
  v._camera.position.set(px,py,pz);v._controls.target.set(tx,ty,tz);
  v._camera.up.set(ux,uy,uz);v._camera.lookAt(v._controls.target);v._controls.update();}"""
JS_WIRE = """(m)=>{const v=window.threejsViewer;let g=0;while(v._shading.wireframeMode!==m&&g++<4)v._shading.cycleWireframe();}"""


def _serve(c):
    h = HTTPServer((c.host, c._http_port), _BlobHandler)
    h.blob_store = c._blob_store
    c._http_server = h
    threading.Thread(target=h.serve_forever, daemon=True).start()
    threading.Thread(target=c._run_server, daemon=True).start()


def _cam(pg, sp, lo, hi):
    dense = sp[lo:hi]
    cx, cy = float(dense[:, 0].mean()), float(dense[:, 1].mean())
    ext = float(max(np.ptp(dense[:, 0]), np.ptp(dense[:, 1]))) + 3 * W0
    d = ext / (2 * math.tan(math.radians(20))) * 1.05
    pg.evaluate(JS_CAM, [cx, cy, d, cx, cy, 0, 0, 1, 0])


def run_input(R, tag):
    """Panel showing just the input data as a polyline (the spine)."""
    port = free()
    c = ViewerClient(port=port, open_browser=False)
    c._http_port = free()
    _serve(c)
    sp, lo, hi = build(R=R)
    with sync_playwright() as p:
        br = p.chromium.launch(headless=True)
        pg = br.new_page(viewport={"width": 1400, "height": 1000})
        pg.goto(f"file://{c.viewer_path.resolve()}?ws_port={c.port}")
        if not c._connected_event.wait(timeout=15):
            raise RuntimeError("viewer never connected")
        c.clear()
        c.add_polyline("t", points=sp, color=0x2266DD, line_width=2.5)
        time.sleep(0.4)
        pg.wait_for_function(
            "()=>window.threejsViewer&&window.threejsViewer._objects.size>0",
            timeout=15000,
        )
        time.sleep(1.0)
        _cam(pg, sp, lo, hi)
        pg.wait_for_timeout(400)
        pg.screenshot(path=str(OUT / f"zr_{tag}_input_top.png"))
        br.close()
    c.disconnect()


def run(R, sc, tag, wire=2):
    port = free()
    c = ViewerClient(port=port, open_browser=False)
    c._http_port = free()
    _serve(c)
    sp, lo, hi = build(R=R)
    w, ht = widths_for(sp, lo, hi)
    with sync_playwright() as p:
        br = p.chromium.launch(headless=True)
        pg = br.new_page(viewport={"width": 1400, "height": 1000})
        pg.goto(f"file://{c.viewer_path.resolve()}?ws_port={c.port}")
        if not c._connected_event.wait(timeout=15):
            raise RuntimeError("viewer never connected")
        c.clear()
        c.add_parametric_tube(
            "t",
            spine=sp,
            widths=w,
            heights=ht,
            color=0x88BCCC,
            anchor="center",
            lod=False,
            strand_collapse=sc,
            roughness=0.45,
            metalness=0.05,
        )
        time.sleep(0.4)
        pg.wait_for_function(
            "()=>window.threejsViewer&&window.threejsViewer._objects.size>0",
            timeout=15000,
        )
        time.sleep(2.0)
        dev = pg.evaluate(PROBE, [lo, hi])
        pg.evaluate(JS_WIRE, wire)  # 2 = solid + black wireframe overlay (mesh visible)
        _cam(pg, sp, lo, hi)
        pg.wait_for_timeout(400)
        pg.screenshot(path=str(OUT / f"zr_{tag}_top.png"))
        br.close()
    c.disconnect()
    return dev


if __name__ == "__main__":
    R = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    print(
        f"rounded ±120 zig-zag, turn radius R={R} mm (bead W={W0}±{W_VAR * 100:.0f}%), mesh visible"
    )
    run_input(R, f"R{R:g}")
    print(f"  input panel -> zr_R{R:g}_input_top.png")
    for sc, lab in [
        (False, "off"),
        ({"max_snap_factor": 0.5}, "0.5"),
        ({"max_snap_factor": 1.0}, "1.0"),
    ]:
        d = run(R, sc, f"R{R:g}_{lab}")
        print(f"  {lab:>4}: {d}")
