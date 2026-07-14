"""Tube-corner quality bench. Scores every (scene x approach) on THREE measures:

  (1) intersecting lines  -> darkPx : interior dark EDGE pixels (wireframe).
                              Fold-fans pile overlapping edges. Lower = cleaner.
  (2) contour fidelity    -> IoU of the blue footprint vs the scene's OFF
                              baseline. ~1.0 = same pixels coloured blue.
  (3) triangle count      -> tris  : reducing triangles WITHOUT losing (1)/(2)
                              is itself a win (lighter mesh, same picture).

Scenes are corner archetypes (rounded zig-zag, 3-point sharp V, hairpin U).
Approaches are spine transform x strand_collapse setting. All render in ONE
browser session at a per-scene top-view camera; pixels are read straight off the
WebGL buffer (no PNG-decode dependency). Add scenes/approaches freely.
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
from zigzag_rounded import build as build_zigzag, widths_for, JS_CAM, JS_WIRE, W0  # noqa: E402

VIEW = {"width": 1200, "height": 900}


# ---------------------------------------------------------------- scenes -----
def _frame(sp, lo, hi, W):
    roi = sp[lo:hi] if hi > lo else sp
    cx, cy = float(roi[:, 0].mean()), float(roi[:, 1].mean())
    ext = float(max(np.ptp(roi[:, 0]), np.ptp(roi[:, 1]))) + 3 * W
    return cx, cy, ext


def scene_zigzag(R=6.0):
    """Rounded, decimated, jittered +120/-120 zig-zag with a +40% bead bump in
    the turn (the case we've been staring at)."""
    sp, lo, hi = build_zigzag(R=R)
    w, ht = widths_for(sp, lo, hi)
    cx, cy, ext = _frame(sp, lo, hi, W0)
    return dict(sp=sp, w=w, ht=ht, lo=lo, hi=hi, cx=cx, cy=cy, ext=ext)


def scene_sharpV(angle_deg=90.0, leg=60.0, W=10.0, H=3.0):
    """3-point sharp corner: two legs meeting at one apex. angle_deg is the
    INTERIOR angle between the legs; 90 deg is a clean right-angle corner whose
    turn (90 deg) stays under the 120 deg miter/bevel limit, so both legs keep a
    constant W width and the footprint is a faithful constant-width bead with one
    corner (a sharper V, e.g. 35 deg, would have the apex miter mangle the leg
    width, making it an invalid contour baseline)."""
    half = math.radians(angle_deg) / 2.0
    B = np.array([0.0, 0.0, 0.0])
    A = B + leg * np.array([math.cos(math.pi - half), math.sin(math.pi - half), 0.0])
    C = B + leg * np.array([math.cos(math.pi + half), math.sin(math.pi + half), 0.0])
    sp = np.array([A, B, C], np.float32)
    w = np.full(3, W, np.float32)
    ht = np.full(3, H, np.float32)
    cx, cy, ext = _frame(sp, 0, 3, W)
    return dict(sp=sp, w=w, ht=ht, lo=0, hi=3, cx=cx, cy=cy, ext=ext)


def scene_hairpin(R=2.0, W=12.0, H=3.0, npts=11):
    """Tight 180 U (wipe loop): radius << bead half-width, so the inside strands
    fold back over each other -> the classic strand-collapse target. Sparse arc
    so rings over-rotate."""
    lead = np.array([[-70.0, R, 0.0], [-6.0, R, 0.0]])
    ang = np.linspace(-math.pi / 2, math.pi / 2, npts)
    arc = np.stack([R * np.cos(ang), R * np.sin(ang), np.zeros_like(ang)], axis=1)
    tail = np.array([[-6.0, -R, 0.0], [-70.0, -R, 0.0]])
    sp = np.vstack([lead, arc, tail]).astype(np.float32)
    lo, hi = 2, 2 + npts
    w = np.full(len(sp), W, np.float32)
    ht = np.full(len(sp), H, np.float32)
    cx, cy, ext = _frame(sp, lo, hi, W)
    return dict(sp=sp, w=w, ht=ht, lo=lo, hi=hi, cx=cx, cy=cy, ext=ext)


def scene_mixed(W=10.0, H=3.0):
    """ONE tube carrying BOTH a genuine SPARSE 90 deg sharp corner AND a DENSE
    smooth rounded turn, so the adaptive gate must discriminate within a single
    object: the sharp vertex stays pinned (contour intact, legs untouched) while
    the dense arc rounds/decimates. The region [lo:hi] brackets both features so
    a single transform call sees them together."""
    pos = np.array([-70.0, -25.0, 0.0])
    heading = 0.0
    pts = [pos.copy()]

    def straight(length, stp):
        nonlocal pos
        s = 0.0
        while s < length - 1e-9:
            ds = min(stp, length - s)
            s += ds
            pos = pos + np.array([math.cos(heading), math.sin(heading), 0.0]) * ds
            pts.append(pos.copy())

    def rounded(total_deg, radius, stp):
        nonlocal pos, heading
        arc = math.radians(abs(total_deg)) * radius
        rate = math.radians(total_deg) / arc
        s = 0.0
        while s < arc - 1e-9:
            ds = min(stp, arc - s)
            s += ds
            heading += rate * ds
            pos = pos + np.array([math.cos(heading), math.sin(heading), 0.0]) * ds
            pts.append(pos.copy())

    straight(30.0, 20.0)  # sparse leg in (heading +x)
    lo = (
        len(pts) - 2
    )  # region opens one vertex BEFORE the corner (corner is INTERIOR -> tests pin-by-angle)
    heading += math.radians(
        90.0
    )  # SHARP 90 deg corner (no vertex added; corner = current pts[-1])
    straight(35.0, 20.0)  # sparse leg out of the corner (heading +y)
    rounded(120.0, 6.0, 0.4)  # DENSE smooth rounded turn (~31 rings)
    hi = len(pts)  # region closes after the dense arc
    straight(30.0, 20.0)  # sparse leg out
    sp = np.array(pts, np.float32)
    w = np.full(len(sp), W, np.float32)
    ht = np.full(len(sp), H, np.float32)
    cx, cy, ext = _frame(sp, lo, hi, W)
    return dict(sp=sp, w=w, ht=ht, lo=lo, hi=hi, cx=cx, cy=cy, ext=ext)


SCENES = [
    ("zigzag", scene_zigzag),
    ("sharpV", scene_sharpV),
    ("hairpin", scene_hairpin),
    ("mixed", scene_mixed),
]


# ------------------------------------------------- spine transforms ----------
def _identity(sp, w, ht, lo, hi):
    return sp, w, ht, lo, hi


def _resample_region(sp, w, ht, lo, hi, step, spline):
    """Resample the [lo:hi] region at ~step spacing, carrying widths by
    arc-length interp. spline=True rounds the corner (Catmull-Rom); False keeps
    the chords (collinear subdivision, contour exact)."""
    P = sp[lo:hi].astype(float)
    if len(P) < 3:
        return sp, w, ht, lo, hi
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]
    if total <= 0:
        return sp, w, ht, lo, hi
    m = max(3, int(math.ceil(total / step)))
    samples = np.linspace(0, total, m)
    pts = []
    for s in samples:
        i = int(np.searchsorted(arc, s) - 1)
        i = min(max(i, 0), len(P) - 2)
        t = (s - arc[i]) / max(seg[i], 1e-9)
        if spline:
            p0 = P[max(i - 1, 0)]
            p1 = P[i]
            p2 = P[i + 1]
            p3 = P[min(i + 2, len(P) - 1)]
            t2 = t * t
            t3 = t2 * t
            pts.append(
                0.5
                * (
                    (2 * p1)
                    + (-p0 + p2) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
                )
            )
        else:
            pts.append(P[i] + (P[i + 1] - P[i]) * t)
    turn = np.asarray(pts, np.float32)
    wn = np.interp(samples, arc, w[lo:hi]).astype(np.float32)
    hn = np.interp(samples, arc, ht[lo:hi]).astype(np.float32)
    sp2 = np.vstack([sp[:lo], turn, sp[hi:]]).astype(np.float32)
    w2 = np.concatenate([w[:lo], wn, w[hi:]]).astype(np.float32)
    h2 = np.concatenate([ht[:lo], hn, ht[hi:]]).astype(np.float32)
    return sp2, w2, h2, lo, lo + len(turn)


def spline(step):
    return lambda sp, w, ht, lo, hi: _resample_region(sp, w, ht, lo, hi, step, True)


def lindens(step):
    return lambda sp, w, ht, lo, hi: _resample_region(sp, w, ht, lo, hi, step, False)


# ---------------------------------------- curvature-adaptive resampler -------
def _catmull_span(P, w, h, step):
    """Round ONE span with a CLAMPED Catmull-Rom (endpoints repeated), sampled
    at ~step arc spacing, carrying width/height by arc-length interp. The clamp
    (`max(i-1,0)`/`min(i+2,L-1)`) is per-span, so the curve passes EXACTLY
    through the span's two endpoints with the incident chord directions
    preserved — i.e. through the pins that bound the span, uncut."""
    P = np.asarray(P, float)
    L = len(P)
    if L < 2:
        return P, np.asarray(w, float), np.asarray(h, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]
    if total <= 0:
        return P, np.asarray(w, float), np.asarray(h, float)
    m = max(3, int(math.ceil(total / step)) + 1)
    samples = np.linspace(0.0, total, m)
    pts = []
    for s in samples:
        i = int(np.searchsorted(arc, s) - 1)
        i = min(max(i, 0), L - 2)
        t = (s - arc[i]) / max(seg[i], 1e-9)
        p0 = P[max(i - 1, 0)]
        p1 = P[i]
        p2 = P[i + 1]
        p3 = P[min(i + 2, L - 1)]
        t2 = t * t
        t3 = t2 * t
        pts.append(
            0.5
            * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
        )
    wn = np.interp(samples, arc, np.asarray(w, float))
    hn = np.interp(samples, arc, np.asarray(h, float))
    return np.asarray(pts, float), wn, hn


def _adaptive_region(sp, w, ht, lo, hi, step, pin_deg, dense_factor):
    """Curvature-adaptive resample of [lo:hi]: PIN genuinely-sharp vertices
    (turn >= pin_deg) as hard span boundaries so the corner is never cut, then
    round ONLY the dense spans (median seg well under the local bead width);
    sparse spans (straights, sharp-corner legs) pass through byte-for-byte.

    Discriminates WITHIN one region: a sharp vertex on sparse legs stays exact
    (identity, same tris), while a dense curved run rounds/decimates like a
    blind spline. Widths/heights ride along by arc-length interp (as
    `_resample_region`), and lo/hi region semantics are preserved."""
    P = sp[lo:hi].astype(float)
    n = len(P)
    if n < 3:
        return sp, w, ht, lo, hi
    wr = w[lo:hi].astype(float)
    hr = ht[lo:hi].astype(float)
    bead = float(np.median(wr)) if len(wr) else 1.0
    d = np.diff(P, axis=0)
    seglen = np.linalg.norm(d, axis=1)  # (n-1,)
    # pins: endpoints + sharp interior turns + degenerate (zero-length) joints
    pin = np.zeros(n, bool)
    pin[0] = pin[-1] = True
    for i in range(1, n - 1):
        la, lb = seglen[i - 1], seglen[i]
        if la < 1e-9 or lb < 1e-9:
            pin[i] = True
            continue
        c = float(np.dot(d[i - 1], d[i]) / (la * lb))
        c = max(-1.0, min(1.0, c))
        if math.degrees(math.acos(c)) >= pin_deg:
            pin[i] = True
    # per-segment density class; span boundaries = pins OR density transitions
    dense_seg = seglen < dense_factor * bead  # (n-1,) bool
    bnd = pin.copy()
    for i in range(1, n - 1):
        if dense_seg[i - 1] != dense_seg[i]:
            bnd[i] = True
    idx = np.nonzero(bnd)[0]
    out_p, out_w, out_h = [], [], []
    for k in range(len(idx) - 1):
        a, b = int(idx[k]), int(idx[k + 1])
        Ps, ws, hs = P[a : b + 1], wr[a : b + 1], hr[a : b + 1]
        is_dense = (b > a) and bool(np.all(dense_seg[a:b]))
        if is_dense and len(Ps) >= 3:
            rp, rw, rh = _catmull_span(Ps, ws, hs, step)
        else:
            rp, rw, rh = Ps, ws, hs
        if k > 0:  # drop shared pin
            rp, rw, rh = rp[1:], rw[1:], rh[1:]
        out_p.append(rp)
        out_w.append(rw)
        out_h.append(rh)
    turn = np.vstack(out_p).astype(np.float32)
    wn = np.concatenate(out_w).astype(np.float32)
    hn = np.concatenate(out_h).astype(np.float32)
    sp2 = np.vstack([sp[:lo], turn, sp[hi:]]).astype(np.float32)
    w2 = np.concatenate([w[:lo], wn, w[hi:]]).astype(np.float32)
    h2 = np.concatenate([ht[:lo], hn, ht[hi:]]).astype(np.float32)
    return sp2, w2, h2, lo, lo + len(turn)


def adaptive(step, pin_deg=35.0, dense_factor=0.75):
    return lambda sp, w, ht, lo, hi: _adaptive_region(
        sp, w, ht, lo, hi, step, pin_deg, dense_factor
    )


# label, spine transform, strand_collapse
APPROACHES = [
    ("off", _identity, False),
    ("sc0.5", _identity, {"max_snap_factor": 0.5}),
    ("sc1.0", _identity, {"max_snap_factor": 1.0}),
    ("spline3", spline(3.0), False),
    ("spline1.5", spline(1.5), False),
    ("spline3+sc0.5", spline(3.0), {"max_snap_factor": 0.5}),
    ("adaptive3", adaptive(3.0), False),
    ("adaptive3+sc0.5", adaptive(3.0), {"max_snap_factor": 0.5}),
]


# ------------------------------------------------------ browser measure ------
JS_MEASURE = r"""
(mode)=>{
  const v=window.threejsViewer, r=v._renderer, gl=r.getContext();
  r.render(v._scene, v._camera);
  const w=gl.drawingBufferWidth, h=gl.drawingBufferHeight;
  const px=new Uint8Array(w*h*4); gl.readPixels(0,0,w,h,gl.RGBA,gl.UNSIGNED_BYTE,px);
  const N=w*h; let foot=0, dark=0; const mask=new Uint8Array(N);
  for(let i=0;i<N;i++){
    const r0=px[i*4], g0=px[i*4+1], b0=px[i*4+2];
    if(Math.abs(r0-34)<=18 && Math.abs(g0-34)<=18 && Math.abs(b0-34)<=18) continue;
    foot++; mask[i]=1; if((r0+g0+b0)/3 < 70) dark++;
  }
  const out={foot,dark};
  if(mode==='solid'){
    if(!window.__baseFoot) window.__baseFoot=mask;
    const base=window.__baseFoot; let inter=0, uni=0;
    for(let i=0;i<N;i++){const a=mask[i],b=base[i]; if(a&&b)inter++; if(a||b)uni++;}
    out.iou = uni? inter/uni : 1.0;
  }
  return out;
}"""
JS_TRIS = (
    "()=>{const g=window.threejsViewer._objects.get('t').geometry;"
    "return {tris: g.index? g.index.count/3 : g.attributes.position.count/3,"
    " verts: g.attributes.position.count};}"
)


def free():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main():
    port = free()
    c = ViewerClient(port=port, open_browser=False)
    c._http_port = free()
    h = HTTPServer((c.host, c._http_port), _BlobHandler)
    h.blob_store = c._blob_store
    c._http_server = h
    threading.Thread(target=h.serve_forever, daemon=True).start()
    threading.Thread(target=c._run_server, daemon=True).start()

    results = {}
    with sync_playwright() as p:
        br = p.chromium.launch(headless=True)
        pg = br.new_page(viewport=VIEW)
        pg.goto(f"file://{c.viewer_path.resolve()}?ws_port={c.port}")
        c._connected_event.wait(timeout=15)
        for scene_name, scene_fn in SCENES:
            sc0 = scene_fn()
            d = sc0["ext"] / (2 * math.tan(math.radians(20))) * 1.05
            pg.evaluate(
                "()=>{window.__baseFoot=null;}"
            )  # reset per-scene contour baseline
            # line-only panel: the raw input spine as a polyline
            c.clear()
            c.add_polyline("t", points=sc0["sp"], color=0x2266DD, line_width=2)
            time.sleep(0.3)
            pg.wait_for_function(
                "()=>window.threejsViewer&&window.threejsViewer._objects.size>0",
                timeout=15000,
            )
            time.sleep(0.6)
            pg.evaluate(
                JS_CAM, [sc0["cx"], sc0["cy"], d, sc0["cx"], sc0["cy"], 0, 0, 1, 0]
            )
            pg.wait_for_timeout(300)
            pg.screenshot(path=str(OUT / f"zrm_{scene_name}_LINE.png"))
            rows = []
            for lab, xform, sc in APPROACHES:
                sp, w, ht, lo, hi = xform(
                    sc0["sp"].copy(),
                    sc0["w"].copy(),
                    sc0["ht"].copy(),
                    sc0["lo"],
                    sc0["hi"],
                )
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
                time.sleep(1.6)
                pg.evaluate(
                    JS_CAM, [sc0["cx"], sc0["cy"], d, sc0["cx"], sc0["cy"], 0, 0, 1, 0]
                )
                tri = pg.evaluate(JS_TRIS)
                pg.evaluate(JS_WIRE, 0)
                pg.wait_for_timeout(280)
                s = pg.evaluate(JS_MEASURE, "solid")
                pg.evaluate(JS_WIRE, 2)
                pg.wait_for_timeout(280)
                wm = pg.evaluate(JS_MEASURE, "wire")
                safe = lab.replace(" ", "").replace(".", "p").replace("+", "_")
                pg.screenshot(path=str(OUT / f"zrm_{scene_name}_{safe}.png"))
                rows.append(
                    (
                        lab,
                        len(sp),
                        tri["tris"],
                        s["foot"],
                        s["iou"],
                        wm["dark"],
                        wm["dark"] / max(s["foot"], 1),
                    )
                )
            results[scene_name] = rows
        br.close()
    c.disconnect()
    return results


GATE_IOU = 0.99


def gate(rows):
    """Success gate for one scene: each approach vs the scene's OFF baseline
    (rows[0]) must hold the contour (IoU >= GATE_IOU) while not ADDING
    triangles or intersecting lines. Returns {label: [failed measure names]}
    (empty list = PASS). This is the acceptance bar a candidate approach must
    clear on EVERY scene before promotion to viewer.js is worth discussing."""
    tris0, dark0 = rows[0][2], rows[0][5]
    out = {}
    for lab, _n, tris, _fp, iou, dp, _df in rows:
        failed = []
        if iou < GATE_IOU:
            failed.append("IoU")
        if tris > tris0:
            failed.append("tris")
        if dp > dark0:
            failed.append("darkPx")
        out[lab] = failed
    return out


if __name__ == "__main__":
    results = main()
    for scene_name, rows in results.items():
        off_dark = rows[0][5]
        print(
            f"\n=== {scene_name} ===  (lower darkPx=fewer intersecting lines; IoU~1=same contour; "
            f"fewer tris at equal quality=win)"
        )
        print(
            f"{'approach':>16} {'rings':>6} {'tris':>7} {'footPx':>8} {'IoU':>7} "
            f"{'darkPx':>8} {'darkFrac':>9} {'vs off':>7}"
        )
        for lab, n, tris, fp, iou, dp, df in rows:
            rel = f"{100 * dp / max(off_dark, 1):.0f}%"
            print(
                f"{lab:>16} {n:>6} {int(tris):>7} {fp:>8} {iou:>7.4f} {dp:>8} {df:>9.4f} {rel:>7}"
            )

    scene_names = list(results)
    verdicts = {sn: gate(results[sn]) for sn in scene_names}
    labs = [r[0] for r in results[scene_names[0]]]
    print(
        f"\n=== success gate ===  (per scene vs off: IoU >= {GATE_IOU}, tris <= off, darkPx <= off; "
        f"cell shows failed measures)"
    )
    print(
        f"{'approach':>16} "
        + " ".join(f"{sn:>10}" for sn in scene_names)
        + f" {'passed':>8}"
    )
    for lab in labs:
        cells, npass = [], 0
        for sn in scene_names:
            failed = verdicts[sn][lab]
            npass += not failed
            cells.append(f"{'PASS' if not failed else ','.join(failed):>10}")
        print(f"{lab:>16} " + " ".join(cells) + f" {f'{npass}/{len(scene_names)}':>8}")
