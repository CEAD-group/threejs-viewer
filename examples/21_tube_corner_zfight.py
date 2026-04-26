"""
Side-by-side demo of ``strand_collapse`` on ``add_parametric_tube``.

The spine is a centripetal Catmull-Rom spline through random polar
control points — a smooth closed curve with continuously varying
curvature that peaks well above 1/half_width in the tight lobes. Sparse
sampling (10 points per segment) is deliberate: it matches the feel of
a real 3D-printed toolpath where per-segment turn is often > 90° in
tight corners.

Both tiles use the same pathological spine (κ·W/2 up to ~9, so the
inner offset curve self-intersects aggressively):

    (left)   strand_collapse=False   — baseline miter (visible folds)
    (right)  strand_collapse=True    — fold detect + snap

``strand_collapse`` scans each per-cross-section-vertex strand polyline
with a sliding window of 10 rings, using a point-to-segment distance
test against a 5%-of-max-cross-section tolerance. Detected fold runs
collapse to their centroid, turning the inner self-intersection into a
clean crease instead of a self-intersecting triangle fan.

Run: uv run python examples/21_tube_corner_zfight.py
"""

import time

import numpy as np

from threejs_viewer import viewer


def blobby_control_points(n, base_r, jitter, seed):
    """Random polar control points on a closed loop: base radius + jitter."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    theta = theta + rng.uniform(-0.35, 0.35, size=n) * (2.0 * np.pi / n)
    r = base_r * (1.0 + rng.uniform(-jitter, jitter, size=n))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def catmull_rom_closed(points, samples_per_segment, alpha=0.5):
    """Centripetal Catmull-Rom through a closed loop of 2D control points."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)

    def knot(ti, a, b):
        d = float(np.linalg.norm(b - a))
        return ti + max(d, 1e-9) ** alpha

    out = []
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        t0 = 0.0
        t1 = knot(t0, p0, p1)
        t2 = knot(t1, p1, p2)
        t3 = knot(t2, p2, p3)
        ts = np.linspace(t1, t2, samples_per_segment, endpoint=False)
        for t in ts:
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            c = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
            out.append(c)
    return np.asarray(out, dtype=np.float64)


def lift_xy(xy, dx=0.0, dy=0.0):
    xyz = np.column_stack([xy[:, 0] + dx, xy[:, 1] + dy, np.zeros(len(xy))])
    return xyz.astype(np.float32)


def hue_ramp(n, offset=0.0):
    """HSV sweep so ring order along the spine is visible."""
    h = (np.linspace(0, 1, n) + offset) % 1.0
    s, v = 0.9, 0.95
    hp = h * 6.0
    c = v * s
    x = c * (1 - np.abs(np.mod(hp, 2) - 1))
    m = v - c
    r = np.zeros(n)
    g = np.zeros(n)
    b = np.zeros(n)
    for i, hi in enumerate(hp):
        ri, gi, bi = {
            0: (c, x[i], 0),
            1: (x[i], c, 0),
            2: (0, c, x[i]),
            3: (0, x[i], c),
            4: (x[i], 0, c),
            5: (c, 0, x[i]),
        }[int(hi) % 6]
        r[i], g[i], b[i] = ri + m, gi + m, bi + m
    r8 = (r * 255).astype(np.uint32)
    g8 = (g * 255).astype(np.uint32)
    b8 = (b * 255).astype(np.uint32)
    return (r8 << 16) | (g8 << 8) | b8


v = viewer()
v.clear()

BEAD_W = 0.70
BEAD_H = 0.30
BASE_R = 1.10
SAMPLES_PER_SEG = 10  # sparse on purpose — realistic toolpath density

ctrl = blobby_control_points(n=13, base_r=BASE_R, jitter=0.45, seed=7)
spine_xy = catmull_rom_closed(ctrl, samples_per_segment=SAMPLES_PER_SEG)
n = len(spine_xy)
colors = hue_ramp(n)
widths = np.full(n, BEAD_W, dtype=np.float32)
heights = np.full(n, BEAD_H, dtype=np.float32)

# Side-by-side layout.
offset = BASE_R * 1.5
ground_w = 2.0 * offset + 2.0 * BASE_R * (1.0 + 0.45) + 0.6
v.add_box(
    "ground",
    width=ground_w,
    height=2.0 * BASE_R * (1.0 + 0.45) + 0.6,
    depth=0.02,
    color=0x1A1A1A,
    position=[0.0, 0.0, -0.01],
)

for tube_id, dx, strand_collapse in [
    ("tube_baseline", -offset, False),
    ("tube_collapse", +offset, True),
]:
    spine = lift_xy(spine_xy, dx=dx, dy=0.0)
    v.add_parametric_tube(
        tube_id,
        spine=spine,
        widths=widths,
        heights=heights,
        colors=colors,
        roughness=0.35,
        metalness=0.05,
        strand_collapse=strand_collapse,
    )

print(f"spine: {n} samples/tile, bead half-width {BEAD_W / 2:.3f}")
print("  left:  strand_collapse=False  (baseline)")
print("  right: strand_collapse=True   (sliding-window fold detect + snap)")
print("Ctrl+C to exit.")

try:
    while True:
        time.sleep(1.0)
except KeyboardInterrupt:
    pass
