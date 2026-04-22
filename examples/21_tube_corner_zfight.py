"""
Miter-frame demo: wide beads around sharp corners.

Parametric tubes now build their cross-section frames as the angle bisector
of the incoming/outgoing segment directions, and scale the width axis by
1/cos(half_turn_angle) — the same miter math SVG stroke rendering uses, with
a miter limit of 4 falling back to bevel (scale=1) past the limit.

Two tubes side-by-side, both using the same miter code path:

* Left  — moderate corners (radius ≈ bead half-width). Miter handles this
  cleanly: each quad fans along the bisector, no ring overlap.
* Right — aggressive corners (radius << bead half-width). Densely sampled
  through the arc so per-segment turn is tiny — miter still applies, but the
  inside offset curve of the bead is *geometrically* inverted at each corner
  (curvature > 1/half-width). Expect a clean outside surface and a visible
  crease on the inside: that inside crease is a real self-intersection no
  frame trick can fix; miter keeps the *outside* from exploding into a
  triangle fan the way the pre-miter build did.

Run: uv run python examples/21_tube_corner_zfight.py
"""

import time

import numpy as np

from threejs_viewer import viewer


def rounded_square(side=2.0, radius=0.3, pts_per_corner=50, pts_per_side=8):
    """Closed rounded-corner square, densely sampled through each corner arc."""
    half = side / 2
    corners = [
        ((+half - radius, +half - radius), 0.0),
        ((-half + radius, +half - radius), np.pi / 2),
        ((-half + radius, -half + radius), np.pi),
        ((+half - radius, -half + radius), 3 * np.pi / 2),
    ]
    pts = []
    for i, ((cx, cy), a0) in enumerate(corners):
        if i > 0:
            prev_cx, prev_cy = corners[i - 1][0]
            prev_a_end = corners[i - 1][1] + np.pi / 2
            start = (
                prev_cx + radius * np.cos(prev_a_end),
                prev_cy + radius * np.sin(prev_a_end),
            )
            end = (cx + radius * np.cos(a0), cy + radius * np.sin(a0))
            t = np.linspace(0, 1, pts_per_side, endpoint=False)[1:]
            xs = start[0] + (end[0] - start[0]) * t
            ys = start[1] + (end[1] - start[1]) * t
            pts.append(np.column_stack([xs, ys]))
        theta = np.linspace(a0, a0 + np.pi / 2, pts_per_corner)
        xs = cx + radius * np.cos(theta)
        ys = cy + radius * np.sin(theta)
        pts.append(np.column_stack([xs, ys]))
    prev_cx, prev_cy = corners[-1][0]
    prev_a_end = corners[-1][1] + np.pi / 2
    start = (
        prev_cx + radius * np.cos(prev_a_end),
        prev_cy + radius * np.sin(prev_a_end),
    )
    first_cx, first_cy = corners[0][0]
    end = (first_cx + radius * np.cos(0.0), first_cy + radius * np.sin(0.0))
    t = np.linspace(0, 1, pts_per_side, endpoint=False)[1:]
    xs = start[0] + (end[0] - start[0]) * t
    ys = start[1] + (end[1] - start[1]) * t
    pts.append(np.column_stack([xs, ys]))

    xy = np.concatenate(pts, axis=0)
    z = np.zeros(len(xy))
    return np.column_stack([xy[:, 0], xy[:, 1], z]).astype(np.float32)


def hue_ramp(n, offset=0.0):
    """HSV sweep so ring order through the corners is visible."""
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

SIDE = 2.0
BEAD_W = 1.00
BEAD_H = 0.30
PTS_PER_CORNER = 60

MODERATE_R = 0.30  # half-width 0.25 ≲ radius 0.30 → miter handles cleanly
AGGRESSIVE_R = 0.08  # half-width 0.25 >> radius 0.08 → miter clamped to limit

v.add_box(
    "ground",
    width=SIDE * 3.5,
    height=SIDE + 1.0,
    depth=0.02,
    color=0x1A1A1A,
    position=[0.0, 0.0, -0.01],
)

# Left: moderate corner radius — miter produces clean corners.
spine_mod = rounded_square(
    side=SIDE, radius=MODERATE_R, pts_per_corner=PTS_PER_CORNER, pts_per_side=8
)
spine_mod[:, 0] -= SIDE * 0.9
n_mod = len(spine_mod)
v.add_parametric_tube(
    "tube_moderate",
    spine=spine_mod,
    widths=np.full(n_mod, BEAD_W, dtype=np.float32),
    heights=np.full(n_mod, BEAD_H, dtype=np.float32),
    colors=hue_ramp(n_mod),
    roughness=0.35,
    metalness=0.05,
)

# Right: aggressive corner radius — miter clamps at the limit, bevelled.
spine_agg = rounded_square(
    side=SIDE, radius=AGGRESSIVE_R, pts_per_corner=PTS_PER_CORNER, pts_per_side=8
)
spine_agg[:, 0] += SIDE * 0.9
n_agg = len(spine_agg)
v.add_parametric_tube(
    "tube_aggressive",
    spine=spine_agg,
    widths=np.full(n_agg, BEAD_W, dtype=np.float32),
    heights=np.full(n_agg, BEAD_H, dtype=np.float32),
    colors=hue_ramp(n_agg),
    roughness=0.35,
    metalness=0.05,
)

print(f"moderate:   corner radius {MODERATE_R:.2f}, bead half-width {BEAD_W / 2:.3f}")
print(f"aggressive: corner radius {AGGRESSIVE_R:.2f}, bead half-width {BEAD_W / 2:.3f}")
print("Both tubes use miter frames. Compare the left (clean) vs right")
print("(bevelled, miter clamped at limit=4). Ctrl+C to exit.")

try:
    while True:
        time.sleep(1.0)
except KeyboardInterrupt:
    pass
