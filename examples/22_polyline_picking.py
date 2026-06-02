"""
Pick a point anywhere along a polyline — from the browser back to Python.

This demonstrates the round trip the viewer normally doesn't do: an event that
originates in the browser (a click on a 3D line) travelling *back* to the
Python process. Enable it with ``enable_polyline_picking()`` /
``on_polyline_pick(callback)``:

  * Hover the cursor near any line. A marker glides to the closest point on it
    and a readout shows the arc-length **fraction** (0–100 %) and the world
    coordinate under the cursor. **Movement is continuous** — the marker slides
    smoothly along the wire and through corners, never snapping to a vertex.
  * Click (a plain left click — dragging still orbits the camera) and the
    picked location is sent to Python. Here the callback prints which object
    was hit (its ``kind`` — "line" or "tube" — and id) plus the fraction/point,
    and drops a red sphere right on the spine, so you can watch the full
    browser → Python → browser loop close.

Four objects of increasing density share the scene — picking works on all of
them at once (the one nearest the cursor in screen space wins):

  * a **square** from just 4 corner points (closed to a 4-edge loop). With
    widely-spaced nodes it's the clearest proof that picking is *continuous*:
    the marker glides down each edge and through each corner without snapping.
    ``fraction`` is arc length around the perimeter, so corners land at 0.25 /
    0.50 / 0.75 / 1.0.
  * a **mixed-detail** line that alternates long, sparse straight runs (two
    points each) with bursts of densely-sampled fine detail (two tight coils
    and a high-frequency ripple). Picking treats both the same — it lands on
    the nearest point of the actual wire regardless of how it was sampled — and
    because ``fraction`` is arc length, the coiled bursts (tiny in space, long
    along the wire) claim a big share of 0→1 while the straight runs barely move
    it.
  * a dense, self-overlapping **spiral vase** (the ball-of-yarn toolpath from
    ``20_line_depth_cues.py``): ripply windings stacked into a vase silhouette.
    Picking a point out of a tangled bundle is the interesting case — and the
    ``D`` / ``Shift+D`` depth cues (fog / eye-dome lighting) help read its
    front-to-back ordering while you aim.
  * a **bead** — a parametric tube (``add_parametric_tube``) on a helix spine.
    Picking works on the extruded body, not just lines: the click resolves a
    point on the tube's full-resolution spine and reports ``kind="tube"``. That
    spine index lines up 1:1 with the per-spine-point arrays you built the tube
    from, so it's the hook for reading *other* per-point data at the pick.

Every line is coloured *by* its own arc-length fraction (turbo), so the colour
under the cursor is roughly the value you'll get back when you pick there.

Run: uv run python examples/22_polyline_picking.py
(then hover/click in the browser; press Ctrl+C in the terminal to quit)
"""

import time

import numpy as np

from threejs_viewer import viewer


def square_path(half=1.6):
    """A square from 4 corner points, standing in the XZ plane, closed back to
    the first corner → 5 vertices / 4 long edges."""
    corners = np.array(
        [
            [-half, 0.0, -half],
            [half, 0.0, -half],
            [half, 0.0, half],
            [-half, 0.0, half],
        ],
        dtype=np.float64,
    )
    return np.vstack([corners, corners[:1]])  # close the loop


def mixed_detail_path():
    """A single open polyline climbing in +Z that alternates sparse straight
    runs with densely-sampled detail (two coils + a ripple)."""
    parts = []
    pos = np.zeros(3, dtype=np.float64)

    def add(pts):
        nonlocal pos
        pts = np.asarray(pts, dtype=np.float64)
        parts.append(pts if not parts else pts[1:])  # drop duplicated joint
        pos = pts[-1].copy()

    def straight(dz, n=2):
        """A long, fairly straight run sampled with just `n` points."""
        to = pos + np.array([0.0, 0.0, dz])
        add(np.linspace(pos, to, n))

    def coil(length, radius, turns, n):
        """A tight helix climbing along +Z — short in Z, long along the wire."""
        t = np.linspace(0.0, 1.0, n)
        ang = t * turns * 2.0 * np.pi
        x = pos[0] + radius * np.sin(ang)
        y = pos[1] + radius * (1.0 - np.cos(ang))  # starts/ends back on the axis
        z = pos[2] + t * length
        add(np.column_stack([x, y, z]))

    def ripple(length, amp, freq, n):
        """A dense high-frequency wiggle in X, fading in/out at the ends."""
        t = np.linspace(0.0, 1.0, n)
        x = pos[0] + amp * np.sin(t * freq * 2.0 * np.pi) * np.sin(t * np.pi)
        y = pos[1] + np.zeros_like(t)
        z = pos[2] + t * length
        add(np.column_stack([x, y, z]))

    straight(1.6, n=2)  # long straight run (2 points)
    coil(2.4, 0.7, 6, 1400)  # tight spring (1400 points)
    straight(2.2, n=2)  # long straight run
    ripple(2.0, 0.9, 14, 1100)  # dense high-frequency ripple
    straight(1.6, n=2)  # long straight run
    coil(1.8, 0.6, 5, 1000)  # second tight spring
    return np.vstack(parts)


def vase_path(n_points=12_000, n_turns=60, height=8.0):
    """A ripply spiral-vase toolpath (the ball-of-yarn case from example 20,
    scaled to a pickable point count): `n_turns` windings stacked into a vase
    silhouette, with multi-frequency angular ripples so the single line overlaps
    itself heavily front-to-back."""
    t = np.linspace(0.0, 1.0, n_points)
    angle = t * n_turns * 2.0 * np.pi
    profile = (
        2.0
        - 0.9 * np.sin(t * np.pi) ** 2  # pinch the waist
        + 1.1 * np.sin(t * np.pi * 0.5) ** 3  # swell the belly / rim
    )
    radius = profile + 0.16 * np.sin(angle * 7.0) + 0.07 * np.sin(angle * 23.0)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = t * height
    return np.column_stack([x, y, z])


def helix_bead(n=400, radius=1.2, turns=3.0, height=4.0):
    """A helix spine + per-point width/height for a parametric tube (the
    "bead"), with a gentle width taper so it reads as a 3D extrusion. Returns
    (spine (N,3), widths (N,), heights (N,))."""
    t = np.linspace(0.0, 1.0, n)
    ang = t * turns * 2.0 * np.pi
    spine = np.column_stack([radius * np.cos(ang), radius * np.sin(ang), t * height])
    widths = (0.45 - 0.25 * t).astype(np.float32)  # taper 0.45 → 0.20
    heights = np.full(n, 0.30, dtype=np.float32)
    return spine, widths, heights


def arclength_fraction(pts):
    """Cumulative arc length normalised to [0, 1] — matches the picked
    ``fraction``, so colouring by it makes the value readable off the line."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    return (cum / total if total > 0 else cum).astype(np.float32)


def placed(pts, x_offset):
    """Centre a path on its own origin, then shift it to a column at `x_offset`
    so the objects sit side by side without overlapping."""
    pts = np.asarray(pts, dtype=np.float64)
    pts = pts - pts.mean(axis=0)
    pts[:, 0] += x_offset
    return pts.astype(np.float32)


v = viewer()
v.clear()
v.unload_animation()

# Line columns, left → right: square, mixed-detail line, dense spiral vase.
# (A parametric-tube "bead" is added as a fourth column just below.)
lines = {
    "square": placed(square_path(), x_offset=-10.0),
    "mixed": placed(mixed_detail_path(), x_offset=0.0),
    "vase": placed(vase_path(), x_offset=11.0),
}
for line_id, pts in lines.items():
    v.add_polyline(
        line_id,
        pts,
        colors=arclength_fraction(pts),
        colormap="turbo",
        line_width=4,
    )

# A fourth column: a parametric tube (the "bead"). Picking works on it too —
# the click resolves a point on the tube's full-resolution spine and reports
# kind="tube". The screen gate widens to the bead's body, so you can click
# anywhere on the extrusion (not just its centre-line).
bead_spine, bead_w, bead_h = helix_bead()
v.add_parametric_tube(
    "bead",
    placed(bead_spine, x_offset=22.0),
    bead_w,
    bead_h,
    color=0x44AACC,
    roughness=0.5,
)


# --- Receive picks from the browser -----------------------------------------
# The callback runs on the client's WebSocket thread; it's fine to call other
# viewer methods (like add_sphere) from inside it.
_pick_count = 0


def on_pick(pick):
    global _pick_count
    x, y, z = pick["point"]
    print(
        f"pick #{_pick_count}: {pick['kind']} {pick['id']!r} "
        f"fraction={pick['fraction']:.3f} ({pick['fraction'] * 100:.1f}% along) "
        f"point=({x:.3f}, {y:.3f}, {z:.3f})  "
        f"[segment {pick['segment']}, t={pick['t']:.3f}]"
    )
    # Drop a persistent marker on the spine where we picked.
    v.add_sphere(
        f"pick_{_pick_count}",
        radius=0.12,
        color=0xFF3344,
        roughness=0.4,
        position=pick["point"],
    )
    _pick_count += 1


# Registering the callback also enables picking in the viewer.
v.on_polyline_pick(on_pick)

total_pts = sum(len(p) for p in lines.values()) + len(bead_spine)
print(__doc__)
print(
    f"Loaded 3 lines + 1 bead ({total_pts:,} points total): square, mixed, vase, bead."
)
print("Picking enabled. Hover for the marker; click to pick a point on any object.")
print("Press F to frame the scene, D / Shift+D for depth cues. Ctrl+C here to quit.")

# Keep the process (and the WebSocket server) alive so picks keep arriving.
try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nDone.")
finally:
    v.disconnect()
