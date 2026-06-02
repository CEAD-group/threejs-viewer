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

Five objects sit in a grid, **each a distinct solid colour** so you can confirm
at a glance that a pick reports the *right* object (the printed ``id`` should
match the colour you clicked). Four are pickable — picking works on all of them
at once, the one nearest the cursor in screen space winning — and the fifth (a
grey **circle**) is added with ``pickable=False`` to demonstrate the per-object
opt-out: hovering or clicking it does nothing, yet it stays drawn like any other
line.

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
  * a grey **circle** added with ``pickable=False`` — the per-object opt-out.
    It renders identically to the other lines but is simply never hit-tested,
    so the marker won't latch onto it and a click on it sends nothing back.
    (Picking is opt-out: every ``add_polyline`` / ``add_parametric_tube``
    defaults to ``pickable=True``.)

Each pick is also logged to this terminal at DEBUG level (the full raw payload)
on top of the human-readable summary — see ``on_pick`` below. That logging
lives entirely in this example; it needs no change to the viewer. Browser-side
*console* logging is a different matter: the only browser-side hook is JS
(``viewer.onPolylinePick`` / ``onPolylineHover``), which has to be registered in
the page, and this Python example has no channel to inject JS into the browser —
so console logging there would require adding an "eval JS" message to the viewer
(out of scope). Note hovers never reach Python at all (only clicks round-trip),
so this terminal log shows clicks only.

Run: uv run python examples/22_polyline_picking.py
(then hover/click in the browser; press Ctrl+C in the terminal to quit)
"""

import logging
import time

import numpy as np

from threejs_viewer import viewer

# Debug logging, scoped to this example's own logger so it doesn't drag in the
# websockets library's (very chatty) DEBUG frames. INFO = the readable one-liner
# per pick, DEBUG = the full raw payload dict. All of this is example-local — the
# viewer is untouched.
log = logging.getLogger("pick-demo")
log.setLevel(logging.DEBUG)
_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)-5s [%(name)s] %(message)s", "%H:%M:%S")
)
log.addHandler(_handler)
log.propagate = False


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


def circle_path(radius=1.6, n=72):
    """A closed circle standing in the XZ plane (n+1 vertices). Used for the
    non-pickable object — a clean, obviously-interactive-looking shape that
    nonetheless never responds to the cursor."""
    ang = np.linspace(0.0, 2.0 * np.pi, n + 1)
    x = radius * np.cos(ang)
    z = radius * np.sin(ang)
    y = np.zeros_like(ang)
    return np.column_stack([x, y, z])


def placed(pts, x_offset, y_offset=0.0):
    """Centre a path on its own origin, then shift it to grid cell
    ``(x_offset, y_offset)`` in the ground plane so the objects sit in a grid
    without overlapping (every object still climbs in +Z)."""
    pts = np.asarray(pts, dtype=np.float64)
    pts = pts - pts.mean(axis=0)
    pts[:, 0] += x_offset
    pts[:, 1] += y_offset
    return pts.astype(np.float32)


v = viewer()
v.clear()
v.unload_animation()

# Five objects in a 3x2 grid (X across, Y back; every object climbs in +Z).
# Each gets its own solid colour so a pick's reported `id` is visually
# verifiable. Grid cell (x, y) per object:
GRID = {
    "square": (-15.0, 7.0),
    "mixed": (0.0, 7.0),
    "vase": (15.0, 7.0),
    "bead": (-15.0, -7.0),
    "circle": (0.0, -7.0),
}
# Distinct solid colours. The grey circle signals "inert" — it's the
# non-pickable one.
COLORS = {
    "square": 0xFF4444,  # red
    "mixed": 0x33DD55,  # green
    "vase": 0x4488FF,  # blue
    "bead": 0xFFAA33,  # orange
    "circle": 0x888888,  # grey — not pickable
}

# Four polylines: square, mixed-detail, dense vase (all pickable) + a grey
# circle added with pickable=False to prove the per-object opt-out.
line_paths = {
    "square": square_path(),
    "mixed": mixed_detail_path(),
    "vase": vase_path(),
    "circle": circle_path(),
}
for line_id, pts in line_paths.items():
    gx, gy = GRID[line_id]
    v.add_polyline(
        line_id,
        placed(pts, gx, gy),
        color=COLORS[line_id],
        line_width=4,
        pickable=(line_id != "circle"),  # the circle opts out of picking
    )

# A parametric tube (the "bead"). Picking works on it too — the click resolves
# a point on the tube's full-resolution spine and reports kind="tube". The
# screen gate widens to the bead's body, so you can click anywhere on the
# extrusion (not just its centre-line).
bead_spine, bead_w, bead_h = helix_bead()
gx, gy = GRID["bead"]
v.add_parametric_tube(
    "bead",
    placed(bead_spine, gx, gy),
    bead_w,
    bead_h,
    color=COLORS["bead"],
    roughness=0.5,
)


# --- Receive picks from the browser -----------------------------------------
# The callback runs on the client's WebSocket thread; it's fine to call other
# viewer methods (like add_sphere) from inside it.
_pick_count = 0


def on_pick(pick):
    global _pick_count
    x, y, z = pick["point"]
    # Human-readable summary (INFO) + the full raw payload (DEBUG). Only clicks
    # reach Python, so this fires once per click — never on hover.
    log.info(
        "pick #%d: %s %r fraction=%.3f (%.1f%% along) "
        "point=(%.3f, %.3f, %.3f) [segment %d, t=%.3f]",
        _pick_count,
        pick["kind"],
        pick["id"],
        pick["fraction"],
        pick["fraction"] * 100,
        x,
        y,
        z,
        pick["segment"],
        pick["t"],
    )
    log.debug("raw pick payload: %r", pick)
    # Drop a persistent (and chunky, so it's easy to see) marker on the spine
    # where we picked.
    v.add_sphere(
        f"pick_{_pick_count}",
        radius=0.4,
        color=0xFF3344,
        roughness=0.4,
        position=pick["point"],
    )
    _pick_count += 1


# Registering the callback also enables picking in the viewer.
v.on_polyline_pick(on_pick)

total_pts = sum(len(p) for p in line_paths.values()) + len(bead_spine)
print(__doc__)
print(
    f"Loaded 4 lines (1 non-pickable) + 1 bead ({total_pts:,} points total): "
    "square, mixed, vase, circle [not pickable], bead."
)
print("Picking enabled. Hover for the marker; click to pick a point on any object.")
print("The grey circle ignores the cursor — clicking it sends nothing back.")
print("Press F to frame the scene, D / Shift+D for depth cues. Ctrl+C here to quit.")

# Keep the process (and the WebSocket server) alive so picks keep arriving.
try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nDone.")
finally:
    v.disconnect()
