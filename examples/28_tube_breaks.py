"""
Parametric Tube Breaks — split one tube at interior discontinuities

Demonstrates ``add_parametric_tube(break_before=...)`` (issue #107): a single
tube whose spine visits several *separate* parts with rapid travel hops between
them. Without a break mask the extruded ribbon stitches straight across each
hop, drawing a stray cone through empty space where no material was deposited.
The break mask cuts the ribbon at those hops so each part renders as its own
disconnected strip, and both open ends of every break get a flat cap so the
strips read as solid.

``break_before`` is an (N,) bool/uint8 mask: a non-zero entry at index ``i``
breaks the ribbon *before* spine point ``i``. Here we mark the first point of
each new part (the landing point of a travel hop).

The scene shows two tubes side by side: the left one bridged (no mask), the
right one broken + capped (same spine + mask). Breaks also survive LOD.

Run: uv run python examples/28_tube_breaks.py
"""

import time

import numpy as np

from threejs_viewer import viewer


def turbo_rgb(values):
    """values in [0, 1] → uint32 0x00RRGGBB (cheap turbo-ish ramp)."""
    v = np.clip(values, 0.0, 1.0)
    r = (np.clip(1.5 - abs(4 * v - 3), 0, 1) * 255).astype(np.uint32)
    g = (np.clip(1.5 - abs(4 * v - 2), 0, 1) * 255).astype(np.uint32)
    b = (np.clip(1.5 - abs(4 * v - 1), 0, 1) * 255).astype(np.uint32)
    return (r << 16) | (g << 8) | b


def arc(cx, cy, radius, a0, a1, n):
    a = np.linspace(a0, a1, n)
    return np.column_stack(
        [cx + radius * np.cos(a), cy + radius * np.sin(a), np.zeros(n)]
    ).astype(np.float32)


# Three near-closed rings laid left-to-right, drawn as ONE spine: the tool
# traces ring 1, hops to ring 2, traces it, hops to ring 3, traces it. Each hop
# is an interior discontinuity — the landing point starts a new part.
parts = [
    arc(-2.6, 0.0, 1.0, 0.0, 1.75 * np.pi, 60),
    arc(0.0, 0.0, 1.0, 0.0, 1.75 * np.pi, 60),
    arc(2.6, 0.0, 1.0, 0.0, 1.75 * np.pi, 60),
]
spine = np.concatenate(parts).astype(np.float32)
n = spine.shape[0]

widths = np.full(n, 0.32, dtype=np.float32)
heights = np.full(n, 0.32, dtype=np.float32)
colors = turbo_rgb(np.linspace(0, 1, n).astype(np.float32))

# Break before the first point of parts 2 and 3 (indices 60 and 120).
break_before = np.zeros(n, dtype=bool)
break_before[60] = True
break_before[120] = True

v = viewer()
v.clear()

# Left: no mask → the two travel hops bridge across empty space (stray cones).
v.add_parametric_tube(
    "bridged",
    spine=spine - np.array([0.0, 3.0, 0.0], dtype=np.float32),
    widths=widths,
    heights=heights,
    colors=colors,
    lod=False,
)

# Right: same spine + break mask → three disconnected, flat-capped strips.
v.add_parametric_tube(
    "broken",
    spine=spine + np.array([0.0, 3.0, 0.0], dtype=np.float32),
    widths=widths,
    heights=heights,
    colors=colors,
    break_before=break_before,
    lod=False,
)

v.wait_for_assets()
print("Top row: break_before splits the tube at each travel hop (capped ends).")
print("Bottom row: no mask — stray cones bridge the hops.")
print("Close the viewer window to exit.")
while True:
    time.sleep(1.0)
