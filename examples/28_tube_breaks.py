"""
Split one tube at interior discontinuities with ``add_parametric_tube(break_before=...)``.

One spine visits three separate rings with travel hops between them. Without a mask
(front row) the ribbon bridges each hop with a stray cone. With ``break_before`` set
on the first point of each new part (back row) it renders as three flat-capped strips.

Run: uv run python examples/28_tube_breaks.py
"""

import numpy as np

from threejs_viewer import Toolpath, viewer


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
colors = Toolpath.from_points(spine, widths, heights).colorize("turbo").packed_colors

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

v.add_grid(
    "floor", cell_size=1.0, extent=20.0, center_color=0x888888, position=[0, 0, -0.3]
)
v.set_view("iso", animate=False)
v.frame_points(
    np.array(np.meshgrid(*zip([-4, -5, -0.5], [4, 5, 0.5]))).reshape(3, -1).T, fill=0.85
)
v.wait_for_assets()
