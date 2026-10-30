"""
Pick a point along a polyline or tube and send it back from the browser to Python.

Hover near an object: a marker glides along it (continuously, never snapping to a
vertex) and a readout shows the arc-length fraction. A plain left click sends the
pick to Python, which drops a sphere in the object's colour on that spot. The grey
circle is added with ``pickable=False`` and is never hit-tested.

Run: uv run python examples/22_polyline_picking.py   (Ctrl+C to quit)
"""

import time

import numpy as np

from threejs_viewer import viewer

COLORS = {"square": 0xFF4444, "mixed": 0x33DD55, "vase": 0x4488FF, "bead": 0xFFAA33}

v = viewer()
v.clear()
v.add_grid("floor", cell_size=1.0, extent=40.0, center_color=0x888888)


def at(pts, x, y):
    """Center a path on its own origin, then place it at (x, y) just above the floor."""
    pts = np.asarray(pts, dtype=np.float64)
    pts = pts - pts.mean(axis=0)
    return (pts + [x, y, 0.0 - pts[:, 2].min() + 0.3]).astype(np.float32)


# Four corners only: widely spaced nodes show the marker is continuous.
h = 1.6
square = np.array([[-h, 0, -h], [h, 0, -h], [h, 0, h], [-h, 0, h], [-h, 0, -h]])

# Straight runs around a dense coil: fraction follows arc length, not extent.
t = np.linspace(0, 1, 1200)
coil = np.column_stack([0.7 * np.sin(t * 30), 0.7 * (1 - np.cos(t * 30)), 2.4 * t])
mixed = np.vstack([[[0, 0, -1.6]], coil, coil[-1] + [[0, 0, 1.6]]])

# A tangled ripply spiral vase, the hard case for picking out of a bundle.
t = np.linspace(0, 1, 12_000)
a = t * 60 * 2 * np.pi
r = 2.0 - 0.9 * np.sin(t * np.pi) ** 2 + 1.1 * np.sin(t * np.pi / 2) ** 3
r += 0.16 * np.sin(7 * a) + 0.07 * np.sin(23 * a)
vase = np.column_stack([r * np.cos(a), r * np.sin(a), 8 * t])

circle_a = np.linspace(0, 2 * np.pi, 73)
circle = np.column_stack([1.6 * np.cos(circle_a), 0 * circle_a, 1.6 * np.sin(circle_a)])

for name, pts, x, y in [
    ("square", square, -8, 5),
    ("mixed", mixed, 0, 5),
    ("vase", vase, 8, 5),
    ("circle", circle, 0, -5),
]:
    v.add_polyline(
        name,
        at(pts, x, y),
        color=COLORS.get(name, 0x888888),
        line_width=4,
        pickable=name != "circle",
    )

# Picking also works on tubes: the click resolves a point on the full-resolution spine.
t = np.linspace(0, 1, 400)
helix = np.column_stack([1.2 * np.cos(t * 19), 1.2 * np.sin(t * 19), 4 * t])
v.add_parametric_tube(
    "bead",
    at(helix, -8, -5),
    (0.45 - 0.25 * t).astype(np.float32),
    np.full(400, 0.3, dtype=np.float32),
    color=COLORS["bead"],
)
v.set_camera(position=[0, -22, 14], target=[0, 0, 2.5])

count = 0


def on_pick(pick):
    global count
    print(f"{pick['kind']} {pick['id']!r} at {pick['fraction']:.1%}  {pick['point']}")
    v.add_sphere(
        f"pick_{count}",
        radius=0.4,
        color=COLORS.get(pick["id"], 0xFFFFFF),
        position=pick["point"],
    )
    count += 1


v.on_polyline_pick(on_pick)  # also enables picking
print("Hover and click objects in the browser. Ctrl+C to quit.")
try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
finally:
    v.disconnect()
