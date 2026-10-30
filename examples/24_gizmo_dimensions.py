"""
1-DOF, 2-DOF and 3-DOF move gizmos pinned at once, in world and object-local space.

Columns are the degrees of freedom (Z rail, XY plane, free). The back row uses
``space="world"``; the front row blocks are tilted and use ``space="local"`` so the
handles turn with them. A translucent ghost marks the grab-time pose during a drag.
Pinned gizmos use ``snap_default=True``: drags snap to 0.5 / 15 degrees and Shift
(Ctrl on Windows) moves freely; Alt (Shift on Windows) rotates.

Run: uv run python examples/24_gizmo_dimensions.py   (Ctrl+C to quit)
"""

import time

import numpy as np

from threejs_viewer import viewer

RAIL, TILE, CUBE = 0x4A90E2, 0x43C873, 0xEF8A3A
TILT = [0.45, 0.0, 0.6]  # Euler radians, so the local gizmos visibly tilt

v = viewer()
v.clear()
v.add_grid("floor", cell_size=1.0, extent=20.0, center_color=0x888888)

# (id, color, size w/h/d, position, rotation, gizmo axes, space)
blocks = [
    ("rail", RAIL, (0.8, 0.8, 2.0), [-3.5, 2, 1.0], None, (0, 0, 1), "world"),
    ("tile", TILE, (1.6, 1.6, 0.3), [0, 2, 0.15], None, (1, 1, 0), "world"),
    ("cube", CUBE, (1.2, 1.2, 1.2), [3.5, 2, 0.6], None, (1, 1, 1), "world"),
    ("rail_l", RAIL, (0.8, 0.8, 2.0), [-3.5, -2, 1.3], TILT, (0, 0, 1), "local"),
    ("tile_l", TILE, (1.6, 1.6, 0.3), [0, -2, 1.0], TILT, (1, 1, 0), "local"),
    ("cube_l", CUBE, (1.2, 1.2, 1.2), [3.5, -2, 1.0], TILT, (1, 1, 1), "local"),
]
for name, color, (w, h, d), pos, rot, (x, y, z), space in blocks:
    v.add_box(name, width=w, height=h, depth=d, color=color, position=pos, rotation=rot)
    v.add_gizmo(name, x=bool(x), y=bool(y), z=bool(z), space=space, snap_default=True)


def on_move(m):
    if m["phase"] == "end":
        x, y, z = m["position"]
        print(f"{m['id']} released at ({x:.2f}, {y:.2f}, {z:.2f})")


# Register after pinning so no extra click-select gizmo is drawn.
v.enable_move_gizmo(translate_snap=0.5, rotate_snap_deg=15, click_select=False)
v.on_object_move(on_move)
v.set_view("iso", animate=False)
v.frame_points(
    np.array(np.meshgrid(*zip([-4.5, -3, 0], [4.5, 3, 2.5]))).reshape(3, -1).T,
    fill=0.85,
)

print("Drag the blocks. Ctrl+C to quit.")
try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
finally:
    v.disconnect()
