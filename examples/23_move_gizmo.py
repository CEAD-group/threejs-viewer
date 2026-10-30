"""
Move and rotate objects from the browser; the new transform is reported to Python.

Click an object to attach the gizmo, drag an arrow to move along an axis or a plane
chip to slide in a plane. Alt (Shift on Windows) rotates, Shift (Ctrl on Windows)
snaps to 0.5 units / 15 degrees. Moves arrive in ``on_object_move`` as throttled
"move" reports and a final "end" report on release.

Run: uv run python examples/23_move_gizmo.py   (Ctrl+C to quit)
"""

import time

import numpy as np

from threejs_viewer import viewer

v = viewer()
v.clear()
v.add_grid("floor", cell_size=1.0, extent=20.0, center_color=0x888888)
v.add_box("red", color=0xEF5468, position=[-3, 0, 0.5])
v.add_sphere("green", radius=0.6, color=0x43C873, position=[0, 0, 0.6])
v.add_box("blue", color=0x4A90E2, position=[3, 0, 0.5])
v.set_view("iso", animate=False)
v.frame_points(
    np.array(np.meshgrid(*zip([-4, -1, 0], [4, 1, 1.2]))).reshape(3, -1).T, fill=0.85
)


def on_move(m):
    if m["phase"] == "end":
        x, y, z = m["position"]
        print(f"{m['id']} released at ({x:.2f}, {y:.2f}, {z:.2f}) [{m['mode']}]")


v.on_object_move(on_move)
v.enable_move_gizmo(translate_snap=0.5, rotate_snap_deg=15)

print("Click an object, then drag. Ctrl+C to quit.")
try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
finally:
    v.disconnect()
