"""
Move and rotate objects from the browser — with the transform reported back.

This is the inverse of the usual flow: instead of Python pushing transforms to
the viewer, the *user* drags an object in the browser and the new transform
travels back to Python. Enable it with ``enable_move_gizmo()`` /
``on_object_move(callback)``.

The gizmo is three.js ``TransformControls``, restyled for a cleaner look
(refined palette, enlarged outlined plane handles). Once enabled:

  * **Click any object** to attach the gizmo to it (``click_select`` is on by
    default; you can also pass ``id=`` to attach to a specific object up front).
  * **Drag** an axis arrow to move along it, or a plane handle to slide in that
    plane.
  * **Hold Alt** while dragging to **rotate** instead of translate.
  * **Hold Shift** to **snap** — translations to a grid (``translate_snap``,
    here ``0.5``), rotations to fixed increments (``rotate_snap_deg``, here
    ``15``). Snapping is live, so you can toggle Shift mid-drag.

As you drag, the moved object's new transform is sent back here and logged
(``on_move`` below): throttled ``"move"`` updates during the drag, then a final
``"end"`` report on release. The callback runs on the client's WebSocket
thread; it's fine to call other viewer methods from inside it.

Run: uv run python examples/23_move_gizmo.py
(then click + drag in the browser; press Ctrl+C in the terminal to quit)
"""

import logging
import time

from threejs_viewer import viewer

log = logging.getLogger("gizmo-demo")
log.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)-5s [%(name)s] %(message)s", "%H:%M:%S")
)
log.addHandler(_handler)
log.propagate = False


v = viewer()
v.clear()
v.unload_animation()

# A few distinctly-coloured objects spread out so there's room to drag.
v.add_box("red", color=0xEF5468, position=[-3, 0, 0.5])
v.add_sphere("green", radius=0.6, color=0x43C873, position=[0, 0, 0.6])
v.add_box("blue", color=0x4A90E2, position=[3, 0, 0.5])


_n = 0


def on_move(m):
    """Receive the moved object's transform (browser → Python)."""
    global _n
    # Log every release in full; throttle the mid-drag spam to the occasional one.
    px, py, pz = m["position"]
    if m["phase"] == "end":
        log.info(
            "%s released at (%.3f, %.3f, %.3f)  quat=%s",
            m["id"],
            px,
            py,
            pz,
            [round(q, 3) for q in m["quaternion"]],
        )
    elif _n % 10 == 0:
        log.info("%s moving…  (%.3f, %.3f, %.3f)", m["id"], px, py, pz)
    _n += 1


# Registering the callback also enables the gizmo. Snap grid 0.5 units / 15°.
v.on_object_move(on_move)
v.enable_move_gizmo(translate_snap=0.5, rotate_snap_deg=15)

print(__doc__)
print("Gizmo enabled. Click an object to attach, then drag.")
print(
    "  Alt = rotate · Shift = snap (0.5 units / 15°) · click another object to switch."
)
print("Press F to frame the scene. Ctrl+C here to quit.")

try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nDone.")
finally:
    v.disconnect()
