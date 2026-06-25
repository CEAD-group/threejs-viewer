"""
1-DOF / 2-DOF / 3-DOF move gizmos, in world-aligned and object-local flavours.

Where ``23_move_gizmo.py`` shows a single interactive gizmo you attach by
clicking, this example *pins* six persistent gizmos at once with ``add_gizmo()``.
Two rows of three, so you can compare both axes of variation directly:

  * **Degrees of freedom** (per column), set by the axis flags:
      - **1-DOF rail** — ``x=False, y=False, z=True`` leaves only the Z arrow.
      - **2-DOF plane** — ``x=True, y=True, z=False`` shows the X and Y arrows
        plus the XY plane chip (slides in a plane, never off it).
      - **3-DOF free** — the full gizmo: three arrows, three plane chips.

  * **Orientation space** (per row), set by ``space=``:
      - **Back row — ``space="world"`` (default).** The handles stay aligned to
        the world axes no matter how the object sits.
      - **Front row — ``space="local"``.** These blocks are *rotated*, and their
        gizmos turn with them, so the arrows follow the tilted object.

While you drag, a **translucent ghost** stays behind at the grab-time pose so
you can see how far the block has travelled; it disappears on release. The plane
chips are flat quads spaced a little out from the gizmo centre.

Each release is reported back here through ``on_object_move`` and logged. Hold
**Alt** to rotate instead of translate, **Shift** to snap (grid ``0.5`` / 15°).

Run: uv run python examples/24_gizmo_dimensions.py
(then drag the blocks in the browser; press F to frame, Ctrl+C to quit)
"""

import logging
import time

from threejs_viewer import viewer

log = logging.getLogger("gizmo-dims")
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
v.show_grid(True)

# One colour per degree-of-freedom column, so world vs local is a row comparison.
RAIL = 0x4A90E2  # blue   — 1-DOF (Z rail)
TILE = 0x43C873  # green  — 2-DOF (XY plane)
CUBE = 0xEF8A3A  # orange — 3-DOF (free)

# The local row is rotated so its object-local gizmos visibly tilt away from the
# world axes (Euler radians: ~26° about X, ~34° about Z).
TILT = [0.45, 0.0, 0.6]


def add_block(name, color, *, w, h, d, pos, rotation=None):
    v.add_box(
        name, width=w, height=h, depth=d, color=color, position=pos, rotation=rotation
    )


# Back row (y = +2): world-aligned gizmos, blocks upright.
add_block("rail", RAIL, w=0.8, h=0.8, d=2.0, pos=[-3.5, 2.0, 1.0])
add_block("tile", TILE, w=1.6, h=1.6, d=0.3, pos=[0, 2.0, 0.15])
add_block("cube", CUBE, w=1.2, h=1.2, d=1.2, pos=[3.5, 2.0, 0.6])

# Front row (y = -2): object-local gizmos, blocks rotated by TILT.
add_block("rail_l", RAIL, w=0.8, h=0.8, d=2.0, pos=[-3.5, -2.0, 1.3], rotation=TILT)
add_block("tile_l", TILE, w=1.6, h=1.6, d=0.3, pos=[0, -2.0, 1.0], rotation=TILT)
add_block("cube_l", CUBE, w=1.2, h=1.2, d=1.2, pos=[3.5, -2.0, 1.0], rotation=TILT)


_n = 0


def on_move(m):
    """Receive a moved block's transform (browser -> Python)."""
    global _n
    px, py, pz = m["position"]
    if m["phase"] == "end":
        log.info(
            "%-6s released at (%.2f, %.2f, %.2f)  quat=%s",
            m["id"],
            px,
            py,
            pz,
            [round(q, 3) for q in m["quaternion"]],
        )
    elif _n % 12 == 0:
        log.info("%-6s moving…  (%.2f, %.2f, %.2f)", m["id"], px, py, pz)
    _n += 1


# Pin the gizmos first, *then* register the callback. With pinned gizmos already
# present, on_object_move only registers the callback (it doesn't also turn on the
# click-select interactive gizmo, which would draw an extra gizmo).
# Back row — world-aligned (the default space).
v.add_gizmo("rail", x=False, y=False, z=True)  # 1-DOF: Z rail
v.add_gizmo("tile", x=True, y=True, z=False)  # 2-DOF: XY plane
v.add_gizmo("cube")  # 3-DOF: free
# Front row — object-local (handles turn with the rotated blocks).
v.add_gizmo("rail_l", x=False, y=False, z=True, space="local")
v.add_gizmo("tile_l", x=True, y=True, z=False, space="local")
v.add_gizmo("cube_l", space="local")

# Snap grid 0.5 units / 15°, applied while Shift is held.
v.enable_move_gizmo(translate_snap=0.5, rotate_snap_deg=15, click_select=False)
v.on_object_move(on_move)

print(__doc__)
print("6 pinned gizmos · columns = 1D/2D/3D · back row = world, front = local.")
print("Drag a block; a ghost stays at the start until you release.")
print("  Alt = rotate · Shift = snap (0.5 units / 15°). Press F to frame.")
print("Ctrl+C here to quit.")

try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nDone.")
finally:
    v.disconnect()
