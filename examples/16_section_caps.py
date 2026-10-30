"""
Section caps: a clipping slab through closed solids, with the cut faces filled.

Primitives, a printed wall (a parametric-tube toolpath of 5200 spine points) and
a swept 5-axis tool body are cut by a slab with set_clipping_caps(True), so each
section reads as solid material instead of a hollow shell. Press C for the
clipping panel (axis, position, Single/Slab, the "Cap cut solids" box); arrow
keys nudge the slab, Up/Down change its thickness. Meshes and loaded models stay
open; caps cover the solids the viewer builds itself.
"""

import numpy as np

from threejs_viewer import Toolpath, viewer

v = viewer()
v.clear()
v.unload_animation()
v.add_grid("floor", cell_size=1.0, extent=40.0, center_color=0x777777)

# Caps of overlapping solids would be coplanar, so the parts only touch.
v.add_sphere("ball", radius=1.6, color=0x2F6FD0, position=[-4.6, 0, 1.6], roughness=0.4)
up = [np.pi / 2, 0, 0]  # cylinders are built along Y; stand them on Z
steel = dict(color=0x9AA4AE, roughness=0.35, metalness=0.5, rotation=up)
for cid, r_bot, r_top, height, z in [
    ("disc", 2.0, 2.0, 0.5, 0.25),
    ("hub", 1.0, 0.7, 2.5, 1.75),
]:
    v.add_cylinder(
        cid,
        radius_top=r_top,
        radius_bottom=r_bot,
        height=height,
        position=[-0.6, 0, z],
        **steel,
    )
for k in range(6):
    a = k * np.pi / 3
    pos = [-0.6 + 1.5 * np.cos(a), 1.5 * np.sin(a), 0.65]
    v.add_cylinder(
        f"bolt_{k}",
        radius_top=0.25,
        radius_bottom=0.25,
        height=0.3,
        position=pos,
        color=0xD08A2E,
        rotation=up,
    )

# Printed wall: a rounded-square spiral, one bead wide, 26 layers of 200 points.
layers, per_layer, bead_h = 26, 200, 0.15
t = np.linspace(0, layers, layers * per_layer, endpoint=False)
ang = 2 * np.pi * t
r = 1.8 / (np.abs(np.cos(ang)) ** 6 + np.abs(np.sin(ang)) ** 6) ** (1 / 6)
wall = np.column_stack([3.8 + r * np.cos(ang), r * np.sin(ang), (t + 1) * bead_h])
tp = Toolpath.from_points(wall.astype(np.float32), bead_width=0.6, bead_height=bead_h)
v.add_toolpath("wall", tp.colorize("viridis"), roughness=0.55, anchor="top")

# Swept tool body: a ball-nose tool crossing the slab while it tilts.
n = 160
s = np.linspace(0, 1, n)
tip = np.column_stack([7.8 + 0.6 * np.sin(np.pi * s), 6.0 * s - 2.0, np.full(n, 0.05)])
lean = 0.4 * np.sin(np.pi * s)
axes = np.column_stack([-np.sin(lean), np.zeros(n), np.cos(lean)])
R = 0.8
h = np.linspace(0, R, 7)
profile = [*zip(h, np.sqrt(np.maximum(R**2 - (R - h) ** 2, 0))), (R, 0.55), (2.6, 0.55)]
v.add_swept_tool(
    "tool",
    tip.astype(np.float32),
    axes.astype(np.float32),
    np.array(profile, dtype=np.float32),
    sections=32,
    color=0xC05A9C,
    roughness=0.4,
)

# A 2.4-thick slab across Y whose front plane runs through every part's centre.
v.set_clipping_caps(True)
v.set_clipping_slab(normal=[0, 1, 0], center=1.2, thickness=2.4, show_helper=False)
v.set_camera(position=[0.5, -16.0, 12.0], target=[3.6, 0.0, 2.2], up=[0, 0, 1])
v.wait_for_assets()
