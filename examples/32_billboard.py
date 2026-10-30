"""
Billboard Demo

`add_billboard` makes a plane the viewer re-orients toward the camera every
frame, for markers and labels that must stay readable from any viewpoint.
Above each box sit two tags:

  * `mode="camera"` (default): parallel to the view plane, never edge-on;
  * `mode="hinge"` with `hinge_world=[0, 0, 1]`: stays upright and only spins
    about world Z, so it tilts away edge-on when you orbit over the top.

The yellow tag rides a rotating arm and still faces you, because the parent's
rotation is divided out. See 34_billboard_modes.py for the full option space.

Run: uv run python examples/32_billboard.py
"""

from itertools import product

import numpy as np

from threejs_viewer import Animation, viewer

v = viewer()
v.clear()
v.unload_animation()
v.add_grid("floor", cell_size=1.0, extent=40.0, fade_start=0.35)

for i, color in enumerate([0xE24A4A, 0x4AE24A, 0x4A90D9]):
    x = (i - 1) * 3.0
    v.add_box(f"box_{i}", width=1, height=1, depth=1, color=color, position=[x, 0, 0.5])
    v.add_billboard(
        f"tag_cam_{i}", width=1.4, height=0.7, color=color, position=[x, 0, 1.6]
    )
    v.add_billboard(
        f"tag_hinge_{i}", width=1.4, height=0.7, color=color, opacity=0.6,
        mode="hinge", hinge="+y", hinge_world=[0, 0, 1], position=[x, 0, 2.6],
    )  # fmt: skip

# A billboard under a rotating parent: the group spins, the billboard does not
v.add_group("spinner", position=[0, 5, 0])
v.add_box("arm", width=4, height=0.2, depth=0.2, color=0x888888, parent="spinner")
v.add_billboard(
    "spinner_tag",
    width=1.2,
    height=0.6,
    color=0xFFC24A,
    position=[2, 0, 0.8],
    parent="spinner",
)

times = np.linspace(0, 6, 181)
angle = 2 * np.pi * times / 6
m = np.tile(np.eye(4), (len(times), 1, 1))
m[:, 0, 0], m[:, 0, 1], m[:, 1, 0], m[:, 1, 1] = (
    np.cos(angle),
    -np.sin(angle),
    np.sin(angle),
    np.cos(angle),
)
m[:, 1, 3] = 5
anim = Animation(loop=True)
anim.set_frame_times(times)
anim.set_transform_data(["spinner"], m.transpose(0, 2, 1).reshape(-1, 1, 16))
v.load_animation(anim)

v.set_view("iso", animate=False)
v.frame_points(list(product([-4, 4], [-1.5, 7], [-0.5, 3])), fill=1.0)
v.wait_for_assets()
