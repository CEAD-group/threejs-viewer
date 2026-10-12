"""Animate a mapped, rising path with bounded local corner repairs.

Run: uv run python examples/39_local_tube_joins.py
For all planar/spatial cases and screenshots: bench/tube_lab/local_joins.py.
"""

import numpy as np
from bead_texture_maker import texture_set

from threejs_viewer import Animation, viewer

v = viewer()
v.clear()
spine = np.array(
    [
        [-30, 0, 0],
        [-20, 0, 0.1],
        [-10, 0, 0.2],
        [0, 0, 0.3],
        [0, 10, 1],
        [0, 20, 3],
        [0, 30, 5],
    ],
    dtype=np.float32,
)
maps = texture_set(color="#698d9e", fibers=True)
v.add_parametric_tube(
    "bead",
    spine,
    np.full(len(spine), 10),
    np.full(len(spine), 2),
    surface="local_join",
    texture=maps["color"],
    normal_map=maps["normal"],
    roughness_map=maps["roughness"],
    texture_length=25,
    roughness=1,
    metalness=0,
)
travel = np.r_[0, np.cumsum(np.linalg.norm(np.diff(spine, axis=0), axis=1))]
anim = Animation(loop=True)
anim.set_frame_times(np.r_[travel / travel[-1] * 8, 10].astype(np.float32))
anim.set_draw_range_data(
    ["bead"], np.r_[np.linspace(0, 1, len(spine)), 1].reshape(-1, 1)
)
v.load_animation(anim)
v.set_camera(position=[60, -65, 70], target=[-5, 10, 1], up=[0, 0, 1])
v.wait_for_assets()
