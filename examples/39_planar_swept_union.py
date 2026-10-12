"""Compare ring stitching and rounded swept union with progressive playback.

Run: uv run python examples/39_planar_swept_union.py
The same planar path has a narrow jog, a retrace and a loop with a hole.
Press M for wireframe; scrub the animation to inspect the moving frontier.
"""

import time

import numpy as np

from threejs_viewer import Animation, viewer

spine = np.array(
    [
        [0, 0, 0],
        [20, 0, 0],
        [21, 1, 0],
        [20, 2, 0],
        [22, 3, 0],
        [40, 3, 0],
        [39, 3, 0],
        [40, 3, 0],
        [50, 3, 0],
        [50, 30, 0],
        [20, 30, 0],
        [20, 10, 0],
        [50, 10, 0],
    ],
    dtype=np.float32,
)
v = viewer()
v.clear()
for name, offset, surface in [("rings", 0, "rings"), ("union", 55, "planar_union")]:
    v.add_parametric_tube(
        name,
        spine + [0, offset, 0],
        np.full(len(spine), 10),
        np.full(len(spine), 2),
        surface=surface,
        lod=False,
        strand_collapse=surface == "rings",
    )
v.wait_for_assets()
anim = Animation(loop=True)
anim.set_frame_times(np.array([0, 8, 10], dtype=np.float32))
anim.set_draw_range_data(
    ["rings", "union"], np.array([[0, 0], [1, 1], [1, 1]], dtype=np.float32)
)
v.load_animation(anim)
print("Lower: ring mesh. Upper: rounded swept union. Press M to compare triangles.")
while True:
    time.sleep(1)
