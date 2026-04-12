"""Minimal parametric tube: straight line with draw_range animation.

For debugging frontier-ring morphing. Wireframe overlay shows ring structure.
"""

import time
import numpy as np
from threejs_viewer import Animation, viewer

v = viewer(open_browser=False)
v.clear()

n_pts = 10
x = np.array([0, 1, 2, 6, 7, 12, 13, 14, 15, 16], dtype=np.float32)
y = 4 * np.sin(x / 3)
z = 3 * np.cos(x / 3)
spine = np.column_stack([x, y, z])
widths = np.full(n_pts, 2, dtype=np.float32)
widths[3:6] = 5
heights = np.full(n_pts, 1, dtype=np.float32)

v.add_parametric_tube("tube", spine=spine, widths=widths, heights=heights, opacity=0.4)
v.add_parametric_tube(
    "wire", spine=spine, widths=widths, heights=heights, color=0x000000, wireframe=True
)

n_frames = 60
anim = Animation(loop=True)
anim.set_frame_times(np.linspace(0, 3, n_frames))
draw = np.linspace(0, 1, n_frames, dtype=np.float32)
anim.set_draw_range_data(["tube", "wire"], np.column_stack([draw, draw]))
v.load_animation(anim)

time.sleep(20)
