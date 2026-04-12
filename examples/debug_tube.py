"""Minimal parametric tube: straight line with draw_range animation.

For debugging frontier-ring morphing. Wireframe overlay shows ring structure.
"""

import time
import numpy as np
from threejs_viewer import Animation, viewer

v = viewer(open_browser=False)
v.clear()

n_pts = 4
x = [0, 4, 8, 12]
y = [0, 0, 0, 0]
z = [0, 0, 0, 0]
spine = np.column_stack([x, y, z])
widths = np.array([8, 8, 2, 2], dtype=np.float32)
heights = np.array([4, 4, 4, 4], dtype=np.float32)

v.add_parametric_tube("tube", spine=spine, widths=widths, heights=heights, opacity=1, color=0xCCCCCC)
v.add_parametric_tube(
    "wire", spine=spine, widths=widths, heights=heights, color=0xFF0000, wireframe=True
)

n_frames = 60
anim = Animation(loop=True)
anim.set_frame_times(np.linspace(0, 3, n_frames))
draw = np.linspace(0, 1, n_frames, dtype=np.float32)
anim.set_draw_range_data(["tube", "wire"], np.column_stack([draw, draw]))
v.load_animation(anim)

time.sleep(20)
