"""
Billboard Demo

Demonstrates `add_billboard`: a flat plane the viewer re-orients toward the
camera every rendered frame — the primitive to use for markers/labels that
must stay readable from any viewpoint.

Two flavours sit side by side above a row of boxes:

  * full billboards (`mode="camera"`, the default) stay parallel to the view
    plane, so they face the camera exactly no matter where you orbit;
  * hinged billboards (`mode="hinge"`, hinge held along world Z) may only
    spin about that axis, so they turn toward you while staying upright —
    orbit up over the top and they tilt away edge-on, which the full ones
    never do.

For the whole option space — including billboarding objects that have their
own rotation — see examples/34_billboard_modes.py.

The third pair is parented to a rotating group to show that a billboard
under a rotating parent keeps facing the camera instead of tumbling with it.

Run: uv run python examples/32_billboard.py
"""

import numpy as np

from threejs_viewer import Animation, Frame, viewer

v = viewer()
v.clear()
v.unload_animation()

v.add_grid("floor", cell_size=1.0, extent=40.0, fade_start=0.35)

COLORS = [0xE24A4A, 0x4AE24A, 0x4A90D9]

for i, color in enumerate(COLORS):
    x = (i - 1) * 3.0
    v.add_box(
        f"box_{i}",
        width=1.0,
        height=1.0,
        depth=1.0,
        color=color,
        position=[x, 0, 0.5],
        roughness=0.5,
    )
    # Full billboard: copies the camera orientation.
    v.add_billboard(
        f"tag_full_{i}",
        width=1.4,
        height=0.7,
        color=color,
        position=[x, 0, 1.6],
    )
    # Hinged billboard: locked upright, spins about world Z only.
    v.add_billboard(
        f"tag_up_{i}",
        width=1.4,
        height=0.7,
        color=color,
        opacity=0.6,
        mode="hinge",
        hinge="+y",
        hinge_world=[0, 0, 1],
        position=[x, 0, 2.6],
    )

# A billboard under a rotating parent: the group spins, the billboard doesn't.
v.add_group("spinner", position=[0, 5, 0])
v.add_box(
    "spinner_arm", width=4.0, height=0.2, depth=0.2, color=0x888888, parent="spinner"
)
v.add_billboard(
    "spinner_tag",
    width=1.2,
    height=0.6,
    color=0xFFC24A,
    position=[2.0, 0, 0.8],
    parent="spinner",
)

frames = []
for t in np.linspace(0, 6.0, 181):
    angle = 2 * np.pi * t / 6.0
    c, s = np.cos(angle), np.sin(angle)
    # Column-major 4x4: rotation about Z at the group's own position.
    frames.append(
        Frame(
            time=float(t),
            transforms={
                "spinner": [c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 5, 0, 1],
            },
        )
    )
v.load_animation(Animation(frames=frames, loop=True))

print("Billboards up. Orbit around: the middle row faces you exactly,")
print("the upper row stays upright and only spins about Z.")
print("The yellow tag rides a rotating arm without tumbling with it.")
