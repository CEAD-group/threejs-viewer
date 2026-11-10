"""
Vector entities

A small DXF-style drawing: a flange outline with bolt holes placed as a
block (DXF BLOCK/INSERT), slots made of bulged polylines, centre lines, an
elliptical recess and a NURBS cam profile.
Curves keep their exact definitions and refine as you zoom in: get close to
any edge and it stays smooth. Layers set colour, linetype and width, and the
last few lines toggle the HIDDEN layer on and off.

Run: uv run python examples/42_vector_entities.py
"""

import math
import time

from threejs_viewer import vector, viewer

v = viewer()
v.clear()
v.unload_animation()
v.set_background(0x1E1E1E)

# DXF-style layer table. Entities default to their layer's style (BYLAYER).
v.set_layer("OUTLINE", color=0xE8E8E8, line_width=2.5)
v.set_layer("HOLES", color=0x5AB4FF, line_width=2)
v.set_layer("CENTER", color=0xFF6E6E, linetype="CENTER", line_width=1)
v.set_layer("HIDDEN", color=0x9A9A9A, linetype="HIDDEN", line_width=1.5)
v.set_layer("CAM", color=0xFFC24A, line_width=2.5)

# Scene units are metres; this drawing is a 200 mm flange.
R = 0.1
entities = [
    vector.circle([0, 0, 0], R, layer="OUTLINE"),
    vector.circle([0, 0, 0], 0.035, layer="HOLES"),
    vector.circle([0, 0, 0], 0.075, layer="CENTER"),  # bolt circle
    vector.line([-0.12, 0, 0], [0.12, 0, 0], layer="CENTER"),
    vector.line([0, -0.12, 0], [0, 0.12, 0], layer="CENTER"),
    # Hidden counterbore behind the centre hole.
    vector.circle([0, 0, 0], 0.045, layer="HIDDEN"),
]

# Six bolt holes on the bolt circle: one HOLE block (a hole with a hidden
# counterbore), placed six times. Its entities have no layer of their own, so
# they take the insert's; the counterbore keeps its HIDDEN linetype.
bolt_centres = [
    [0.075 * math.cos(i * math.pi / 3), 0.075 * math.sin(i * math.pi / 3), 0]
    for i in range(6)
]
entities.append(vector.insert("HOLE", bolt_centres, layer="HOLES"))
blocks = {
    "HOLE": [
        vector.circle([0, 0, 0], 0.008),
        vector.circle([0, 0, 0], 0.011, linetype="HIDDEN"),
    ]
}

# Two slots: straight sides joined by half-circle ends (bulge = 1).
for sign in (-1, 1):
    y = sign * 0.05
    entities.append(
        vector.polyline(
            [
                [-0.02, y - 0.006, 0],
                [0.02, y - 0.006, 0],
                [0.02, y + 0.006, 0],
                [-0.02, y + 0.006, 0],
            ],
            bulges=[0, 1, 0, 1],
            closed=True,
            layer="OUTLINE",
        )
    )

# An elliptical recess behind the bore, and an arc marking a keep-out zone.
entities.append(vector.ellipse([0.0, -0.0, 0], [0.018, 0.0, 0], 0.45, layer="HIDDEN"))
entities.append(
    vector.arc([0, 0, 0], 0.09, math.radians(200), math.radians(340), layer="CENTER")
)

v.add_vector_entities("flange", entities, blocks=blocks)

# A rational NURBS cam profile beside the flange, in its own object.
cam = [
    [0.18, 0.0, 0],
    [0.24, 0.06, 0],
    [0.30, 0.02, 0],
    [0.33, -0.05, 0],
    [0.26, -0.09, 0],
    [0.19, -0.06, 0],
]
v.add_spline("cam", cam, degree=3, weights=[1, 2, 1, 3, 1, 1], closed=True, layer="CAM")
# Its control polygon, dotted, with the control points.
v.add_vector_entities(
    "cam_ctrl",
    [
        vector.points(cam, color=0x777777),
        vector.polyline(cam, closed=True, color=0x555555, linetype="DOT"),
    ],
)

v.set_view("top", animate=False)
v.frame_points([[-0.12, -0.12, 0], [0.34, 0.12, 0]])

# Toggle the hidden lines a few times.
for _ in range(3):
    time.sleep(1.5)
    v.set_layer("HIDDEN", visible=False)
    time.sleep(1.5)
    v.set_layer("HIDDEN", visible=True)
