"""
Grazing Opacity

`add_mesh(grazing_opacity=...)` fades a translucent mesh's opacity toward a
second value where its surface turns edge-on. Two identical translucent
domes stand over a floor grid with an opaque post inside each. The left one
has constant alpha, so from a low angle the grid shows through it as strongly
as head-on and the dome reads as a flat tint. The right one rises to 0.85 at
its silhouette: the grid fades out toward the rim while the post stays
visible through the faint centre.

Run: uv run python examples/41_grazing_opacity.py
"""

import numpy as np

from threejs_viewer import viewer


def dome(cx, radius=2.0, height=2.6, n=64):
    """A closed half-ellipsoid on z=0: curved shell plus a flat base."""
    theta = np.linspace(0, np.pi / 2, n // 2 + 1)
    phi = np.linspace(0, 2 * np.pi, n + 1)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    unit = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1)
    scale = np.array([radius, radius, height])
    pos = (unit * scale).reshape(-1, 3) + [cx, 0, 0]
    nrm = unit / scale
    nrm = (nrm / np.linalg.norm(nrm, axis=-1, keepdims=True)).reshape(-1, 3)
    cols = n + 1
    rows = np.arange(len(theta) - 1)[:, None] * cols + np.arange(n)
    a = rows.ravel()
    shell = np.stack([a, a + cols, a + 1, a + 1, a + cols, a + cols + 1], 1).reshape(
        -1, 3
    )
    # Base: a fan around a centre vertex, facing down.
    rim = (len(theta) - 1) * cols + np.arange(n)
    centre = len(pos)
    base = np.stack([np.full(n, centre), rim + 1, rim], 1)
    pos = np.vstack([pos, [cx, 0, 0]])
    nrm = np.vstack([nrm, [0, 0, -1]])
    tris = np.vstack([shell, base]).astype(np.uint32)
    return pos.astype(np.float32), nrm.astype(np.float32), tris.ravel()


v = viewer()
v.clear()
v.add_grid("floor", cell_size=0.5, extent=40.0, fade_start=0.4)
for name, cx, grazing in (("constant", -2.6, None), ("grazing", 2.6, 0.85)):
    v.add_cylinder(
        f"{name}_post",
        radius_top=0.25,
        radius_bottom=0.25,
        height=1.8,
        color=0xFF5500,
        position=[cx, 0, 0.9],
        rotation=[np.pi / 2, 0, 0],
    )
    pos, nrm, tris = dome(cx)
    v.add_mesh(
        name,
        positions=pos,
        indices=tris,
        normals=nrm,
        color=0x3D8BD9,
        opacity=0.25,
        grazing_opacity=grazing,
        grazing_power=2.0,
        pickable=False,
    )

v.set_camera(position=[0, -11, 2.2], target=[0, 0, 1.0])
v.wait_for_assets()
