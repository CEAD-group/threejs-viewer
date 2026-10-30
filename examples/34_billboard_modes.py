"""
Billboard Modes

`set_billboard` makes any object face the viewer every frame:
R_final = R_aim · R_own, where R_own is the object's own rotation (here a spin
driven by a transforms channel), applied first in its local frame. Checkered
donuts make roll and spin visible. Orbit and compare the rows:

  z=7.0   MODES  camera | world_up | aim | hinge. Only `aim` keeps the spin;
          the 3-DOF modes ignore it, a world-pinned hinge absorbs it.
  z=3.6   WHEEL  aim plus its own spin: the axle tracks you, the donut turns.
  z=0.6   FACE   face="+z" | "+y" | "+x", axes are read in the donut's frame.
  z=-2.6  PIVOT  rotate about the centre | about a point on the rim.
  z=-5.8  HINGE  free hinge (own rotation aims it) | pinned to world +Z.

Run: uv run python examples/34_billboard_modes.py
"""

from itertools import product

import numpy as np

from threejs_viewer import Animation, viewer

R_MAJOR, R_MINOR = 0.70, 0.28


def checkered_torus(v, id, dark, light=0xE8E8E8, nu=72, nv=36, **kwargs):
    """Donut with a hard-edged 12x6 checker (hole axis +Z), four own vertices per quad."""
    uu, ww = np.meshgrid(
        *[np.linspace(0, 2 * np.pi, n + 1) for n in (nu, nv)], indexing="ij"
    )
    ring = R_MAJOR + R_MINOR * np.cos(ww)
    grid = np.stack([ring * np.cos(uu), ring * np.sin(uu), R_MINOR * np.sin(ww)], -1)
    nrm = np.stack([np.cos(ww) * np.cos(uu), np.cos(ww) * np.sin(uu), np.sin(ww)], -1)
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    quad = lambda a: np.stack([a[i:, j:][:nu, :nv] for i, j in corners], axis=2)  # noqa: E731
    rgb = lambda c: np.array([c >> 16 & 255, c >> 8 & 255, c & 255]) / 255  # noqa: E731
    is_dark = ((np.arange(nu)[:, None] // 12) + (np.arange(nv)[None, :] // 6)) % 2 == 0
    colors = np.where(is_dark[..., None, None], rgb(dark), rgb(light))
    colors = np.broadcast_to(colors, (nu, nv, 4, 3))
    idx = (np.arange(nu * nv)[:, None] * 4 + [0, 1, 2, 0, 2, 3]).ravel()
    v.add_mesh(
        id, quad(grid).reshape(-1, 3).astype(np.float32), idx.astype(np.uint32),
        normals=quad(nrm).reshape(-1, 3).astype(np.float32),
        colors=colors.reshape(-1, 3).astype(np.float32),
        roughness=0.45, metalness=0.1, **kwargs,
    )  # fmt: skip


BLUE, GOLD, GREEN, PURPLE, RED = 0x2E6FB7, 0xB4600A, 0x2C8C56, 0x8C3BA0, 0xC0392B
UPRIGHT = dict(hinge="+y", hinge_world=[0, 0, 1])
# id, colour, position, spin turns per loop, mounting tilt about X, set_billboard options
DONUTS = [
    ("mode_camera", BLUE, [-6.3, 0, 7], 0.30, 0, dict(mode="camera")),
    ("mode_world_up", BLUE, [-2.1, 0, 7], 0.45, 0, dict(mode="world_up")),
    ("mode_aim", BLUE, [2.1, 0, 7], 0.60, 0, dict(mode="aim")),
    ("mode_hinge", BLUE, [6.3, 0, 7], 0.75, 0, dict(mode="hinge", **UPRIGHT)),
    ("wheel", GOLD, [0, 0, 3.6], 1.60, 0, dict(mode="aim")),
    ("face_z", GREEN, [-2.6, 0, 0.6], 0.50, 0, dict(mode="aim", face="+z")),
    ("face_y", GREEN, [0, 0, 0.6], 0.85, 0, dict(mode="aim", face="+y")),
    ("face_x", GREEN, [2.6, 0, 0.6], 1.20, 0, dict(mode="aim", face="+x")),
    ("pivot_centre", PURPLE, [-2, 0, -2.6], 0.40, 0, dict(mode="aim", pivot=[0, 0, 0])),
    ("pivot_rim", PURPLE, [2, 0, -2.6], 0.40, 0, dict(mode="aim", pivot=[R_MAJOR + R_MINOR, 0, 0])),
    ("hinge_own", RED, [-2, 0, -5.8], 0.35, 0.52, dict(mode="hinge", hinge="+y")),
    ("hinge_world", RED, [2, 0, -5.8], 0.35, 0.52, dict(mode="hinge", **UPRIGHT)),
]  # fmt: skip

v = viewer()
v.clear()
v.unload_animation()
for id, color, pos, _, tilt, options in DONUTS:
    checkered_torus(v, id, color, position=pos, rotation=[tilt, 0, 0])
    v.set_billboard(id, **{"face": "+z", **options})

# Spins as one transforms channel: the animation writes obj.matrix directly,
# which a billboard has to compose with rather than overwrite.
ids, _, pos, turns, tilt, _ = zip(*DONUTS)
n_frames, duration = 181, 6.0
times = np.linspace(0, duration, n_frames)
a = 2 * np.pi * np.array(turns) * times[:, None] / duration  # (F, K)
c, s, z, o = np.cos(a), np.sin(a), np.zeros_like(a), np.ones_like(a)
spin = np.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(*a.shape, 3, 3)
ct, st = np.cos(tilt), np.sin(tilt)
tilt_x = np.stack(
    [np.ones_like(ct), 0 * ct, 0 * ct, 0 * ct, ct, -st, 0 * ct, st, ct], -1
)
m = np.zeros((*a.shape, 4, 4))
m[..., :3, :3] = (
    tilt_x.reshape(-1, 3, 3) @ spin
)  # tilt is the mount, spin is about the donut's own axis
m[..., :3, 3] = pos
m[..., 3, 3] = 1
anim = Animation(loop=True)
anim.set_frame_times(times)
anim.set_transform_data(
    list(ids), m.transpose(0, 1, 3, 2).reshape(n_frames, -1, 16).astype(np.float32)
)
v.load_animation(anim)

v.set_view("front", animate=False)
v.frame_points(list(product([-7.3, 7.3], [-1, 1], [-8, 8.2])), fill=0.9)
v.wait_for_assets()
