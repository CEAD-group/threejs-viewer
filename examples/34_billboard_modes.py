"""
Billboard Modes — checkered donuts in every configuration

`set_billboard` makes *any* object orient itself toward the viewer, every
frame. Every mode is one composition:

    R_final = R_aim · R_own

`R_own` is the object's own rotation (what you set, or what an animation
channel is driving). It acts **first, in the object's local frame** — which is
why the spinning donut below keeps spinning about its own axle while that axle
points at you: a spin about an axis leaves that axis fixed, so it never fights
the aim.

The donuts are checkered on purpose. A plain torus is rotationally symmetric,
so you cannot see roll or spin on one; the checker pattern breaks the symmetry
and makes every degree of freedom visible.

Orbit the camera and watch each row react:

  1. MODES     camera / world_up / aim / hinge, spread along X so the
               difference shows: `camera` keeps every donut parallel to the
               screen, `aim` turns each one toward where you actually are.
               Every donut here spins at its own rate, so the two 3-DOF modes
               give themselves away — they override the spin and hang still.
  2. WHEEL     `aim` plus its own spin — the headline case, and the fastest.
  3. FACE      the same mode with face="+z" / "+y" / "+x": axes are read in the
               donut's own frame, so a different part of it points at you.
  4. PIVOT     rotation origin at the centre vs out on the rim: the rim donut
               swings about that point instead of turning in place.
  5. HINGE     1 DOF. hinge_world=None keeps the hinge wherever the donut's own
               rotation put it; hinge_world=[0,0,1] pins it upright.

Run: uv run python examples/34_billboard_modes.py
"""

import numpy as np

from threejs_viewer import Animation, viewer

R_MAJOR = 0.70
R_MINOR = 0.28
RIM = R_MAJOR + R_MINOR


def checkered_torus(v, id, color_a, color_b, nu=72, nv=36, checks=(12, 6), **kwargs):
    """Add a donut whose surface is a hard-edged checker.

    The hole axis is local +Z. Each check is emitted as its own four vertices
    so the colours meet at a crisp edge instead of blending across a shared
    vertex; normals stay analytic, so the shading is still smooth.
    """
    u = np.linspace(0, 2 * np.pi, nu + 1)
    w = np.linspace(0, 2 * np.pi, nv + 1)
    uu, ww = np.meshgrid(u, w, indexing="ij")

    ring = R_MAJOR + R_MINOR * np.cos(ww)
    grid = np.stack(
        [ring * np.cos(uu), ring * np.sin(uu), R_MINOR * np.sin(ww)], axis=-1
    )
    normals = np.stack(
        [np.cos(ww) * np.cos(uu), np.cos(ww) * np.sin(uu), np.sin(ww)], axis=-1
    )

    # Four corners per quad -> (nu, nv, 4, 3), so no vertex is shared.
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    positions = np.stack([grid[a:, b:][:nu, :nv] for a, b in corners], axis=2)
    quad_normals = np.stack([normals[a:, b:][:nu, :nv] for a, b in corners], axis=2)

    cu, cv = checks
    i = np.arange(nu)[:, None]
    j = np.arange(nv)[None, :]
    dark = ((i // cu) + (j // cv)) % 2 == 0
    palette = np.array(
        [
            [
                (color_b >> 16 & 255) / 255,
                (color_b >> 8 & 255) / 255,
                (color_b & 255) / 255,
            ],
            [
                (color_a >> 16 & 255) / 255,
                (color_a >> 8 & 255) / 255,
                (color_a & 255) / 255,
            ],
        ],
        dtype=np.float32,
    )
    colors = np.repeat(palette[dark.astype(int)][:, :, None, :], 4, axis=2)

    quads = np.arange(nu * nv)[:, None] * 4
    indices = (quads + np.array([0, 1, 2, 0, 2, 3])).ravel().astype(np.uint32)

    v.add_mesh(
        id,
        positions.reshape(-1, 3).astype(np.float32),
        indices,
        normals=quad_normals.reshape(-1, 3).astype(np.float32),
        colors=colors.reshape(-1, 3),
        roughness=0.45,
        metalness=0.1,
        **kwargs,
    )


v = viewer()
v.clear()
v.unload_animation()

v.add_grid("floor", cell_size=1.0, extent=60.0, fade_start=0.4)


# Every donut gets its own spin, each at a different rate (turns per 6 s loop).
# That is the point of the demo: the spin IS the object's own rotation, so you
# can read straight off the scene what each mode does with it.
#
# Which donuts keep turning tells you what each mode does with that rotation:
#   aim       keeps it as the roll                        -> turns
#   camera    3 DOF, pose fully determined by the camera  -> ignores it
#   world_up  3 DOF, same                                 -> ignores it
#   hinge + hinge_world   aligning the hinge and then solving the spin leaves
#             no freedom at all, so the own rotation is absorbed exactly -> still
#   hinge, free hinge     the own rotation is what decides where the hinge
#             points, so it does show                     -> turns
# The still ones are not stuck: they still track you as you orbit. (All of this
# is measured with tmp/probe_spin.py, not assumed.)
def donut(id, dark, pos, turns, tilt=0.0, **billboard):
    return dict(id=id, dark=dark, pos=pos, turns=turns, tilt=tilt, bb=billboard)


BLUE, GOLD, GREEN, PURPLE, RED = 0x2E6FB7, 0xB4600A, 0x2C8C56, 0x8C3BA0, 0xC0392B

DONUTS = [
    # 1. MODES — spread wide so `camera` (which stays parallel to the view
    #    plane) separates visibly from `aim` (which turns toward where you are).
    donut("mode_camera", BLUE, [-6.3, 0, 7.0], 0.30, mode="camera", face="+z"),
    donut("mode_world_up", BLUE, [-2.1, 0, 7.0], 0.45, mode="world_up", face="+z"),
    donut("mode_aim", BLUE, [2.1, 0, 7.0], 0.60, mode="aim", face="+z"),
    donut(
        "mode_hinge",
        BLUE,
        [6.3, 0, 7.0],
        0.75,
        mode="hinge",
        face="+z",
        hinge="+y",
        hinge_world=[0, 0, 1],
    ),
    # 2. WHEEL — the headline case, and the fastest.
    donut("wheel", GOLD, [0, 0, 3.6], 1.60, mode="aim", face="+z"),
    # 3. FACE AXIS — same mode, different axis of the donut's OWN frame.
    donut("face_z", GREEN, [-2.6, 0, 0.6], 0.50, mode="aim", face="+z"),
    donut("face_y", GREEN, [0.0, 0, 0.6], 0.85, mode="aim", face="+y"),
    donut("face_x", GREEN, [2.6, 0, 0.6], 1.20, mode="aim", face="+x"),
    # 4. PIVOT — rotation origin at the centre vs out on the rim. Same rate, so
    #    the only difference you see is what each one turns *about*.
    donut(
        "pivot_centre",
        PURPLE,
        [-2.0, 0, -2.6],
        0.40,
        mode="aim",
        face="+z",
        pivot=[0, 0, 0],
    ),
    donut(
        "pivot_rim",
        PURPLE,
        [2.0, 0, -2.6],
        0.40,
        mode="aim",
        face="+z",
        pivot=[RIM, 0, 0],
    ),
    # 5. HINGE — 1 DOF, hinge free vs pinned to world +Z.
    donut(
        "hinge_own",
        RED,
        [-2.0, 0, -5.8],
        0.35,
        tilt=0.52,
        mode="hinge",
        face="+z",
        hinge="+y",
    ),
    donut(
        "hinge_world",
        RED,
        [2.0, 0, -5.8],
        0.35,
        tilt=0.52,
        mode="hinge",
        face="+z",
        hinge="+y",
        hinge_world=[0, 0, 1],
    ),
]

for d in DONUTS:
    checkered_torus(
        v, d["id"], 0xE8E8E8, d["dark"], position=d["pos"], rotation=[d["tilt"], 0, 0]
    )
    v.set_billboard(d["id"], **d["bb"])

# --- The spins, as one binary transforms channel -------------------------
# This is also the path that pins matrixAutoUpdate off and writes obj.matrix
# directly, which a billboard has to compose with rather than overwrite.
n_frames, duration = 181, 6.0
times = np.linspace(0.0, duration, n_frames)
ids = [d["id"] for d in DONUTS]
spin = np.zeros((n_frames, len(ids), 16), dtype=np.float32)
for i, t in enumerate(times):
    for k, d in enumerate(DONUTS):
        a = 2 * np.pi * d["turns"] * t / duration
        c, s_ = np.cos(a), np.sin(a)
        spin_r = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        ct, st = np.cos(d["tilt"]), np.sin(d["tilt"])
        tilt_x = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
        m = np.eye(4)
        # Tilt is the pose the donut is mounted at; the spin happens about its
        # own axis inside that mount, so it multiplies on the right.
        m[:3, :3] = tilt_x @ spin_r
        m[:3, 3] = d["pos"]
        spin[i, k] = m.T.flatten()

anim = Animation(frames=[])
anim.set_frame_times(times)
anim.set_transform_data(ids, spin)
v.load_animation(anim, loop=True)

print("Checkered donuts, one row per option. Every donut spins at its own rate,")
print("so you can see what each mode does with the donut's own rotation.")
print("  z=7.0  MODES  camera | world_up | aim | hinge  (left to right)")
print("         Only `aim` keeps the spin. camera/world_up are 3 DOF and ignore")
print("         it; hinge (pinned to world +Z) absorbs it exactly. Orbit and all")
print("         four still track you — they are not stuck.")
print("  z=3.6  WHEEL  aim + its own spin: the axle tracks you, the donut turns.")
print("  z=0.6  FACE   face=+z | +y | +x — the axis is read in the donut's own frame.")
print(
    "  z=-2.6 PIVOT  centre (turns in place) | rim (swings about a point on its edge)."
)
print("  z=-5.8 HINGE  hinge free (own rotation aims the hinge, so it turns)")
print("                | pinned to world +Z (own rotation fully absorbed).")

# The whole animation goes over in one push: `wait_for_assets` blocks until
# the browser has fetched the binary channel, then shuts the server down. From
# there the browser plays it on its own clock — this script does not need to
# stay connected, and the scene survives it exiting.
v.wait_for_assets()
print()
print("Sent. Python is done — the browser keeps animating on its own.")
