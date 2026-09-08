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
  2. WHEEL     `aim` plus its own spin — the headline case.
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

# --- 1. MODES ------------------------------------------------------------
# Spread along X so `camera` (always parallel to the view plane) separates
# visibly from `aim` (turns toward where the camera actually is).
MODES = ["camera", "world_up", "aim", "hinge"]
for k, mode in enumerate(MODES):
    x = (k - 1.5) * 4.2
    checkered_torus(v, f"mode_{mode}", 0xE8E8E8, 0x2E6FB7, position=[x, 0, 7.0])
    v.set_billboard(
        f"mode_{mode}", mode=mode, face="+z", hinge="+y", hinge_world=[0, 0, 1]
    )

# --- 2. WHEEL: aim + its own spin ----------------------------------------
# The donut spins about its own axle (+Z) while the axle tracks the camera.
checkered_torus(v, "wheel", 0xFFD24A, 0xB4600A, position=[0, 0, 3.6])
v.set_billboard("wheel", mode="aim", face="+z")

# --- 3. FACE AXIS: axes are read in the donut's own frame ----------------
for k, face in enumerate(["+z", "+y", "+x"]):
    x = (k - 1) * 2.6
    checkered_torus(v, f"face_{k}", 0xE8E8E8, 0x2C8C56, position=[x, 0, 0.6])
    v.set_billboard(f"face_{k}", mode="aim", face=face)

# --- 4. PIVOT: rotation origin -------------------------------------------
# Left turns in place; right swings about a point out on its rim.
checkered_torus(v, "pivot_centre", 0xE8E8E8, 0x8C3BA0, position=[-2.0, 0, -2.6])
v.set_billboard("pivot_centre", mode="aim", face="+z", pivot=[0, 0, 0])
checkered_torus(v, "pivot_rim", 0xE8E8E8, 0x8C3BA0, position=[2.0, 0, -2.6])
v.set_billboard("pivot_rim", mode="aim", face="+z", pivot=[RIM, 0, 0])

# --- 5. HINGE: 1 DOF ------------------------------------------------------
# Left: the hinge stays where the donut's own rotation put it (tilted 30 deg).
# Right: the hinge is pinned to world +Z, so it stays upright.
checkered_torus(
    v, "hinge_own", 0xE8E8E8, 0xC0392B, position=[-2.0, 0, -5.8], rotation=[0.52, 0, 0]
)
v.set_billboard("hinge_own", mode="hinge", hinge="+y", face="+z")
checkered_torus(
    v, "hinge_world", 0xE8E8E8, 0xC0392B, position=[2.0, 0, -5.8], rotation=[0.52, 0, 0]
)
v.set_billboard(
    "hinge_world", mode="hinge", hinge="+y", hinge_world=[0, 0, 1], face="+z"
)

# --- The spin, as a binary transforms channel ----------------------------
# This is also the path that pins matrixAutoUpdate off and writes obj.matrix
# directly, which a billboard has to compose with rather than overwrite.
n_frames, duration = 121, 6.0
times = np.linspace(0.0, duration, n_frames)
spin = np.zeros((n_frames, 1, 16), dtype=np.float32)
for i, t in enumerate(times):
    a = 2 * np.pi * t / duration
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4)
    m[:2, :2] = [[c, -s], [s, c]]  # about the donut's own +Z axle
    m[:3, 3] = [0, 0, 3.6]
    spin[i, 0] = m.T.flatten()

anim = Animation(frames=[])
anim.set_frame_times(times)
anim.set_transform_data(["wheel"], spin)
v.load_animation(anim, loop=True)

print("Checkered donuts, one row per option. Orbit and compare:")
print("  z=7.0  MODES  camera | world_up | aim | hinge  (left to right)")
print("         `camera` stays parallel to the screen; `aim` turns toward you.")
print(
    "  z=3.6  WHEEL  aim + its own spin: the axle tracks you, the donut keeps turning."
)
print("  z=0.6  FACE   face=+z | +y | +x — the axis is read in the donut's own frame.")
print(
    "  z=-2.6 PIVOT  centre (turns in place) | rim (swings about a point on its edge)."
)
print("  z=-5.8 HINGE  hinge_world=None (tilted, hinge follows the donut)")
print("                | hinge_world=+Z (hinge pinned upright).")
