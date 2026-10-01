"""
Bead texture on a printed part — an extruded-plastic look on a real toolpath

A low, wide part printed in one continuous bead (8 x 3 mm, 14 layers), the
kind of shape large-format extrusion makes:

- the outline is a 200 mm square with one round side: three straight walls
  with two rounded corners, and a bulging arc in place of the fourth,
- the wall is a single bead; inside it a zig-zag rib crosses back and forth
  between two opposite walls and runs a short way along each one it meets,
  so there are never more than two beads side by side,
- every turn is rounded: the nozzle never makes a sharp corner,
- the round side overhangs: it leans outward steeply at the bed and
  straightens up toward the top, so the step between stacked beads changes
  with height, and the zig-zag follows it.

Layers alternate wall-then-rib and rib-then-wall (the rib reversed), so each
layer ends where the next begins and the whole part is one unbroken bead.

The bead wears the texture set from ``bead_texture_maker``: colour, normal
and roughness maps repeating every 25 mm along the path
(``add_parametric_tube(texture=, normal_map=, roughness_map=)``). The
streaks run along the bead and the top is wiped smoother by the nozzle.
A draw-range animation prints the part; the image is laid down in place
as the frontier advances.

Run: uv run python examples/38_bead_texture.py
"""

import numpy as np
from bead_texture_maker import texture_set

from threejs_viewer import Animation, viewer

BEAD_W = 8.0  # mm
BEAD_H = 3.0  # mm
N_LAYERS = 14
SIDE = 200.0  # the square's side, mm
ZIG_PITCH = 45.0  # distance along the part between rib turns
ZIG_RUN = 15.0  # how far the rib runs along a wall at each turn
SEAM_X = 20.0  # where on the bottom wall each loop starts and ends
CORNER_R = 20.0  # corner radius; tighter where the legs are too short for it
TILE_MM = 25.0  # length of bead one repeat of the texture covers
STEP_MM = 3.0  # spacing of the toolpath points
PRINT_SECONDS = 40.0


def bulge(z):
    """How far the round side reaches past the square at height ``z``: it
    leans out at about 40 degrees at the bed and is vertical at the top."""
    return 70.0 + 22.0 * np.sin(0.5 * np.pi * z / (N_LAYERS * BEAD_H))


def wall_loop(z):
    """The outline, counter-clockwise from the seam on the bottom wall,
    ending just past where it started so the loop closes."""
    a = np.radians(np.linspace(-90, 90, 61))
    arc = np.column_stack(
        [SIDE + bulge(z) * np.cos(a), SIDE / 2 + SIDE / 2 * np.sin(a)]
    )
    return np.vstack([[SEAM_X, 0], arc, [0, SIDE], [0, 0], [SEAM_X + 6, 0]])


def zig_zag(z):
    """The rib: it runs along the bottom wall, crosses to the top wall, runs
    along that, crosses back, and so on across the square, one bead width
    inside the wall. Its last crossing lands on the round side and follows
    the inside of the arc a short way, so the rib ends against the wall at
    every layer however far the arc leans out."""
    half = SIDE / 2 - BEAD_W
    pts = []
    k = 0
    while SEAM_X + 2 * BEAD_W + k * ZIG_PITCH + ZIG_RUN <= SIDE:
        xa = SEAM_X + 2 * BEAD_W + k * ZIG_PITCH
        y = SIDE / 2 + (half if k % 2 else -half)
        pts += [[xa, y], [xa + ZIG_RUN, y]]
        k += 1
    side = 1 if k % 2 else -1  # the wall the next turn would have met
    a = side * np.radians(np.linspace(50, 35, 6))
    reach = bulge(z) - BEAD_W
    arc = np.column_stack([SIDE + reach * np.cos(a), SIDE / 2 + half * np.sin(a)])
    return np.vstack([pts, arc])


def round_corners(pts, radius):
    """Replace every corner of a polyline with a tangent arc of ``radius``,
    or the largest one the two legs have room for."""
    out = [pts[0]]
    for a, p, b in zip(pts[:-2], pts[1:-1], pts[2:]):
        u_in, u_out = p - a, b - p
        len_in, len_out = np.linalg.norm(u_in), np.linalg.norm(u_out)
        u_in, u_out = u_in / len_in, u_out / len_out
        turn = np.arccos(np.clip(u_in @ u_out, -1, 1))
        if turn < np.radians(8):  # already part of a smooth curve
            out.append(p)
            continue
        tan_half = np.tan(turn / 2)
        setback = min(radius * tan_half, 0.45 * len_in, 0.45 * len_out)
        r = setback / tan_half
        start, end = p - setback * u_in, p + setback * u_out
        inward = u_out - u_in
        centre = p + inward / np.linalg.norm(inward) * (r / np.cos(turn / 2))
        a0 = np.arctan2(*(start - centre)[::-1])
        sweep = np.arctan2(*(end - centre)[::-1]) - a0
        sweep = (sweep + np.pi) % (2 * np.pi) - np.pi  # the short way round
        angles = a0 + sweep * np.linspace(0, 1, int(np.ceil(turn / np.radians(8))) + 1)
        out.extend(centre + r * np.column_stack([np.cos(angles), np.sin(angles)]))
    out.append(pts[-1])
    return np.array(out)


def subdivide(pts, step):
    """Add points so no segment is longer than ``step``; corners are kept."""
    out = [pts[:1]]
    for a, b in zip(pts[:-1], pts[1:]):
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.append(a + np.outer(np.arange(1, k + 1) / k, b - a))
    return np.concatenate(out)


def make_toolpath():
    spine = []
    for layer in range(N_LAYERS):
        wall = wall_loop(layer * BEAD_H)
        rib = zig_zag(layer * BEAD_H)
        # Even layers: wall, then the rib out to the round side. Odd layers
        # come back along the rib and finish with the wall.
        path = np.vstack([wall, rib] if layer % 2 == 0 else [rib[::-1], wall])
        path = subdivide(round_corners(path, CORNER_R), STEP_MM)
        z = (layer + 1) * BEAD_H  # nozzle height: the top of this layer's bead
        spine.append(np.column_stack([path, np.full(len(path), z)]))
    return np.concatenate(spine).astype(np.float32)


v = viewer()
v.clear()

spine = make_toolpath()
n = len(spine)

v.add_grid("floor", cell_size=TILE_MM, extent=1200.0, position=[SIDE / 2, SIDE / 2, 0])

maps = texture_set(color="#0b0c0c", width=BEAD_W, height=BEAD_H, length=TILE_MM)
v.add_parametric_tube(
    "bead",
    spine,
    np.full(n, BEAD_W, dtype=np.float32),
    np.full(n, BEAD_H, dtype=np.float32),
    anchor="top",  # the spine is the nozzle tip; the bead hangs below it
    texture=maps["color"],
    normal_map=maps["normal"],
    roughness_map=maps["roughness"],
    texture_length=TILE_MM,
    roughness=0.9,
    metalness=0.0,
)

# Print the part: one keyframe per toolpath point, paced by distance.
travelled = np.concatenate(
    [[0.0], np.cumsum(np.linalg.norm(np.diff(spine, axis=0), axis=1))]
)
anim = Animation(loop=True)
anim.set_frame_times((travelled / travelled[-1] * PRINT_SECONDS).astype(np.float32))
anim.set_draw_range_data(["bead"], np.linspace(0.0, 1.0, n).reshape(-1, 1))
v.load_animation(anim)

print(
    f"{n} toolpath points, {travelled[-1] / 1000:.1f} m of bead, "
    f"texture repeating every {TILE_MM:g} mm."
)
v.wait_for_assets()
