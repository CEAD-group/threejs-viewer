"""Regenerate materials.gltf / materials.bin: one test body per material.

The test body is an upright cylinder standing on the floor (axis +Z — the
file keeps Blender's Z-up frame, like the original export) with a 90° "pac-man"
wedge removed (0°..90°), so two flat radial cut faces are exposed. The closed
top rim is treated two ways over the remaining 270°: a quarter-round fillet on
the first half and a 45° chamfer on the second half, both sized 25% of the
height, so fillet and chamfer are compared on one material. The bottom rim is
sharp. Normals are smooth around the circumference and across the fillet
(tangent-continuous into wall and cap); the chamfer, the cut faces, and the
bottom rim are hard edges. The bodies sit at the same grid positions as the
original Blender export (``CENTRES``).

The material name is embossed on the flat top cap in a built-in 5x7 block font
(no external font files): every lit pixel becomes a box standing on the cap,
horizontal runs and identical stacked runs are merged into one box each, and
the bottom face is omitted (it sits flush on the cap). The name is split on
``_`` into up to ``TEXT_MAX_LINES`` lines laid out in the lower half of the cap
disc, reading correctly from +Z looking down with +Y up, and each node's pixel
size is the largest (capped at ``TEXT_PIXEL_MAX``) that fits every line inside
the flat disc. So each node carries its own geometry; the material definitions
and node -> material mapping are carried over unchanged.

The material definitions are read from the existing materials.gltf, so this
script only replaces the geometry and layout:

    uv run python examples/materials/make_materials_gltf.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
GLTF = HERE / "materials.gltf"
BIN = HERE / "materials.bin"

RADIUS = 1.29  # footprint of the original cubes
HEIGHT = 2.0
EDGE = 0.25 * HEIGHT  # fillet radius / chamfer size on the top rim
SEGMENTS = 64  # columns per full circle
WEDGE_DEG = 90.0  # pac-man wedge removed, starting at 0°
RIM_STEPS = 12  # rings across the fillet arc (and the chamfer, for a uniform strip)

TEXT_HEIGHT = 0.06  # emboss height above the cap
TEXT_PIXEL_MAX = 0.04  # largest pixel size (glyph height = 7 px)
TEXT_MAX_LINES = 3
TEXT_MARGIN = 0.94  # fraction of the flat cap radius the text may reach
TEXT_LINE_GAP = 3  # blank pixel rows between lines
FONT_ROWS = 7

# Node name -> centre, taken from the original Blender export's bounding boxes.
CENTRES: dict[str, list[float]] = {
    "ANODIZED_BLACK": [0.0, 0.0, 0.0],
    "ANODIZED_CLEAR": [3.877000093460083, 0.0, 0.0],
    "ANODIZED_VIBRANT_BLUE": [7.7540998458862305, 0.0, 0.0],
    "ANODIZED_CHARCOAL_GREY": [11.631099700927734, 0.0, 0.0],
    "BEADBLASTED_STEEL": [27.139299392700195, 0.0, 0.0],
    "BEADBLASTED_ALUMINIUM": [31.016399383544922, 0.0, 0.0],
    "POWDERCOAT_WHITE": [3.877000093460083, 3.877000093460083, 0.0],
    "POWDERCOAT_BLACK": [7.7540998458862305, 3.877000093460083, 0.0],
    "SPRAYPAINTED_BLACK": [11.631099700927734, 3.877000093460083, 0.0],
    "SPRAYPAINTED_WHITE": [15.508199691772461, 3.877000093460083, 0.0],
    "HARDCHROMATIC": [19.38520050048828, 3.877000093460083, 0.0],
    "BLACK_OXIDE": [34.89339828491211, 0.0, 0.0],
    "COLORDYE_BLACK": [23.262300491333008, 3.877000093460083, 0.0],
    "POLISHED_STEEL": [7.7540998458862305, 7.7540998458862305, 0.0],
    "GOLD": [31.016399383544922, 3.877000093460083, 0.0],
    "COPPER": [15.508199691772461, 7.7540998458862305, 0.0],
    "TITANIUM": [19.38520050048828, 7.7540998458862305, 0.0],
    "BRASS": [23.262300491333008, 7.7540998458862305, 0.0],
    "STEEL": [15.508199691772461, 11.631099700927734, 0.0],
    "PLASTIC_BLUE": [34.89339828491211, 3.877000093460083, 0.0],
    "PA630GF": [0.0, 7.7540998458862305, 0.0],
    "PLASTIC_BLACK": [3.877000093460083, 7.7540998458862305, 0.0],
    "PC": [0.0, 11.631099700927734, 0.0],
    "PVC": [3.877000093460083, 11.631099700927734, 0.0],
    "HDPE": [7.7540998458862305, 11.631099700927734, 0.0],
    "POM": [11.631099700927734, 11.631099700927734, 0.0],
    "PA630GF.001": [0.0, 7.7540998458862305, 0.0],
    "POM.001": [11.631099700927734, 11.631099700927734, 0.0],
    "PAINT_GREY": [34.89339828491211, 7.7540998458862305, 0.0],
    "PAINT_LIGHT_GREY": [27.139299392700195, 7.7540998458862305, 0.0],
}

# 5x7 block font: '#' = lit pixel, rows top to bottom. Glyphs are trimmed to
# their lit columns (proportional advance) with a one-pixel gap between them.
FONT: dict[str, list[str]] = {
    "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "B": ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."],
    "C": [".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."],
    "D": ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."],
    "E": ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
    "F": ["#####", "#....", "#....", "####.", "#....", "#....", "#...."],
    "G": [".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".####"],
    "H": ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "I": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "#####"],
    "J": ["..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."],
    "K": ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"],
    "L": ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
    "M": ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"],
    "N": ["#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"],
    "O": [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "P": ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."],
    "Q": [".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"],
    "R": ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
    "S": [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
    "T": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    "U": ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "V": ["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."],
    "W": ["#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"],
    "X": ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"],
    "Y": ["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."],
    "Z": ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"],
    "0": [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    "3": ["#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."],
    "4": ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    "5": ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    "6": ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."],
    "7": ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    "8": [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    "9": [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
    ".": [".....", ".....", ".....", ".....", ".....", ".##..", ".##.."],
    "-": [".....", ".....", ".....", "#####", ".....", ".....", "....."],
    "_": [".....", ".....", ".....", ".....", ".....", ".....", "#####"],
    "?": [".###.", "#...#", "....#", "...#.", "..#..", ".....", "..#.."],
}


def glyph(ch: str) -> list[str]:
    """Rows of the glyph trimmed to its lit columns (unknown chars -> '?')."""
    rows = FONT.get(ch.upper(), FONT["?"])
    lit = [i for i in range(len(rows[0])) if any(r[i] == "#" for r in rows)]
    return [r[lit[0] : lit[-1] + 1] for r in rows]


def line_pixels(text: str) -> tuple[set[tuple[int, int]], int]:
    """Lit (col, row) pixels of a text line (row 0 = top) and its width in px."""
    pixels: set[tuple[int, int]] = set()
    x = 0
    for ch in text:
        rows = glyph(ch)
        for r, row in enumerate(rows):
            pixels.update((x + i, r) for i, c in enumerate(row) if c == "#")
        x += len(rows[0]) + 1
    return pixels, x - 1


def pixel_boxes(pixels: set[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    """Merge lit pixels into (x0, x1, r0, r1) boxes, half-open on x1 / r1.

    Horizontal runs per row first; identical runs on consecutive rows are then
    stacked into one box.
    """
    runs: dict[int, set[tuple[int, int]]] = {}
    for r in sorted({p[1] for p in pixels}):
        cols = sorted(c for c, rr in pixels if rr == r)
        row_runs, start = set(), cols[0]
        for a, b in zip(cols, cols[1:] + [None]):
            if b != a + 1:
                row_runs.add((start, a + 1))
                start = b
        runs[r] = row_runs
    boxes: list[tuple[int, int, int, int]] = []
    open_boxes: dict[tuple[int, int], int] = {}  # (x0, x1) -> top row of open box
    for r in range(max(runs) + 2):
        here = runs.get(r, set())
        for key, r0 in list(open_boxes.items()):
            if key not in here:
                boxes.append((key[0], key[1], r0, r))
                del open_boxes[key]
        for key in sorted(here):
            open_boxes.setdefault(key, r)
    return sorted(boxes)


def wrap_name(name: str) -> list[str]:
    """Split on '_' into at most TEXT_MAX_LINES lines (re-joining the tail)."""
    parts = name.split("_")
    if len(parts) > TEXT_MAX_LINES:
        head, tail = parts[: TEXT_MAX_LINES - 1], parts[TEXT_MAX_LINES - 1 :]
        parts = head + ["_".join(tail)]
    return parts


def layout_text(name: str, cap_radius: float):
    """Return (pixel size, [(line pixels, width px, x0, y_top)]) for the name.

    Lines are stacked downwards from just below the cap centre (pitch 7 rows +
    TEXT_LINE_GAP) inside the lower half of the flat disc, so the removed wedge
    (first quadrant) is never touched. The pixel size is the largest, capped at
    TEXT_PIXEL_MAX, at which every line's box fits inside the chord of the disc
    at the line's lowest edge.
    """
    lines = [line_pixels(t) for t in wrap_name(name)]
    reach = TEXT_MARGIN * cap_radius

    def placed(p):
        out, y_top = [], -TEXT_LINE_GAP * p
        for pixels, width in lines:
            out.append((pixels, width, -width * p / 2, y_top))
            y_top -= (FONT_ROWS + TEXT_LINE_GAP) * p
        return out

    def fits(p):
        for _, width, _, y_top in placed(p):
            y_bot = y_top - FONT_ROWS * p
            if abs(y_bot) >= reach:
                return False
            if width * p / 2 > np.sqrt(reach**2 - y_bot**2):
                return False
        return True

    p = TEXT_PIXEL_MAX
    while not fits(p):
        p -= 0.0005
        if p <= 0:
            raise ValueError(f"cannot fit {name!r} on the cap")
    return p, placed(p)


def emboss_text(name: str, z_top: float, cap_radius: float):
    """Return (positions, normals, tris) of the raised name on the cap.

    One box per merged pixel run: top face plus four sides with hard normals,
    no bottom face (it sits flush on the cap at z_top).
    """
    p, lines = layout_text(name, cap_radius)
    pos, nrm, tris = [], [], []

    def quad(corners, n):
        base = len(pos)
        pos.extend(corners)
        nrm.extend([n] * 4)
        tris.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])

    z0, z1 = z_top, z_top + TEXT_HEIGHT
    for pixels, _, x0, y_top in lines:
        for bx0, bx1, br0, br1 in pixel_boxes(pixels):
            xa, xb = x0 + bx0 * p, x0 + bx1 * p
            ya, yb = y_top - br1 * p, y_top - br0 * p
            quad([(xa, ya, z1), (xb, ya, z1), (xb, yb, z1), (xa, yb, z1)], (0, 0, 1))
            quad([(xa, ya, z0), (xb, ya, z0), (xb, ya, z1), (xa, ya, z1)], (0, -1, 0))
            quad([(xa, yb, z0), (xb, yb, z0), (xb, yb, z1), (xa, yb, z1)], (0, 1, 0))
            quad([(xa, ya, z0), (xa, yb, z0), (xa, yb, z1), (xa, ya, z1)], (-1, 0, 0))
            quad([(xb, ya, z0), (xb, yb, z0), (xb, yb, z1), (xb, ya, z1)], (1, 0, 0))
    return np.asarray(pos, np.float64), np.asarray(nrm, np.float64), tris


def pacman_cylinder(name: str):
    """Return (positions (N,3) f32, normals (N,3) f32, indices (M,) u32).

    The test body with ``name`` embossed on its top cap.
    """
    top, bot = HEIGHT / 2, -HEIGHT / 2  # centre at z=0 like the original cubes
    start = np.radians(WEDGE_DEG)
    ncol = int(round(SEGMENTS * (360.0 - WEDGE_DEG) / 360.0)) + 1
    ang = np.linspace(start, 2 * np.pi, ncol)  # open sweep, both ends kept
    c, s_ = np.cos(ang), np.sin(ang)
    radial = np.column_stack([c, s_, np.zeros(ncol)])
    up = np.array([0.0, 0.0, 1.0])
    chamfer_col = ang >= (start + 2 * np.pi) / 2  # second half of the sweep

    pos, nrm, tris = [], [], []

    def add(p, n):
        pos.extend(np.atleast_2d(p))
        nrm.extend(np.atleast_2d(n))

    def ring(radius, z):
        return np.column_stack([radius * c, radius * s_, np.full(ncol, z)])

    def strip(rings):
        """Open quad strip between consecutive rings of ncol columns."""
        base = len(pos) - len(rings) * ncol
        for r in range(len(rings) - 1):
            a0, b0 = base + r * ncol, base + (r + 1) * ncol
            for i in range(ncol - 1):
                tris.extend(
                    [(a0 + i, a0 + i + 1, b0 + i), (b0 + i, a0 + i + 1, b0 + i + 1)]
                )

    # Wall.
    add(ring(RADIUS, bot), radial)
    add(ring(RADIUS, top - EDGE), radial)
    strip([0, 1])

    # Rim: fillet arc on the first half, straight 45° chamfer on the second.
    rim_rings = []
    for k in range(RIM_STEPS + 1):
        t = np.pi / 2 * k / RIM_STEPS
        n_fillet = np.cos(t) * radial + np.sin(t) * up
        p_fillet = ring(RADIUS - EDGE, top - EDGE) + EDGE * n_fillet
        f = k / RIM_STEPS
        p_chamfer = ring(RADIUS - EDGE * f, top - EDGE * (1 - f))
        n_chamfer = (radial + up) / np.sqrt(2.0)
        p = np.where(chamfer_col[:, None], p_chamfer, p_fillet)
        n = np.where(chamfer_col[:, None], n_chamfer, n_fillet)
        rim_rings.append(p)
        add(p, n)
    strip(rim_rings)

    # Top cap: own rim vertices (hard against the chamfer, seamless on the fillet).
    centre = len(pos)
    add([0.0, 0.0, top], up)
    add(ring(RADIUS - EDGE, top), np.tile(up, (ncol, 1)))
    for i in range(ncol - 1):
        tris.append((centre, centre + 1 + i, centre + 2 + i))

    # Bottom cap.
    centre = len(pos)
    add([0.0, 0.0, bot], -up)
    add(ring(RADIUS, bot), np.tile(-up, (ncol, 1)))
    for i in range(ncol - 1):
        tris.append((centre, centre + 2 + i, centre + 1 + i))

    # Two flat radial cut faces, each a fan from the bottom axis point over the
    # column's full profile (rim -> wall top -> rim strip -> top axis).
    for col in (0, ncol - 1):
        profile = [pos[col]] + [
            r[col] for r in rim_rings
        ]  # rim_rings[0] is the wall top
        profile += [[0.0, 0.0, top]]
        # Outward = away from the solid, which lies at increasing angle for the
        # first cut and decreasing angle for the last.
        tangent = np.array([-s_[col], c[col], 0.0])
        n = -tangent if col == 0 else tangent
        base = len(pos)
        add([0.0, 0.0, bot], n)
        add(np.asarray(profile), np.tile(n, (len(profile), 1)))
        for i in range(len(profile) - 1):
            tris.append((base, base + 1 + i, base + 2 + i))

    # Embossed material name standing on the flat cap.
    t_pos, t_nrm, t_tris = emboss_text(name, top, RADIUS - EDGE)
    base = len(pos)
    add(t_pos, t_nrm)
    tris.extend((base + a, base + b, base + c) for a, b, c in t_tris)

    pos = np.asarray(pos, np.float32)
    nrm = np.asarray(nrm, np.float32)
    tris = np.asarray(tris, np.int64)
    # Orient every triangle to agree with its (per-surface constant) vertex normal.
    t = pos[tris]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    flip = (fn * nrm[tris].mean(1)).sum(1) < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return pos, nrm, tris.ravel().astype(np.uint32)


def main() -> None:
    old = json.loads(GLTF.read_text())
    materials = old["materials"]
    # Preserve node name -> material mapping from the current file.
    node_mats = [
        (n["name"], old["meshes"][n["mesh"]]["primitives"][0].get("material"))
        for n in old["nodes"]
    ]

    # One geometry per node (the embossed name differs), packed back to back.
    blob = b""
    buffer_views, accessors, meshes, nodes = [], [], [], []
    n_verts = n_tris = 0
    for i, (name, mat) in enumerate(node_mats):
        positions, normals, indices = pacman_cylinder(name)
        n_verts += len(positions)
        n_tris += len(indices) // 3
        acc_base = len(accessors)
        for arr, target in ((positions, 34962), (normals, 34962), (indices, 34963)):
            data = arr.tobytes()
            buffer_views.append(
                {
                    "buffer": 0,
                    "byteOffset": len(blob),
                    "byteLength": len(data),
                    "target": target,
                }
            )
            blob += data + b"\0" * (-len(data) % 4)
        accessors += [
            {
                "bufferView": acc_base,
                "componentType": 5126,
                "count": len(positions),
                "type": "VEC3",
                "min": positions.min(0).tolist(),
                "max": positions.max(0).tolist(),
            },
            {
                "bufferView": acc_base + 1,
                "componentType": 5126,
                "count": len(normals),
                "type": "VEC3",
            },
            {
                "bufferView": acc_base + 2,
                "componentType": 5125,
                "count": len(indices),
                "type": "SCALAR",
            },
        ]
        meshes.append(
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": {"POSITION": acc_base, "NORMAL": acc_base + 1},
                        "indices": acc_base + 2,
                        "material": mat,
                    }
                ],
            }
        )
        nodes.append({"name": name, "mesh": i, "translation": CENTRES[name]})
    BIN.write_bytes(blob)

    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "threejs-viewer examples/materials/make_materials_gltf.py",
            "copyright": old["asset"].get("copyright", ""),
        },
        "extensionsUsed": old.get("extensionsUsed", []),
        "scene": 0,
        "scenes": [{"name": "Scene", "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(blob), "uri": BIN.name}],
    }
    GLTF.write_text(json.dumps(gltf, indent=1) + "\n")
    print(f"{GLTF.name}: {len(nodes)} nodes, {n_verts} verts, {n_tris} tris")


if __name__ == "__main__":
    main()
