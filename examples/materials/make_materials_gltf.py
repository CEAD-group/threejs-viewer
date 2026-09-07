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
the bottom face is omitted (it sits flush on the cap). The name is wrapped at
its ``_`` separators into up to ``TEXT_MAX_LINES`` lines, reading correctly
from +Z looking down with +Y up; the wrap, the block's vertical offset, and the
pixel size (capped at ``TEXT_PIXEL_MAX``) are chosen per name to maximise the
pixel size such that every line fits the flat 270° disc — a line may use the
full chord where it lies below the cap centre and only the left half where it
lies above it (the removed wedge is the +x/+y quadrant); each line is centred
in its own allowed span.

Geometry is shared through the glTF node graph to keep the file small: the base
body is one set of accessors used by every material's mesh, and every glyph is
one set of accessors (boxes in pixel units, glyph top-left at the origin)
referenced by a per-(glyph, material) mesh, with one child node per character
placing it under its material's body node (translation = pen position, scale =
[pixel, pixel, 1]). So the bin holds one body plus one copy of each glyph used,
not a copy per character. The material definitions and node -> material
mapping are carried over unchanged.

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

TEXT_HEIGHT = 0.08  # emboss height above the cap
TEXT_PIXEL_MAX = 0.05  # largest pixel size (glyph height = 7 px)
TEXT_PIXEL_STEP = 0.0005  # pixel-size search resolution
TEXT_MAX_LINES = 3
TEXT_MARGIN = 0.96  # fraction of the flat cap radius the text may reach
TEXT_LINE_GAP = 2  # blank pixel rows between lines
TEXT_OFFSET_STEPS = 48  # candidate block-top positions between the rim and centre
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


def glyph_pixels(ch: str) -> set[tuple[int, int]]:
    """Lit (col, row) pixels of one trimmed glyph (row 0 = top)."""
    rows = glyph(ch)
    return {(i, r) for r, row in enumerate(rows) for i, c in enumerate(row) if c == "#"}


def line_glyphs(text: str) -> tuple[list[tuple[str, int]], int]:
    """(char, pen x in px) per character of a text line, and its width in px."""
    pens, x = [], 0
    for ch in text:
        pens.append((ch, x))
        x += len(glyph(ch)[0]) + 1
    return pens, x - 1


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


def wrap_candidates(name: str) -> list[list[str]]:
    """Every way to wrap the name at its '_' separators into <= TEXT_MAX_LINES.

    Lines keep the separators they absorb (``A_B`` stays ``A_B``), so the text
    on the cap is always the literal name; only the line breaks vary.
    """
    tokens = name.split("_")
    out: list[list[str]] = []

    def rec(i: int, lines: list[str]) -> None:
        if i == len(tokens):
            out.append(lines)
            return
        if len(lines) == TEXT_MAX_LINES:
            return
        for j in range(i + 1, len(tokens) + 1):
            rec(j, lines + ["_".join(tokens[i:j])])

    rec(0, [])
    return out


def line_span(y_top: float, y_bot: float, reach: float) -> tuple[float, float] | None:
    """Allowed x-range for a text line occupying y in [y_bot, y_top].

    The flat cap is a disc of radius ``reach`` minus the +x/+y quadrant: below
    the centre a line may use the full chord at its farthest y; a line that
    reaches above the centre must stop at x = 0 on the right.
    """
    far = max(abs(y_top), abs(y_bot))
    if far >= reach:
        return None
    chord = float(np.sqrt(reach**2 - far**2))
    return -chord, (0.0 if y_top > 0 else chord)


def layout_text(name: str, cap_radius: float):
    """Return (pixel size, [(line glyph pens, x0, y_top)]) for the name.

    Searches the wrap (``wrap_candidates``), the block's vertical offset (from
    the block touching the rim down to its top at the centre) and the pixel
    size, for the largest pixel size (capped at TEXT_PIXEL_MAX) at which every
    line fits its ``line_span``; ties prefer fewer lines, then the block
    centred closest to the cap centre. Each line is centred in its own span.
    """
    reach = TEXT_MARGIN * cap_radius
    candidates = [[line_glyphs(t) for t in w] for w in wrap_candidates(name)]
    pitch = FONT_ROWS + TEXT_LINE_GAP

    def place(lines, p, block_top):
        out, y_top = [], block_top
        for pens, width in lines:
            span = line_span(y_top, y_top - FONT_ROWS * p, reach)
            if span is None or span[1] - span[0] < width * p:
                return None
            out.append((pens, (span[0] + span[1] - width * p) / 2, y_top))
            y_top -= pitch * p
        return out

    p = TEXT_PIXEL_MAX
    while p > 0:
        best = None
        for lines in candidates:
            height = (len(lines) * pitch - TEXT_LINE_GAP) * p
            for block_top in np.linspace(reach, 0.0, TEXT_OFFSET_STEPS):
                placed = place(lines, p, block_top)
                if placed is None:
                    continue
                key = (len(lines), abs(block_top - height / 2))
                if best is None or key < best[0]:
                    best = (key, placed)
        if best is not None:
            return p, best[1]
        p = round(p - TEXT_PIXEL_STEP, 6)
    raise ValueError(f"cannot fit {name!r} on the cap")


def text_placements(
    name: str, cap_radius: float
) -> list[tuple[str, float, float, float]]:
    """(char, x, y_top, pixel size) for every character of the name on the cap."""
    p, lines = layout_text(name, cap_radius)
    return [
        (ch, x0 + pen * p, y_top, p) for pens, x0, y_top in lines for ch, pen in pens
    ]


def glyph_geometry(ch: str):
    """Return (positions, normals, indices) of one glyph as raised boxes.

    Pixel units, glyph top-left at the origin (x right, y down from 0 to
    -FONT_ROWS), z from 0 to TEXT_HEIGHT (unscaled: the placing node scales
    x/y by the pixel size and z by 1). One box per merged pixel run: top face
    plus four sides with hard normals, no bottom face (flush on the cap).
    Same array contract as ``pacman_cylinder``.
    """
    pos, nrm, tris = [], [], []

    def quad(corners, n):
        base = len(pos)
        pos.extend(corners)
        nrm.extend([n] * 4)
        tris.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])

    z0, z1 = 0.0, TEXT_HEIGHT
    for xa, xb, br0, br1 in pixel_boxes(glyph_pixels(ch)):
        ya, yb = -br1, -br0
        quad([(xa, ya, z1), (xb, ya, z1), (xb, yb, z1), (xa, yb, z1)], (0, 0, 1))
        quad([(xa, ya, z0), (xb, ya, z0), (xb, ya, z1), (xa, ya, z1)], (0, -1, 0))
        quad([(xa, yb, z0), (xb, yb, z0), (xb, yb, z1), (xa, yb, z1)], (0, 1, 0))
        quad([(xa, ya, z0), (xa, yb, z0), (xa, yb, z1), (xa, ya, z1)], (-1, 0, 0))
        quad([(xb, ya, z0), (xb, yb, z0), (xb, yb, z1), (xb, ya, z1)], (1, 0, 0))
    return orient(pos, nrm, tris)


def orient(pos, nrm, tris):
    """Pack to f32/u32, flipping every triangle to agree with its vertex normals."""
    pos = np.asarray(pos, np.float32)
    nrm = np.asarray(nrm, np.float32)
    tris = np.asarray(tris, np.int64)
    t = pos[tris]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    flip = (fn * nrm[tris].mean(1)).sum(1) < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    idx_dtype = np.uint16 if len(pos) <= 0xFFFF else np.uint32
    return pos, nrm, tris.ravel().astype(idx_dtype)


def pacman_cylinder():
    """Return (positions (N,3) f32, normals (N,3) f32, indices (M,) u32)."""
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

    # Orient every triangle to agree with its (per-surface constant) vertex normal.
    return orient(pos, nrm, tris)


def main() -> None:
    old = json.loads(GLTF.read_text())
    materials = old["materials"]
    # Preserve node name -> material mapping from the current file (its scene
    # roots are the bodies; glyph child nodes carry no name).
    node_mats = []
    for i in old["scenes"][old.get("scene", 0)]["nodes"]:
        n = old["nodes"][i]
        node_mats.append(
            (n["name"], old["meshes"][n["mesh"]]["primitives"][0].get("material"))
        )

    blob = b""
    buffer_views, accessors = [], []

    def add_geometry(positions, normals, indices) -> int:
        """Append one geometry's three accessors; return the first accessor index."""
        nonlocal blob
        first = len(accessors)
        index_type = {np.uint16: 5123, np.uint32: 5125}[indices.dtype.type]
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
        accessors.append(
            {
                "bufferView": first,
                "componentType": 5126,
                "count": len(positions),
                "type": "VEC3",
                "min": positions.min(0).tolist(),
                "max": positions.max(0).tolist(),
            }
        )
        accessors.append(
            {
                "bufferView": first + 1,
                "componentType": 5126,
                "count": len(normals),
                "type": "VEC3",
            }
        )
        accessors.append(
            {
                "bufferView": first + 2,
                "componentType": index_type,
                "count": len(indices),
                "type": "SCALAR",
            }
        )
        return first

    def primitive(first: int, mat) -> dict:
        return {
            "attributes": {"POSITION": first, "NORMAL": first + 1},
            "indices": first + 2,
            "material": mat,
        }

    # Shared base body; glyph accessors and (glyph, material) meshes on demand.
    body = pacman_cylinder()
    body_acc = add_geometry(*body)
    meshes, nodes = [], []
    glyph_acc: dict[str, int] = {}
    glyph_mesh: dict[tuple[str, int | None], int] = {}
    n_tris = len(body[2]) // 3
    for name, mat in node_mats:
        children = []
        for ch, x, y_top, p in text_placements(name, RADIUS - EDGE):
            key = ch.upper() if ch.upper() in FONT else "?"
            if key not in glyph_acc:
                glyph_acc[key] = add_geometry(*glyph_geometry(key))
            if (key, mat) not in glyph_mesh:
                glyph_mesh[key, mat] = len(meshes)
                meshes.append(
                    {
                        "name": f"glyph_{key}",
                        "primitives": [primitive(glyph_acc[key], mat)],
                    }
                )
            n_tris += accessors[glyph_acc[key] + 2]["count"] // 3
            children.append(len(nodes))
            nodes.append(
                {
                    "mesh": glyph_mesh[key, mat],
                    "translation": [x, y_top, HEIGHT / 2],
                    "scale": [p, p, 1.0],
                }
            )
        meshes.append({"name": name, "primitives": [primitive(body_acc, mat)]})
        nodes.append(
            {
                "name": name,
                "mesh": len(meshes) - 1,
                "translation": CENTRES[name],
                "children": children,
            }
        )
    roots = [i for i, n in enumerate(nodes) if "name" in n]
    BIN.write_bytes(blob)

    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "threejs-viewer examples/materials/make_materials_gltf.py",
            "copyright": old["asset"].get("copyright", ""),
        },
        "extensionsUsed": old.get("extensionsUsed", []),
        "scene": 0,
        "scenes": [{"name": "Scene", "nodes": roots}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(blob), "uri": BIN.name}],
    }
    GLTF.write_text(json.dumps(gltf, indent=1) + "\n")
    print(
        f"{GLTF.name}: {len(roots)} bodies, {len(nodes) - len(roots)} glyph nodes, "
        f"{len(glyph_acc)} glyphs, {n_tris} tris drawn "
        f"(body {len(body[2]) // 3}), {BIN.name} {len(blob)} bytes"
    )


if __name__ == "__main__":
    main()
