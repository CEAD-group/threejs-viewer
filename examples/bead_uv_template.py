"""
Export two sample beads as an OBJ, for painting the bead texture in Blender
(or any other paint tool) and seeing how it behaves on real geometry.

Both beads are built the way the viewer builds ``add_parametric_tube``: the
same chamfered hexagonal cross-section, constant-up frames, mitred corners,
and the same UV layout as ``texture=...``:

- u is the arc length along the bead divided by ``--length`` (so it runs
  past 1 and the image repeats; its left edge must match its right edge),
- v runs 0..1 across the bead, from one side over the top to the other; the
  bottom half mirrors the top, so a stroke on the top also paints the bottom.

``bead_wavy`` is a run of S-bends whose width and height swell and shrink
along the way. ``bead_corners`` has a sharp 90 degree corner, a tight U-turn
and a climbing leg, with the section ramping wider and taller. The ends are
left open. Coordinates are written Y-up, which Blender's OBJ importer
converts by default, so the beads lie in the XY plane with Z up.

Run: uv run python examples/bead_uv_template.py [--length 25] [-o beads.obj]
"""

import argparse
from pathlib import Path

import numpy as np

MITER_LIMIT = 2.0  # as the viewer's TUBE_MITER_LIMIT
UP = np.array([0.0, 0.0, 1.0])


def section(width, height):
    """The 6 cross-section points (u, v), as the viewer's sampleChamferedRect."""
    hw, hh = width / 2, height / 2
    c = min(hw, hh)
    if width >= height:
        pts = [
            (hw, 0),
            (hw - c, hh),
            (c - hw, hh),
            (-hw, 0),
            (c - hw, -hh),
            (hw - c, -hh),
        ]
    else:
        pts = [
            (hw, c - hh),
            (hw, hh - c),
            (0, hh),
            (-hw, hh - c),
            (-hw, c - hh),
            (0, -hh),
        ]
    return np.array(pts, dtype=float)


def section_tex_v(width, height):
    """Across-the-bead texture coordinate, as the viewer's tubeSectionTexV."""
    hw, hh = width / 2, height / 2
    c = min(hw, hh)
    diag = c * np.sqrt(2)
    if width >= height:
        a = diag / (2 * diag + (width - 2 * c))
        return [0, a, 1 - a, 1, 1 - a, a]
    wall = hh - c
    a = wall / (2 * (wall + diag))
    return [a, a, 0.5, 1 - a, 1 - a, 0.5]


def build_bead(spine, widths, heights, tile_length):
    """Ring vertices (N, 6, 3) and UVs (N, 6, 2) of a bead along ``spine``."""
    spine = np.asarray(spine, dtype=float)
    n = len(spine)
    seg = np.diff(spine, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    dirs = seg / seg_len[:, None]
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    verts = np.empty((n, 6, 3))
    uvs = np.empty((n, 6, 2))
    for i in range(n):
        d_in = dirs[max(i - 1, 0)]
        d_out = dirs[min(i, n - 2)]
        # Unit-bisector tangent, V from the constant up vector, U = V x T.
        t = d_in + d_out
        t /= np.linalg.norm(t)
        v_axis = UP - UP.dot(t) * t
        v_axis /= np.linalg.norm(v_axis)
        u_axis = np.cross(v_axis, t)
        # Mitre: stretch the section by 1/cos(half turn) along the turn
        # direction, dropped (bevel) past the mitre limit.
        cos_half = np.sqrt(max(0.0, (1 + d_in.dot(d_out)) / 2))
        scale = 1.0 if cos_half < 1 / MITER_LIMIT else 1 / cos_half
        pts = section(widths[i], heights[i])
        m = d_out - d_in
        m_uv = np.array([m.dot(u_axis), m.dot(v_axis)])
        if scale != 1.0 and np.linalg.norm(m_uv) > 1e-12:
            m_uv /= np.linalg.norm(m_uv)
            pts = pts + np.outer((scale - 1) * pts.dot(m_uv), m_uv)
        verts[i] = spine[i] + np.outer(pts[:, 0], u_axis) + np.outer(pts[:, 1], v_axis)
        uvs[i, :, 0] = arc[i] / tile_length
        uvs[i, :, 1] = section_tex_v(widths[i], heights[i])
    return verts, uvs


def resample(points, step):
    """Subdivide a polyline so no segment is longer than ``step`` (corner
    points are kept), giving the varying section enough rings to show."""
    points = np.asarray(points, dtype=float)
    out = [points[:1]]
    for a, b in zip(points[:-1], points[1:]):
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.append(a + np.outer(np.arange(1, k + 1) / k, b - a))
    return np.concatenate(out)


def wavy_bead():
    x = np.linspace(0, 150, 151)
    spine = np.column_stack([x, 15 * np.sin(2 * np.pi * x / 75), np.zeros_like(x)])
    widths = 9 + 3 * np.sin(2 * np.pi * x / 50)  # 6..12
    heights = 4 + 1 * np.cos(2 * np.pi * x / 60)  # 3..5
    return spine, widths, heights


def corners_bead():
    # Up the first leg, over a U-turn (it leaves heading back down), then a
    # second sharp corner into the climbing leg.
    a = np.radians(np.linspace(180, 0, 25))
    u_turn = np.column_stack([52 + 12 * np.cos(a), -40 + 12 * np.sin(a), np.zeros(25)])
    corners = np.concatenate(
        [
            [[0, -80, 0], [40, -80, 0]],  # sharp 90 degree corner at (40, -80)
            u_turn,  # tight U-turn, radius 12
            [[64, -70, 0], [150, -70, 12]],  # sharp corner, then the climbing leg
        ]
    )
    spine = resample(corners, 2.5)
    f = np.linspace(0, 1, len(spine))
    widths = 8 + 6 * np.sin(np.pi * f)  # 8 -> 14 -> 8
    heights = 4 + 2 * f  # 4 -> 6
    return spine, widths, heights


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--length", type=float, default=25.0, help="texture_length (mm)")
    ap.add_argument("-o", "--output", type=Path, default=Path("bead_section.obj"))
    args = ap.parse_args()

    lines = [f"# sample beads, texture tile {args.length:g} mm"]
    base = 1  # OBJ indices are 1-based and global across objects
    for name, (spine, widths, heights) in (
        ("bead_wavy", wavy_bead()),
        ("bead_corners", corners_bead()),
    ):
        verts, uvs = build_bead(spine, widths, heights, args.length)
        n = len(verts)
        lines.append(f"o {name}")
        # Scene is Z-up, OBJ is Y-up: (x, y, z) -> (x, z, -y).
        lines += [f"v {x:.5f} {z:.5f} {-y:.5f}" for x, y, z in verts.reshape(-1, 3)]
        lines += [f"vt {u:.6f} {v:.6f}" for u, v in uvs.reshape(-1, 2)]
        lines.append("s 1")
        for i in range(n - 1):
            a0, b0 = base + i * 6, base + (i + 1) * 6
            for j in range(6):
                jn = (j + 1) % 6
                quad = (a0 + j, a0 + jn, b0 + jn, b0 + j)
                lines.append("f " + " ".join(f"{k}/{k}" for k in quad))
        base += n * 6
        print(f"{name}: {n} rings, {uvs[-1, 0, 0]:.2f} tiles long")
    args.output.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
