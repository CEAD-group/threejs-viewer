"""Bead texture handedness: v follows the world side of the bead, not the ring's U."""

import numpy as np
import pytest

from test_parametric_tube import _png_bytes, _wait_for_object
from test_tube_geometry_regressions import worker_code  # noqa: F401

pytestmark = pytest.mark.browser
_TUBE_SIDE_INFO = """(id) => {
    const o = window.threejsViewer._objects.get(id), g = o.geometry, md = o.userData.tubeMorphData;
    return {uv: Array.from(g.getAttribute('uv').array), pos: Array.from(g.getAttribute('position').array),
        frames: Array.from(md.localFrames), tangents: Array.from(md.tangents),
        n: o.userData.tubeNumSpinePoints, mirror: md.texMirror ? Array.from(md.texMirror) : null};
}"""


def _assert_texture_v_follows_world_side(info, min_flipped):
    """The ring vertex carrying v = 0 (the tile's first side) lies on the
    +(V x T) side of every ring, whichever way the ring's U points."""
    n = info["n"]
    uv = np.asarray(info["uv"]).reshape(-1, 2)
    pos = np.asarray(info["pos"]).reshape(-1, 3)
    frames = np.asarray(info["frames"]).reshape(-1, 6)
    tangents = np.asarray(info["tangents"]).reshape(-1, 3)
    ref = np.cross(frames[:, 3:], tangents)
    flipped = np.einsum("ij,ij->i", frames[:, :3], ref) < 0
    assert flipped.sum() >= min_flipped, "the spine must contain a frame flip"
    if info.get("mirror") is not None:
        np.testing.assert_array_equal(np.asarray(info["mirror"]) == 1, flipped)
    rings_v = uv[: n * 6, 1].reshape(n, 6)
    rings_p = pos[: n * 6].reshape(n, 6, 3)
    checked = 0
    for i in range(n):
        if np.linalg.norm(ref[i]) < 1e-6:
            continue
        k0 = int(rings_v[i].argmin())
        side = (rings_p[i, k0] - rings_p[i].mean(axis=0)) @ ref[i]
        assert side > 0, (i, bool(flipped[i]), side)
        checked += 1
    assert checked > n // 2
    return flipped


@pytest.mark.parametrize("joins", ["miter", "round"])
def test_textured_tube_keeps_one_handedness_across_a_reversal(
    viewer_client,
    viewer_page,
    tmp_path,
    worker_code,  # noqa: F811
    joins,
):
    """After a hairpin the frame sweep leaves U flipped for the rest of the
    run; v is mirrored on those rings so the tile's sides stay on the same
    world side (boat layers 48 and 52 read as differently shaded bands),
    on the main-thread build, the worker build, and the draw-range morph."""
    # A straight run, an exact retrace and an L turn: the sweep flips U at
    # the reversal and every later ring keeps the flipped frame.
    out = [[x, 0.0, 0.0] for x in range(0, 21)]
    back = [[x, 0.0, 0.0] for x in range(19, 9, -1)]
    turn = [[10.0, y, 0.0] for y in range(1, 11)]
    spine = np.asarray(out + back + turn, dtype=np.float32)
    n = len(spine)
    widths, heights = np.full(n, 9.46, np.float32), np.full(n, 2.5, np.float32)
    png = tmp_path / "map.png"
    png.write_bytes(_png_bytes())
    viewer_client.add_parametric_tube(
        "t", spine, widths, heights, lod=False, anchor="top", texture=png, joins=joins
    )
    _wait_for_object(viewer_page, "t")
    info = viewer_page.evaluate(_TUBE_SIDE_INFO, "t")
    flipped = _assert_texture_v_follows_world_side(info, min_flipped=3)

    # Draw-range morph with the frontier on a flipped ring keeps the mirror.
    i = int(np.nonzero(flipped)[0][0]) + 2
    assert flipped[i]
    viewer_page.evaluate(
        "v => window.threejsViewer._objects.get('t').userData.parametricTube.setDrawRange(v)",
        (i - 0.5) / (n - 1),
    )
    _assert_texture_v_follows_world_side(viewer_page.evaluate(_TUBE_SIDE_INFO, "t"), 3)

    # Worker build: same invariant through the injected helpers.
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(spine, axis=0), axis=1))]
    )
    worker = viewer_page.evaluate(
        """({code, spine, widths, heights, arc, tol}) => {
            const run = new Function('self', 'spine', 'widths', 'heights', 'arc', 'tol', code + `
                const n = widths.length;
                const geo = buildGeometry(spine, widths, heights, [0, 0, 1], null, null, null, tol, null, 5, 0.15);
                const uv = new Float32Array(geo.totalVerts * 2), alt = new Float32Array(geo.totalVerts);
                const mirror = tubeTexMirrorFlags(geo.localFrames, geo.tangents, n);
                writeTubeUVs(uv, alt, arc, widths, heights, n, 6, N_CAP_RINGS, null, 0,
                             geo.jointVerts, geo.jointVertBase, geo.jointRings, mirror);
                return {uv: Array.from(uv), pos: Array.from(geo.positions), frames: Array.from(geo.localFrames),
                    tangents: Array.from(geo.tangents), n, mirror: Array.from(mirror)};`);
            return run({}, new Float32Array(spine.flat()), new Float32Array(widths), new Float32Array(heights),
                new Float64Array(arc), tol);
        }""",
        dict(
            code=worker_code,
            spine=spine.tolist(),
            widths=widths.tolist(),
            heights=heights.tolist(),
            arc=arc.tolist(),
            tol=0.015 if joins == "round" else 0,
        ),
    )
    _assert_texture_v_follows_world_side(worker, min_flipped=3)
