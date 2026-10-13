"""Round tube joins: the z-buffer union of segment prisms and revolved corners."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from test_parametric_tube import _png_bytes, _wait_for_object
from test_tube_geometry_regressions import _assert_outward, worker_code  # noqa: F401

pytestmark = pytest.mark.browser
DATA = Path(__file__).parent / "data"

READ = """(id) => {
    const o = window.threejsViewer._objects.get(id), g = o.geometry, ud = o.userData;
    return {positions: Array.from(g.attributes.position.array),
        normals: Array.from(g.attributes.normal.array),
        indices: Array.from(g.index.array), drawCount: g.drawRange.count,
        joints: Array.from(ud.tubeJointVerts || []), jointRings: ud.tubeJointRings,
        jointIndexCount: ud.tubeJointIndexCount, jointVertBase: ud.tubeJointVertBase,
        capPer: ud.tubeCapIndicesPerCap, perPair: ud.tubeIndicesPerRingPair,
        ringPairs: ud.tubeRingPairs, total: ud.totalIndexCount, joins: ud.tubeJoins,
        widths: Array.from(ud.tubeMorphData.widths), heights: Array.from(ud.tubeMorphData.heights)};
}"""


def _issue_spine(issue):
    return np.asarray(
        json.loads((DATA / f"tube_issue_{issue}.json").read_text()), dtype=np.float32
    )


def _dist_to_polyline(points, spine):
    """Distance from each point to the spine polyline (segment-wise closest point)."""
    a, b = spine[:-1], spine[1:]
    ab = b - a
    ab2 = np.maximum((ab * ab).sum(axis=1), 1e-20)
    best = np.full(len(points), np.inf)
    for i in range(len(a)):
        t = np.clip(((points - a[i]) @ ab[i]) / ab2[i], 0.0, 1.0)
        d = np.linalg.norm(points - (a[i] + t[:, None] * ab[i]), axis=1)
        best = np.minimum(best, d)
    return best


def _pair_offset(g, k):
    """Mirror of tubePairOffset: index offset of ring pair k."""
    joints_before = sum(1 for v in g["joints"] if v <= k)
    return g["capPer"] + k * g["perPair"] + joints_before * g["jointIndexCount"]


@pytest.mark.parametrize("issue, w, h", [(277, 11.83, 2.0), (278, 9.46, 2.5)])
def test_round_joins_keep_every_vertex_inside_the_bead_envelope(
    viewer_client, viewer_page, issue, w, h
):
    """No vertex of a round-join tube lies farther from the spine than the bead
    half-extent plus the join tolerance: a miter spike or a fold fin would."""
    spine = _issue_spine(issue)
    n = len(spine)
    result = {}
    for joins in ("round", "miter"):
        viewer_client.add_parametric_tube(
            joins, spine, np.full(n, w), np.full(n, h), lod=False, joins=joins
        )
        _wait_for_object(viewer_page, joins)
        g = viewer_page.evaluate(READ, joins)
        points = np.asarray(g["positions"]).reshape(-1, 3)
        result[joins] = (g, _dist_to_polyline(points, spine.astype(float)) / (w / 2))
    g, ratio = result["round"]
    assert g["joins"] == "round" and len(g["joints"]) > 0
    assert np.isfinite(ratio).all()
    # A kept miter deviates by at most the join tolerance (1.5%); the
    # deposition bias scales the bead by up to 0.1% on top.
    assert ratio.max() < 1.017, ratio.max()
    _assert_outward(g)
    assert (
        g["total"]
        == len(g["indices"])
        == (
            2 * g["capPer"]
            + g["ringPairs"] * g["perPair"]
            + len(g["joints"]) * g["jointIndexCount"]
        )
    )
    g, ratio = result["miter"]
    assert g["joins"] == "miter" and g["joints"] == []
    assert np.isfinite(ratio).all()
    assert (
        g["total"]
        == len(g["indices"])
        == 2 * g["capPer"] + g["ringPairs"] * g["perPair"]
    )


def test_round_joins_keep_an_anchored_exact_retrace_below_the_spine(
    viewer_client, viewer_page
):
    """At an exact 180 degree retrace the outgoing segment flips T while the
    sweep keeps U, so a joint frame with V = T x U pointed down and an
    anchor="top" ring sat a bead height above the spine (the hairpins test
    data). Both joint rings must follow the neighbouring strip ring's U and
    V, so the outgoing ring is ring v+1 moved back along the segment."""
    fwd = [(i * 0.8, 0.0, 7.5) for i in range(8)]
    back = [((6 - i) * 0.8, 0.0, 7.5) for i in range(7)]
    spine = np.array(fwd + back, dtype=np.float32)
    n, w, h = len(spine), 9.9, 2.475
    viewer_client.add_parametric_tube(
        "t", spine, np.full(n, w), np.full(n, h), lod=False, anchor="top", joins="round"
    )
    _wait_for_object(viewer_page, "t")
    g = viewer_page.evaluate(READ, "t")
    assert g["joints"] == [7]
    points = np.asarray(g["positions"]).reshape(-1, 3)
    # Anchored at the top: nothing above the spine, nothing below one bead.
    assert points[:, 2].max() < 7.5 + 0.02
    assert points[:, 2].min() > 7.5 - h - 0.02
    base = g["jointVertBase"]
    outgoing = points[base : base + 6]
    ring_next = points[8 * 6 : 9 * 6]
    # Strip rings carry the deposition bias (up to 0.1% of the bead), the
    # joint rings do not.
    delta = outgoing - ring_next
    assert np.allclose(delta[:, 1:], 0.0, atol=0.02), delta
    assert np.allclose(delta[:, 0], 0.8, atol=0.02), delta
    _assert_outward(g)


def _corner_spine():
    """Three 90 degree corners with short legs, a bead wider than the legs."""
    pts = [
        (0, 0, 0),
        (10, 0, 0),
        (10, 6, 0),
        (20, 6, 0),
        (20, 0, 0),
        (30, 0, 0),
        (40, 0, 0),
    ]
    return np.array(pts, dtype=np.float32)


def test_round_joins_sit_between_pairs_and_draw_range_restores(
    viewer_client, viewer_page
):
    spine = _corner_spine()
    n = len(spine)
    viewer_client.add_parametric_tube(
        "c", spine, np.full(n, 8.0), np.full(n, 3.0), lod=False, joins="round"
    )
    _wait_for_object(viewer_page, "c")
    initial = viewer_page.evaluate(READ, "c")
    assert initial["joints"] == [1, 2, 3, 4]
    assert initial["jointIndexCount"] == 2 * initial["jointRings"] * 6 * 6
    # A joint's first band stitches its vertex's ring to the body.
    off = _pair_offset(initial, 1) - initial["jointIndexCount"]
    band = initial["indices"][off : off + 36]
    ring1 = set(range(6, 12))
    assert ring1 & set(band)
    assert max(band) >= initial["jointVertBase"]

    def set_range(value):
        viewer_page.evaluate(
            "v => window.threejsViewer._objects.get('c').userData.parametricTube.setDrawRange(v)",
            value,
        )
        return viewer_page.evaluate(READ, "c")

    cap = initial["capPer"]
    pairs = initial["ringPairs"]
    for k in range(1, pairs + 1):
        g = set_range(k / pairs)
        # Pairs 0..k-1 plus the cap right after pair k-1; the joint at vertex k
        # is hidden under the relocated cap until the frontier moves on.
        assert g["drawCount"] == _pair_offset(g, k - 1) + g["perPair"] + cap, k
    for value in (0.3, 0.55, 0.62, 0.9, 0.2):
        g = set_range(value)
        m = int(value * pairs)
        assert g["drawCount"] == _pair_offset(g, m) + g["perPair"] + cap, value
        _assert_outward({**g, "indices": g["indices"][: g["drawCount"]]})
    g = set_range(0.0)
    assert g["drawCount"] == 0
    g = set_range(1.0)
    assert g["drawCount"] == g["total"]
    np.testing.assert_array_equal(g["indices"], initial["indices"])
    np.testing.assert_allclose(g["positions"], initial["positions"], atol=1e-5)


def test_round_joins_skip_break_ends_and_zero_sections(viewer_client, viewer_page):
    spine = _corner_spine()
    n = len(spine)
    widths = np.full(n, 8.0)
    widths[4] = 0.0  # a zero-width corner has no body
    mask = np.zeros(n, dtype=np.uint8)
    mask[2] = 1  # the ribbon breaks before vertex 2: vertices 1 and 2 are strip ends
    viewer_client.add_parametric_tube(
        "b", spine, widths, np.full(n, 3.0), lod=False, break_before=mask, joins="round"
    )
    _wait_for_object(viewer_page, "b")
    g = viewer_page.evaluate(READ, "b")
    assert g["joints"] == [3]
    _assert_outward(g)


def test_round_joins_follow_the_lod_epsilon(viewer_client, viewer_page):
    """Under LOD the join tolerance is floored by the LOD epsilon, so the same
    corners carry joints up close and none from far away."""
    n = 30000
    t = np.linspace(0, 1, n)
    # A square wave in plan: sharp corners every 3000 points on a long path.
    x = t * 300.0
    y = 20.0 * (np.floor(t * 10) % 2)
    spine = np.column_stack([x, y, np.zeros(n)]).astype(np.float32)
    viewer_client.add_parametric_tube(
        "l", spine, np.full(n, 8.0), np.full(n, 3.0), joins="round"
    )
    _wait_for_object(viewer_page, "l")
    viewer_client.frame_object("l")
    time.sleep(0.5)

    def joints_after(camera_distance):
        viewer_client.set_camera(
            position=[150, 10, camera_distance], target=[150, 10, 0]
        )
        deadline = time.time() + 8
        last = None
        while time.time() < deadline:
            time.sleep(0.4)
            last = viewer_page.evaluate(
                "() => { const ud = window.threejsViewer._objects.get('l').userData;"
                " return {joints: (ud.tubeJointVerts || []).length, lod: !!ud.tubeLOD,"
                " n: ud.tubeNumSpinePoints}; }"
            )
            if (camera_distance > 1e4) == (last["joints"] == 0):
                return last
        return last

    near = joints_after(60.0)
    assert near["lod"] and near["n"] < n
    assert near["joints"] > 0, near
    far = joints_after(5e5)
    assert far["joints"] == 0, far
    near = joints_after(60.0)
    assert near["joints"] > 0, near


def test_round_joins_take_colour_updates_and_textures(
    viewer_client, viewer_page, tmp_path
):
    spine = _corner_spine()
    n = len(spine)
    mask = np.zeros(n, dtype=np.uint8)
    mask[5] = 1
    png = tmp_path / "map.png"
    png.write_bytes(_png_bytes())
    viewer_client.add_parametric_tube(
        "t",
        spine,
        np.full(n, 8.0),
        np.full(n, 3.0),
        lod=False,
        colors=np.full(n, 0xFF0000, dtype=np.uint32),
        break_before=mask,
        texture=png,
        joins="round",
    )
    _wait_for_object(viewer_page, "t")
    viewer_client.update_parametric_tube_colors(
        "t", np.full(n, 0x0000FF, dtype=np.uint32)
    )
    deadline = time.time() + 5
    while time.time() < deadline:
        time.sleep(0.2)
        state = viewer_page.evaluate(
            """() => { const g = window.threejsViewer._objects.get('t').geometry;
            const c = g.attributes.color.array, uv = g.attributes.uv.array;
            let red = 0; for (let i = 0; i < c.length; i += 3) if (c[i] > 0.5 || c[i + 2] < 0.5) red++;
            let bad = 0; for (const v of uv) if (!Number.isFinite(v)) bad++;
            return {red, verts: g.attributes.position.count, colorCount: g.attributes.color.count,
                uvCount: uv.length / 2, bad, joints: window.threejsViewer._objects.get('t').userData.tubeJointVerts.length}; }"""
        )
        if state["red"] == 0:
            break
    assert state["joints"] > 0
    assert state["red"] == 0, state
    assert state["colorCount"] == state["verts"] == state["uvCount"]
    assert state["bad"] == 0


def test_round_joins_worker_build_matches_main_thread(
    viewer_client,
    viewer_page,
    worker_code,  # noqa: F811
):
    spine = _issue_spine(277)
    n = len(spine)
    viewer_client.add_parametric_tube(
        "w", spine, np.full(n, 11.83), np.full(n, 2.0), lod=False, joins="round"
    )
    _wait_for_object(viewer_page, "w")
    main = viewer_page.evaluate(READ, "w")
    worker = viewer_page.evaluate(
        """({code, spine, widths, heights, rings}) => {
            const run = new Function('self', 'spine', 'widths', 'heights', 'rings', code + `
                const geo = buildGeometry(spine, widths, heights, [0, 0, 1], null, null, null, 0.015, null, rings);
                return {positions: Array.from(geo.positions), indices: Array.from(geo.indices),
                    joints: Array.from(geo.jointVerts)};`);
            return run({}, new Float32Array(spine.flat()), new Float32Array(widths),
                new Float32Array(heights), rings);
        }""",
        dict(
            code=worker_code,
            spine=spine.tolist(),
            widths=main["widths"],
            heights=main["heights"],
            rings=main["jointRings"],
        ),
    )
    assert worker["joints"] == main["joints"] == [12, 13, 15, 16]
    np.testing.assert_array_equal(worker["indices"], main["indices"])
    np.testing.assert_allclose(worker["positions"], main["positions"], atol=1e-6)


def _zigzag(turns_deg, seg=5.0):
    """Planar path turning by each angle in turn; vertex k+1 turns by turns_deg[k]."""
    pts = [np.zeros(3)]
    heading = 0.0
    for k, t in enumerate(turns_deg):
        pts.append(pts[-1] + seg * np.array([np.cos(heading), np.sin(heading), 0.0]))
        heading += np.radians(t) * (1 if k % 2 else -1)
    pts.append(pts[-1] + seg * np.array([np.cos(heading), np.sin(heading), 0.0]))
    return np.asarray(pts, dtype=np.float32)


def test_round_join_budget_keeps_the_sharpest_corners(viewer_client, viewer_page):
    """The joint budget bounds the joints per tube (at least 64) and spends
    them on the corners with the largest relative miter spike; None lifts it."""
    turns = [150.0 if k % 3 == 0 else 40.0 for k in range(300)]
    spine = _zigzag(turns)
    n = len(spine)
    sharp = {k + 1 for k, t in enumerate(turns) if t == 150.0}
    for name, budget in (("capped", 0.15), ("open", None)):
        viewer_client.add_parametric_tube(
            name,
            spine,
            np.full(n, 0.8),
            np.full(n, 0.3),
            lod=False,
            joins="round",
            join_budget=budget,
        )
        _wait_for_object(viewer_page, name)
    capped = viewer_page.evaluate(READ, "capped")
    opened = viewer_page.evaluate(READ, "open")
    assert len(opened["joints"]) == 300, "every corner exceeds the tolerance"
    # 0.15 * 300 / (2 * 5 rings) = 4 is under the floor of 64 joints.
    assert len(capped["joints"]) == 64
    assert set(capped["joints"]) <= sharp
    assert capped["joints"] == sorted(capped["joints"])
    assert capped["total"] < opened["total"]
