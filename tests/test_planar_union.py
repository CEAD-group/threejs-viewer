"""Swept-envelope tests use distance to the path, not the old ring silhouette."""

import json
from pathlib import Path

import numpy as np
import pytest

from threejs_viewer import ViewerClient

DATA = Path(__file__).parent / "data"


def test_planar_union_validation():
    client = ViewerClient(port=0, open_browser=False)
    base = dict(
        id="t",
        spine=[[0, 0, 0], [10, 0, 0]],
        widths=[10, 10],
        heights=[2, 2],
        surface="planar_union",
    )
    cases = [
        ({"spine": [[0, 0, 0], [10, 0, 1]]}, "plane"),
        ({"heights": [2, 3]}, "constant"),
        ({"widths": [2, 2]}, "widths greater"),
        ({"up_vector": [0, 0, 0]}, "up_vector"),
        ({"surface_tolerance": 0}, "surface_tolerance"),
        ({"surface_tolerance": 1e-12}, "512"),
        ({"colors": [0, 0]}, "uniform color"),
        ({"surface": "unknown"}, "surface must"),
    ]
    for patch, message in cases:
        with pytest.raises(ValueError, match=message):
            client.add_parametric_tube(**(base | patch))


def _wait(page, value=1, id="t"):
    page.wait_for_function(
        """([id, value]) => window.threejsViewer._objects.get(id)
            ?.userData.unionRenderedValue === value""",
        arg=[id, value],
        timeout=30000,
    )


def _geometry(page, id="t"):
    data = page.evaluate(
        """id => {
        const o = window.threejsViewer._objects.get(id), g = o.geometry;
        return {p: Array.from(g.attributes.position.array),
            i: Array.from(g.index.array), segments: o.userData.unionSegmentCount};
    }""",
        id,
    )
    return np.asarray(data["p"]).reshape(-1, 3), np.asarray(data["i"]).reshape(-1, 3)


def _dist(points, spine):
    a, d = spine[:-1, :2], np.diff(spine[:, :2], axis=0)
    delta = points[:, None, :2] - a
    t = np.clip(
        np.einsum("nsi,si->ns", delta, d) / np.maximum((d * d).sum(1), 1e-20), 0, 1
    )
    return np.linalg.norm(delta - t[:, :, None] * d, axis=2).min(1)


def _assert_closed(p, indices):
    assert np.isfinite(p).all()
    # Weld geometric seams (top/chamfer and upper/lower need distinct normals).
    _, welded = np.unique(np.round(p, 5), axis=0, return_inverse=True)
    tri = welded[indices]
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2), np.unique(counts, return_counts=True)
    a, b, c = p[indices].transpose(1, 0, 2)
    volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    assert volume > 0


@pytest.mark.browser
@pytest.mark.parametrize("issue,width,height", [(277, 11.83, 2), (278, 9.46, 2.5)])
def test_reported_paths_are_closed_rounded_envelopes(
    viewer_client, viewer_page, issue, width, height
):
    spine = np.asarray(json.loads((DATA / f"tube_issue_{issue}.json").read_text()))
    viewer_client.add_parametric_tube(
        "t",
        spine,
        np.full(len(spine), width),
        np.full(len(spine), height),
        surface="planar_union",
    )
    _wait(viewer_page)
    p, indices = _geometry(viewer_page)
    _assert_closed(p, indices)
    # Every boundary vertex lies on the requested swept bead, within outline
    # chord tolerance. A miter blade fails this independent distance oracle.
    d = _dist(p, spine)
    z = np.abs(p[:, 2] - spine[0, 2])
    assert np.max(d + z - width / 2) < width / 500 + 2e-4
    assert z.max() == pytest.approx(height / 2, abs=2e-5)
    assert len(indices) < 4000


@pytest.mark.browser
def test_union_retrace_is_single_solid_and_sampling_does_not_grow_mesh(
    viewer_client, viewer_page
):
    for name, x in [
        ("simple", [0, 50]),
        ("dense", np.linspace(0, 50, 1001)),
        ("retrace", [0, 50, 0, 50]),
    ]:
        spine = np.column_stack([x, np.zeros(len(x)), np.zeros(len(x))])
        viewer_client.add_parametric_tube(
            name, spine, np.full(len(x), 10), np.full(len(x), 2), surface="planar_union"
        )
        _wait(viewer_page, id=name)
    geometries = [
        _geometry(viewer_page, name) for name in ("simple", "dense", "retrace")
    ]
    for p, i in geometries:
        _assert_closed(p, i)
    assert len(geometries[0][1]) == len(geometries[1][1]) == len(geometries[2][1])
    assert len(geometries[0][1]) < 700


@pytest.mark.browser
def test_prefix_holes_backward_seek_and_empty(viewer_client, viewer_page):
    spine = np.array([[0, 0, 0], [20, 0, 0], [20, 20, 0], [0, 20, 0], [0, 0, 0]])
    viewer_client.add_parametric_tube(
        "t", spine, np.full(5, 6), np.full(5, 2), surface="planar_union"
    )
    _wait(viewer_page)
    full = _geometry(viewer_page)
    _assert_closed(*full)
    # No top triangle covers the hole's centre.
    p, i = full
    for a, b, c in p[i][:, :, :2]:

        def cross(u, v):
            return u[0] * v[1] - u[1] * v[0]

        q = np.array([10, 10])
        signs = np.array(
            [cross(b - a, q - a), cross(c - b, q - b), cross(a - c, q - c)]
        )
        assert not (np.all(signs > 1e-8) or np.all(signs < -1e-8))
    for value in (0.375, 0.8, 0.12, 1):
        viewer_client.set_draw_range("t", value)
        _wait(viewer_page, value)
        p, indices = _geometry(viewer_page)
        _assert_closed(p, indices)
        at = value * 4
        if value < 1:
            end = int(at)
            prefix = np.vstack(
                [
                    spine[: end + 1],
                    spine[end] + (at - end) * (spine[end + 1] - spine[end]),
                ]
            )
            assert _dist(p, prefix).max() < 3.001
    np.testing.assert_array_equal(_geometry(viewer_page)[0], full[0])
    viewer_client.set_draw_range("t", 0)
    _wait(viewer_page, 0)
    assert len(_geometry(viewer_page)[1]) == 0


@pytest.mark.browser
def test_union_rotated_plane_anchor_and_break(viewer_client, viewer_page):
    spine = np.array([[5, 0, 0], [5, 20, 0], [5, 60, 0], [5, 80, 0]])
    viewer_client.add_parametric_tube(
        "t",
        spine,
        [8] * 4,
        [2] * 4,
        surface="planar_union",
        up_vector=[1, 0, 0],
        anchor="top",
        break_before=[0, 0, 1, 0],
    )
    _wait(viewer_page)
    p, i = _geometry(viewer_page)
    _assert_closed(p, i)
    assert p[:, 0].min() == pytest.approx(3)
    assert p[:, 0].max() == pytest.approx(5)
    assert not np.any((p[:, 1] > 24.01) & (p[:, 1] < 55.99))


@pytest.mark.browser
@pytest.mark.parametrize("kind", ["taper", "closing_hole", "duplicates"])
def test_union_variable_width_and_topology_change(viewer_client, viewer_page, kind):
    if kind == "taper":
        spine = np.array([[0, 0, 0], [20, 0, 0], [20, 20, 0]])
        widths = [6, 14, 8]
    elif kind == "closing_hole":
        spine = np.array([[0, 0, 0], [20, 0, 0], [20, 20, 0], [0, 20, 0], [0, 0, 0]])
        widths = [22] * len(spine)
    else:
        spine = np.array([[0, 0, 0], [0, 0, 0], [20, 0, 0], [20, 0, 0], [20, 20, 0]])
        widths = [8] * len(spine)
    height = 4
    viewer_client.add_parametric_tube(
        "t", spine, widths, [height] * len(spine), surface="planar_union"
    )
    _wait(viewer_page)
    p, indices = _geometry(viewer_page)
    _assert_closed(p, indices)
    assert np.abs(p[:, 2]).max() <= height / 2 + 1e-6
    if kind == "closing_hole":
        # The mid-plane footprint fills the hole, but the top contour still has
        # a hole. Its centre must be below the flat top, not capped at z=2.
        centre = p[np.linalg.norm(p[:, :2] - [10, 10], axis=1) < 2]
        assert len(centre) > 0
        assert np.abs(centre[:, 2]).min() < 1.5


@pytest.mark.browser
def test_union_coalesced_playback_wire_overlay_and_dispose(viewer_client, viewer_page):
    viewer_client.add_parametric_tube(
        "t",
        [[0, 0, 0], [10, 0, 0], [10, 20, 0]],
        [8] * 3,
        [2] * 3,
        surface="planar_union",
    )
    _wait(viewer_page)
    viewer_page.evaluate("""() => {
        const v = window.threejsViewer;
        v._shading.wireframeMode = 2; v._shading.applyWireframe();
        for (const t of [.8, .2, .95, .4, .1]) v._setDrawRange('t', t);
    }""")
    _wait(viewer_page, 0.1)
    assert viewer_page.evaluate("""() => {
        const o = window.threejsViewer._objects.get('t');
        return o.geometry === o.userData.wireframeOverlay.geometry;
    }""")
    p, i = _geometry(viewer_page)
    _assert_closed(p, i)
    assert p[:, 0].max() <= 6.001
    viewer_page.evaluate("""() => {
        const v = window.threejsViewer;
        window.oldUnionController = v._objects.get('t').userData.parametricTube;
        v._setDrawRange('t', .9); v._deleteObject('t');
    }""")
    assert viewer_page.evaluate("() => window.oldUnionController.disposed")


@pytest.mark.browser
def test_union_top_coverage_matches_disk_sweep(viewer_client, viewer_page):
    spine = np.asarray(json.loads((DATA / "tube_issue_277.json").read_text()))
    width, height = 11.83, 2
    viewer_client.add_parametric_tube(
        "t", spine, [width] * len(spine), [height] * len(spine), surface="planar_union"
    )
    _wait(viewer_page)
    p, indices = _geometry(viewer_page)
    triangles = p[indices]
    top = triangles[
        np.all(np.abs(triangles[:, :, 2] - (spine[0, 2] + 1)) < 1e-5, axis=1)
    ]
    x, y = np.meshgrid(
        np.linspace(p[:, 0].min(), p[:, 0].max(), 101),
        np.linspace(p[:, 1].min(), p[:, 1].max(), 103),
    )
    samples = np.column_stack([x.ravel(), y.ravel()])
    coverage = np.zeros(len(samples), dtype=int)
    for triangle in top:
        signs = []
        for a, b in zip(triangle[:, :2], np.roll(triangle[:, :2], -1, axis=0)):
            signs.append(
                (b[0] - a[0]) * (samples[:, 1] - a[1])
                - (b[1] - a[1]) * (samples[:, 0] - a[0])
            )
        coverage += np.all(np.array(signs) > 1e-7, axis=0)
    margin = _dist(samples, spine) - (width - height) / 2
    away_from_boundary = np.abs(margin) > 2 * width / 500
    np.testing.assert_array_equal(
        coverage[away_from_boundary], (margin[away_from_boundary] < 0).astype(int)
    )


@pytest.mark.browser
def test_union_oblique_plane(viewer_client, viewer_page):
    up = np.array([1, 2, 3], dtype=float)
    up /= np.linalg.norm(up)
    u = np.cross(up, [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(up, u)
    origin = np.array([300, 500, 100])
    spine = origin + np.array([[0, 0], [20, 0], [20, 20]]) @ np.array([u, v])
    viewer_client.add_parametric_tube(
        "t", spine, [8] * 3, [2] * 3, surface="planar_union", up_vector=up
    )
    _wait(viewer_page)
    p, i = _geometry(viewer_page)
    _assert_closed(p, i)
    np.testing.assert_allclose(
        [((p - origin) @ up).min(), ((p - origin) @ up).max()], [-1, 1], atol=5e-5
    )
