"""Tests for the parametric_tube primitive."""

import time

import numpy as np
import pytest


def _wait_for_object(page, obj_id, timeout=5.0):
    """Poll until a named object appears in the viewer's _objects map."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        exists = page.evaluate(
            "(id) => window.threejsViewer && window.threejsViewer._objects.has(id)",
            obj_id,
        )
        if exists:
            return
        time.sleep(0.05)
    raise TimeoutError(f"Object '{obj_id}' never appeared in viewer")


# --- Python API unit tests ---


def test_add_parametric_tube_validates_spine_length():
    from threejs_viewer import ViewerClient

    c = ViewerClient(port=0, open_browser=False)
    spine = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    widths = np.array([1.0], dtype=np.float32)
    heights = np.array([1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="spine points"):
        c.add_parametric_tube("t", spine, widths, heights)


def test_add_parametric_tube_validates_widths_length():
    from threejs_viewer import ViewerClient

    c = ViewerClient(port=0, open_browser=False)
    spine = np.zeros((4, 3), dtype=np.float32)
    widths = np.ones(3, dtype=np.float32)
    heights = np.ones(4, dtype=np.float32)
    with pytest.raises(ValueError, match="widths/heights"):
        c.add_parametric_tube("t", spine, widths, heights)


def test_add_parametric_tube_validates_colors_length():
    from threejs_viewer import ViewerClient

    c = ViewerClient(port=0, open_browser=False)
    spine = np.zeros((4, 3), dtype=np.float32)
    widths = np.ones(4, dtype=np.float32)
    heights = np.ones(4, dtype=np.float32)
    colors = np.zeros(3, dtype=np.uint32)
    with pytest.raises(ValueError, match="colors"):
        c.add_parametric_tube("t", spine, widths, heights, colors=colors)


def test_add_parametric_tube_validates_n_cross_section_verts():
    from threejs_viewer import ViewerClient

    c = ViewerClient(port=0, open_browser=False)
    spine = np.zeros((4, 3), dtype=np.float32)
    spine[:, 0] = [0, 1, 2, 3]
    widths = np.ones(4, dtype=np.float32)
    heights = np.ones(4, dtype=np.float32)
    with pytest.raises(ValueError, match="n_cross_section_verts"):
        c.add_parametric_tube("t", spine, widths, heights, n_cross_section_verts=2)


def test_add_parametric_tube_validates_cross_section():
    from threejs_viewer import ViewerClient

    c = ViewerClient(port=0, open_browser=False)
    spine = np.zeros((4, 3), dtype=np.float32)
    spine[:, 0] = [0, 1, 2, 3]
    widths = np.ones(4, dtype=np.float32)
    heights = np.ones(4, dtype=np.float32)
    with pytest.raises(ValueError, match="cross_section"):
        c.add_parametric_tube("t", spine, widths, heights, cross_section="circle")


def test_add_parametric_tube_validates_corner_radius_frac():
    from threejs_viewer import ViewerClient

    c = ViewerClient(port=0, open_browser=False)
    spine = np.zeros((4, 3), dtype=np.float32)
    spine[:, 0] = [0, 1, 2, 3]
    widths = np.ones(4, dtype=np.float32)
    heights = np.ones(4, dtype=np.float32)
    with pytest.raises(ValueError, match="corner_radius_frac"):
        c.add_parametric_tube("t", spine, widths, heights, corner_radius_frac=0.8)


def test_add_parametric_tube_validates_widths_positive():
    from threejs_viewer import ViewerClient

    c = ViewerClient(port=0, open_browser=False)
    spine = np.zeros((4, 3), dtype=np.float32)
    spine[:, 0] = [0, 1, 2, 3]
    widths = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)
    heights = np.ones(4, dtype=np.float32)
    with pytest.raises(ValueError, match="widths"):
        c.add_parametric_tube("t", spine, widths, heights)


def test_add_parametric_tube_validates_heights_finite():
    from threejs_viewer import ViewerClient

    c = ViewerClient(port=0, open_browser=False)
    spine = np.zeros((4, 3), dtype=np.float32)
    spine[:, 0] = [0, 1, 2, 3]
    widths = np.ones(4, dtype=np.float32)
    heights = np.array([1.0, np.inf, 1.0, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="heights"):
        c.add_parametric_tube("t", spine, widths, heights)


# --- Browser integration tests ---


def _straight_spine(n=20, length=2.0):
    """Horizontal spine along +X so the derived frame has width=+Y, height=+Z."""
    x = np.linspace(0.0, length, n, dtype=np.float32)
    spine = np.column_stack(
        [x, np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)]
    )
    return spine


@pytest.mark.browser
def test_parametric_tube_builds_expected_geometry(viewer_client, viewer_page):
    """Creating a parametric_tube produces a mesh with the expected vertex
    and index counts, and the width parameter shows up in the bounds."""
    n = 20
    n_cs = 8
    spine = _straight_spine(n=n, length=2.0)
    widths = np.full(n, 0.4, dtype=np.float32)
    heights = np.full(n, 0.2, dtype=np.float32)

    viewer_client.add_parametric_tube(
        "tube",
        spine=spine,
        widths=widths,
        heights=heights,
        n_cross_section_verts=n_cs,
        corner_radius_frac=0.0,
    )
    _wait_for_object(viewer_page, "tube")

    info = viewer_page.evaluate(
        """(id) => {
            const v = window.threejsViewer;
            const obj = v._objects.get(id);
            if (!obj) return null;
            const geom = obj.geometry;
            geom.computeBoundingBox();
            const bb = geom.boundingBox;
            return {
                isTube: obj.userData.isParametricTube === true,
                nSpine: obj.userData.tubeNumSpinePoints,
                nCs: obj.userData.tubeNCs,
                ringPairs: obj.userData.tubeRingPairs,
                perPair: obj.userData.tubeIndicesPerRingPair,
                totalIndex: obj.userData.totalIndexCount,
                vertexCount: geom.getAttribute('position').count,
                indexCount: geom.getIndex().count,
                bbLength: bb.max.x - bb.min.x,
                bbWidth: bb.max.y - bb.min.y,
                bbHeight: bb.max.z - bb.min.z,
            };
        }""",
        "tube",
    )
    assert info is not None, "tube object was not created"
    assert info["isTube"] is True
    assert info["nSpine"] == n
    assert info["nCs"] == n_cs
    assert info["ringPairs"] == n - 1
    assert info["perPair"] == n_cs * 6
    n_cap_rings = 3  # dome cap latitude rings
    cap_indices = n_cap_rings * n_cs * 6 + n_cs * 3  # dome per cap
    assert info["totalIndex"] == 2 * cap_indices + (n - 1) * n_cs * 6
    # Tube ring verts + 2 dome caps (each: nCapRings * nCs + 1 pole)
    assert info["vertexCount"] == n * n_cs + 2 * (n_cap_rings * n_cs + 1)
    assert info["indexCount"] == 2 * cap_indices + (n - 1) * n_cs * 6
    # Spine along +X. Frame derives width=+Y, height=+Z. Sharp rectangle
    # of 0.4 x 0.2 sampled at 8 angles: outermost samples land on the
    # flat edges so the bounding box matches the parameter values.
    # Dome caps extend min(w,h)/2 = 0.1 beyond each spine endpoint
    assert abs(info["bbLength"] - 2.2) < 0.05, info
    assert abs(info["bbWidth"] - 0.4) < 0.05, info
    assert abs(info["bbHeight"] - 0.2) < 0.05, info


@pytest.mark.browser
def test_parametric_tube_draw_range_morphs_frontier(viewer_client, viewer_page):
    """draw_range morphs the frontier ring, showing one extra ring pair beyond
    the floor so the tube grows smoothly."""
    n = 10
    n_cs = 6
    spine = _straight_spine(n=n, length=1.0)
    widths = np.full(n, 0.3, dtype=np.float32)
    heights = np.full(n, 0.2, dtype=np.float32)

    viewer_client.add_parametric_tube(
        "tube2",
        spine=spine,
        widths=widths,
        heights=heights,
        n_cross_section_verts=n_cs,
    )
    _wait_for_object(viewer_page, "tube2")

    # 0.37 * 9 ring pairs = 3.33 → 3 complete + 1 morphed frontier = 4 ring pairs visible
    viewer_client.set_draw_range("tube2", 0.37)
    time.sleep(0.1)

    count = viewer_page.evaluate(
        """(id) => window.threejsViewer._objects.get(id).geometry.drawRange.count""",
        "tube2",
    )
    n_cap_rings = 3
    cap = n_cap_rings * n_cs * 6 + n_cs * 3  # dome per cap
    expected = 2 * cap + 4 * n_cs * 6  # start cap + 4 ring pairs + end cap
    assert count == expected, f"expected {expected}, got {count}"


@pytest.mark.browser
def test_parametric_tube_frontier_morph_positions(viewer_client, viewer_page):
    """The frontier ring's positions are interpolated between adjacent spine points."""
    n = 10
    n_cs = 6
    spine = _straight_spine(n=n, length=9.0)  # 1.0 spacing between spine points
    widths = np.full(n, 0.3, dtype=np.float32)
    heights = np.full(n, 0.2, dtype=np.float32)

    viewer_client.add_parametric_tube(
        "tube_morph",
        spine=spine,
        widths=widths,
        heights=heights,
        n_cross_section_verts=n_cs,
    )
    _wait_for_object(viewer_page, "tube_morph")

    # 0.5 * 9 ring pairs = 4.5 → frontier ring is ring 5, morphed to 50% between ring 4 and 5
    viewer_client.set_draw_range("tube_morph", 0.5)
    time.sleep(0.1)

    x_avg = viewer_page.evaluate(
        """(id) => {
            const obj = window.threejsViewer._objects.get(id);
            const pos = obj.geometry.getAttribute('position').array;
            const nCs = obj.userData.tubeNCs;
            const frontierRing = 5;
            let sum = 0;
            for (let j = 0; j < nCs; j++) sum += pos[(frontierRing * nCs + j) * 3];
            return sum / nCs;
        }""",
        "tube_morph",
    )
    # Ring 4 center at x=4.0, ring 5 center at x=5.0, morphed at 0.5 → x≈4.5
    assert abs(x_avg - 4.5) < 0.05, f"Expected frontier ring center X ~4.5, got {x_avg}"


@pytest.mark.browser
def test_parametric_tube_frontier_restores_on_full(viewer_client, viewer_page):
    """When draw_range reaches 1.0, the morphed frontier ring is restored."""
    n = 10
    n_cs = 6
    spine = _straight_spine(n=n, length=9.0)
    widths = np.full(n, 0.3, dtype=np.float32)
    heights = np.full(n, 0.2, dtype=np.float32)

    viewer_client.add_parametric_tube(
        "tube_restore",
        spine=spine,
        widths=widths,
        heights=heights,
        n_cross_section_verts=n_cs,
    )
    _wait_for_object(viewer_page, "tube_restore")

    # Morph ring 5 by setting draw_range to 0.5
    viewer_client.set_draw_range("tube_restore", 0.5)
    time.sleep(0.1)

    # Now set to 1.0 — all rings should be at original positions
    viewer_client.set_draw_range("tube_restore", 1.0)
    time.sleep(0.1)

    x_avg = viewer_page.evaluate(
        """(id) => {
            const obj = window.threejsViewer._objects.get(id);
            const pos = obj.geometry.getAttribute('position').array;
            const nCs = obj.userData.tubeNCs;
            const ring5 = 5;
            let sum = 0;
            for (let j = 0; j < nCs; j++) sum += pos[(ring5 * nCs + j) * 3];
            return sum / nCs;
        }""",
        "tube_restore",
    )
    assert abs(x_avg - 5.0) < 0.05, (
        f"Expected restored ring 5 center X ~5.0, got {x_avg}"
    )


@pytest.mark.browser
def test_parametric_tube_color_swap(viewer_client, viewer_page):
    """update_parametric_tube_colors replaces the color attribute without
    rebuilding positions/indices."""
    n = 8
    n_cs = 6
    spine = _straight_spine(n=n, length=1.0)
    widths = np.full(n, 0.2, dtype=np.float32)
    heights = np.full(n, 0.1, dtype=np.float32)
    initial_colors = np.full(n, 0xFF0000, dtype=np.uint32)

    viewer_client.add_parametric_tube(
        "tube3",
        spine=spine,
        widths=widths,
        heights=heights,
        colors=initial_colors,
        n_cross_section_verts=n_cs,
    )
    _wait_for_object(viewer_page, "tube3")

    before = viewer_page.evaluate(
        """(id) => {
            const g = window.threejsViewer._objects.get(id).geometry;
            const pos = g.getAttribute('position');
            const col = g.getAttribute('color');
            return {
                posHash: pos.array[0] + pos.array[1] * 7 + pos.array[2] * 13,
                colLen: col.array.length,
                // First vertex color channels (red).
                r0: col.array[0], g0: col.array[1], b0: col.array[2],
            };
        }""",
        "tube3",
    )
    assert abs(before["r0"] - 1.0) < 1e-3
    assert before["g0"] < 0.01

    new_colors = np.full(n, 0x0000FF, dtype=np.uint32)
    viewer_client.update_parametric_tube_colors("tube3", new_colors)
    time.sleep(0.3)

    after = viewer_page.evaluate(
        """(id) => {
            const g = window.threejsViewer._objects.get(id).geometry;
            const pos = g.getAttribute('position');
            const col = g.getAttribute('color');
            return {
                posHash: pos.array[0] + pos.array[1] * 7 + pos.array[2] * 13,
                colLen: col.array.length,
                r0: col.array[0], g0: col.array[1], b0: col.array[2],
            };
        }""",
        "tube3",
    )
    assert after["colLen"] == before["colLen"]
    assert after["posHash"] == before["posHash"]  # positions untouched
    assert after["r0"] < 0.01
    assert abs(after["b0"] - 1.0) < 1e-3
