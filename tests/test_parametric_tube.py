"""Tests for the parametric_tube primitive."""

import time

import numpy as np
import pytest

from threejs_viewer import Animation


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


def test_add_parametric_tube_validates_widths_non_negative():
    from threejs_viewer import ViewerClient

    c = ViewerClient(port=0, open_browser=False)
    spine = np.zeros((4, 3), dtype=np.float32)
    spine[:, 0] = [0, 1, 2, 3]
    # Zero is allowed (travel/cap), negative is not
    widths = np.array([1.0, -0.1, 1.0, 1.0], dtype=np.float32)
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


def test_add_parametric_tube_anchor_rejects_invalid():
    from threejs_viewer import ViewerClient

    c = ViewerClient(port=0, open_browser=False)
    spine = np.zeros((4, 3), dtype=np.float32)
    spine[:, 0] = [0, 1, 2, 3]
    widths = np.ones(4, dtype=np.float32)
    heights = np.ones(4, dtype=np.float32)
    with pytest.raises(ValueError, match="anchor"):
        c.add_parametric_tube("t", spine, widths, heights, anchor="bottom")


def test_add_parametric_tube_anchor_forwards_height_offset():
    """anchor="top" must send heightOffset=-0.5 so the spine sits at the top
    surface (+cv is up, so a negative shift moves the bead downward)."""
    from threejs_viewer import ViewerClient

    c = ViewerClient(port=0, open_browser=False)
    c._binary_messages = []
    c._send_binary = lambda h, p: c._binary_messages.append((h, p))

    spine = np.zeros((4, 3), dtype=np.float32)
    spine[:, 0] = [0, 1, 2, 3]
    widths = np.ones(4, dtype=np.float32)
    heights = np.ones(4, dtype=np.float32)

    c.add_parametric_tube("t_center", spine, widths, heights)
    header_center, _ = c._binary_messages[-1]
    assert "heightOffset" not in header_center  # default centered → omitted

    c.add_parametric_tube("t_top", spine, widths, heights, anchor="top")
    header_top, _ = c._binary_messages[-1]
    assert header_top["heightOffset"] == -0.5


def _capture_client():
    from threejs_viewer import ViewerClient

    c = ViewerClient(port=0, open_browser=False)
    c._binary_messages = []
    c._send_binary = lambda h, p: c._binary_messages.append((h, p))
    return c


def _simple_tube_args():
    spine = np.zeros((4, 3), dtype=np.float32)
    spine[:, 0] = [0, 1, 2, 3]
    widths = np.ones(4, dtype=np.float32)
    heights = np.ones(4, dtype=np.float32)
    return spine, widths, heights


def test_add_parametric_tube_lod_default_omits_header_key():
    c = _capture_client()
    spine, widths, heights = _simple_tube_args()
    c.add_parametric_tube("t", spine, widths, heights)
    header, _ = c._binary_messages[-1]
    assert "lod" not in header


def test_add_parametric_tube_lod_false_serializes_false():
    c = _capture_client()
    spine, widths, heights = _simple_tube_args()
    c.add_parametric_tube("t", spine, widths, heights, lod=False)
    header, _ = c._binary_messages[-1]
    assert header["lod"] is False


def test_add_parametric_tube_lod_epsilon_divisor_only():
    c = _capture_client()
    spine, widths, heights = _simple_tube_args()
    c.add_parametric_tube("t", spine, widths, heights, lod={"epsilon_divisor": 10000})
    header, _ = c._binary_messages[-1]
    assert header["lod"] == {"epsilonDivisor": 10000.0}


def test_add_parametric_tube_lod_both_keys_serialize_camel():
    c = _capture_client()
    spine, widths, heights = _simple_tube_args()
    c.add_parametric_tube(
        "t",
        spine,
        widths,
        heights,
        lod={"epsilon_divisor": 5000, "threshold": 0},
    )
    header, _ = c._binary_messages[-1]
    assert header["lod"] == {"epsilonDivisor": 5000.0, "threshold": 0}


def test_add_parametric_tube_lod_true_rejected():
    c = _capture_client()
    spine, widths, heights = _simple_tube_args()
    with pytest.raises(ValueError, match="lod must be"):
        c.add_parametric_tube("t", spine, widths, heights, lod=True)


def test_add_parametric_tube_lod_non_dict_rejected():
    c = _capture_client()
    spine, widths, heights = _simple_tube_args()
    with pytest.raises(ValueError, match="lod must be"):
        c.add_parametric_tube("t", spine, widths, heights, lod="bogus")


def test_add_parametric_tube_lod_unknown_key_rejected():
    c = _capture_client()
    spine, widths, heights = _simple_tube_args()
    with pytest.raises(ValueError, match="unknown keys"):
        c.add_parametric_tube("t", spine, widths, heights, lod={"foo": 1})


def test_add_parametric_tube_lod_epsilon_divisor_zero_rejected():
    c = _capture_client()
    spine, widths, heights = _simple_tube_args()
    with pytest.raises(ValueError, match="epsilon_divisor"):
        c.add_parametric_tube("t", spine, widths, heights, lod={"epsilon_divisor": 0})


def test_add_parametric_tube_lod_epsilon_divisor_negative_rejected():
    c = _capture_client()
    spine, widths, heights = _simple_tube_args()
    with pytest.raises(ValueError, match="epsilon_divisor"):
        c.add_parametric_tube("t", spine, widths, heights, lod={"epsilon_divisor": -1})


def test_add_parametric_tube_lod_threshold_negative_rejected():
    c = _capture_client()
    spine, widths, heights = _simple_tube_args()
    with pytest.raises(ValueError, match="threshold"):
        c.add_parametric_tube("t", spine, widths, heights, lod={"threshold": -5})


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
    n_cs = 6
    spine = _straight_spine(n=n, length=2.0)
    widths = np.full(n, 0.4, dtype=np.float32)
    heights = np.full(n, 0.2, dtype=np.float32)

    viewer_client.add_parametric_tube(
        "tube",
        spine=spine,
        widths=widths,
        heights=heights,
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
    n_cap_rings = 8
    cap_indices = n_cap_rings * n_cs * 6  # spoke quads only (no pole)
    assert info["totalIndex"] == 2 * cap_indices + (n - 1) * n_cs * 6
    # Tube ring verts + 2 caps (each: nCapRings * nCs, no pole)
    assert info["vertexCount"] == n * n_cs + 2 * (n_cap_rings * n_cs)
    assert info["indexCount"] == 2 * cap_indices + (n - 1) * n_cs * 6
    # Spine along +X. Frame derives width=+Y, height=+Z. Chamfered hex
    # with w=0.4 > h=0.2: right/left tips at ±hw, flat top/bottom at ±hh.
    # Revolution cap extends ~hw beyond each endpoint.
    assert info["bbLength"] > 2.3, info
    assert info["bbLength"] < 2.5, info
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
    )
    _wait_for_object(viewer_page, "tube2")

    # 0.37 * 9 ring pairs = 3.33 → 3 complete + 1 morphed frontier = 4 ring pairs visible
    viewer_client.set_draw_range("tube2", 0.37)
    time.sleep(0.1)

    count = viewer_page.evaluate(
        """(id) => window.threejsViewer._objects.get(id).geometry.drawRange.count""",
        "tube2",
    )
    n_cap_rings = 8
    cap = n_cap_rings * n_cs * 6  # spoke quads per cap (no pole)
    expected = 2 * cap + 4 * n_cs * 6  # start cap + 4 ring pairs + end cap
    assert count == expected, f"expected {expected}, got {count}"


@pytest.mark.browser
def test_parametric_tube_frontier_morph_positions(viewer_client, viewer_page):
    """The frontier ring's positions are interpolated between adjacent spine points."""
    n = 10
    spine = _straight_spine(n=n, length=9.0)  # 1.0 spacing between spine points
    widths = np.full(n, 0.3, dtype=np.float32)
    heights = np.full(n, 0.2, dtype=np.float32)

    viewer_client.add_parametric_tube(
        "tube_morph",
        spine=spine,
        widths=widths,
        heights=heights,
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
    spine = _straight_spine(n=n, length=9.0)
    widths = np.full(n, 0.3, dtype=np.float32)
    heights = np.full(n, 0.2, dtype=np.float32)

    viewer_client.add_parametric_tube(
        "tube_restore",
        spine=spine,
        widths=widths,
        heights=heights,
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


@pytest.mark.browser
def test_parametric_tube_color_swap_during_draw_range(viewer_client, viewer_page):
    """Swapping colors while draw_range animation is playing must update ALL
    visible rings — not just the rings outside the frontier morph zone.

    This reproduces the bug from example 18 where update_parametric_tube_colors
    mid-animation leaves some rings with stale colors.
    """
    n = 20
    spine = _straight_spine(n=n, length=4.0)
    widths = np.full(n, 0.3, dtype=np.float32)
    heights = np.full(n, 0.2, dtype=np.float32)
    initial_colors = np.full(n, 0xFF0000, dtype=np.uint32)  # all red

    viewer_client.add_parametric_tube(
        "tube_anim",
        spine=spine,
        widths=widths,
        heights=heights,
        colors=initial_colors,
    )
    _wait_for_object(viewer_page, "tube_anim")

    # Set draw_range to 0.5 to trigger frontier morphing (like animation mid-play)
    viewer_client.set_draw_range("tube_anim", 0.5)
    time.sleep(0.15)

    # Verify frontier is morphed (savedRingIndex should be set)
    morph_state = viewer_page.evaluate(
        """(id) => {
            const obj = window.threejsViewer._objects.get(id);
            const md = obj.userData.tubeMorphData;
            return {
                savedRingIndex: md ? md.savedRingIndex : null,
                hasRingColors: md ? !!md.ringColors : false,
                hasSavedRingColors: md ? !!md.savedRingColors : false,
            };
        }""",
        "tube_anim",
    )
    assert morph_state["savedRingIndex"] is not None, (
        "Frontier ring should be morphed at draw_range=0.5"
    )

    # Now swap colors to blue (while frontier is morphed)
    new_colors = np.full(n, 0x0000FF, dtype=np.uint32)  # all blue
    viewer_client.update_parametric_tube_colors("tube_anim", new_colors)
    time.sleep(0.5)

    # Sample colors at multiple rings — ALL should be blue now
    color_check = viewer_page.evaluate(
        """(id) => {
            const obj = window.threejsViewer._objects.get(id);
            const col = obj.geometry.getAttribute('color');
            const nCs = obj.userData.tubeNCs;
            const nSpine = obj.userData.tubeNumSpinePoints;
            const results = [];
            // Sample first vertex of each ring
            for (let ring = 0; ring < nSpine; ring++) {
                const base = ring * nCs * 3;
                results.push({
                    ring: ring,
                    r: col.array[base],
                    g: col.array[base + 1],
                    b: col.array[base + 2],
                });
            }
            // Also check md.ringColors
            const md = obj.userData.tubeMorphData;
            const mdColors = [];
            if (md && md.ringColors) {
                for (let i = 0; i < nSpine; i++) {
                    mdColors.push({
                        ring: i,
                        r: md.ringColors[i * 3],
                        g: md.ringColors[i * 3 + 1],
                        b: md.ringColors[i * 3 + 2],
                    });
                }
            }
            return { vertexColors: results, ringColors: mdColors };
        }""",
        "tube_anim",
    )

    # Check that ALL rings in the vertex buffer are blue (b≈1, r≈0)
    stale_rings = []
    for entry in color_check["vertexColors"]:
        if entry["r"] > 0.1 or entry["b"] < 0.9:
            stale_rings.append(entry)
    assert not stale_rings, (
        f"Rings still have stale (non-blue) colors after swap: {stale_rings}"
    )

    # Check that md.ringColors is also fully updated
    stale_md = []
    for entry in color_check["ringColors"]:
        if entry["r"] > 0.1 or entry["b"] < 0.9:
            stale_md.append(entry)
    assert not stale_md, f"md.ringColors still has stale colors: {stale_md}"


@pytest.mark.browser
def test_parametric_tube_color_swap_then_advance_draw_range(viewer_client, viewer_page):
    """After a color swap during draw_range animation, advancing draw_range
    further must use the NEW colors for frontier lerp, not stale saved colors."""
    n = 20
    spine = _straight_spine(n=n, length=4.0)
    widths = np.full(n, 0.3, dtype=np.float32)
    heights = np.full(n, 0.2, dtype=np.float32)
    initial_colors = np.full(n, 0xFF0000, dtype=np.uint32)  # all red

    viewer_client.add_parametric_tube(
        "tube_advance",
        spine=spine,
        widths=widths,
        heights=heights,
        colors=initial_colors,
    )
    _wait_for_object(viewer_page, "tube_advance")

    # Set draw_range to 0.3 to trigger frontier morphing
    viewer_client.set_draw_range("tube_advance", 0.3)
    time.sleep(0.15)

    # Swap to blue
    new_colors = np.full(n, 0x0000FF, dtype=np.uint32)
    viewer_client.update_parametric_tube_colors("tube_advance", new_colors)
    time.sleep(0.3)

    # Now advance draw_range to 0.7 — this triggers more morphing
    viewer_client.set_draw_range("tube_advance", 0.7)
    time.sleep(0.15)

    # Then set to 1.0 to show all rings
    viewer_client.set_draw_range("tube_advance", 1.0)
    time.sleep(0.15)

    # ALL rings should be blue
    color_check = viewer_page.evaluate(
        """(id) => {
            const obj = window.threejsViewer._objects.get(id);
            const col = obj.geometry.getAttribute('color');
            const nCs = obj.userData.tubeNCs;
            const nSpine = obj.userData.tubeNumSpinePoints;
            const stale = [];
            for (let ring = 0; ring < nSpine; ring++) {
                const base = ring * nCs * 3;
                const r = col.array[base];
                const b = col.array[base + 2];
                if (r > 0.1 || b < 0.9) {
                    stale.push({ ring, r, g: col.array[base + 1], b });
                }
            }
            return stale;
        }""",
        "tube_advance",
    )
    assert not color_check, (
        f"Rings have stale colors after swap + draw_range advance: {color_check}"
    )


@pytest.mark.browser
def test_parametric_tube_color_swap_during_looping_animation(
    viewer_client, viewer_page
):
    """Reproduces example 18: color swap while a looping draw_range animation
    is actively playing.  After the swap, ALL rings must show the new color —
    checked by pausing the animation and sampling the buffer."""
    n = 20
    spine = _straight_spine(n=n, length=4.0)
    widths = np.full(n, 0.3, dtype=np.float32)
    heights = np.full(n, 0.2, dtype=np.float32)
    initial_colors = np.full(n, 0xFF0000, dtype=np.uint32)  # all red

    viewer_client.add_parametric_tube(
        "tube_loop",
        spine=spine,
        widths=widths,
        heights=heights,
        colors=initial_colors,
    )
    _wait_for_object(viewer_page, "tube_loop")

    # Build a looping draw_range animation (like example 18)
    n_frames = 60
    frame_times = np.linspace(0, 3.0, n_frames, dtype=np.float32)
    draw_fracs = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)

    anim = Animation(loop=True)
    anim.set_frame_times(frame_times)
    anim.set_draw_range_data(["tube_loop"], draw_fracs.reshape(n_frames, 1))
    viewer_client.load_animation(anim)

    # Let the animation play for a bit so frontier morphing is actively happening
    time.sleep(1.0)

    # Swap colors to blue while animation is playing
    new_colors = np.full(n, 0x0000FF, dtype=np.uint32)
    viewer_client.update_parametric_tube_colors("tube_loop", new_colors)

    # Wait for the color update to be processed
    time.sleep(0.5)

    # Pause animation and set draw_range to 1.0 so all rings are visible
    viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            // Stop animation playback
            v._animationPlaying = false;
        }"""
    )
    viewer_client.set_draw_range("tube_loop", 1.0)
    time.sleep(0.2)

    # Check ALL ring vertex colors — should all be blue
    color_check = viewer_page.evaluate(
        """(id) => {
            const obj = window.threejsViewer._objects.get(id);
            const col = obj.geometry.getAttribute('color');
            const nCs = obj.userData.tubeNCs;
            const nSpine = obj.userData.tubeNumSpinePoints;
            const md = obj.userData.tubeMorphData;
            const stale = [];
            for (let ring = 0; ring < nSpine; ring++) {
                const base = ring * nCs * 3;
                const r = col.array[base];
                const g = col.array[base + 1];
                const b = col.array[base + 2];
                if (r > 0.1 || b < 0.9) {
                    stale.push({ ring, r, g, b });
                }
            }
            // Also check md.ringColors
            const mdStale = [];
            if (md && md.ringColors) {
                for (let i = 0; i < nSpine; i++) {
                    const r = md.ringColors[i * 3];
                    const b = md.ringColors[i * 3 + 2];
                    if (r > 0.1 || b < 0.9) {
                        mdStale.push({ ring: i, r, g: md.ringColors[i * 3 + 1], b });
                    }
                }
            }
            // Check savedRingColors if frontier is morphed
            let savedInfo = null;
            if (md && md.savedRingIndex != null && md.savedRingColors) {
                savedInfo = {
                    savedRingIndex: md.savedRingIndex,
                    r: md.savedRingColors[0],
                    g: md.savedRingColors[1],
                    b: md.savedRingColors[2],
                };
            }
            return { staleVertexRings: stale, staleMdRings: mdStale, savedInfo };
        }""",
        "tube_loop",
    )
    assert not color_check["staleVertexRings"], (
        f"Vertex buffer has stale ring colors: {color_check['staleVertexRings']}"
    )
    assert not color_check["staleMdRings"], (
        f"md.ringColors has stale entries: {color_check['staleMdRings']}"
    )
    if color_check["savedInfo"]:
        si = color_check["savedInfo"]
        assert si["r"] < 0.1 and si["b"] > 0.9, f"savedRingColors has stale color: {si}"


@pytest.mark.browser
def test_parametric_tube_color_update_clears_pending_update_ranges(
    viewer_client, viewer_page
):
    """Race condition: morphFrontierRing adds addUpdateRange on the color attr.
    If update_parametric_tube_colors runs before the renderer consumes those
    ranges, Three.js only uploads the partial ranges instead of the full buffer.

    The fix: color update sets _colorFullUploadNeeded, and
    applyParametricTubeDrawRange clears all color update ranges at the end
    when the flag is set.

    This test simulates the exact frame sequence:
    1. morphFrontierRing adds partial color ranges (via set_draw_range)
    2. Color update writes full buffer, sets flag
    3. Next set_draw_range runs — morphFrontierRing adds ranges again,
       but cleanup at end clears them
    """
    n = 20
    spine = _straight_spine(n=n, length=4.0)
    widths = np.full(n, 0.3, dtype=np.float32)
    heights = np.full(n, 0.2, dtype=np.float32)
    initial_colors = np.full(n, 0xFF0000, dtype=np.uint32)

    viewer_client.add_parametric_tube(
        "tube_race",
        spine=spine,
        widths=widths,
        heights=heights,
        colors=initial_colors,
    )
    _wait_for_object(viewer_page, "tube_race")

    result = viewer_page.evaluate(
        """(id) => {
            const v = window.threejsViewer;
            const obj = v._objects.get(id);
            const ud = obj.userData;
            const nCs = ud.tubeNCs;
            const nSpine = ud.tubeNumSpinePoints;
            const colAttr = obj.geometry.getAttribute('color');
            const md = ud.tubeMorphData;

            // Phase 1: morphFrontierRing adds partial color ranges
            v._setDrawRange(id, 0.5);
            const rangesFromMorph = colAttr.updateRanges.length;

            // Phase 2: Simulate color update (clears ranges, writes full buffer, sets flag)
            const out = colAttr.array;
            for (let i = 0; i < nSpine * nCs; i++) {
                out[i * 3] = 0; out[i * 3 + 1] = 0; out[i * 3 + 2] = 1;
            }
            colAttr.clearUpdateRanges();
            colAttr.needsUpdate = true;
            ud._colorFullUploadNeeded = true;
            if (md && md.ringColors) {
                for (let i = 0; i < nSpine; i++) {
                    md.ringColors[i * 3] = 0;
                    md.ringColors[i * 3 + 1] = 0;
                    md.ringColors[i * 3 + 2] = 1;
                }
            }

            // Phase 3: Next draw_range update — morphFrontierRing runs again,
            // but cleanup should clear all color ranges at the end
            v._setDrawRange(id, 0.6);

            return {
                rangesFromMorph,
                rangesAfterFix: colAttr.updateRanges.length,
                flagCleared: !ud._colorFullUploadNeeded,
            };
        }""",
        "tube_race",
    )

    # Phase 1 should have added color ranges from morphFrontierRing
    assert result["rangesFromMorph"] > 0, (
        "morphFrontierRing should add color update ranges"
    )
    # Phase 3 cleanup should have cleared all color ranges
    assert result["rangesAfterFix"] == 0, (
        f"Expected 0 pending color ranges after fix, got {result['rangesAfterFix']}"
    )
    assert result["flagCleared"], "_colorFullUploadNeeded flag was not consumed"
