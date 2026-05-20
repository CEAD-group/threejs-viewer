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


def _wait_for_collapse(page, obj_id, timeout=20.0):
    """Wait until the LOD worker's collapseOnly pass has stashed both buffers
    on the tube's userData. Required before toggling strand_collapse off."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        ready = page.evaluate(
            """(id) => {
                const obj = window.threejsViewer
                    && window.threejsViewer._objects.get(id);
                return !!(obj && obj.userData
                    && obj.userData.uncollapsedPositions
                    && obj.userData.collapsedPositions);
            }""",
            obj_id,
        )
        if ready:
            return
        time.sleep(0.05)
    raise TimeoutError(f"strand_collapse buffers never landed for '{obj_id}'")


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


def test_add_parametric_tube_strand_collapse_default_omits_header_key():
    c = _capture_client()
    spine, widths, heights = _simple_tube_args()
    c.add_parametric_tube("t", spine, widths, heights)
    header, _ = c._binary_messages[-1]
    assert "strandCollapse" not in header


def test_add_parametric_tube_strand_collapse_true_sets_header():
    c = _capture_client()
    spine, widths, heights = _simple_tube_args()
    c.add_parametric_tube("t", spine, widths, heights, strand_collapse=True)
    header, _ = c._binary_messages[-1]
    assert header["strandCollapse"] is True


def test_add_parametric_tube_strand_collapse_dict_max_snap_factor_in_header():
    c = _capture_client()
    spine, widths, heights = _simple_tube_args()
    c.add_parametric_tube(
        "t", spine, widths, heights, strand_collapse={"max_snap_factor": 1.5}
    )
    header, _ = c._binary_messages[-1]
    assert header["strandCollapse"] == {"maxSnapFactor": 1.5}


def test_add_parametric_tube_strand_collapse_max_snap_factor_negative_rejected():
    c = _capture_client()
    spine, widths, heights = _simple_tube_args()
    with pytest.raises(ValueError, match="max_snap_factor"):
        c.add_parametric_tube(
            "t", spine, widths, heights, strand_collapse={"max_snap_factor": -1.0}
        )


def test_add_parametric_tube_strand_collapse_max_snap_factor_non_number_rejected():
    c = _capture_client()
    spine, widths, heights = _simple_tube_args()
    with pytest.raises(ValueError, match="max_snap_factor"):
        c.add_parametric_tube(
            "t", spine, widths, heights, strand_collapse={"max_snap_factor": "big"}
        )


def test_add_parametric_tube_strand_collapse_empty_dict_enables_defaults():
    c = _capture_client()
    spine, widths, heights = _simple_tube_args()
    c.add_parametric_tube("t", spine, widths, heights, strand_collapse={})
    header, _ = c._binary_messages[-1]
    # `{}` means "enabled with defaults", not "disabled" — must serialize
    # as True so the worker runs the collapse pass.
    assert header["strandCollapse"] is True


def test_add_parametric_tube_strand_collapse_unknown_key_rejected():
    c = _capture_client()
    spine, widths, heights = _simple_tube_args()
    with pytest.raises(ValueError, match="unknown keys"):
        c.add_parametric_tube("t", spine, widths, heights, strand_collapse={"foo": 1})


def test_add_parametric_tube_strand_collapse_non_dict_non_bool_rejected():
    c = _capture_client()
    spine, widths, heights = _simple_tube_args()
    with pytest.raises(ValueError, match="strand_collapse must be"):
        c.add_parametric_tube("t", spine, widths, heights, strand_collapse="bogus")


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


def test_add_parametric_tube_lod_threshold_float_rejected():
    """threshold is documented as integer — reject floats to avoid silent
    truncation (1.9 → 1)."""
    c = _capture_client()
    spine, widths, heights = _simple_tube_args()
    with pytest.raises(ValueError, match="threshold must be an integer"):
        c.add_parametric_tube("t", spine, widths, heights, lod={"threshold": 1.9})


def test_add_parametric_tube_lod_accepts_numpy_scalars():
    """numpy scalar types (np.float32, np.int64, ...) must be accepted —
    the rest of the client API accepts numpy-coercible numerics."""
    c = _capture_client()
    spine, widths, heights = _simple_tube_args()
    c.add_parametric_tube(
        "t",
        spine,
        widths,
        heights,
        lod={"epsilon_divisor": np.float32(7500), "threshold": np.int64(1000)},
    )
    header, _ = c._binary_messages[-1]
    assert header["lod"] == {"epsilonDivisor": 7500.0, "threshold": 1000}
    assert isinstance(header["lod"]["threshold"], int)
    assert isinstance(header["lod"]["epsilonDivisor"], float)


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


# 100-spine-point bead extracted from the ribweaver dump tube_8f5bba97.
# Triggers the strand_collapse cross-link and wide-bead-corner bugs that
# previously rendered as a flat triangulated diamond fan / cube-cluster.
# See repro_min.py for the manual visual repro context.
_REPRO_SPINE_W_H_B64 = (
    "IAa3PYAKIb7ALya7oHu2PQApIL7O2SO7IC+2PYCrH74+CBq74OG1PSAuH75m5gK7AJa1PUCwHr7U"
    "9My6AEi1PTAzHr5IInG6wEazPcDaGr4gmvs5QFCxPXCCF76g2fM5gGetPRDNEL6gzvQ5QJClPUhp"
    "A77Ap/Q5AOqVPbAt0b2gpvQ5gEttPUCgS72ghfQ5QOpqPYBoRr0Ag/Q5QMNlPaD1Q71AlPQ5AEwj"
    "O8BAA71gnPQ5ACLHvADjxbzgHvQ5YHMavcCxpbwgnfY5oOQ1vUCFlbwgue85gJ1DvUB+jbzA/v05"
    "YFxRvUCUhbxAnvO5gDBTvYCLhLy4uHi64ANVvcB5g7xwfLC6wNZWvQBogrw8ete6IKxYvUBjgbyc"
    "5ue6IHhbvYAdf7ywr+m6QEJevYBOe7ywuem6QKdjvYD/cbyMsem6AH1ovYCQY7yQxem6wEBrvYBC"
    "TrwYuum6QLBrvQBiOrzkyum6oJ9rvYBlMLxAium64DNrvQB+Jrw4Sd+6oOxqvYD8HrxASMK6oKRq"
    "vYBxF7y0GJq6oDZqvYATELxodUC6wMJpvQCrCLzAire5wI9kvQD6e7sA7f05oDxfvQDgKzrgse05"
    "YGRUvYCBHTwAAvc5IAM/vQAR4jzA8vQ5wHIUvVBCgj2AB/U5IKERvbCGhj3gnvU5AN0HvfAYiz2g"
    "mfQ5wKT9vACajD0AdvQ5AKSsPBCeqz2gpPQ5gNFAPSAruz1gsfQ5gPl1PXD/wj3AqfQ5IE6IPQDh"
    "xj3ArvQ54PGOPTDXyD1gq/Q5QKaVPZCZyj0ArfQ5YJuWPTDLyj1ArPQ54JGXPZD5yj3grPQ5YIiY"
    "PZAlyz2grPQ5wIGZPWA4yz3ArPQ54LmaPTBayz3ArPQ5IPCbPXBeyz3ArPQ5AF+ePTAeyz3ArPQ5"
    "ALigPeAWyj3ArPQ5QKmhPWAxyT3ArPQ5AD2iPTAPyD3ArPQ5IJaiPeAhxz3ArPQ5IKqiPRAtxj3A"
    "rPQ5QLuiPYAxxT3ArPQ5QKeiPQA4xD3ArPQ54EKgPaCiuj3ArPQ5QHadPaAfsT3ArPQ5QOmXPVAf"
    "nj3ArPQ5gMSMPYApcD3ArPQ5wPNsPUDcrzzArPQ5gIRqPYBwpTzArPQ5AE9lPQCboDzArPQ5AGgw"
    "OwDrgjvArPQ5wCPEvADOebvArPQ5oJEYvQDJ/LvArPQ5oNgzvYBnHrzArPQ5IHhBvQBXLrzArPQ5"
    "IDVPvQDVPLzArPQ5gCdRvQCRPrzArPQ5gB5TvQD7P7zArPQ5QBZVvQBPQbzArPQ5gA5XvYCgQrzA"
    "rPQ5ILdZvYBwRLzArPQ5IGZcvYDDRbzArPQ5AL5hvYCaSLzArPQ5YORnvYCzTbzArPQ5ABptvYAf"
    "W7zArPQ54ORvvQC/a7zArPQ5oPZxvQDwfbzArPQ5YKZyvYC9grzArPQ5gGpzvUBohrzArPQ54AF0"
    "vQBCirygrPQ5AKN0vcACjrzArPQ5oKp4vUAfqbyArPQ5IMV8vcDlw7xArfQ50D+CvUBw+rwArPQ5"
    "4FGGvaAMGL2ArfQ5IBeLvQByMb3AsfQ5UO6MvcBiM72gsfQ5IJSPveB4Mr0AmvQ5kAKSvaAfK72A"
    "lvQ5XPyCPHJTgjzIgH087S9xPBLVYTwFQ1I8f2o8PH9qPDx/ajw8f2o8PH9qPDx/ajw8f2o8PH9q"
    "PDx/ajw8f2o8PH9qPDx/ajw8f2o8POS0ZTyNUXk8fGaFPODAizxNgI486sqOPOrKjjzqyo486sqO"
    "POrKjjzqyo486sqOPPgLjTzhYog8slyBPF7mcTxkR2A8f2o8PH9qPDx/ajw8f2o8PH9qPDx/ajw8"
    "f2o8PH9qPDx/ajw8f2o8PH9qPDx/ajw8f2o8PJIyXDzRcXA8WfiBPJKdiTxu54086sqOPOrKjjzq"
    "yo486sqOPOrKjjz6uY48uNaLPGpdhTxEnng8VjdkPH9qPDx/ajw8f2o8PH9qPDx/ajw8f2o8PH9q"
    "PDx/ajw8f2o8PH9qPDx/ajw8f2o8PJO+WzxodXA8MC6CPCDbiTyxCY486sqOPOrKjjzqyo486sqO"
    "POrKjjzqyo48WnyNPGhaiTw9iYI8JD10PIb3YTx/ajw8f2o8PH9qPDx/ajw8f2o8PH9qPDx/ajw8"
    "f2o8PKabRDsmX0Y7A+1PO61iYDsZ6HQ7ptuEO7x0kzuMZ5M7vHSTO2x0kztadJM7SnKTOyBykzs0"
    "c5M7t3OTO+Brkzu8dJM7giWTO7x0kzu3QGg7tCVKO0L2LzvFlhw7MFcUO7x0Ezu8dBM7vHQTOxpx"
    "Ezu8dBM7cW4TO7x0EztFrxg7TOYmO4ZHOzvTv1U7Tn1wO7x0kzsPBZM7vHSTO7x0kzu8dJM7vHST"
    "O4pzkztQcZM7O3STO7x0kzuNdJM7vHSTO6V0kzvXrnY7LfVXO25fOjsyKyM7JicWO7x0Ezu8dBM7"
    "vHQTO7x0Ezu8dBM7IqgTO9FrHDvJETA72Y1LO7yDaju8dJM7vHSTO7x0kzu8dJM7vHSTO7x0kzu8"
    "dJM7vHSTO7x0kzu8dJM7vHSTO7x0kzveXnc7uu9XOwW8OTtgcCI7Kr8VO7x0Ezu8dBM7vHQTO7x0"
    "Ezu8dBM7vHQTOyRsFzsM9yM7sKc4OywzUjuJ7W07t3STO7x0kzuxdJM7vHSTO7x0kzu8dJM7knOT"
    "O1hzkzs="
)


def _decode_repro_bead(scale=100.0):
    import base64

    n = 100
    buf = np.frombuffer(base64.b64decode(_REPRO_SPINE_W_H_B64), dtype=np.float32)
    spine = buf[: n * 3].reshape(n, 3).copy() * scale
    widths = buf[n * 3 : n * 3 + n].copy() * scale
    heights = buf[n * 3 + n :].copy() * scale
    return spine, widths, heights


@pytest.mark.browser
def test_parametric_tube_strand_collapse_repro_renders_clean(
    viewer_client, viewer_page
):
    """Regression for the strand_collapse cross-link / wide-bead-corner bugs.

    The 100-pt bead previously rendered as a flat triangulated diamond fan /
    cube-cluster across multi-corner spans when strand_collapse=True. After
    the fix, the mesh must (a) have no NaN/Inf positions and (b) have a
    bounding box that matches strand_collapse=False within tolerance — the
    fold only pulls inside-bend strands inward, so the outer envelope is
    preserved.
    """
    spine, widths, heights = _decode_repro_bead()

    viewer_client.add_parametric_tube(
        "repro_baseline",
        spine=spine,
        widths=widths,
        heights=heights,
        anchor="top",
        lod=False,
    )
    _wait_for_object(viewer_page, "repro_baseline")

    viewer_client.add_parametric_tube(
        "repro_collapsed",
        spine=spine,
        widths=widths,
        heights=heights,
        anchor="top",
        lod=False,
        strand_collapse=True,
    )
    _wait_for_object(viewer_page, "repro_collapsed")
    _wait_for_collapse(viewer_page, "repro_collapsed")

    info = viewer_page.evaluate(
        """(ids) => {
            const out = {};
            for (const id of ids) {
                const obj = window.threejsViewer._objects.get(id);
                const pos = obj.geometry.getAttribute('position').array;
                let nonFinite = 0;
                for (let i = 0; i < pos.length; i++) {
                    if (!Number.isFinite(pos[i])) nonFinite++;
                }
                obj.geometry.computeBoundingBox();
                const bb = obj.geometry.boundingBox;
                const u = obj.userData.uncollapsedPositions;
                let maxMove = 0;
                if (u && u.length === pos.length) {
                    for (let i = 0; i < pos.length; i += 3) {
                        const dx = pos[i] - u[i];
                        const dy = pos[i + 1] - u[i + 1];
                        const dz = pos[i + 2] - u[i + 2];
                        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
                        if (d > maxMove) maxMove = d;
                    }
                }
                out[id] = {
                    nonFinite,
                    w: bb.max.x - bb.min.x,
                    h: bb.max.y - bb.min.y,
                    d: bb.max.z - bb.min.z,
                    maxMove,
                };
            }
            return out;
        }""",
        ["repro_baseline", "repro_collapsed"],
    )

    base = info["repro_baseline"]
    coll = info["repro_collapsed"]

    # The genuine fold must still fire under the default max_snap_factor=1.0
    # guard — at least one ring vertex should have moved noticeably from its
    # mitered baseline. Use the smallest width in the bead (~50 mm at scale)
    # as a conservative lower bound; the actual fold pulls strands by a
    # larger fraction of W on this dataset.
    min_w = float(np.min(widths))
    assert coll["maxMove"] > 0.1 * min_w, (
        "strand_collapse fold did not fire: max ring movement "
        f"{coll['maxMove']:.3f} ≤ 0.1·minW ({0.1 * min_w:.3f}) — "
        "default max_snap_factor=1.0 may have over-rejected the genuine fold"
    )

    assert base["nonFinite"] == 0, f"baseline has {base['nonFinite']} NaN/Inf positions"
    assert coll["nonFinite"] == 0, (
        f"strand_collapse has {coll['nonFinite']} NaN/Inf positions"
    )

    # Outer envelope is set by the cross-section's outside strands, which
    # strand_collapse never touches. Bbox must match within a couple percent
    # of the larger dimension. Before the cross-link guard, the diamond fan
    # warped the envelope by snapping inside-bend strands of unrelated
    # corners together (visually flat, geometrically detectable as bbox
    # drift on this dataset).
    tol = 0.02 * max(base["w"], base["h"], base["d"])
    assert abs(coll["w"] - base["w"]) < tol, (
        f"bbox width drifted: baseline={base['w']:.3f}, collapsed={coll['w']:.3f}"
    )
    assert abs(coll["h"] - base["h"]) < tol, (
        f"bbox height drifted: baseline={base['h']:.3f}, collapsed={coll['h']:.3f}"
    )
    assert abs(coll["d"] - base["d"]) < tol, (
        f"bbox depth drifted: baseline={base['d']:.3f}, collapsed={coll['d']:.3f}"
    )


def _u_shaped_cross_toolpath(separation_factor=0.3, W=10.0, leg_n=20):
    """Two parallel legs separated by ``separation_factor * W`` in Y.

    The spines pass the existing FOLD_SEP_FACTOR=0.5 guard (separation < 0.5·W)
    but the seg-seg midpoint between offset strands lands far from the rings,
    which used to produce > 2·W ring displacement. The snap-distance guard
    introduced here rejects that snap; max(|pos - uncollapsed|) must remain
    ≤ max(W, H).
    """
    x_fwd = np.linspace(0.0, 50.0, leg_n, dtype=np.float32)
    x_back = x_fwd[::-1]
    fwd = np.column_stack([x_fwd, np.zeros_like(x_fwd), np.zeros_like(x_fwd)])
    back = np.column_stack(
        [x_back, np.full_like(x_back, separation_factor * W), np.zeros_like(x_back)]
    )
    spine = np.vstack([fwd, back]).astype(np.float32)
    widths = np.full(spine.shape[0], W, dtype=np.float32)
    heights = np.full(spine.shape[0], W, dtype=np.float32)
    return spine, widths, heights


@pytest.mark.browser
def test_parametric_tube_strand_collapse_snap_distance_guarded(
    viewer_client, viewer_page
):
    """Cross-toolpath coincidence — snap target sits far from rings.

    With ``max_snap_factor=1.0`` no ring should move more than ``max(W, H)``
    from its mitered baseline. Without the guard, this same input produced
    displacements > 2·W on real ribweaver dumps.
    """
    spine, widths, heights = _u_shaped_cross_toolpath()
    W = float(widths[0])

    viewer_client.add_parametric_tube(
        "u_guard",
        spine=spine,
        widths=widths,
        heights=heights,
        anchor="top",
        lod=False,
        strand_collapse={"max_snap_factor": 1.0},
    )
    _wait_for_object(viewer_page, "u_guard")
    _wait_for_collapse(viewer_page, "u_guard")

    max_move = viewer_page.evaluate(
        """(id) => {
            const obj = window.threejsViewer._objects.get(id);
            const pos = obj.geometry.getAttribute('position').array;
            const u = obj.userData.uncollapsedPositions;
            let maxMove = 0;
            for (let i = 0; i < pos.length; i += 3) {
                const dx = pos[i] - u[i];
                const dy = pos[i + 1] - u[i + 1];
                const dz = pos[i + 2] - u[i + 2];
                const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
                if (d > maxMove) maxMove = d;
            }
            return maxMove;
        }""",
        "u_guard",
    )
    assert max_move <= W + 1e-3, (
        f"strand_collapse moved a ring by {max_move:.3f} mm — exceeds "
        f"max_snap_factor * max(W, H) = {W:.3f} mm"
    )


@pytest.mark.browser
def test_parametric_tube_set_strand_collapse_enabled_round_trips(
    viewer_client, viewer_page
):
    """Toggling strand_collapse off then back on restores the collapsed buffer.

    The viewer keeps both buffers alive in userData; toggling is an O(N)
    copy from the right buffer into the geometry's position attribute.
    """
    spine, widths, heights = _decode_repro_bead()
    tube_id = "toggle_tube"

    viewer_client.add_parametric_tube(
        tube_id,
        spine=spine,
        widths=widths,
        heights=heights,
        anchor="top",
        lod=False,
        strand_collapse=True,
    )
    _wait_for_object(viewer_page, tube_id)
    _wait_for_collapse(viewer_page, tube_id)

    def positions():
        return viewer_page.evaluate(
            """(id) => Array.from(
                window.threejsViewer._objects.get(id)
                    .geometry.getAttribute('position').array
            )""",
            tube_id,
        )

    collapsed = np.asarray(positions(), dtype=np.float32)

    viewer_client.set_strand_collapse_enabled(tube_id, False)
    time.sleep(0.2)
    uncollapsed = np.asarray(positions(), dtype=np.float32)
    assert not np.allclose(collapsed, uncollapsed), (
        "toggle off did not change the rendered position buffer"
    )

    viewer_client.set_strand_collapse_enabled(tube_id, True)
    time.sleep(0.2)
    re_collapsed = np.asarray(positions(), dtype=np.float32)
    np.testing.assert_allclose(collapsed, re_collapsed)
