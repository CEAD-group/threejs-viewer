"""Tests for Toolpath class."""

import numpy as np
import pytest

from threejs_viewer import Toolpath


# ---------------------------------------------------------------------------
# Toolpath.from_gcode
# ---------------------------------------------------------------------------


def _gcode_row(x, y, z, E, F):
    return [x, y, z, E, F]


def test_from_gcode_zero_len_connector_within_extrusion():
    """Zero-length connector between two extruding segments must NOT become w=0.

    Regression: arc→straight join points share the same xyz, so the segment
    arriving at them has length 0 → dE=0 → the point was incorrectly assigned
    width=0, creating a spurious cap ring in the middle of the bead.

    Toolpath:  A --extrude--> B (same as C) --extrude--> D
    B and C are at the same location (zero-length connector).
    All three segments (A→B, B→C, C→D) should produce full-width rings.
    """
    bw, bh = 2.0, 0.9
    F = 3000.0
    # Segment A→B: length=10, extruding.  dE = bw*bh*10/1000 = 0.018
    # Segment B→C: length=0, extruding (zero-length join).  dE = 0
    # Segment C→D: length=10, extruding.  dE = 0.018
    E_AB = bw * bh * 10.0 / 1000.0
    E_CD = E_AB + bw * bh * 10.0 / 1000.0
    raw = np.array(
        [
            _gcode_row(0.0, 0.0, 0.0, 0.0, F),  # A
            _gcode_row(10.0, 0.0, 0.0, E_AB, F),  # B (last of first seg)
            _gcode_row(10.0, 0.0, 0.0, E_AB, F),  # C (same xyz as B, zero-len join)
            _gcode_row(20.0, 0.0, 0.0, E_CD, F),  # D
        ],
        dtype=np.float32,
    )
    tp = Toolpath.from_gcode(raw, bw, bh)

    # B and C (indices 1 and 2) are co-located; C is the zero-len connector.
    # C should have full width because the segment C→D is extruding.
    assert tp.widths[2] == pytest.approx(bw), (
        f"Zero-len connector C should have w={bw}, got {tp.widths[2]}"
    )
    assert tp.heights[2] == pytest.approx(bh)

    # A (index 0) is the start cap — should have w=0
    assert tp.widths[0] == pytest.approx(0.0), "Start point should be a zero-width cap"

    # B, D should be full width
    assert tp.widths[1] == pytest.approx(bw)
    assert tp.widths[3] == pytest.approx(bw)


def test_from_gcode_zero_len_connector_extrusion_to_travel():
    """Zero-length connector at extrusion→travel transition must remain w=0.

    The zero-length segment joining the last extruding point to the first travel
    point is a genuine cap — the departing segment is NOT extruding, so the
    point stays at width=0.
    """
    bw, bh = 2.0, 0.9
    F = 3000.0
    E_ext = bw * bh * 10.0 / 1000.0
    raw = np.array(
        [
            _gcode_row(0.0, 0.0, 0.0, 0.0, F),  # A — start cap
            _gcode_row(10.0, 0.0, 0.0, E_ext, F),  # B — last extrusion ring
            _gcode_row(10.0, 0.0, 0.0, E_ext, F),  # C — same xyz, travel cap
            _gcode_row(20.0, 0.0, 0.0, E_ext, F),  # D — travel (E constant)
        ],
        dtype=np.float32,
    )
    tp = Toolpath.from_gcode(raw, bw, bh)

    # B (index 1) — last extruding point — should be full width
    assert tp.widths[1] == pytest.approx(bw)

    # C (index 2) — zero-len connector to travel — must be width 0 (cap)
    assert tp.widths[2] == pytest.approx(0.0), (
        f"Extrusion→travel cap should have w=0, got {tp.widths[2]}"
    )

    # D (index 3) — travel — should be width 0
    assert tp.widths[3] == pytest.approx(0.0)


def test_from_gcode_pill_no_spurious_zero_width():
    """Pill toolpath: zero-width points only at genuine travel/cap locations.

    The pill has zero-length connectors at arc→straight→arc joins within the
    extrusion section.  None of those should be zero-width.
    The only zero-width points allowed are:
    - index 0 (start cap)
    - the zero-len cap at the extrusion→travel transition
    - the actual travel segment points
    """
    bw, bh = 2.0, 0.9
    n_arc = 8  # small for speed
    n_layers = 2
    radius = 12.0
    half_length = 15.0
    layer_dz = 0.9
    print_speed = 3000.0
    travel_factor = 3.0

    z_layers = (np.arange(n_layers, dtype=np.float32) + 1) * layer_dz
    right_angles = np.linspace(-np.pi / 2, np.pi / 2, n_arc)
    left_angles = np.linspace(np.pi / 2, 3 * np.pi / 2, n_arc)

    ra = np.empty((n_layers, n_arc, 3), dtype=np.float32)
    ra[:, :, 0] = half_length + radius * np.cos(right_angles)
    ra[:, :, 1] = radius * np.sin(right_angles)
    ra[:, :, 2] = z_layers[:, None]

    ts = np.empty((n_layers, 2, 3), dtype=np.float32)
    ts[:, 0, :2] = [half_length, radius]
    ts[:, 1, :2] = [-half_length, radius]
    ts[:, :, 2] = z_layers[:, None]

    la = np.empty((n_layers, n_arc, 3), dtype=np.float32)
    la[:, :, 0] = -half_length + radius * np.cos(left_angles)
    la[:, :, 1] = radius * np.sin(left_angles)
    la[:, :, 2] = z_layers[:, None]

    bs = np.empty((n_layers, 2, 3), dtype=np.float32)
    bs[:, 0, :2] = [-half_length, -radius]
    bs[:, 1, :2] = [half_length, -radius]
    bs[:, :, 2] = z_layers[:, None]

    xyz = np.concatenate([ra, ts, la, bs], axis=1).reshape(-1, 3)
    n_ext = 2 * n_arc + 2
    is_ext_next = np.tile([True] * n_ext + [False] * 2, n_layers)
    speed_next = np.where(is_ext_next, print_speed, print_speed * travel_factor)
    F = np.concatenate([[print_speed], speed_next[:-1]]).astype(np.float32)
    seg_len_next = np.linalg.norm(np.diff(xyz, axis=0, append=xyz[-1:]), axis=1)
    dE_next = np.where(is_ext_next, bw * bh * seg_len_next / 1000.0, 0.0)
    E_cc = np.concatenate([[0.0], np.cumsum(dE_next[:-1])]).astype(np.float32)
    raw = np.column_stack([xyz, E_cc, F]).astype(np.float32)

    tp = Toolpath.from_gcode(raw, bw, bh)
    w = tp.widths

    # No zero-width point should have both neighbours extruding (spurious cap)
    spurious = []
    for i in range(1, len(w) - 1):
        if w[i] == 0.0 and w[i - 1] == bw and w[i + 1] == bw:
            spurious.append(i)

    assert spurious == [], (
        f"Spurious zero-width rings found at indices {spurious} "
        f"(surrounded by full-width extrusion on both sides)"
    )


# ---------------------------------------------------------------------------
# Toolpath.colorize
# ---------------------------------------------------------------------------


def _simple_tp(n=10):
    pts = np.zeros((n, 3), dtype=np.float32)
    pts[:, 0] = np.linspace(0, 1, n)
    return Toolpath.from_points(pts, bead_width=0.2, bead_height=0.1)


def test_colorize_string_returns_self():
    tp = _simple_tp()
    result = tp.colorize("viridis")
    assert result is tp


def test_colorize_string_shape():
    tp = _simple_tp(10)
    tp.colorize("viridis")
    assert tp.colors is not None
    assert tp.colors.shape == (10, 3)
    assert tp.colors.dtype == np.float32


def test_colorize_string_all_colormaps():
    tp = _simple_tp()
    for name in ("viridis", "plasma", "turbo"):
        tp.colorize(name)
        assert tp.colors is not None


def test_colorize_string_unknown_raises():
    tp = _simple_tp()
    with pytest.raises(ValueError, match="Unknown colormap"):
        tp.colorize("rainbow")


def test_colorize_string_travel_color_applied():
    pts = np.zeros((5, 3), dtype=np.float32)
    pts[:, 0] = np.arange(5, dtype=np.float32)
    widths = np.array([0.2, 0.2, 0.0, 0.0, 0.2], dtype=np.float32)
    heights = np.array([0.1, 0.1, 0.0, 0.0, 0.1], dtype=np.float32)
    tp = Toolpath.from_points(pts, bead_width=widths, bead_height=heights)
    tp.colorize("viridis", travel_color=(0.25, 0.25, 0.25))
    # points 2 and 3 are travel (w=h=0) → must get travel color
    assert np.allclose(tp.colors[2], [0.25, 0.25, 0.25], atol=1e-6)
    assert np.allclose(tp.colors[3], [0.25, 0.25, 0.25], atol=1e-6)


def test_colorize_hex_int():
    tp = _simple_tp(5)
    tp.colorize(0xFF0000)
    assert tp.colors.shape == (5, 3)
    assert np.allclose(tp.colors[:, 0], 1.0, atol=1e-3)
    assert np.allclose(tp.colors[:, 1], 0.0, atol=1e-3)
    assert np.allclose(tp.colors[:, 2], 0.0, atol=1e-3)


def test_colorize_tuple_rgb():
    tp = _simple_tp(5)
    tp.colorize((0.1, 0.5, 0.9))
    assert tp.colors.shape == (5, 3)
    assert np.allclose(tp.colors[0], [0.1, 0.5, 0.9], atol=1e-6)
    assert np.allclose(tp.colors[-1], [0.1, 0.5, 0.9], atol=1e-6)


def test_colorize_ndarray_3_is_solid():
    tp = _simple_tp(5)
    tp.colorize(np.array([0.2, 0.4, 0.6], dtype=np.float32))
    assert tp.colors.shape == (5, 3)
    assert np.allclose(tp.colors[2], [0.2, 0.4, 0.6], atol=1e-6)


def test_colorize_per_point_array():
    tp = _simple_tp(8)
    values = np.linspace(0, 1, 8, dtype=np.float32)
    tp.colorize(values, colormap="plasma")
    assert tp.colors is not None
    assert tp.colors.shape == (8, 3)


def test_colorize_rgb_array():
    tp = _simple_tp(5)
    rgb = np.random.rand(5, 3).astype(np.float32)
    tp.colorize(rgb)
    assert tp.colors is not None
    assert np.allclose(tp.colors, rgb)


def test_colorize_bad_array_shape_raises():
    tp = _simple_tp(5)
    with pytest.raises(ValueError, match="shape"):
        tp.colorize(np.ones((3, 3), dtype=np.float32))  # wrong N


def test_to_mesh_single_point_raises():
    # Bypass constructor validation to test to_mesh() guard directly
    tp = Toolpath(np.zeros((1, 6), dtype=np.float32))
    with pytest.raises(ValueError, match="at least 2"):
        tp.to_mesh()


def test_from_points_single_point_raises():
    pts = np.zeros((1, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="at least 2"):
        Toolpath.from_points(pts, bead_width=0.2, bead_height=0.1)


def test_from_points_bad_shape_raises():
    pts = np.zeros((4, 2), dtype=np.float32)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        Toolpath.from_points(pts, bead_width=0.2, bead_height=0.1)


def test_from_gcode_bad_shape_raises():
    raw = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match=r"\(N, 5\)"):
        Toolpath.from_gcode(raw, bead_width=0.2, bead_height=0.1)


def test_from_gcode_too_few_points_raises():
    raw = np.zeros((1, 5), dtype=np.float32)
    raw[0, 4] = 1000.0  # feedrate
    with pytest.raises(ValueError, match="at least 2"):
        Toolpath.from_gcode(raw, bead_width=0.2, bead_height=0.1)


def test_colors_setter():
    tp = _simple_tp(4)
    assert tp.colors is None
    arr = np.ones((4, 3), dtype=np.float32)
    tp.colors = arr
    assert tp.colors is not None
    tp.colors = None
    assert tp.colors is None


# ---------------------------------------------------------------------------
# Toolpath.to_mesh
# ---------------------------------------------------------------------------


def test_to_mesh_output_keys():
    tp = _simple_tp(5)
    mesh = tp.to_mesh()
    assert set(mesh.keys()) == {"positions", "indices", "normals", "colors"}


def test_to_mesh_shapes():
    N = 6
    tp = _simple_tp(N)
    mesh = tp.to_mesh()
    P = 6
    assert mesh["positions"].shape == (N * P, 3)
    assert mesh["normals"].shape == (N * P, 3)
    assert mesh["indices"].shape == ((N - 1) * P * 6,)
    assert mesh["colors"] is None


def test_to_mesh_with_colors():
    N = 5
    tp = _simple_tp(N)
    tp.colorize("viridis")
    mesh = tp.to_mesh()
    P = 6
    assert mesh["colors"] is not None
    assert mesh["colors"].shape == (N * P, 3)


def test_to_mesh_no_colors_is_none():
    tp = _simple_tp(4)
    mesh = tp.to_mesh()
    assert mesh["colors"] is None


def test_to_mesh_plane_normal_default_matches_z_up():
    """plane_normal=None (default) produces same result as passing [0,0,1]."""
    N = 5
    tp = _simple_tp(N)
    mesh_default = tp.to_mesh()
    mesh_explicit = tp.to_mesh(plane_normal=np.array([0.0, 0.0, 1.0]))
    assert np.allclose(mesh_default["positions"], mesh_explicit["positions"], atol=1e-6)
    assert np.allclose(mesh_default["normals"], mesh_explicit["normals"], atol=1e-6)


def test_to_mesh_plane_normal_y_up():
    """plane_normal=[0,1,0] produces different geometry than the default Z-up.

    Path along X, plane_normal=Y:
      Z-up: binormal=[0,-1,0], height along Z → Z spread = bead_height
      Y-up: binormal=[0,0,1], height along Y → Z spread = bead_width

    So Z spread should be larger with Y-up (bead_width=0.2) than with Z-up
    (bead_height=0.1).
    """
    pts = np.zeros((3, 3), dtype=np.float32)
    pts[:, 0] = [0.0, 1.0, 2.0]  # path along X, all at Y=Z=0
    tp = Toolpath.from_points(pts, bead_width=0.2, bead_height=0.1)

    mesh_zup = tp.to_mesh()
    mesh_yup = tp.to_mesh(plane_normal=np.array([0.0, 1.0, 0.0]))

    # Z-up: height (bead_height=0.1) goes along Z
    assert abs(np.ptp(mesh_zup["positions"][:, 2]) - 0.1) < 0.01, "Z-up: Z spread ≈ bead_height"
    # Y-up: width (bead_width=0.2) goes along Z (binormal), height along Y
    assert abs(np.ptp(mesh_yup["positions"][:, 2]) - 0.2) < 0.01, "Y-up: Z spread ≈ bead_width"
    assert abs(np.ptp(mesh_yup["positions"][:, 1]) - 0.1) < 0.01, "Y-up: Y spread ≈ bead_height"


def test_to_mesh_zero_width_ring_collapsed():
    """W=H=0 ring: all 6 vertices collapse to the path point."""
    pts = np.zeros((4, 3), dtype=np.float32)
    pts[:, 0] = np.arange(4, dtype=np.float32)
    widths = np.array([0.2, 0.0, 0.2, 0.2], dtype=np.float32)
    heights = np.array([0.1, 0.0, 0.1, 0.1], dtype=np.float32)
    tp = Toolpath.from_points(pts, bead_width=widths, bead_height=heights)
    mesh = tp.to_mesh()
    P = 6
    ring1_verts = mesh["positions"][1 * P : 2 * P]
    assert np.allclose(ring1_verts, ring1_verts[0], atol=1e-6)


# ---------------------------------------------------------------------------
# Toolpath.merge — color interpolation
# ---------------------------------------------------------------------------


def test_merge_preserves_colors():
    pts = np.zeros((5, 3), dtype=np.float32)
    pts[:, 0] = np.linspace(0, 4, 5)
    tp = Toolpath.from_points(pts, bead_width=0.2, bead_height=0.1, duration=4.0)
    tp.colorize("viridis")

    frame_times = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    merged, _ = tp.merge(frame_times)

    assert merged.colors is not None
    assert merged.colors.shape == (len(merged), 3)
    assert merged.colors.dtype == np.float32


def test_merge_no_colors_stays_none():
    pts = np.zeros((5, 3), dtype=np.float32)
    pts[:, 0] = np.linspace(0, 4, 5)
    tp = Toolpath.from_points(pts, bead_width=0.2, bead_height=0.1, duration=4.0)

    frame_times = np.array([1.0, 2.0], dtype=np.float32)
    merged, _ = tp.merge(frame_times)

    assert merged.colors is None
