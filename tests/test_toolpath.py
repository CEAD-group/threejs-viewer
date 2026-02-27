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
