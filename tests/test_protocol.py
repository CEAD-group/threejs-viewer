"""Tests for the wire protocol — verifies JSON payloads sent by ViewerClient."""

import numpy as np
import pytest

from threejs_viewer import Toolpath, ViewerClient


@pytest.fixture
def client():
    """Create a ViewerClient with mocked _send and _send_binary."""
    c = ViewerClient()
    c._messages = []
    c._binary_messages = []

    def capture_send(data):
        c._messages.append(data)

    def capture_send_binary(header, payload):
        c._binary_messages.append((header, payload))

    c._send = capture_send
    c._send_binary = capture_send_binary
    return c


# === add_group ===


def test_add_group_basic(client):
    client.add_group("g1")
    assert client._messages == [{"type": "add_group", "id": "g1"}]


def test_add_group_with_parent(client):
    client.add_group("g1", parent="p")
    msg = client._messages[0]
    assert msg["type"] == "add_group"
    assert msg["id"] == "g1"
    assert msg["parent"] == "p"


def test_add_group_with_transform(client):
    client.add_group(
        "g1", position=[1, 2, 3], rotation=[0.1, 0.2, 0.3], scale=[2, 2, 2]
    )
    msg = client._messages[0]
    assert msg["transform"]["position"] == [1, 2, 3]
    assert msg["transform"]["rotation"] == [0.1, 0.2, 0.3]
    assert msg["transform"]["scale"] == [2, 2, 2]


def test_add_group_invisible(client):
    client.add_group("g1", visible=False)
    assert client._messages[0]["visible"] is False


def test_add_group_visible_default_omitted(client):
    client.add_group("g1")
    assert "visible" not in client._messages[0]


# === Primitives with parent ===


def test_add_box_with_parent(client):
    client.add_box("b", parent="g")
    msg = client._messages[0]
    assert msg["type"] == "add_object"
    assert msg["parent"] == "g"
    assert msg["object"]["primitive"] == "box"


def test_add_sphere_with_parent(client):
    client.add_sphere("s", parent="g")
    msg = client._messages[0]
    assert msg["type"] == "add_object"
    assert msg["parent"] == "g"
    assert msg["object"]["primitive"] == "sphere"


def test_add_cylinder_with_parent(client):
    client.add_cylinder("c", parent="g")
    msg = client._messages[0]
    assert msg["type"] == "add_object"
    assert msg["parent"] == "g"
    assert msg["object"]["primitive"] == "cylinder"


def test_add_capsule_with_parent(client):
    client.add_capsule("c", parent="g")
    msg = client._messages[0]
    assert msg["type"] == "add_object"
    assert msg["parent"] == "g"
    assert msg["object"]["primitive"] == "capsule"


# === Primitives without parent — backward compat ===


def test_add_box_no_parent(client):
    client.add_box("b")
    msg = client._messages[0]
    assert "parent" not in msg
    assert msg["type"] == "add_object"
    assert msg["id"] == "b"
    assert msg["object"]["primitive"] == "box"
    assert msg["object"]["params"]["color"] == 0x4A90D9


def test_add_box_default_payload(client):
    """Regression guard: verify complete default payload for add_box."""
    client.add_box("b")
    msg = client._messages[0]
    assert msg == {
        "type": "add_object",
        "id": "b",
        "object": {
            "primitive": "box",
            "params": {
                "width": 1,
                "height": 1,
                "depth": 1,
                "color": 0x4A90D9,
                "opacity": 1.0,
            },
            "transform": None,
        },
    }


# === add_grid ===


def test_add_grid_default_payload(client):
    """Regression guard: verify complete default payload for add_grid."""
    client.add_grid("floor")
    msg = client._messages[0]
    assert msg == {
        "type": "add_grid",
        "id": "floor",
        "cellSize": 1.0,
        "extent": 100.0,
        "lineWidth": 1.5,
        "color": 0x555555,
        "backgroundOpacity": 0.0,
        "fadeStart": 0.5,
    }


def test_add_grid_full_payload(client):
    client.add_grid(
        "floor",
        cell_size=250.0,
        extent=8000.0,
        line_width=2.0,
        color=0x336699,
        center_color=0xFFFFFF,
        background_color=0x111111,
        background_opacity=0.25,
        fade_start=0.6,
        position=[1, 2, 3],
        rotation=[0.5, 0, 0],
        parent="cell",
    )
    msg = client._messages[0]
    assert msg["type"] == "add_grid"
    assert msg["cellSize"] == 250.0
    assert msg["extent"] == 8000.0
    assert msg["lineWidth"] == 2.0
    assert msg["color"] == 0x336699
    assert msg["centerColor"] == 0xFFFFFF
    assert msg["backgroundColor"] == 0x111111
    assert msg["backgroundOpacity"] == 0.25
    assert msg["fadeStart"] == 0.6
    assert msg["transform"] == {"position": [1, 2, 3], "rotation": [0.5, 0, 0]}
    assert msg["parent"] == "cell"


def test_add_grid_optional_colors_omitted(client):
    client.add_grid("floor")
    msg = client._messages[0]
    assert "centerColor" not in msg
    assert "backgroundColor" not in msg
    assert "transform" not in msg
    assert "parent" not in msg


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cell_size": 0},
        {"cell_size": -1},
        {"extent": 0},
        {"line_width": 0},
        {"background_opacity": -0.1},
        {"background_opacity": 1.1},
        {"fade_start": -0.1},
        {"fade_start": 1.5},
    ],
)
def test_add_grid_validation(client, kwargs):
    with pytest.raises(ValueError):
        client.add_grid("floor", **kwargs)
    assert client._messages == []


# === add_model ===


def test_add_model_with_parent(client):
    client.add_model("m", "http://example.com/model.glb", parent="g")
    msg = client._messages[0]
    assert msg["type"] == "add_object"
    assert msg["parent"] == "g"
    assert msg["object"]["model"] == "http://example.com/model.glb"


def test_add_model_no_parent(client):
    client.add_model("m", "http://example.com/model.glb")
    assert "parent" not in client._messages[0]


def test_add_model_y_up_true(client):
    client.add_model("m", "http://example.com/model.glb", y_up=True)
    assert client._messages[0]["object"]["yUp"] is True


def test_add_model_y_up_default_omitted(client):
    client.add_model("m", "http://example.com/model.glb")
    assert "yUp" not in client._messages[0]["object"]


# === add_model_binary ===


def test_add_model_binary_with_parent(client):
    client.add_model_binary("m", b"\x00" * 10, parent="g")
    header, payload = client._binary_messages[0]
    assert header["type"] == "add_model_binary"
    assert header["parent"] == "g"
    assert payload == b"\x00" * 10


def test_add_model_binary_no_parent(client):
    client.add_model_binary("m", b"\x00" * 10)
    header, _ = client._binary_messages[0]
    assert "parent" not in header


def test_add_model_binary_y_up_true(client):
    client.add_model_binary("m", b"\x00" * 10, y_up=True)
    header, _ = client._binary_messages[0]
    assert header["yUp"] is True


def test_add_model_binary_y_up_default_omitted(client):
    client.add_model_binary("m", b"\x00" * 10)
    header, _ = client._binary_messages[0]
    assert "yUp" not in header


# === add_polyline ===


def test_add_polyline_with_parent(client):
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    client.add_polyline("pl", pts, parent="g")
    header, _ = client._binary_messages[0]
    assert header["type"] == "add_polyline_binary"
    assert header["parent"] == "g"


def test_add_polyline_no_parent(client):
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    client.add_polyline("pl", pts)
    header, _ = client._binary_messages[0]
    assert "parent" not in header


def test_update_polyline_colors_rgb(client):
    rgb = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    client.update_polyline_colors("pl", rgb)
    header, payload = client._binary_messages[0]
    assert header["type"] == "update_polyline_colors"
    assert header["id"] == "pl"
    assert header["numPoints"] == 2
    # Two points × 3 channels = 6 uint8 bytes — red, then blue.
    assert payload == bytes([255, 0, 0, 0, 0, 255])


def test_update_polyline_colors_scalar_uses_colormap(client):
    # Scalar input gets passed through `_apply_colormap`, so two distinct
    # values produce two distinct (non-equal) RGB entries.
    scalars = np.array([0.0, 1.0], dtype=np.float32)
    client.update_polyline_colors("pl", scalars, colormap="viridis")
    header, payload = client._binary_messages[0]
    assert header["type"] == "update_polyline_colors"
    assert header["numPoints"] == 2
    assert len(payload) == 6
    # First and last viridis stops differ on every channel.
    assert payload[:3] != payload[3:]


def test_update_polyline_colors_rejects_bad_shapes(client):
    # (N, 4) RGBA — must raise rather than ship a misaligned uint8 blob.
    rgba = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="N, 3"):
        client.update_polyline_colors("pl", rgba)
    # (N, 2) — likewise.
    bad = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        client.update_polyline_colors("pl", bad)


# === add_mesh ===


def test_add_mesh_with_parent(client):
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    idx = np.array([0, 1, 2], dtype=np.uint32)
    client.add_mesh("m", pos, idx, parent="g")
    header, _ = client._binary_messages[0]
    assert header["type"] == "add_mesh_binary"
    assert header["parent"] == "g"


def test_add_mesh_no_parent(client):
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    idx = np.array([0, 1, 2], dtype=np.uint32)
    client.add_mesh("m", pos, idx)
    header, _ = client._binary_messages[0]
    assert "parent" not in header


def test_add_mesh_opacity_forwarded(client):
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    idx = np.array([0, 1, 2], dtype=np.uint32)
    client.add_mesh("m", pos, idx, opacity=0.5)
    header, _ = client._binary_messages[0]
    assert header["opacity"] == 0.5


def test_add_mesh_opacity_default_is_one(client):
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    idx = np.array([0, 1, 2], dtype=np.uint32)
    client.add_mesh("m", pos, idx)
    header, _ = client._binary_messages[0]
    assert header["opacity"] == 1.0


def test_add_mesh_with_position(client):
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    idx = np.array([0, 1, 2], dtype=np.uint32)
    client.add_mesh("m", pos, idx, position=[1, 2, 3])
    header, _ = client._binary_messages[0]
    assert header["transform"] == {"position": [1, 2, 3]}


def test_add_mesh_with_matrix(client):
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    idx = np.array([0, 1, 2], dtype=np.uint32)
    mat = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1]
    client.add_mesh("m", pos, idx, matrix=mat)
    header, _ = client._binary_messages[0]
    assert header["transform"] == {"matrix": mat}


def test_add_mesh_no_transform(client):
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    idx = np.array([0, 1, 2], dtype=np.uint32)
    client.add_mesh("m", pos, idx)
    header, _ = client._binary_messages[0]
    assert "transform" not in header


# === add_points ===


def test_add_points_flat_color(client):
    pts = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
    client.add_points("pc", pts, color=0xFF8800, size=3.0)
    header, payload = client._binary_messages[0]
    assert header["type"] == "add_points_binary"
    assert header["id"] == "pc"
    assert header["numPoints"] == 3
    assert header["hasVertexColors"] is False
    assert header["color"] == 0xFF8800
    assert header["size"] == 3.0
    assert header["sizeAttenuation"] is True
    # No vertex colors → payload is positions only (3 points × 3 × float32).
    assert len(payload) == 3 * 3 * 4


def test_add_points_size_attenuation_off(client):
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    client.add_points("pc", pts, size_attenuation=False)
    header, _ = client._binary_messages[0]
    assert header["sizeAttenuation"] is False


def test_add_points_rgb_colors(client):
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    rgb = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    client.add_points("pc", pts, colors=rgb)
    header, payload = client._binary_messages[0]
    assert header["hasVertexColors"] is True
    assert header["numPoints"] == 2
    # positions (2×3 float32 = 24 bytes) then colors (2×3 uint8 = 6 bytes).
    assert len(payload) == 24 + 6
    assert payload[24:] == bytes([255, 0, 0, 0, 0, 255])


def test_add_points_scalar_colors_use_colormap(client):
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    scalars = np.array([0.0, 1.0], dtype=np.float32)
    client.add_points("pc", pts, colors=scalars, colormap="viridis")
    header, payload = client._binary_messages[0]
    assert header["hasVertexColors"] is True
    color_bytes = payload[24:]
    assert len(color_bytes) == 6
    # First and last viridis stops differ on every channel.
    assert color_bytes[:3] != color_bytes[3:]


def test_add_points_rejects_bad_color_shape(client):
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    rgba = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="N, 3"):
        client.add_points("pc", pts, colors=rgba)


def test_add_points_rejects_color_length_mismatch(client):
    pts = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
    # 2 scalars for 3 points — would pack a misaligned color blob (NaN colors
    # in the browser) without this guard.
    with pytest.raises(ValueError, match="length 3"):
        client.add_points("pc", pts, colors=np.array([0.0, 1.0], dtype=np.float32))
    # (N, 3) RGB with the wrong row count too.
    with pytest.raises(ValueError, match="length 3"):
        client.add_points("pc", pts, colors=np.zeros((2, 3), dtype=np.float32))
    # A 0-D scalar must raise a clean ValueError, not an IndexError on shape[0].
    with pytest.raises(ValueError, match="length 3"):
        client.add_points("pc", pts, colors=np.float32(0.5))


def test_add_points_rejects_bad_positions_shape(client):
    # 1D length not a multiple of 3 — must raise, not silently truncate via // 3.
    with pytest.raises(ValueError):
        client.add_points("pc", np.zeros(7, dtype=np.float32))


def test_add_points_time_windows_payload_layout(client):
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    birth = np.array([0.5, 1.5], dtype=np.float32)
    removal = np.array([2.0, 3.0], dtype=np.float32)
    client.add_points("pc", pts, birth_times=birth, removal_times=removal)
    header, payload = client._binary_messages[0]
    assert header["hasBirthTimes"] is True
    assert header["hasRemovalTimes"] is True
    # positions (24 B) + birth (8 B) + removal (8 B), no colors.
    assert len(payload) == 24 + 8 + 8
    np.testing.assert_array_equal(
        np.frombuffer(payload[24:32], dtype=np.float32), birth
    )
    np.testing.assert_array_equal(
        np.frombuffer(payload[32:40], dtype=np.float32), removal
    )


def test_add_points_time_windows_after_colors(client):
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    rgb = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    removal = np.array([2.0, 3.0], dtype=np.float32)
    client.add_points("pc", pts, colors=rgb, removal_times=removal)
    header, payload = client._binary_messages[0]
    assert "hasBirthTimes" not in header
    assert header["hasRemovalTimes"] is True
    # positions (24 B) + colors (6 B) + removal (8 B).
    assert len(payload) == 24 + 6 + 8
    np.testing.assert_array_equal(
        np.frombuffer(payload[30:38], dtype=np.float32), removal
    )


def test_add_points_time_windows_flags_omitted_by_default(client):
    pts = np.array([[0, 0, 0]], dtype=np.float32)
    client.add_points("pc", pts)
    header, _ = client._binary_messages[0]
    assert "hasBirthTimes" not in header
    assert "hasRemovalTimes" not in header


def test_add_points_time_nan_inf_map_to_unbounded_sentinels(client):
    # NaN/±inf never reach the float32 attribute — GLSL NaN compares are
    # undefined. NaN birth = "always existed" (−FLT_MAX); NaN removal =
    # "never removed" (+FLT_MAX); ±inf clamp likewise.
    flt_max = float(np.finfo(np.float32).max)
    pts = np.zeros((3, 3), dtype=np.float32)
    birth = np.array([np.nan, -np.inf, 1.0])
    removal = np.array([np.nan, np.inf, 2.0])
    client.add_points("pc", pts, birth_times=birth, removal_times=removal)
    _, payload = client._binary_messages[0]
    packed_birth = np.frombuffer(payload[36:48], dtype=np.float32)
    packed_removal = np.frombuffer(payload[48:60], dtype=np.float32)
    assert packed_birth[0] == -flt_max
    assert packed_birth[1] == -flt_max
    assert packed_birth[2] == 1.0
    assert packed_removal[0] == flt_max
    assert packed_removal[1] == flt_max
    assert packed_removal[2] == 2.0


def test_add_points_rejects_time_length_mismatch(client):
    pts = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="birth_times must have length 3"):
        client.add_points("pc", pts, birth_times=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="removal_times must have length 3"):
        client.add_points("pc", pts, removal_times=np.zeros(4))


def test_set_points_time(client):
    client.set_points_time("pc", 2.5)
    assert client._messages == [{"type": "set_points_time", "id": "pc", "time": 2.5}]


def test_set_points_time_rejects_non_finite(client):
    with pytest.raises(ValueError, match="finite"):
        client.set_points_time("pc", float("nan"))
    with pytest.raises(ValueError, match="finite"):
        client.set_points_time("pc", float("inf"))


def test_set_clip_time(client):
    client.set_clip_time("model", 1.25)
    assert client._messages == [{"type": "set_clip_time", "id": "model", "time": 1.25}]


def test_set_clip_progress(client):
    client.set_clip_progress("model", 0.5)
    assert client._messages == [{"type": "set_clip_progress", "id": "model", "t": 0.5}]


def test_set_clip_progress_coerces_to_float(client):
    client.set_clip_progress("model", 1)
    assert client._messages == [{"type": "set_clip_progress", "id": "model", "t": 1.0}]
    assert isinstance(client._messages[0]["t"], float)


def test_set_clip_progress_rejects_non_finite(client):
    with pytest.raises(ValueError, match="finite"):
        client.set_clip_progress("model", float("nan"))
    with pytest.raises(ValueError, match="finite"):
        client.set_clip_progress("model", float("inf"))
    with pytest.raises(ValueError, match="finite"):
        client.set_clip_progress("model", float("-inf"))
    assert client._messages == []


def test_add_points_with_parent(client):
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    client.add_points("pc", pts, parent="g")
    header, _ = client._binary_messages[0]
    assert header["parent"] == "g"


def test_add_points_no_parent(client):
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    client.add_points("pc", pts)
    header, _ = client._binary_messages[0]
    assert "parent" not in header


# === add_swept_tool ===


def _swept_args():
    pos = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    axes = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
    profile = np.array([[0, 0.0], [0.5, 0.5], [0.5, 0.4], [4.0, 0.4]], dtype=np.float32)
    return pos, axes, profile


def test_add_swept_tool_header_and_payload(client):
    pos, axes, profile = _swept_args()
    client.add_swept_tool("tool", pos, axes, profile, sections=20)
    header, payload = client._binary_messages[0]
    assert header["type"] == "add_swept_tool_binary"
    assert header["numStations"] == 3
    assert header["numProfile"] == 4
    assert header["sections"] == 20
    assert header["hasColors"] is False
    # positions (3×3 f32) + axes (3×3 f32) + profile (4×2 f32), no colors.
    assert len(payload) == 3 * 3 * 4 + 3 * 3 * 4 + 4 * 2 * 4


def test_add_swept_tool_normalizes_axes(client):
    pos, _, profile = _swept_args()
    axes = np.array([[0, 0, 5], [0, 0, 5], [0, 0, 5]], dtype=np.float32)
    client.add_swept_tool("tool", pos, axes, profile)
    _, payload = client._binary_messages[0]
    # The axis block is the second 3×3 f32 section; each axis must be unit-length.
    axis_block = np.frombuffer(payload[36:72], dtype=np.float32).reshape(3, 3)
    assert np.allclose(np.linalg.norm(axis_block, axis=1), 1.0, atol=1e-5)


def test_add_swept_tool_colors_packed(client):
    pos, axes, profile = _swept_args()
    colors = np.array([0xFF0000, 0x00FF00, 0x0000FF], dtype=np.uint32)
    client.add_swept_tool("tool", pos, axes, profile, colors=colors)
    header, payload = client._binary_messages[0]
    assert header["hasColors"] is True
    tail = np.frombuffer(payload[-12:], dtype=np.uint32)
    assert list(tail) == [0xFF0000, 0x00FF00, 0x0000FF]


def test_add_swept_tool_material_and_transform(client):
    pos, axes, profile = _swept_args()
    client.add_swept_tool(
        "tool",
        pos,
        axes,
        profile,
        opacity=0.5,
        metalness=0.2,
        roughness=0.7,
        parent="g",
        position=[1, 2, 3],
    )
    header, _ = client._binary_messages[0]
    assert header["opacity"] == 0.5
    assert header["metalness"] == 0.2
    assert header["roughness"] == 0.7
    assert header["parent"] == "g"
    assert header["transform"] == {"position": [1, 2, 3]}


def test_add_swept_tool_validates(client):
    pos, axes, profile = _swept_args()
    with pytest.raises(ValueError, match="stations"):
        client.add_swept_tool("t", pos[:1], axes[:1], profile)
    with pytest.raises(ValueError, match="profile"):
        client.add_swept_tool("t", pos, axes, np.array([[0, 1.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="non-zero"):
        client.add_swept_tool("t", pos, np.zeros((3, 3), dtype=np.float32), profile)
    with pytest.raises(ValueError, match="sections"):
        client.add_swept_tool("t", pos, axes, profile, sections=2)
    with pytest.raises(ValueError, match="length"):
        client.add_swept_tool("t", pos, axes[:2], profile)
    # Heights must be non-decreasing (the docstring's contract).
    with pytest.raises(ValueError, match="non-decreasing"):
        client.add_swept_tool(
            "t", pos, axes, np.array([[0, 0.5], [2, 0.4], [1, 0.3]], dtype=np.float32)
        )
    # Equal consecutive heights (a vertical step) must be allowed.
    client.add_swept_tool(
        "t", pos, axes, np.array([[0, 0.5], [1, 0.5], [1, 0.3]], dtype=np.float32)
    )


# === Toolpath.add_toolpath ===


def test_add_toolpath_with_parent(client):
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    tp = Toolpath.from_points(pts, bead_width=1.0, bead_height=0.5)
    client.add_toolpath("bd", tp, parent="g")
    header, _ = client._binary_messages[0]
    assert header["type"] == "add_parametric_tube_binary"
    assert header["parent"] == "g"


def test_add_toolpath_no_parent(client):
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    tp = Toolpath.from_points(pts, bead_width=1.0, bead_height=0.5)
    client.add_toolpath("bd", tp)
    header, _ = client._binary_messages[0]
    assert header["type"] == "add_parametric_tube_binary"
    assert "parent" not in header


def test_add_toolpath_colorize_sets_colors(client):
    """colorize() -> add_toolpath sends hasColors=True."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    tp = Toolpath.from_points(pts, bead_width=1.0, bead_height=0.5)
    tp.colorize("plasma")
    client.add_toolpath("bd", tp)
    header, _ = client._binary_messages[0]
    assert header["hasColors"] is True


def test_add_toolpath_no_colorize_no_colors(client):
    """Without colorize(), add_toolpath sends hasColors=False."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    tp = Toolpath.from_points(pts, bead_width=1.0, bead_height=0.5)
    client.add_toolpath("bd", tp)
    header, _ = client._binary_messages[0]
    assert header["hasColors"] is False


def test_add_toolpath_with_zero_width_travel(client):
    """Toolpath with zero-width travel splits into segment group."""
    pts = np.zeros((6, 3), dtype=np.float32)
    pts[:, 0] = np.arange(6, dtype=np.float32)
    widths = np.array([0.2, 0.2, 0.0, 0.0, 0.2, 0.2], dtype=np.float32)
    heights = np.array([0.1, 0.1, 0.0, 0.0, 0.1, 0.1], dtype=np.float32)
    tp = Toolpath.from_points(pts, bead_width=widths, bead_height=heights)
    client.add_toolpath("b", tp)
    # Should create: add_group, 2x add_parametric_tube_binary, register_toolpath_group
    group_msg = client._messages[0]
    assert group_msg["type"] == "add_group"
    assert group_msg["id"] == "b"
    assert len(client._binary_messages) == 2
    assert client._binary_messages[0][0]["numSpinePoints"] == 2
    assert client._binary_messages[1][0]["numSpinePoints"] == 2
    reg_msg = client._messages[1]
    assert reg_msg["type"] == "register_toolpath_group"
    assert reg_msg["segmentIds"] == ["b_seg_0", "b_seg_1"]


def test_add_toolpath_passes_kwargs(client):
    """Extra kwargs (roughness, metalness, etc.) are forwarded."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    tp = Toolpath.from_points(pts, bead_width=1.0, bead_height=0.5)
    client.add_toolpath("bd", tp, roughness=0.4, metalness=0.1)
    header, _ = client._binary_messages[0]
    assert header["roughness"] == 0.4
    assert header["metalness"] == 0.1


def test_add_toolpath_forwards_lod_false(client):
    """lod=False on add_toolpath passes through to add_parametric_tube header."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    tp = Toolpath.from_points(pts, bead_width=1.0, bead_height=0.5)
    client.add_toolpath("bd", tp, lod=False)
    header, _ = client._binary_messages[0]
    assert header["lod"] is False


# === query_scene ===


def test_query_scene_sends_correct_message(client):
    """Verify query_scene sends the right request type with a requestId."""
    # Patch _send to also simulate an immediate response so query_scene doesn't block
    sent = []

    def mock_send(data):
        sent.append(data)
        # Simulate viewer response immediately
        if data.get("type") == "query_scene":
            rid = data["requestId"]
            if rid in client._pending_responses:
                client._responses[rid] = {
                    "type": "query_scene_response",
                    "requestId": rid,
                    "tree": {},
                }
                client._pending_responses[rid].set()

    client._send = mock_send
    result = client.query_scene(timeout=1.0)

    assert len(sent) == 1
    assert sent[0]["type"] == "query_scene"
    assert "requestId" in sent[0]
    assert result == {"objects": {}, "meta": {}}


# === Clipping plane ===


def test_set_clipping_plane_with_normal(client):
    client.set_clipping_plane(normal=[0, 0, 1], distance=2.0)
    assert client._messages == [
        {
            "type": "set_clipping_plane",
            "normal": [0, 0, 1],
            "distance": 2.0,
            "show_helper": True,
        }
    ]


def test_set_clipping_plane_no_normal(client):
    client.set_clipping_plane(distance=1.0)
    msg = client._messages[0]
    assert msg == {
        "type": "set_clipping_plane",
        "distance": 1.0,
        "show_helper": True,
    }
    assert "normal" not in msg


def test_set_clipping_slab(client):
    client.set_clipping_slab(normal=[0, 0, 1], center=2.0, thickness=1.0)
    assert client._messages == [
        {
            "type": "set_clipping_slab",
            "normal": [0, 0, 1],
            "center": 2.0,
            "thickness": 1.0,
            "show_helper": True,
        }
    ]


def test_disable_clipping_plane(client):
    client.disable_clipping_plane()
    assert client._messages == [{"type": "disable_clipping_plane"}]


def test_set_clipping_defaults(client):
    client.set_clipping_defaults(normal=[0, 0, -1], distance=3.0)
    assert client._messages == [
        {
            "type": "set_clipping_defaults",
            "normal": [0, 0, -1],
            "distance": 3.0,
        }
    ]


# === camera pose API ===


def test_set_camera_full_message(client):
    client.set_camera(
        position=[1, 2, 3], target=[0, 0.5, 0], up=[0, 0, 1], fov=50, zoom=2.0
    )
    assert client._messages == [
        {
            "type": "set_camera",
            "position": [1.0, 2.0, 3.0],
            "target": [0.0, 0.5, 0.0],
            "up": [0.0, 0.0, 1.0],
            "fov": 50.0,
            "zoom": 2.0,
        }
    ]


def test_set_camera_partial_omits_unset_fields(client):
    client.set_camera(target=[1, 1, 1])
    assert client._messages == [{"type": "set_camera", "target": [1.0, 1.0, 1.0]}]


def test_set_camera_validation(client):
    with pytest.raises(ValueError, match="3-vector"):
        client.set_camera(position=[1, 2])
    with pytest.raises(ValueError, match="finite"):
        client.set_camera(target=[float("nan"), 0, 0])
    with pytest.raises(ValueError, match="fov"):
        client.set_camera(fov=200)
    with pytest.raises(ValueError, match="zoom"):
        client.set_camera(zoom=0)
    assert client._messages == []


# === set_view ===


def test_set_view_message(client):
    client.set_view("top")
    assert client._messages == [{"type": "set_view", "name": "top"}]


def test_set_view_no_animate(client):
    client.set_view("iso", animate=False)
    assert client._messages == [{"type": "set_view", "name": "iso", "animate": False}]


def test_set_view_accepts_all_predefined_names(client):
    for name in ["top", "bottom", "front", "back", "left", "right", "iso", "home"]:
        client.set_view(name)
    assert [m["name"] for m in client._messages] == [
        "top",
        "bottom",
        "front",
        "back",
        "left",
        "right",
        "iso",
        "home",
    ]


def test_set_view_rejects_unknown_name(client):
    with pytest.raises(ValueError, match="view must be one of"):
        client.set_view("sideways")
    with pytest.raises(ValueError, match="view must be one of"):
        client.set_view("TOP")  # case-sensitive on purpose
    assert client._messages == []


# === follow path ===


def test_set_follow_path_payload(client):
    times = np.array([0.0, 1.0, 3.0])
    pos = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0]], dtype=np.float32)
    axes = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
    client.set_follow_path("tool", times, pos, axes)
    header, payload = client._binary_messages[0]
    assert header["type"] == "set_follow_path"
    assert header["id"] == "tool"
    assert header["count"] == 3
    # Layout: (K,) f64 times, then (K, 6) f32 [px, py, pz, ax, ay, az].
    t = np.frombuffer(payload[: 3 * 8], dtype=np.float64)
    rows = np.frombuffer(payload[3 * 8 :], dtype=np.float32).reshape(3, 6)
    np.testing.assert_array_equal(t, times)
    np.testing.assert_array_equal(rows[:, 0:3], pos)
    np.testing.assert_array_equal(rows[:, 3:6], axes)


def test_set_follow_path_times_keep_float64_precision(client):
    # Millisecond-spaced keys late in an hours-long timeline: float32's ulp
    # at t=160,000 s is ~15.6 ms, so f32 packing would collapse these to
    # equal (or 1-ulp-stepped) values. The f64 payload must round-trip them
    # bit-exactly.
    times = 160_000.0 + np.array([0.0, 0.005, 0.010])
    pos = np.zeros((3, 3), dtype=np.float32)
    axes = np.tile(np.array([0, 0, 1], dtype=np.float32), (3, 1))
    client.set_follow_path("tool", times, pos, axes)
    _, payload = client._binary_messages[0]
    t = np.frombuffer(payload[: 3 * 8], dtype=np.float64)
    np.testing.assert_array_equal(t, times)
    assert np.all(np.diff(t) > 0), "adjacent keys must stay distinct"


def test_animation_float64_channels_layout_and_precision(client):
    """draw_ranges/clip_times/point_times pack as float64: the packer sorts
    channels size-descending so the float64 block leads the payload (8-byte
    aligned for the browser's Float64Array view), and fraction values that
    would collapse to 1.0 in float32 stay distinct."""
    from threejs_viewer import Animation

    anim = Animation(loop=False)
    anim.set_frame_times(np.array([0.0, 1.0, 2.0]))
    # uint8 channel added FIRST — the packer must still emit float64 first.
    anim.add_channel(
        "visibility", ["a"], np.ones((3, 1), dtype=np.uint8), dtype="uint8"
    )
    # Adjacent fractions ~1e-9 apart near 1.0: one float32 ulp there is
    # ~6e-8, so f32 packing would quantize all three to exactly 1.0.
    fracs = 1.0 - np.array([[3e-9], [2e-9], [1e-9]])
    anim.set_draw_range_data(["a"], fracs)
    client.load_animation(anim, autoplay=False)

    header = next(m for m in client._messages if m["type"] == "load_animation_http")
    assert [ch["dtype"] for ch in header["channels"]] == ["float64", "uint8"]
    blob_key = "/" + header["blob_url"].rsplit("/", 1)[1]
    payload = client._blob_store[blob_key]
    packed = np.frombuffer(payload[: 3 * 8], dtype=np.float64)
    np.testing.assert_array_equal(packed, fracs.ravel())
    assert np.all(np.diff(packed) > 0), "fractions must stay distinct"
    assert not np.any(packed == 1.0)


def test_set_follow_path_validation(client):
    ok3 = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="K>=2"):
        client.set_follow_path("t", [0.0], ok3[:1], ok3[:1])
    with pytest.raises(ValueError, match="K>=2"):
        client.set_follow_path("t", [0.0, 1.0, 2.0], ok3[:2], ok3)
    with pytest.raises(ValueError, match="non-decreasing"):
        client.set_follow_path("t", [0.0, 2.0, 1.0], ok3, ok3)
    assert client._binary_messages == []


# === add_polyline segments / add_toolpath travel line ===


def test_add_polyline_segments_header_and_constraints(client):
    pairs = np.array([[0, 0, 0], [1, 0, 0], [5, 0, 0], [6, 0, 0]], dtype=np.float32)
    client.add_polyline("seg", pairs, segments=True, fat=True, line_width=4)
    header, payload = client._binary_messages[0]
    assert header["segments"] is True
    assert header["fat"] is False  # segments implies the native path
    assert header["pickable"] is False  # edge soups have no arc length
    assert header["numPoints"] == 4
    assert len(payload) == 4 * 12
    # odd point count rejected
    with pytest.raises(ValueError, match="even point count"):
        client.add_polyline("bad", pairs[:3], segments=True)


def test_add_toolpath_travel_line(client):
    """travel="line" adds one LineSegments child covering the travel edges
    and registers it with ascending end-fraction thresholds."""
    pts = np.zeros((8, 3), dtype=np.float32)
    pts[:, 0] = np.arange(8, dtype=np.float32)
    widths = np.array([0.2, 0.2, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0], dtype=np.float32)
    heights = np.where(widths > 0, 0.1, 0.0).astype(np.float32)
    tp = Toolpath.from_points(pts, bead_width=widths, bead_height=heights)
    client.add_toolpath("b", tp, travel="line", travel_color=0x334455)

    # group + register with travel mapping
    assert client._messages[0]["type"] == "add_group"
    reg = client._messages[-1]
    assert reg["type"] == "register_toolpath_group"
    assert reg["travelId"] == "b_travel"
    # travel edges: every edge not interior to an extrusion run =
    # (1,2) (2,3) (3,4) and the trailing (5,6) (6,7)
    # fractions use the draw_range convention index/(n-1); dividing by n
    # skewed the recovered frontier index by `value` points (the
    # nozzle-vs-frontier desync on long moves)
    assert reg["travelEndFracs"] == pytest.approx([2 / 7, 3 / 7, 4 / 7, 6 / 7, 7 / 7])
    assert reg["travelEndFracs"] == sorted(reg["travelEndFracs"])
    flat = [x for pair in reg["segmentRanges"] for x in pair]
    assert flat == pytest.approx([0 / 7, 1 / 7, 4 / 7, 5 / 7])

    # the travel polyline itself: segments, parented to the group
    travel_headers = [
        h for h, _ in client._binary_messages if h["type"] == "add_polyline_binary"
    ]
    assert len(travel_headers) == 1
    th = travel_headers[0]
    assert th["id"] == "b_travel"
    assert th["segments"] is True
    assert th["parent"] == "b"
    assert th["color"] == 0x334455
    assert th["numPoints"] == 10  # 5 edges x 2 endpoints

    # 2 bead segments still present
    tubes = [
        h
        for h, _ in client._binary_messages
        if h["type"] == "add_parametric_tube_binary"
    ]
    assert len(tubes) == 2


def test_add_toolpath_travel_line_single_segment_still_groups(client):
    """A single extrusion run with leading travel still gets the group +
    travel line (previously a bare tube, which had nowhere to hang the
    travel mapping)."""
    pts = np.zeros((5, 3), dtype=np.float32)
    pts[:, 0] = np.arange(5, dtype=np.float32)
    widths = np.array([0.0, 0.0, 0.2, 0.2, 0.2], dtype=np.float32)
    heights = np.where(widths > 0, 0.1, 0.0).astype(np.float32)
    tp = Toolpath.from_points(pts, bead_width=widths, bead_height=heights)

    client.add_toolpath("solo", tp, travel="line")
    assert client._messages[0]["type"] == "add_group"
    assert client._messages[-1]["travelId"] == "solo_travel"

    # without travel: unchanged single-tube fast path (no group)
    client._messages.clear()
    client._binary_messages.clear()
    client.add_toolpath("solo2", tp)
    assert client._messages == []
    assert client._binary_messages[0][0]["type"] == "add_parametric_tube_binary"


def test_add_toolpath_travel_validation_and_noop(client):
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    tp = Toolpath.from_points(pts, bead_width=1.0, bead_height=0.5)
    with pytest.raises(ValueError, match="travel"):
        client.add_toolpath("b", tp, travel="dashed")
    # no travel stretches: travel="line" is a no-op (plain single tube)
    client.add_toolpath("b", tp, travel="line")
    assert client._binary_messages[0][0]["type"] == "add_parametric_tube_binary"
    assert client._messages == []


def test_add_toolpath_group_transform_covers_travel_line(client):
    """Transform kwargs land on the GROUP, not on each child, so the travel
    line (which has no transform args) stays aligned with the beads
    (Copilot review on #94)."""
    pts = np.zeros((8, 3), dtype=np.float32)
    pts[:, 0] = np.arange(8, dtype=np.float32)
    widths = np.array([0.2, 0.2, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0], dtype=np.float32)
    heights = np.where(widths > 0, 0.1, 0.0).astype(np.float32)
    tp = Toolpath.from_points(pts, bead_width=widths, bead_height=heights)
    client.add_toolpath(
        "b", tp, travel="line", position=[1, 2, 3], rotation=[0, 0, 0.5]
    )
    group_msg = client._messages[0]
    assert group_msg["type"] == "add_group"
    assert group_msg["transform"]["position"] == [1, 2, 3]
    assert group_msg["transform"]["rotation"] == [0, 0, 0.5]
    # no child (tube or travel line) carries its own transform
    for header, _ in client._binary_messages:
        assert "position" not in header and "rotation" not in header
        assert "matrix" not in header


# === add_toolpath single-point extrusion runs (issue #103) ===


def _issue_103_toolpath():
    """Exact minimal repro from issue #103: travel, one isolated extruding
    point, travel, then a normal 2-point extrusion run."""
    data = np.array(
        [
            # t   x    y  z  w  h
            [0.0, 0.0, 0, 0, 0, 0],
            [1.0, 10.0, 0, 0, 4, 2],  # isolated extruding point -> 1-point run
            [2.0, 20.0, 0, 0, 0, 0],
            [3.0, 30.0, 0, 0, 4, 2],
            [4.0, 40.0, 0, 0, 4, 2],
        ],
        dtype=np.float32,
    )
    return Toolpath(data)


def test_add_toolpath_single_point_run_skipped(client):
    """Issue #103 repro must not raise: the 1-point run is skipped and only
    the 2-point run renders (single-tube fast path, global bias kept)."""
    tp = _issue_103_toolpath()
    client.add_toolpath("tp", tp)  # raised ValueError before the fix
    assert client._messages == []  # single remaining run -> no group
    assert len(client._binary_messages) == 1
    header, _ = client._binary_messages[0]
    assert header["type"] == "add_parametric_tube_binary"
    assert header["id"] == "tp"
    assert header["numSpinePoints"] == 2
    # bias ramp stays global (spine indices 3..4 of 5)
    assert header["biasIndexOffset"] == 3
    assert header["biasIndexTotal"] == 5


def test_add_toolpath_single_point_run_travel_covers_skipped_point(client):
    """With travel="line" the skipped isolated point stays covered by the
    travel line: both edges flanking it are travel edges, and
    travelEndFracs stay ascending / consistent with segmentRanges."""
    tp = _issue_103_toolpath()
    client.add_toolpath("tp", tp, travel="line")

    assert client._messages[0] == {"type": "add_group", "id": "tp"}
    reg = client._messages[-1]
    assert reg["type"] == "register_toolpath_group"
    assert reg["segmentIds"] == ["tp_seg_0"]
    # only the 2-point run (spine indices 3..4), global-index fractions
    assert len(reg["segmentRanges"]) == 1
    assert reg["segmentRanges"][0] == pytest.approx([3 / 4, 4 / 4])
    assert reg["travelId"] == "tp_travel"
    # travel edges (0,1) (1,2) (2,3): through the isolated point at index 1
    assert reg["travelEndFracs"] == pytest.approx([1 / 4, 2 / 4, 3 / 4])
    assert reg["travelEndFracs"] == sorted(reg["travelEndFracs"])

    tubes = [
        h
        for h, _ in client._binary_messages
        if h["type"] == "add_parametric_tube_binary"
    ]
    assert len(tubes) == 1
    assert tubes[0]["numSpinePoints"] == 2
    travel_headers = [
        h for h, _ in client._binary_messages if h["type"] == "add_polyline_binary"
    ]
    assert len(travel_headers) == 1
    assert travel_headers[0]["numPoints"] == 6  # 3 edges x 2 endpoints


def test_add_toolpath_all_single_point_runs_noop(client):
    """Every extrusion run a single point and no travel line requested:
    nothing drawable -> clean no-op, no messages, no crash."""
    pts = np.zeros((5, 3), dtype=np.float32)
    pts[:, 0] = np.arange(5, dtype=np.float32)
    widths = np.array([0.0, 0.2, 0.0, 0.2, 0.0], dtype=np.float32)
    heights = np.where(widths > 0, 0.1, 0.0).astype(np.float32)
    tp = Toolpath.from_points(pts, bead_width=widths, bead_height=heights)
    client.add_toolpath("tp", tp)
    assert client._messages == []
    assert client._binary_messages == []


def test_add_toolpath_all_single_point_runs_travel_only(client):
    """Every extrusion run a single point + travel="line": degrade to a
    travel-only group (no bead tubes, all edges are travel edges)."""
    pts = np.zeros((5, 3), dtype=np.float32)
    pts[:, 0] = np.arange(5, dtype=np.float32)
    widths = np.array([0.0, 0.2, 0.0, 0.2, 0.0], dtype=np.float32)
    heights = np.where(widths > 0, 0.1, 0.0).astype(np.float32)
    tp = Toolpath.from_points(pts, bead_width=widths, bead_height=heights)
    client.add_toolpath("tp", tp, travel="line")

    assert client._messages[0] == {"type": "add_group", "id": "tp"}
    reg = client._messages[-1]
    assert reg["type"] == "register_toolpath_group"
    assert reg["segmentIds"] == []
    assert reg["segmentRanges"] == []
    assert reg["travelId"] == "tp_travel"
    assert reg["travelEndFracs"] == pytest.approx([1 / 4, 2 / 4, 3 / 4, 4 / 4])

    tubes = [
        h
        for h, _ in client._binary_messages
        if h["type"] == "add_parametric_tube_binary"
    ]
    assert tubes == []
    travel_headers = [
        h for h, _ in client._binary_messages if h["type"] == "add_polyline_binary"
    ]
    assert len(travel_headers) == 1
    assert travel_headers[0]["numPoints"] == 8  # 4 edges x 2 endpoints


# === polyline picking ===


def test_enable_polyline_picking_defaults(client):
    """Default enable stores no decimation (maxPickPoints=0 = exact full scan)."""
    client.enable_polyline_picking()
    msg = client._polyline_picking
    assert msg["type"] == "set_polyline_picking"
    assert msg["enabled"] is True
    assert msg["maxPickPoints"] == 0


def test_enable_polyline_picking_max_pick_points(client):
    """max_pick_points is forwarded as an int maxPickPoints for the coarse-scan
    decimation (issue #111)."""
    client.enable_polyline_picking(
        marker_color=0x112233, threshold_px=8.0, max_pick_points=50_000
    )
    msg = client._polyline_picking
    assert msg["markerColor"] == 0x112233
    assert msg["thresholdPx"] == 8.0
    assert msg["maxPickPoints"] == 50_000
    assert isinstance(msg["maxPickPoints"], int)


@pytest.mark.parametrize("raw", [1, 0, -5])
def test_enable_polyline_picking_max_pick_points_below_two_normalizes_to_zero(
    client, raw
):
    """Values below 2 can't form a coarse segment; normalize to 0 (off) so the
    stored/replayed state matches what the viewer itself does (Copilot review
    on #112)."""
    client.enable_polyline_picking(max_pick_points=raw)
    assert client._polyline_picking["maxPickPoints"] == 0
