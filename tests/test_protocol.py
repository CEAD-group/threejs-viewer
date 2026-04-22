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
