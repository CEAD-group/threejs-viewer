"""Tests for the wire protocol — verifies JSON payloads sent by ViewerClient."""

import uuid

import numpy as np
import pytest

from threejs_viewer import ViewerClient


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
    client.add_group("g1", position=[1, 2, 3], rotation=[0.1, 0.2, 0.3], scale=[2, 2, 2])
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


# === add_bead ===


def test_add_bead_with_parent(client):
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    client.add_bead("bd", pts, width=1.0, height=0.5, parent="g")
    # add_bead delegates to add_mesh, which calls _send_binary
    header, _ = client._binary_messages[0]
    assert header["type"] == "add_mesh_binary"
    assert header["parent"] == "g"


def test_add_bead_no_parent(client):
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    client.add_bead("bd", pts, width=1.0, height=0.5)
    header, _ = client._binary_messages[0]
    assert "parent" not in header


# === query_scene ===


def test_query_scene_sends_correct_message(client):
    """Verify query_scene sends the right request type with a requestId."""
    # Patch _send to also simulate an immediate response so query_scene doesn't block
    original_send = client._send
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
    assert result == {}
