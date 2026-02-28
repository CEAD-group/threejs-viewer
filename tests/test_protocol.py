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


# === Toolpath.to_mesh + add_mesh ===


def test_to_mesh_add_mesh_with_parent(client):
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    tp = Toolpath.from_points(pts, bead_width=1.0, bead_height=0.5)
    client.add_mesh("bd", parent="g", **tp.to_mesh())
    header, _ = client._binary_messages[0]
    assert header["type"] == "add_mesh_binary"
    assert header["parent"] == "g"


def test_to_mesh_add_mesh_no_parent(client):
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    tp = Toolpath.from_points(pts, bead_width=1.0, bead_height=0.5)
    client.add_mesh("bd", **tp.to_mesh())
    header, _ = client._binary_messages[0]
    assert "parent" not in header


def test_to_mesh_colorize_sets_vertex_colors(client):
    """colorize() → to_mesh() → add_mesh sends hasVertexColors=True."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    tp = Toolpath.from_points(pts, bead_width=1.0, bead_height=0.5)
    tp.colorize("plasma")
    client.add_mesh("bd", **tp.to_mesh())
    header, _ = client._binary_messages[0]
    assert header["hasVertexColors"] is True


def test_to_mesh_no_colorize_no_vertex_colors(client):
    """Without colorize(), to_mesh() → add_mesh sends hasVertexColors=False."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    tp = Toolpath.from_points(pts, bead_width=1.0, bead_height=0.5)
    client.add_mesh("bd", **tp.to_mesh())
    header, _ = client._binary_messages[0]
    assert header["hasVertexColors"] is False


def _extract_bead_indices(header, payload):
    """Extract the index buffer from an add_mesh_binary binary message."""
    nv = header["numVertices"]
    ni = header["numIndices"]
    offset = nv * 3 * 4  # positions
    if header["hasNormals"]:
        offset += nv * 3 * 4
    if header["hasVertexColors"]:
        offset += nv * 3 * 4
    return np.frombuffer(payload[offset : offset + ni * 4], dtype=np.uint32).copy()


def test_add_bead_scalar_all_segments_active(client):
    """Scalar width/height: every segment should have non-zero triangle indices."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
    tp = Toolpath.from_points(pts, bead_width=0.2, bead_height=0.1)
    client.add_mesh("b", **tp.to_mesh())
    header, payload = client._binary_messages[0]
    indices = _extract_bead_indices(header, payload)

    P = 6  # vertices per ring
    n_segs = len(pts) - 1  # 3 segments
    # Each segment contributes P * 6 indices; none should be all-zero for a real mesh
    for seg in range(n_segs):
        seg_indices = indices[seg * P * 6 : (seg + 1) * P * 6]
        assert np.any(seg_indices != 0), f"Segment {seg} should have active geometry"


def test_add_bead_array_width_travel_degenerate(client):
    """Array width with zeros for travel: zero-width rings collapse to a point.

    Travel move geometry is handled purely by vertex positions — W=H=0 collapses
    all 6 ring vertices to the same point, producing zero-area triangles that GPUs
    discard. No explicit index zeroing is needed or done.
    """
    # 6 points: [ext, ext, travel, travel, ext, ext]
    pts = np.zeros((6, 3), dtype=np.float32)
    pts[:, 0] = np.arange(6, dtype=np.float32)
    widths = np.array([0.2, 0.2, 0.0, 0.0, 0.2, 0.2], dtype=np.float32)
    heights = np.array([0.1, 0.1, 0.0, 0.0, 0.1, 0.1], dtype=np.float32)
    tp = Toolpath.from_points(pts, bead_width=widths, bead_height=heights)
    client.add_mesh("b", **tp.to_mesh())

    header, payload = client._binary_messages[0]

    # Extract vertex positions to verify collapsed rings (positions are first in payload)
    nv = header["numVertices"]
    positions = np.frombuffer(payload[: nv * 3 * 4], dtype=np.float32).reshape(-1, 3)

    P = 6
    # Travel rings (points 2 and 3): all 6 vertices must collapse to the path point
    for ring_idx in [2, 3]:
        ring_verts = positions[ring_idx * P : (ring_idx + 1) * P]
        assert np.allclose(ring_verts, ring_verts[0], atol=1e-6), (
            f"Ring {ring_idx} (travel, W=H=0) should have all vertices at same point"
        )
    # Extruding rings (points 0, 1, 4, 5): vertices must be spread (non-collapsed)
    for ring_idx in [0, 1, 4, 5]:
        ring_verts = positions[ring_idx * P : (ring_idx + 1) * P]
        assert not np.allclose(ring_verts, ring_verts[0], atol=1e-6), (
            f"Ring {ring_idx} (extruding) should have spread vertices"
        )


def test_add_bead_array_width_index_count_unchanged(client):
    """Array vs scalar width: same number of indices (draw_range mapping preserved)."""
    pts = np.zeros((5, 3), dtype=np.float32)
    pts[:, 0] = np.arange(5, dtype=np.float32)

    tp_scalar = Toolpath.from_points(pts, bead_width=0.2, bead_height=0.1)
    client.add_mesh("scalar", **tp_scalar.to_mesh())
    h_scalar, _ = client._binary_messages[0]

    widths = np.array([0.2, 0.0, 0.0, 0.2, 0.2], dtype=np.float32)
    tp_array = Toolpath.from_points(pts, bead_width=widths, bead_height=0.1)
    client.add_mesh("array", **tp_array.to_mesh())
    h_array, _ = client._binary_messages[1]

    assert h_scalar["numIndices"] == h_array["numIndices"], (
        "Index count must be identical so draw_range fractions map the same way"
    )


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
    assert result == {}
