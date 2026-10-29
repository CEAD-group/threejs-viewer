"""add_usda and the usd.py array helpers (no usd_brep needed)."""

import numpy as np
import pytest

from threejs_viewer import ViewerClient, usd
from threejs_viewer.usd import UsdBody, edge_segments, triangle_corners


@pytest.fixture
def client():
    """Create a ViewerClient with mocked _send and _send_binary."""
    c = ViewerClient()
    c._messages = []
    c._binary_messages = []
    c._send = c._messages.append
    c._send_binary = lambda header, payload: c._binary_messages.append(
        (header, payload)
    )
    return c


def _body(edges: bool = True) -> UsdBody:
    return UsdBody(
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 2, 3]], dtype=np.float32),
        normals=np.array([[0, 1, 0]] * 3, dtype=np.float32),
        indices=np.arange(3, dtype=np.uint32),
        edges=np.array(
            [[0, 0, 0], [1, 0, 0]] if edges else [], dtype=np.float32
        ).reshape(-1, 3),
    )


@pytest.fixture
def bodies(monkeypatch):
    """Stub the tessellation; the returned list is what add_usda will load."""
    loaded = [_body()]
    monkeypatch.setattr(usd, "load_bodies", lambda source, **kw: list(loaded))
    return loaded


def _ids(client):
    return [header["id"] for header, _ in client._binary_messages]


def test_triangle_corners_all_triangles():
    vertices, corners = triangle_corners([3, 0, 1, 2, 3, 2, 1, 3])
    assert vertices.tolist() == [0, 1, 2, 2, 1, 3]
    assert corners.tolist() == [0, 1, 2, 3, 4, 5]


def test_triangle_corners_fans_polygons():
    vertices, corners = triangle_corners([4, 5, 6, 7, 8, 3, 0, 1, 2])
    assert vertices.tolist() == [5, 6, 7, 5, 7, 8, 0, 1, 2]
    assert corners.tolist() == [0, 1, 2, 0, 2, 3, 4, 5, 6]


def test_edge_segments_splits_per_edge():
    points = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (9, 9, 9), (5, 0, 0), (5, 1, 0)]
    segments = edge_segments(points, [3, 1, 2])  # the 1-point edge is degenerate
    assert segments[:, 0].tolist() == [0, 1, 1, 2, 5, 5]
    assert edge_segments([], []).shape == (0, 3)


def test_add_usda_sends_group_mesh_and_edges(client, bodies):
    assert client.add_usda("part.usda") == "part"
    assert client._messages == [{"type": "add_group", "id": "part"}]
    mesh, edges = (header for header, _ in client._binary_messages)
    assert mesh["type"] == "add_mesh_binary"
    assert (mesh["id"], mesh["parent"]) == ("part/body0", "part")
    assert mesh["hasNormals"] and mesh["polygonOffset"]
    assert edges["type"] == "add_polyline_binary"
    assert (edges["id"], edges["parent"]) == ("part/body0/edges", "part")
    assert edges["segments"] is True


def test_add_usda_default_id_for_objects(client, bodies):
    assert client.add_usda(object()) == "usd"


def test_add_usda_without_edges(client, bodies):
    client.add_usda("part.usda", edges=False)
    assert _ids(client) == ["part/body0"]
    assert "polygonOffset" not in client._binary_messages[0][0]


def test_add_usda_reload_keeps_group_and_deletes_stale(client, bodies):
    bodies.append(_body())
    client.add_usda("part.usda")
    client._messages.clear()
    client._binary_messages.clear()

    del bodies[1]
    bodies[0] = _body(edges=False)
    client.add_usda("part.usda")
    assert _ids(client) == ["part/body0"]
    deleted = {m["id"] for m in client._messages if m["type"] == "delete_object"}
    assert deleted == {"part/body0/edges", "part/body1", "part/body1/edges"}
    assert not any(m["type"] == "add_group" for m in client._messages)


@pytest.mark.parametrize("forget", ["delete", "clear"])
def test_add_usda_resends_group_after_delete_or_clear(client, bodies, forget):
    client.add_usda("part.usda")
    client.delete("part") if forget == "delete" else client.clear()
    client._messages.clear()
    client.add_usda("part.usda")
    assert {"type": "add_group", "id": "part"} in client._messages


def test_add_usda_y_up_converts_to_z_up(client, bodies):
    client.add_usda("part.usda", y_up=True)
    _, payload = client._binary_messages[0]
    positions = np.frombuffer(payload[:36], dtype=np.float32).reshape(3, 3)
    normals = np.frombuffer(payload[36:72], dtype=np.float32).reshape(3, 3)
    assert positions[2].tolist() == [0, -3, 2]
    assert normals[0].tolist() == [0, 0, 1]


def test_load_bodies_without_usd_brep(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "usd_brep", None)
    with pytest.raises(ImportError, match="usd_brep"):
        usd.load_bodies("part.usda")
