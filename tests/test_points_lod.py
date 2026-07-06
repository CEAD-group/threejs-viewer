"""Tests for the point-cloud octree LOD builder and its wire protocol."""

import numpy as np
import pytest

from threejs_viewer import ViewerClient
from threejs_viewer.points_lod import (
    HIERARCHY_DTYPE,
    NO_CHILD,
    build_points_octree,
    pack_node_payload,
)


@pytest.fixture
def client():
    """ViewerClient with mocked _send/_send_binary (blob store stays real)."""
    c = ViewerClient()
    c._messages = []
    c._binary_messages = []
    c._send = lambda data: c._messages.append(data)
    c._send_binary = lambda header, payload: c._binary_messages.append(
        (header, payload)
    )
    return c


def _cloud(n=60_000, seed=7):
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 3)).astype(np.float32) * [8.0, 3.0, 1.5]
    birth = pts[:, 0] + rng.random(n).astype(np.float32)
    removal = birth + 1.0
    return pts, birth, removal


# === builder invariants ===


def test_octree_permutation_and_tiling():
    pts, birth, _ = _cloud()
    o = build_points_octree(pts, strat_times=birth, node_capacity=5000)
    n = len(pts)
    # order is a permutation of all points
    assert np.array_equal(np.sort(o.order), np.arange(n))
    # node ranges tile [0, N) exactly, in slot order
    assert o.offsets[0] == 0
    ends = o.offsets + o.counts
    assert ends[-1] == n
    assert np.array_equal(o.offsets[1:], ends[:-1])


def test_octree_points_inside_node_cubes():
    pts, birth, _ = _cloud()
    o = build_points_octree(pts, strat_times=birth, node_capacity=5000)
    # Cubes must actually halve per level — a node's cube is root_half /
    # 2^level (regression: children were once allocated at the parent's
    # size, which inflated projected sizes and blew the tree depth).
    np.testing.assert_allclose(
        o.half_sizes, o.half_sizes[0] / 2.0 ** o.levels.astype(np.float64), rtol=1e-6
    )
    pos_r = pts[o.order]
    for i in range(o.n_nodes):
        lo, hi = int(o.offsets[i]), int(o.offsets[i]) + int(o.counts[i])
        if hi == lo:
            continue
        overflow = np.abs(pos_r[lo:hi] - o.centers[i]) - o.half_sizes[i]
        assert overflow.max() <= 1e-4 * o.half_sizes[i] + 1e-6
    # And with correct cube sizes the tree stays shallow for a
    # near-uniform cloud (depth explosion is the bug signature).
    assert o.max_level <= 6


def test_octree_capacity_and_topology():
    pts, birth, _ = _cloud()
    cap = 5000
    o = build_points_octree(pts, strat_times=birth, node_capacity=cap)
    assert o.n_nodes > 1
    # interior nodes hold exactly a capacity-sized sample; leaves at most
    interior = o.first_child != NO_CHILD
    assert (o.counts[interior] == cap).all()
    assert (o.counts[~interior] <= cap).all()
    # children are consecutive slots one level deeper, count = popcount(mask)
    for i in np.flatnonzero(interior):
        k = bin(int(o.child_mask[i])).count("1")
        assert k >= 1
        fc = int(o.first_child[i])
        assert fc + k <= o.n_nodes
        assert (o.levels[fc : fc + k] == o.levels[i] + 1).all()
    # every non-root node is someone's child, exactly once
    child_slots = []
    for i in np.flatnonzero(interior):
        k = bin(int(o.child_mask[i])).count("1")
        child_slots.extend(range(int(o.first_child[i]), int(o.first_child[i]) + k))
    assert sorted(child_slots) == list(range(1, o.n_nodes))


def test_octree_time_stratified_root_sample():
    """The root sample must span the full time range proportionally — the
    property that keeps time-filtered density honest at coarse LOD."""
    pts, birth, _ = _cloud(n=80_000)
    o = build_points_octree(pts, strat_times=birth, node_capacity=4000)
    root_birth = birth[o.order[: int(o.counts[0])]]
    for q in (0.1, 0.5, 0.9):
        assert abs(
            float(np.quantile(root_birth, q)) - float(np.quantile(birth, q))
        ) < 0.15 * (birth.max() - birth.min())


def test_octree_small_cloud_single_node():
    pts = np.random.default_rng(0).random((100, 3)).astype(np.float32)
    o = build_points_octree(pts, node_capacity=1000)
    assert o.n_nodes == 1
    assert o.counts[0] == 100
    assert o.first_child[0] == NO_CHILD


def test_octree_deterministic_for_seed():
    pts, birth, _ = _cloud(n=20_000)
    a = build_points_octree(pts, strat_times=birth, node_capacity=3000, seed=5)
    b = build_points_octree(pts, strat_times=birth, node_capacity=3000, seed=5)
    assert np.array_equal(a.order, b.order)


def test_octree_rejects_empty_and_bad_capacity():
    with pytest.raises(ValueError, match="empty"):
        build_points_octree(np.zeros((0, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="node_capacity"):
        build_points_octree(np.zeros((10, 3), dtype=np.float32), node_capacity=0)


# === payload + hierarchy packing ===


def test_pack_node_payload_layout_and_quantization():
    pts, birth, removal = _cloud(n=500)
    o = build_points_octree(pts, node_capacity=1000)
    center, half = o.centers[0], float(o.half_sizes[0])
    colors = np.zeros((500, 3), dtype=np.uint8)
    blob = pack_node_payload(pts, colors, birth, removal, center, half)
    assert len(blob) == 500 * (6 + 3 + 4 + 4)
    # int16 positions dequantize back within one quantum of the original
    q = np.frombuffer(blob[: 500 * 6], dtype="<i2").reshape(500, 3)
    dequant = center + (q / 32767.0) * half
    assert np.abs(dequant - pts).max() <= half / 32767.0 + 1e-6


def test_pack_hierarchy_record_layout_and_time_bounds():
    pts, birth, removal = _cloud(n=30_000)
    o = build_points_octree(pts, strat_times=birth, node_capacity=4000)
    birth_r, removal_r = birth[o.order], removal[o.order]
    blob = o.pack_hierarchy(birth_r, removal_r)
    rec = np.frombuffer(blob, dtype=HIERARCHY_DTYPE)
    assert len(rec) == o.n_nodes
    assert np.array_equal(rec["offset"], o.offsets)
    assert np.array_equal(rec["count"], o.counts)
    # per-node time bounds match the node's own slice
    for i in range(o.n_nodes):
        lo, hi = int(o.offsets[i]), int(o.offsets[i]) + int(o.counts[i])
        assert rec["tmin"][i] == pytest.approx(birth_r[lo:hi].min(), rel=1e-6)
        assert rec["tmax"][i] == pytest.approx(removal_r[lo:hi].max(), rel=1e-6)


# === protocol / client integration ===


def test_add_points_lod_header_and_blobs(client):
    pts, birth, removal = _cloud(n=30_000)
    client.add_points(
        "cloud",
        pts,
        colors=pts[:, 2],
        birth_times=birth,
        removal_times=removal,
        lod={"node_capacity": 4000, "point_budget": 100_000, "refine_pixels": 8},
    )
    assert client._binary_messages == []  # LOD path does not use _send_binary
    (msg,) = client._messages
    assert msg["type"] == "add_points_lod"
    assert msg["numPoints"] == 30_000
    assert msg["pointBudget"] == 100_000
    assert msg["refinePixels"] == 8.0
    assert msg["hasVertexColors"] is True
    assert msg["hasBirthTimes"] is True
    assert msg["hasRemovalTimes"] is True
    assert msg["nodeCount"] >= 1
    # hierarchy blob registered and sized nodeCount * 40
    key_base = client._points_lod["cloud"]
    hierarchy = client._blob_store[f"{key_base}/hierarchy"]
    assert len(hierarchy) == msg["nodeCount"] * HIERARCHY_DTYPE.itemsize
    assert msg["hierarchy_url"].endswith(f"{key_base}/hierarchy")
    assert msg["node_url_base"].endswith(f"{key_base}/")
    # node providers are lazy callables producing the advertised layout
    rec = np.frombuffer(hierarchy, dtype=HIERARCHY_DTYPE)
    for i in (0, msg["nodeCount"] - 1):
        payload = client._blob_store[f"{key_base}/{i}"]()
        assert len(payload) == int(rec["count"][i]) * (6 + 3 + 4 + 4)


def test_add_points_lod_replaced_and_released(client):
    pts, _, _ = _cloud(n=5_000)
    client.add_points("cloud", pts, lod=True)
    first_base = client._points_lod["cloud"]
    client.add_points("cloud", pts, lod=True)
    second_base = client._points_lod["cloud"]
    assert first_base != second_base
    assert not any(k.startswith(first_base) for k in client._blob_store)
    client.delete("cloud")
    assert "cloud" not in client._points_lod
    assert not any(k.startswith(second_base) for k in client._blob_store)


def test_add_points_lod_released_on_clear(client):
    pts, _, _ = _cloud(n=5_000)
    client.add_points("cloud", pts, lod=True)
    base = client._points_lod["cloud"]
    client.clear()
    assert client._points_lod == {}
    assert not any(k.startswith(base) for k in client._blob_store)


def test_add_points_lod_rejects_unknown_options(client):
    pts, _, _ = _cloud(n=1_000)
    with pytest.raises(ValueError, match="Unknown lod option"):
        client.add_points("cloud", pts, lod={"epsilon_divisor": 5})


def test_add_points_lod_rejects_budget_below_capacity(client):
    pts, _, _ = _cloud(n=1_000)
    with pytest.raises(ValueError, match="point_budget"):
        client.add_points("cloud", pts, lod={"node_capacity": 1000, "point_budget": 10})


def test_add_points_lod_flat_color_no_times(client):
    pts, _, _ = _cloud(n=5_000)
    client.add_points("cloud", pts, color=0x3366FF, lod={"node_capacity": 2000})
    (msg,) = client._messages
    assert msg["hasVertexColors"] is False
    assert "hasBirthTimes" not in msg
    assert "hasRemovalTimes" not in msg
    key_base = client._points_lod["cloud"]
    rec = np.frombuffer(
        client._blob_store[f"{key_base}/hierarchy"], dtype=HIERARCHY_DTYPE
    )
    payload = client._blob_store[f"{key_base}/0"]()
    assert len(payload) == int(rec["count"][0]) * 6  # positions only


def test_add_points_lod_empty_dict_enables_lod(client):
    """lod={} means "LOD with all defaults" — it must not fall through to
    the flat path via dict truthiness (Copilot review, PR #80)."""
    pts, _, _ = _cloud(n=2_000)
    client.add_points("cloud", pts, lod={})
    (msg,) = client._messages
    assert msg["type"] == "add_points_lod"
    assert client._binary_messages == []
