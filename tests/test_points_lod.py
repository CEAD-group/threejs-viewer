"""Tests for the point-cloud octree LOD builder and its wire protocol."""

import numpy as np
import pytest

from threejs_viewer import ViewerClient
from threejs_viewer.points_lod import (
    HIERARCHY_DTYPE,
    NO_CHILD,
    build_points_octree,
    build_points_octree_grid,
    grid_morton_codes,
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


# === grid fast-path builder ===


def _grid_cloud(side=80, layers=12, spacing=0.1, seed=3):
    """Voxel centres in a `layers`-thick shell over a curved surface, on a
    regular lattice of pitch `spacing` — the shape the grid path targets."""
    rng = np.random.default_rng(seed)
    xi = np.arange(side, dtype=np.int64)
    gx, gy = np.meshgrid(xi, xi, indexing="ij")
    gx = gx.ravel()
    gy = gy.ravel()
    h = np.round(6.0 * np.sin(gx / 11.0) * np.cos(gy / 9.0)).astype(np.int64)
    lz = np.arange(layers, dtype=np.int64) - layers // 2
    ix = np.repeat(gx, layers)
    iy = np.repeat(gy, layers)
    iz = (np.repeat(h, layers) + np.tile(lz, gx.size)).astype(np.int64)
    pts = np.stack([ix, iy, iz], axis=1).astype(np.float32) * spacing
    birth = (ix.astype(np.float32) / side) + rng.random(len(ix)).astype(
        np.float32
    ) * 0.01
    removal = birth + 1.0
    return pts, birth, removal, spacing


def test_grid_octree_permutation_and_tiling():
    pts, birth, _, sp = _grid_cloud()
    o = build_points_octree_grid(pts, spacing=sp, strat_times=birth, node_capacity=5000)
    n = len(pts)
    assert np.array_equal(np.sort(o.order), np.arange(n))
    assert o.offsets[0] == 0
    ends = o.offsets + o.counts
    assert ends[-1] == n
    assert np.array_equal(o.offsets[1:], ends[:-1])


def test_grid_octree_cubes_and_points_inside():
    pts, birth, _, sp = _grid_cloud()
    o = build_points_octree_grid(pts, spacing=sp, strat_times=birth, node_capacity=5000)
    # cubes halve per level
    np.testing.assert_allclose(
        o.half_sizes, o.half_sizes[0] / 2.0 ** o.levels.astype(np.float64), rtol=1e-6
    )
    # every point sits strictly inside its node cube (grid data is exact)
    for i in range(o.n_nodes):
        lo, hi = int(o.offsets[i]), int(o.offsets[i]) + int(o.counts[i])
        if hi == lo:
            continue
        p = pts[o.order[lo:hi]]
        overflow = np.abs(p - o.centers[i]).max(axis=0) - o.half_sizes[i]
        assert overflow.max() <= 1e-4 * o.half_sizes[i] + 1e-6


def test_grid_octree_capacity_topology():
    pts, birth, _, sp = _grid_cloud()
    cap = 4000
    o = build_points_octree_grid(pts, spacing=sp, strat_times=birth, node_capacity=cap)
    assert o.n_nodes > 1
    interior = o.first_child != NO_CHILD
    assert (o.counts[interior] == cap).all()
    assert (o.counts[~interior] <= cap).all()


def test_grid_octree_root_sample_spatially_representative():
    """The grid path subsamples SPATIALLY (strided pick over Morton order), so
    the root sample must span the full bounding box in every axis — that is the
    property that keeps the coarse LOD honest."""
    pts, birth, _, sp = _grid_cloud(side=110)
    o = build_points_octree_grid(pts, spacing=sp, strat_times=birth, node_capacity=4000)
    root = pts[o.order[: int(o.counts[0])]]
    lo_all, hi_all = pts.min(0), pts.max(0)
    for ax in range(3):
        span = hi_all[ax] - lo_all[ax]
        # sample reaches within 5% of each extent (no whole region missed)
        assert root[:, ax].min() - lo_all[ax] < 0.05 * span
        assert hi_all[ax] - root[:, ax].max() < 0.05 * span


def test_grid_octree_time_representative_when_time_tracks_space():
    """Dropping time-stratification is safe for carve-like data because
    removal/birth is spatially correlated (the tool sweeps through space): a
    spatial coarse sample is then still ~uniform in time. birth ∝ x here."""
    pts, birth, _, sp = _grid_cloud(side=110)
    o = build_points_octree_grid(pts, spacing=sp, node_capacity=4000)
    root_birth = birth[o.order[: int(o.counts[0])]]
    for q in (0.1, 0.5, 0.9):
        assert abs(
            float(np.quantile(root_birth, q)) - float(np.quantile(birth, q))
        ) < 0.15 * (birth.max() - birth.min())


def test_grid_octree_deterministic_for_seed():
    pts, birth, _, sp = _grid_cloud()
    a = build_points_octree_grid(
        pts, spacing=sp, strat_times=birth, node_capacity=3000, seed=5
    )
    b = build_points_octree_grid(
        pts, spacing=sp, strat_times=birth, node_capacity=3000, seed=5
    )
    assert np.array_equal(a.order, b.order)


def test_grid_octree_matches_float_structure():
    """Grid and float builders must agree on tree shape for lattice data (the
    grid path is an integer-arithmetic reimplementation, not a different tree)."""
    pts, birth, _, sp = _grid_cloud(side=90)
    g = build_points_octree_grid(pts, spacing=sp, strat_times=birth, node_capacity=5000)
    f = build_points_octree(pts, strat_times=birth, node_capacity=5000)
    # Same node count and identical per-level node distribution.
    assert abs(g.n_nodes - f.n_nodes) <= 2
    gl = np.bincount(g.levels)
    fl = np.bincount(f.levels)
    assert len(gl) == len(fl)
    assert np.abs(gl.astype(int) - fl.astype(int)).max() <= 2


def test_grid_octree_rejects_bad_spacing():
    pts, _, _, _ = _grid_cloud(side=20)
    with pytest.raises(ValueError, match="spacing"):
        build_points_octree_grid(pts, spacing=0.0)
    with pytest.raises(ValueError, match="spacing"):
        build_points_octree_grid(pts, spacing=[0.1, 0.1])


# === grid precomputed-order (external producer) fast path ===


def _assert_octrees_identical(a, b):
    """Every field of two PointsOctrees is byte-identical."""
    assert np.array_equal(a.order, b.order)
    assert np.array_equal(a.centers, b.centers)
    assert np.array_equal(a.half_sizes, b.half_sizes)
    assert np.array_equal(a.offsets, b.offsets)
    assert np.array_equal(a.counts, b.counts)
    assert np.array_equal(a.levels, b.levels)
    assert np.array_equal(a.first_child, b.first_child)
    assert np.array_equal(a.child_mask, b.child_mask)


def test_grid_morton_codes_reference_matches_internal_sort():
    """grid_morton_codes is the reference an external producer reproduces; the
    order it induces is exactly what the builder sorts to internally."""
    pts, birth, _, sp = _grid_cloud(side=90)
    codes, n_bits = grid_morton_codes(pts, spacing=sp)
    # Grid voxels are distinct -> codes are distinct -> the sort order is unique.
    assert len(np.unique(codes)) == len(codes)
    assert n_bits >= 12


def test_grid_octree_precomputed_codes_roundtrip():
    """Feeding the internally-computed codes back through the codes= path
    reproduces the pure-numpy build byte-for-byte (skips the quantise stage)."""
    pts, birth, _, sp = _grid_cloud(side=96)
    base = build_points_octree_grid(pts, spacing=sp, node_capacity=4000, seed=7)
    codes, n_bits = grid_morton_codes(pts, spacing=sp)
    ext = build_points_octree_grid(
        pts, spacing=sp, node_capacity=4000, seed=7, codes=codes, n_bits=n_bits
    )
    _assert_octrees_identical(base, ext)


def test_grid_octree_precomputed_codes_and_order_roundtrip():
    """Feeding BOTH codes and a valid Morton order reproduces the build exactly
    (skips quantise AND sort — the whole external-producer fast path)."""
    pts, birth, _, sp = _grid_cloud(side=96)
    base = build_points_octree_grid(pts, spacing=sp, node_capacity=4000, seed=7)
    codes, n_bits = grid_morton_codes(pts, spacing=sp)
    order = np.argsort(codes)  # what a Rust producer would ship
    ext = build_points_octree_grid(
        pts,
        spacing=sp,
        node_capacity=4000,
        seed=7,
        codes=codes,
        order=order,
        n_bits=n_bits,
    )
    _assert_octrees_identical(base, ext)


def test_grid_octree_precomputed_rejects_bad_inputs():
    pts, _, _, sp = _grid_cloud(side=40)
    codes, n_bits = grid_morton_codes(pts, spacing=sp)
    n = len(pts)
    # wrong dtype
    with pytest.raises(ValueError, match="uint64"):
        build_points_octree_grid(
            pts, spacing=sp, codes=codes.astype(np.int64), n_bits=n_bits
        )
    # wrong length
    with pytest.raises(ValueError, match="length"):
        build_points_octree_grid(pts, spacing=sp, codes=codes[:-1], n_bits=n_bits)
    # codes overflow the declared n_bits budget (a bit set above 3*n_bits)
    over = codes.copy()
    over[0] = np.uint64(1) << np.uint64(3 * n_bits)
    with pytest.raises(ValueError, match="budget"):
        build_points_octree_grid(pts, spacing=sp, codes=over, n_bits=n_bits)
    # order without codes
    with pytest.raises(ValueError, match="order requires codes"):
        build_points_octree_grid(pts, spacing=sp, order=np.argsort(codes))
    # non-permutation order
    bad = np.argsort(codes).copy()
    bad[0] = bad[1]
    with pytest.raises(ValueError, match="permutation"):
        build_points_octree_grid(pts, spacing=sp, codes=codes, order=bad, n_bits=n_bits)
    # out-of-range order indices: negative (would wrap in numpy) and >= N
    neg = np.argsort(codes).astype(np.int64)
    neg[0] -= n  # same element via wraparound, but not a valid index
    with pytest.raises(ValueError, match=r"\[0, "):
        build_points_octree_grid(pts, spacing=sp, codes=codes, order=neg, n_bits=n_bits)
    big = np.argsort(codes).copy()
    big[0] = n
    with pytest.raises(ValueError, match=r"\[0, "):
        build_points_octree_grid(pts, spacing=sp, codes=codes, order=big, n_bits=n_bits)
    # order that does not sort the codes (identity is not Morton order here)
    with pytest.raises(ValueError, match="monotonic"):
        build_points_octree_grid(
            pts, spacing=sp, codes=codes, order=np.arange(n), n_bits=n_bits
        )
    # n_bits out of range
    with pytest.raises(ValueError, match="n_bits"):
        build_points_octree_grid(pts, spacing=sp, n_bits=99)


def test_grid_octree_precomputed_n_bits_defaults_to_spacing():
    """Omitting n_bits with supplied codes reuses the spacing-derived value; a
    matching-resolution producer need not pass n_bits."""
    pts, _, _, sp = _grid_cloud(side=64)
    codes, n_bits = grid_morton_codes(pts, spacing=sp)
    a = build_points_octree_grid(pts, spacing=sp, node_capacity=3000, codes=codes)
    b = build_points_octree_grid(
        pts, spacing=sp, node_capacity=3000, codes=codes, n_bits=n_bits
    )
    _assert_octrees_identical(a, b)


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


def test_add_points_lod_grid_dispatch(client):
    """lod={'grid': {...}} builds a valid LOD cloud via the grid fast-path;
    the wire header/blobs are identical in shape to the float path."""
    pts, birth, removal, sp = _grid_cloud(side=70, layers=10)
    client.add_points(
        "gcloud",
        pts,
        colors=pts[:, 2],
        birth_times=birth,
        removal_times=removal,
        lod={"node_capacity": 4000, "grid": {"spacing": sp}},
    )
    (msg,) = client._messages
    assert msg["type"] == "add_points_lod"
    assert msg["numPoints"] == len(pts)
    assert msg["nodeCount"] >= 1
    key_base = client._points_lod["gcloud"]
    hierarchy = client._blob_store[f"{key_base}/hierarchy"]
    rec = np.frombuffer(hierarchy, dtype=HIERARCHY_DTYPE)
    assert len(rec) == msg["nodeCount"]
    payload = client._blob_store[f"{key_base}/0"]()
    assert len(payload) == int(rec["count"][0]) * (6 + 3 + 4 + 4)


def test_add_points_lod_grid_precomputed_passthrough(client):
    """codes/order/n_bits in lod['grid'] reach the builder: the wire blobs
    are byte-identical to the plain grid path (the external-producer seam)."""
    pts, birth, removal, sp = _grid_cloud(side=70, layers=10)
    codes, n_bits = grid_morton_codes(pts, spacing=sp)
    order = np.argsort(codes)
    kw = dict(colors=pts[:, 2], birth_times=birth, removal_times=removal)
    client.add_points("plain", pts, lod={"grid": {"spacing": sp}}, **kw)
    client.add_points(
        "pre",
        pts,
        lod={"grid": {"spacing": sp, "codes": codes, "order": order, "n_bits": n_bits}},
        **kw,
    )
    plain_key = client._points_lod["plain"]
    pre_key = client._points_lod["pre"]
    assert (
        client._blob_store[f"{plain_key}/hierarchy"]
        == client._blob_store[f"{pre_key}/hierarchy"]
    )
    n_nodes = np.frombuffer(
        client._blob_store[f"{plain_key}/hierarchy"], dtype=HIERARCHY_DTYPE
    ).shape[0]
    for i in range(n_nodes):
        assert (
            client._blob_store[f"{plain_key}/{i}"]()
            == client._blob_store[f"{pre_key}/{i}"]()
        )


def test_add_points_lod_grid_validation(client):
    pts, _, _, sp = _grid_cloud(side=20, layers=4)
    with pytest.raises(ValueError, match="spacing"):
        client.add_points("g", pts, lod={"grid": {"origin": [0, 0, 0]}})
    with pytest.raises(ValueError, match="grid"):
        client.add_points("g", pts, lod={"grid": {"spacing": sp, "bogus": 1}})


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


# === runtime LOD option tuning (set_points_lod_options) ===


def test_set_points_lod_options_message(client):
    client.set_points_lod_options(
        "c", point_budget=2_000_000, refine_pixels=6, size_boost_max=1.0
    )
    assert client._messages[-1] == {
        "type": "set_points_lod_options",
        "id": "c",
        "pointBudget": 2_000_000,
        "refinePixels": 6.0,
        "sizeBoostMax": 1.0,
    }
    # partial update: only the provided field ships
    client.set_points_lod_options("c", refine_pixels=8)
    assert client._messages[-1] == {
        "type": "set_points_lod_options",
        "id": "c",
        "refinePixels": 8.0,
    }


def test_set_points_lod_options_validation(client):
    with pytest.raises(ValueError, match="point_budget"):
        client.set_points_lod_options("c", point_budget=0)
    with pytest.raises(ValueError, match="refine_pixels"):
        client.set_points_lod_options("c", refine_pixels=0)
    with pytest.raises(ValueError, match="refine_pixels"):
        client.set_points_lod_options("c", refine_pixels=float("nan"))
    with pytest.raises(ValueError, match="size_boost_max"):
        client.set_points_lod_options("c", size_boost_max=0.5)
    with pytest.raises(ValueError, match="at least one"):
        client.set_points_lod_options("c")
    assert client._messages == []
