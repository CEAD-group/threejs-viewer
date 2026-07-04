"""Octree LOD builder for point clouds (issue #79, plans/points-octree-lod.md).

Potree-style additive sampled octree, built provider-side in numpy:

- Every node owns a *sample* of the points inside its cube; the root is a
  coarse sample of the whole cloud, children add detail (never replace it),
  and no point is stored twice. Every rendered point at every LOD is a real
  point carrying its true attributes — the per-point time window (birth /
  removal) stays exactly correct at all distances, which is why sampled
  real points were chosen over aggregated voxel rollups.
- Sampling is pseudo-random (seeded) and **time-stratified**: candidates
  are ordered by time with a random tiebreak and picked at even strides, so
  a time filter thins every LOD level proportionally instead of punching
  patchy holes (the one genuinely novel requirement — no prior art covers
  LOD density under a per-point lifetime filter).
- Within each node the sample is Morton-sorted then shuffled in batches of
  ``VERTEX_BATCH`` points ("shuffled Morton", Schütz 2021): raster-order
  locality inside a batch, spatially spread batches — up to ~4x raw
  GL_POINTS throughput and no pathological slow viewpoints.

The build reorders the cloud into one permutation (``order``) where each
node's sample is a contiguous ``[offset, offset+count)`` range, so the
serving side can slice zero-copy views per node request.

Node payloads are quantized per node: positions as int16 over the node
cube (dequantized for free by the GPU via normalized attributes + the node
object's center/scale transform), colors u8, times f32 (u16 quantization
deferred: an "unbounded" sentinel needs an immortal flag the shader can
read — see the plan doc).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Hierarchy wire record, 40 bytes/node, little-endian. Nodes are emitted in
# BFS order so a node's existing children occupy consecutive slots starting
# at first_child (bit k of child_mask set => octant k present, in ascending
# octant order).
HIERARCHY_DTYPE = np.dtype(
    [
        ("cx", "<f4"),
        ("cy", "<f4"),
        ("cz", "<f4"),
        ("half", "<f4"),
        ("offset", "<u4"),
        ("count", "<u4"),
        ("tmin", "<f4"),
        ("tmax", "<f4"),
        ("first_child", "<u4"),
        ("child_mask", "u1"),
        ("level", "u1"),
        ("_pad", "<u2"),
    ]
)
assert HIERARCHY_DTYPE.itemsize == 40

NO_CHILD = 0xFFFFFFFF
VERTEX_BATCH = 128  # shuffled-Morton batch size (Schütz 2021)
_FLT_MAX = float(np.finfo(np.float32).max)

DEFAULT_NODE_CAPACITY = 15_000
DEFAULT_POINT_BUDGET = 1_500_000
DEFAULT_REFINE_PIXELS = 12.0
DEFAULT_MAX_DEPTH = 12


@dataclass
class PointsOctree:
    """Build result: a permutation plus flat per-node arrays (BFS order)."""

    order: np.ndarray  # (N,) int64 — original index for each reordered slot
    centers: np.ndarray  # (M, 3) float32 — node cube centers
    half_sizes: np.ndarray  # (M,) float32 — node cube half edge lengths
    offsets: np.ndarray  # (M,) uint32 — into the reordered arrays
    counts: np.ndarray  # (M,) uint32 — node's own (sample) point count
    levels: np.ndarray  # (M,) uint8
    first_child: np.ndarray  # (M,) uint32 — NO_CHILD for leaves
    child_mask: np.ndarray  # (M,) uint8

    @property
    def n_nodes(self) -> int:
        return len(self.counts)

    @property
    def max_level(self) -> int:
        return int(self.levels.max()) if len(self.levels) else 0

    def pack_hierarchy(
        self,
        birth: Optional[np.ndarray] = None,
        removal: Optional[np.ndarray] = None,
    ) -> bytes:
        """Pack the 40-byte-per-node hierarchy blob.

        ``birth``/``removal`` are the *reordered* per-point time arrays;
        each node's [tmin, tmax) = [min own birth, max own removal) is used
        by the viewer to skip fetching/drawing nodes whose own points are
        all unborn or all removed at the current scrub time. Unbounded ends
        default to ±FLT_MAX (never time-culled).
        """
        m = self.n_nodes
        rec = np.zeros(m, dtype=HIERARCHY_DTYPE)
        rec["cx"] = self.centers[:, 0]
        rec["cy"] = self.centers[:, 1]
        rec["cz"] = self.centers[:, 2]
        rec["half"] = self.half_sizes
        rec["offset"] = self.offsets
        rec["count"] = self.counts
        rec["first_child"] = self.first_child
        rec["child_mask"] = self.child_mask
        rec["level"] = self.levels
        rec["tmin"] = -_FLT_MAX
        rec["tmax"] = _FLT_MAX
        for i in range(m):
            lo = int(self.offsets[i])
            hi = lo + int(self.counts[i])
            if birth is not None and hi > lo:
                rec["tmin"][i] = birth[lo:hi].min()
            if removal is not None and hi > lo:
                rec["tmax"][i] = removal[lo:hi].max()
        return rec.tobytes()


def _shuffled_morton_order(rel: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vertex order within one node: Morton-sort, then shuffle whole batches
    of VERTEX_BATCH consecutive points (locality inside a batch, batches
    spread across the raster pipeline).

    ``rel`` is (n, 3) positions relative to the node cube in [-1, 1].
    """
    n = len(rel)
    if n <= VERTEX_BATCH:
        return np.arange(n)
    # 10 bits per axis is plenty for raster-locality purposes.
    q = np.clip(((rel + 1.0) * 0.5 * 1023.0).astype(np.uint32), 0, 1023)
    key = np.zeros(n, dtype=np.uint64)
    for b in range(10):
        for axis in range(3):
            bit = ((q[:, axis] >> np.uint32(b)) & np.uint32(1)).astype(np.uint64)
            key |= bit << np.uint64(3 * b + axis)
    morton = np.argsort(key, kind="stable")
    n_batches = (n + VERTEX_BATCH - 1) // VERTEX_BATCH
    batch_order = rng.permutation(n_batches)
    out = np.empty(n, dtype=np.int64)
    pos = 0
    for b in batch_order:
        lo = int(b) * VERTEX_BATCH
        hi = min(lo + VERTEX_BATCH, n)
        out[pos : pos + hi - lo] = morton[lo:hi]
        pos += hi - lo
    return out


def build_points_octree(
    positions: np.ndarray,
    strat_times: Optional[np.ndarray] = None,
    node_capacity: int = DEFAULT_NODE_CAPACITY,
    max_depth: int = DEFAULT_MAX_DEPTH,
    seed: int = 0,
) -> PointsOctree:
    """Build the additive sampled octree over ``positions`` (N, 3).

    ``strat_times``: optional (N,) per-point times to stratify node samples
    over (typically birth times); None = plain random sampling.

    Nodes with ≤ ``node_capacity`` points (or at ``max_depth``) keep all
    their points; larger nodes keep a stratified sample of
    ``node_capacity`` and push the rest down by octant.
    """
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    n = len(positions)
    if n == 0:
        raise ValueError("positions must not be empty")
    if node_capacity < 1:
        raise ValueError(f"node_capacity must be >= 1 (got {node_capacity})")
    rng = np.random.default_rng(seed)

    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    center0 = ((lo + hi) * 0.5).astype(np.float64)
    # Cube the box on the max extent; tiny pad so boundary points stay
    # strictly inside and a degenerate (single-point / flat) cloud still
    # gets a positive half size.
    half0 = float((hi - lo).max()) * 0.5 * 1.0001 + 1e-9

    # BFS. Each queue entry: (candidate original-indices, center, half,
    # level, node_slot). Node slots are allocated for children while
    # processing the parent, so children are consecutive (first_child).
    centers: list[np.ndarray] = []
    halfs: list[float] = []
    levels: list[int] = []
    first_child: list[int] = []
    child_mask: list[int] = []
    samples: list[np.ndarray] = []  # per-slot original indices (node's own)

    def _alloc(center, half, level) -> int:
        centers.append(np.asarray(center, dtype=np.float64))
        halfs.append(float(half))
        levels.append(int(level))
        first_child.append(NO_CHILD)
        child_mask.append(0)
        samples.append(None)  # type: ignore[arg-type]
        return len(centers) - 1

    # One global candidate order, computed once: time-sorted with a random
    # tiebreak when stratifying (a strided pick then gives every node an
    # unbiased-per-time-slice sample, keeping time-filtered density honest
    # at every LOD), plain random permutation otherwise (strided pick over
    # a random order = random sample). Boolean-mask partitioning below
    # preserves relative order, so every node's candidate list inherits
    # this order for free — no per-node sort. (A per-node lexsort here
    # dominated build time: the root alone re-sorted all N points.)
    if strat_times is not None:
        strat_arr = np.asarray(strat_times).reshape(-1)
        if strat_arr.shape[0] != n:
            raise ValueError(
                f"strat_times must have length {n} (got {strat_arr.shape[0]})"
            )
        base_order = np.lexsort((rng.random(n), strat_arr))
    else:
        base_order = rng.permutation(n)

    root = _alloc(center0, half0, 0)
    queue: list[tuple[np.ndarray, int]] = [(base_order.astype(np.int64), root)]
    qi = 0
    while qi < len(queue):
        idx, slot = queue[qi]
        qi += 1
        center = centers[slot]
        half = halfs[slot]
        level = levels[slot]
        m = len(idx)

        if m <= node_capacity or level >= max_depth:
            samples[slot] = idx
            continue

        # Candidates arrive pre-ordered (see base_order above): an even
        # stride through them is the stratified/random sample.
        picks = (np.arange(node_capacity, dtype=np.float64) * m / node_capacity).astype(
            np.int64
        )
        keep = np.zeros(m, dtype=bool)
        keep[picks] = True
        samples[slot] = idx[keep]
        rest = idx[~keep]

        p = positions[rest]
        octant = (
            (p[:, 0] > center[0]).astype(np.int8)
            | ((p[:, 1] > center[1]).astype(np.int8) << 1)
            | ((p[:, 2] > center[2]).astype(np.int8) << 2)
        )
        mask = 0
        quarter = half * 0.5
        for k in range(8):
            child_idx = rest[octant == k]
            if len(child_idx) == 0:
                continue
            child_center = center + quarter * np.array(
                [
                    1.0 if k & 1 else -1.0,
                    1.0 if k & 2 else -1.0,
                    1.0 if k & 4 else -1.0,
                ]
            )
            # Child cube: half the parent's edge, centered a quarter-edge
            # (= child half) away along each split axis.
            child_slot = _alloc(child_center, quarter, level + 1)
            if mask == 0:
                first_child[slot] = child_slot
            mask |= 1 << k
            queue.append((child_idx, child_slot))
        child_mask[slot] = mask

    # Emit: concatenate node samples in BFS slot order; shuffled-Morton
    # vertex order inside each node.
    n_nodes = len(centers)
    offsets = np.zeros(n_nodes, dtype=np.uint32)
    counts = np.zeros(n_nodes, dtype=np.uint32)
    parts = []
    pos = 0
    for slot in range(n_nodes):
        own = samples[slot]
        c = len(own)
        if c > 0:
            rel = (positions[own].astype(np.float64) - centers[slot]) / halfs[slot]
            own = own[_shuffled_morton_order(rel, rng)]
        offsets[slot] = pos
        counts[slot] = c
        parts.append(own)
        pos += c
    order = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    assert pos == n

    return PointsOctree(
        order=order,
        centers=np.array(centers, dtype=np.float32).reshape(n_nodes, 3),
        half_sizes=np.array(halfs, dtype=np.float32),
        offsets=offsets,
        counts=counts,
        levels=np.array(levels, dtype=np.uint8),
        first_child=np.array(first_child, dtype=np.uint32),
        child_mask=np.array(child_mask, dtype=np.uint8),
    )


# ── Grid fast-path ──────────────────────────────────────────────────────────
# When the caller declares the cloud is voxel-centres on a regular lattice
# (mill-sim's carve view), octree assignment is pure integer arithmetic: a
# single global Morton code over the lattice, then every octant split and
# every node's Morton emit-order is a bit-slice of that code — no per-node
# float gathers, no per-node argsort, no O(N log N) lexsort for
# stratification. The general float builder above stays the default (it makes
# no grid assumption, D6); this is opt-in via ``lod={"grid": {...}}``.
#
# Correctness note: the split planes are power-of-two subdivisions of the root
# cube, so a lattice-quantised bit-slice reproduces the float ``pos > centre``
# test EXACTLY for points that sit at cell centres (which grid voxels do). For
# arbitrary float clouds a point within one lattice cell of a split plane could
# fall on the other side of the bit boundary than of the float centre — which
# is why this path is gated on the grid promise rather than made universal.

_MORTON_MAX_BITS = 21  # 3 * 21 = 63 fits a uint64 Morton code


def _spread_bits_u64(x: np.ndarray) -> np.ndarray:
    """Spread the low 21 bits of each value so bit i lands at position 3*i
    (Morton "Part1By2", vectorised over the whole array). Input/out uint64."""
    x = x.astype(np.uint64) & np.uint64(0x1FFFFF)
    x = (x | (x << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    x = (x | (x << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    x = (x | (x << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    x = (x | (x << np.uint64(2))) & np.uint64(0x1249249249249249)
    return x


def _grid_root_cube(positions: np.ndarray) -> tuple[np.ndarray, float]:
    """Root cube (center0 float64 (3,), half0 float) for a grid cloud — the
    SAME bounds-derived cube the float builder uses. An external code producer
    must reproduce this exactly (it fixes the lattice origin, below)."""
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    center0 = ((lo + hi) * 0.5).astype(np.float64)
    half0 = float((hi - lo).max()) * 0.5 * 1.0001 + 1e-9
    return center0, half0


def _grid_n_bits(half0: float, pitch: float, max_depth: int) -> int:
    """Morton bits per axis: enough that one lattice cell ⊂ one Morton cell (so
    distinct voxels get distinct codes), but at least ``max_depth`` (octant
    extraction needs one bit per level) and at most ``_MORTON_MAX_BITS``."""
    cube_edge = 2.0 * half0
    need_bits = int(np.ceil(np.log2(max(2.0, cube_edge / pitch))))
    return int(np.clip(max(max_depth, need_bits), max_depth, _MORTON_MAX_BITS))


def grid_morton_codes(
    positions: np.ndarray,
    spacing,
    max_depth: int = DEFAULT_MAX_DEPTH,
    n_bits: Optional[int] = None,
) -> tuple[np.ndarray, int]:
    """The exact global Morton code per point that :func:`build_points_octree_grid`
    uses internally — the reference an external producer (e.g. mill-sim's Rust
    kernel) reproduces to feed the precomputed-order fast path.

    Returns ``(codes, n_bits)`` where ``codes`` is ``(N,)`` uint64. The mapping,
    pinned so it is bit-reproducible, is: quantise each point to a lattice index
    ``q = clip(floor((p - (center0 - half0)) / cell), 0, 2**n_bits - 1)`` per
    axis (``center0``/``half0`` from :func:`_grid_root_cube`, ``cell =
    2*half0 / 2**n_bits``), then interleave with **x at bit 0 of each triple, y
    at bit 1, z at bit 2** (``code = Σ_i (x_i<<3i) | (y_i<<3i+1) | (z_i<<3i+2)``).
    ``n_bits`` defaults to :func:`_grid_n_bits`; pass it explicitly to pin it.
    """
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    sp = np.asarray(spacing, dtype=np.float64).reshape(-1)
    if sp.size not in (1, 3) or not np.all(sp > 0):
        raise ValueError(
            f"grid spacing must be a positive scalar or 3-vector (got {spacing!r})"
        )
    center0, half0 = _grid_root_cube(positions)
    if n_bits is None:
        n_bits = _grid_n_bits(half0, float(sp.min()), max_depth)
    return _grid_morton_codes(positions, center0, half0, n_bits), n_bits


def _grid_morton_codes(
    positions: np.ndarray, center0: np.ndarray, half0: float, n_bits: int
) -> np.ndarray:
    """Vectorised quantise+interleave (see :func:`grid_morton_codes`). Builds the
    code axis by axis, freeing each column so peak memory stays near one uint64
    code array (8 B/pt) rather than a full (N,3) int64 lattice."""
    cube_lo = center0 - half0
    cell = (2.0 * half0) / float(1 << n_bits)
    inv_cell = 1.0 / cell
    hi_idx = (1 << n_bits) - 1
    n = len(positions)
    mcode = np.zeros(n, dtype=np.uint64)
    for axis in range(3):
        qa = np.floor(
            (positions[:, axis].astype(np.float64) - cube_lo[axis]) * inv_cell
        )
        np.clip(qa, 0, hi_idx, out=qa)
        mcode |= _spread_bits_u64(qa.astype(np.uint64)) << np.uint64(axis)
        del qa
    return mcode


def _batch_shuffle(order_in: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffled-Morton emit order (Schütz 2021): keep raster locality inside a
    VERTEX_BATCH run but spread the batches across the raster pipeline. Takes an
    already-Morton-sorted index order and permutes it in whole batches."""
    c = len(order_in)
    if c <= VERTEX_BATCH:
        return order_in
    n_batches = (c + VERTEX_BATCH - 1) // VERTEX_BATCH
    out = np.empty(c, dtype=order_in.dtype)
    pos = 0
    for b in rng.permutation(n_batches):
        lo = int(b) * VERTEX_BATCH
        hi = min(lo + VERTEX_BATCH, c)
        out[pos : pos + hi - lo] = order_in[lo:hi]
        pos += hi - lo
    return out


def build_points_octree_grid(
    positions: np.ndarray,
    spacing,
    strat_times: Optional[np.ndarray] = None,
    origin: Optional[np.ndarray] = None,
    node_capacity: int = DEFAULT_NODE_CAPACITY,
    max_depth: int = DEFAULT_MAX_DEPTH,
    seed: int = 0,
    *,
    codes: Optional[np.ndarray] = None,
    order: Optional[np.ndarray] = None,
    n_bits: Optional[int] = None,
) -> PointsOctree:
    """Grid-aware build for lattice-aligned clouds (voxel centres).

    Same additive sampled octree as :func:`build_points_octree` (identical node
    cubes, per-node capacity, tiling and shuffled-Morton emit order), but built
    by sorting every point ONCE by its global Morton code and then slicing the
    sorted array into node ranges — no recursive octant re-partition, no
    per-node sort. The build's only super-linear step is that one sort; the rest
    is O(N) range bookkeeping, so it is markedly faster than both the float
    builder and the earlier top-down grid build, and holds flat as N grows.

    The LOD sample is a plain strided (spatial) subsample within each node's
    range. Unlike the float builder this does NOT time-stratify: each point
    keeps its own birth/removal and the viewer's per-point time window culls
    independently, and in a carve removal_time is spatially correlated (the tool
    sweeps through space), so a spatial coarse sample stays honest under a time
    scrub. ``strat_times`` is accepted for API symmetry but ignored.

    ``spacing``: lattice pitch — scalar or (3,). Only its smallest component is
    used, to size the Morton resolution so each voxel maps to its own cell.
    ``origin`` is accepted for API symmetry/validation but not required (the
    root cube is derived from the point bounds like the float path).

    Precomputed-order fast path (``codes`` / ``order`` / ``n_bits``)
    ----------------------------------------------------------------
    An external producer (e.g. mill-sim's Rust kernel) can supply the two
    super-linear stages so the builder skips them:

    - ``codes``: ``(N,)`` uint64 global Morton codes, exactly as
      :func:`grid_morton_codes` would compute them (see that function for the
      bit-reproducible mapping). Skips the internal quantise. Must be paired with
      the ``n_bits`` those codes were built with (pass ``n_bits``, or leave it
      ``None`` to reuse the spacing-derived value — it must match).
    - ``order``: ``(N,)`` integer permutation with ``codes[order]`` non-
      decreasing (the Morton sort). Requires ``codes``. Skips the internal sort.
      For a genuine grid cloud every voxel has a distinct code, so this order is
      unique and the result is byte-identical regardless of who sorted; if codes
      collide (multiple points per cell — violates the grid promise) the tie
      order is free and only node *membership*, not the intra-cell emit slot, is
      guaranteed to match.

    All three are optional and validated loudly; omitting them reproduces the
    pure-numpy build exactly. See docs/points-lod-grid-api.md for the full
    external-producer contract.
    """
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    n = len(positions)
    if n == 0:
        raise ValueError("positions must not be empty")
    if node_capacity < 1:
        raise ValueError(f"node_capacity must be >= 1 (got {node_capacity})")
    sp = np.asarray(spacing, dtype=np.float64).reshape(-1)
    if sp.size not in (1, 3) or not np.all(sp > 0):
        raise ValueError(
            f"grid spacing must be a positive scalar or 3-vector (got {spacing!r})"
        )
    pitch = float(sp.min())
    rng = np.random.default_rng(seed)

    center0, half0 = _grid_root_cube(positions)

    # Morton resolution: enough bits that one lattice cell ⊂ one Morton cell
    # (so voxels never collide), but at least max_depth (octant extraction
    # needs one bit per level) and at most _MORTON_MAX_BITS (uint64 budget).
    # An explicit n_bits (from a code producer) pins it; else derive from pitch.
    derived_bits = _grid_n_bits(half0, pitch, max_depth)
    if n_bits is None:
        n_bits = derived_bits
    else:
        n_bits = int(n_bits)
        if not (max_depth <= n_bits <= _MORTON_MAX_BITS):
            raise ValueError(
                f"n_bits must be in [max_depth={max_depth}, {_MORTON_MAX_BITS}] (got {n_bits})"
            )

    # strat_times is intentionally unused (see docstring): the grid path samples
    # spatially, each point keeps its own time, so no time-stratified sort.
    del strat_times

    # Stage 1 — global Morton codes. Skipped when a producer supplies them.
    if codes is None:
        # Lattice indices fit uint32 (n_bits <= 21). Build the code axis by axis,
        # freeing each column so peak memory stays near one uint64 array (8 B/pt).
        mcode = _grid_morton_codes(positions, center0, half0, n_bits)
    else:
        mcode = np.asarray(codes)
        if mcode.dtype != np.uint64:
            raise ValueError(f"codes must be uint64 (got dtype {mcode.dtype})")
        if mcode.ndim != 1 or mcode.shape[0] != n:
            raise ValueError(
                f"codes must be 1-D length {n} to match positions (got shape {mcode.shape})"
            )
        code_ceiling = (
            np.uint64((1 << (3 * n_bits)) - 1) if 3 * n_bits < 64 else np.uint64(-1)
        )
        if n and mcode.max() > code_ceiling:
            raise ValueError(
                f"codes exceed the {3 * n_bits}-bit budget for n_bits={n_bits}; the "
                f"producer's n_bits/geometry disagrees with this build"
            )

    # Stage 2 — sort ONCE by Morton code (LBVH/Karras style), or accept a
    # producer's order. After this every octree cell at every level is a
    # CONTIGUOUS RANGE in the sorted order: a node's octant children are adjacent
    # sub-ranges (sliced at the level's bit-triple), so the recursion never
    # re-partitions or gathers and each node's points are already Morton-sorted
    # (emit needs only a batch-shuffle, no per-node argsort). This is the whole
    # build's one O(N log N) step; everything below is O(N) range bookkeeping.
    if order is None:
        sort = np.argsort(
            mcode
        )  # quicksort uint64; distinct grid codes => unique order
        sorted_idx = sort.astype(np.int64)
    else:
        if codes is None:
            raise ValueError(
                "order requires codes (the builder cannot slice octants without them)"
            )
        ordr = np.asarray(order)
        if ordr.dtype.kind not in "iu":
            raise ValueError(f"order must be an integer array (got dtype {ordr.dtype})")
        if ordr.ndim != 1 or ordr.shape[0] != n:
            raise ValueError(
                f"order must be 1-D length {n} to match positions (got shape {ordr.shape})"
            )
        seen = np.zeros(n, dtype=bool)
        seen[ordr] = True  # a duplicate index leaves some slot unseen => caught below
        if not seen.all():
            raise ValueError(
                "order must be a permutation of range(N) (missing/duplicate indices)"
            )
        sorted_idx = ordr.astype(np.int64)
    sorted_codes = mcode[sorted_idx]
    if order is not None and n > 1 and np.any(sorted_codes[1:] < sorted_codes[:-1]):
        raise ValueError(
            "order must sort codes non-decreasing (codes[order] is not monotonic)"
        )

    centers = [center0]
    halfs = [half0]
    levels = [0]
    first_child = [NO_CHILD]
    child_mask = [0]
    samples: list = [None]

    # Queue carries each node's candidate (index, code) arrays, both in Morton
    # order — contiguous slices of the globally-sorted arrays, minus the strided
    # samples removed by ancestors (so still ascending, still sliceable).
    queue: list = [(sorted_idx, sorted_codes, 0)]
    qi = 0
    arange9 = np.arange(9)
    while qi < len(queue):
        idx, codes, slot = queue[qi]
        qi += 1
        level = levels[slot]
        m = len(idx)
        if m <= node_capacity or level >= max_depth:
            samples[slot] = idx  # own = all, already Morton-sorted
            continue
        # Strided (spatial) subsample of this node's Morton-sorted candidates.
        picks = (np.arange(node_capacity, dtype=np.float64) * m / node_capacity).astype(
            np.int64
        )
        keep = np.zeros(m, dtype=bool)
        keep[picks] = True
        samples[slot] = idx[keep]
        notkeep = ~keep
        rest_idx = idx[notkeep]
        rest_codes = codes[notkeep]  # subsequence of a sorted array → still sorted
        # Octant = the level's bit-triple; since rest_codes ascend, the 8 octant
        # groups are contiguous — split by searchsorted, no per-octant gather.
        shift = np.uint64(3 * (n_bits - 1 - level))
        oct_rest = ((rest_codes >> shift) & np.uint64(7)).astype(np.int64)
        bnd = np.searchsorted(oct_rest, arange9)
        center = centers[slot]
        quarter = halfs[slot] * 0.5
        mask = 0
        for k in range(8):
            a = int(bnd[k])
            b = int(bnd[k + 1])
            if b <= a:
                continue
            child_center = center + quarter * np.array(
                [1.0 if k & 1 else -1.0, 1.0 if k & 2 else -1.0, 1.0 if k & 4 else -1.0]
            )
            centers.append(child_center)
            halfs.append(quarter)
            levels.append(level + 1)
            first_child.append(NO_CHILD)
            child_mask.append(0)
            samples.append(None)
            child_slot = len(centers) - 1
            if mask == 0:
                first_child[slot] = child_slot
            mask |= 1 << k
            queue.append((rest_idx[a:b], rest_codes[a:b], child_slot))
        child_mask[slot] = mask

    n_nodes = len(centers)
    offsets = np.zeros(n_nodes, dtype=np.uint32)
    counts = np.zeros(n_nodes, dtype=np.uint32)
    parts = []
    pos = 0
    for slot in range(n_nodes):
        own = samples[slot]
        c = len(own)
        if c > VERTEX_BATCH:
            # Already Morton-sorted (subsequence of the global sort) — the emit
            # order is free; batch-shuffle for GPU raster locality (Schütz 2021).
            own = _batch_shuffle(own, rng)
        offsets[slot] = pos
        counts[slot] = c
        parts.append(own)
        pos += c
    order = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    assert pos == n

    return PointsOctree(
        order=order,
        centers=np.array(centers, dtype=np.float32).reshape(n_nodes, 3),
        half_sizes=np.array(halfs, dtype=np.float32),
        offsets=offsets,
        counts=counts,
        levels=np.array(levels, dtype=np.uint8),
        first_child=np.array(first_child, dtype=np.uint32),
        child_mask=np.array(child_mask, dtype=np.uint8),
    )


def pack_node_payload(
    positions: np.ndarray,
    colors_u8: Optional[np.ndarray],
    birth: Optional[np.ndarray],
    removal: Optional[np.ndarray],
    center: np.ndarray,
    half: float,
) -> bytes:
    """Pack one node's point block for the wire.

    Layout (all little-endian, count = len(positions)):
      int16 xyz * count      — node-local, normalized to the node cube
                               (dequantized on the GPU via normalized
                               attributes + the node object's transform)
      uint8 rgb * count      — when the cloud has vertex colors
      float32 birth * count  — when the cloud has birth times
      float32 removal * count
    """
    rel = (positions.astype(np.float64) - np.asarray(center, dtype=np.float64)) / half
    q = np.clip(np.rint(rel * 32767.0), -32767, 32767).astype("<i2")
    parts = [q.tobytes()]
    if colors_u8 is not None:
        parts.append(np.ascontiguousarray(colors_u8, dtype=np.uint8).tobytes())
    if birth is not None:
        parts.append(np.ascontiguousarray(birth, dtype="<f4").tobytes())
    if removal is not None:
        parts.append(np.ascontiguousarray(removal, dtype="<f4").tobytes())
    return b"".join(parts)
