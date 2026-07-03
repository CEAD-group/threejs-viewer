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


def _stratified_sample(
    n: int,
    capacity: int,
    strat_values: Optional[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick ``capacity`` of ``n`` candidate slots.

    With ``strat_values`` (per-candidate times): order by time with a random
    tiebreak and pick at even strides — an unbiased-per-time-slice sample,
    so a time filter later thins this node uniformly. Without: plain seeded
    random sample.
    """
    if strat_values is None:
        return rng.permutation(n)[:capacity]
    order = np.lexsort((rng.random(n), strat_values))
    picks = (np.arange(capacity, dtype=np.float64) * n / capacity).astype(np.int64)
    return order[picks]


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

    root = _alloc(center0, half0, 0)
    queue: list[tuple[np.ndarray, int]] = [(np.arange(n, dtype=np.int64), root)]
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

        strat = strat_times[idx] if strat_times is not None else None
        sel = _stratified_sample(m, node_capacity, strat, rng)
        keep = np.zeros(m, dtype=bool)
        keep[sel] = True
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
