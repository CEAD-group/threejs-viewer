# Grid octree LOD — precomputed-order external producer API

`build_points_octree_grid` (in `src/threejs_viewer/points_lod.py`) builds the
additive sampled octree for a **lattice-aligned** point cloud (voxel centres,
e.g. mill-sim's carve view) by computing one global Morton code per point,
sorting once, and slicing the sorted array into node ranges. The two
super-linear stages — the **quantise** (code build) and the **sort** — can be
supplied by an external producer (e.g. mill-sim's Rust kernel) via the `codes`,
`order`, and `n_bits` parameters, so the builder skips them and goes straight to
range-slicing + strided sampling.

This document pins down everything a producer needs to generate **byte-identical
input** so the resulting octree matches the pure-numpy build exactly.

---

## 1. Function signature

```python
def build_points_octree_grid(
    positions: np.ndarray,          # (N, 3) float32 — voxel centres, world units
    spacing,                        # scalar or (3,) lattice pitch, world units
    strat_times=None,               # accepted for API symmetry; IGNORED (spatial sampling)
    origin=None,                    # accepted for API symmetry; IGNORED (cube from bounds)
    node_capacity=15_000,           # per-node sample cap
    max_depth=12,                   # max octree depth
    seed=0,                         # RNG seed for the shuffled-Morton emit order
    *,
    codes: Optional[np.ndarray] = None,   # (N,) uint64 global Morton codes  (skips quantise)
    order: Optional[np.ndarray] = None,   # (N,) integer permutation, codes[order] ascending (skips sort)
    n_bits: Optional[int] = None,         # Morton bits/axis the codes use   (else derived from spacing)
) -> PointsOctree
```

Reference producer in Python: `grid_morton_codes(positions, spacing, max_depth=12,
n_bits=None) -> (codes, n_bits)` returns exactly the codes the builder would
compute internally. A Rust producer must reproduce these bit-for-bit.

Passing rules:

| supplied            | builder skips        | notes |
|---------------------|----------------------|-------|
| nothing             | — (pure numpy build) | default, fully backward compatible |
| `codes` (+`n_bits`) | quantise             | still sorts internally |
| `codes` + `order` (+`n_bits`) | quantise **and** sort | the full fast path |
| `order` alone       | — (raises)           | `order` requires `codes` |

`n_bits` may be omitted whenever the producer used the spacing-derived value
(§4); pass it explicitly to pin the resolution. Validation is loud (see §9).

### Client-level entry point

`ViewerClient.add_points(..., lod={"grid": {...}})` forwards the same
three keys: `lod={"grid": {"spacing": pitch, "codes": codes, "order": order,
"n_bits": n_bits}}`. The arrays never go over the wire — they are consumed
locally by the octree build; the wire format is unchanged.

---

## 2. Root cube (lattice origin) — derived from the point bounds

The octree root cube is derived from the point AABB, **not** from `origin`
(which is ignored). Reproduce with float32 positions and this exact arithmetic
(`_grid_root_cube`):

```
lo      = positions.min(axis=0)          # float32 reduction over the f32 positions
hi      = positions.max(axis=0)          # float32
center0 = ((lo + hi) * 0.5).astype(f64)  # add/mul in float32, THEN upcast to float64
half0   = float((hi - lo).max()) * 0.5 * 1.0001 + 1e-9   # (hi-lo).max() in f32, upcast, rest in f64
```

Notes for a bit-exact producer:

- `positions` is first coerced to **float32** (`np.asarray(positions,
  dtype=np.float32)`). Do the min/max on the float32 values.
- `center0` is computed in float32 (`(lo+hi)*0.5`) and only then cast to float64.
- `half0` upcasts the float32 max extent to a Python float (float64) **before**
  the `*0.5 * 1.0001 + 1e-9`. The `1.0001` pad keeps boundary points strictly
  inside the cube; the `+1e-9` keeps a degenerate (single-cell / flat) cloud's
  half positive.
- The cube is **axis-aligned and cubic**: the same `half0` on all three axes.

`cube_lo = center0 - half0` (float64) is the lattice origin used below.

---

## 3. Morton code layout (uint64)

Per point, per axis `a ∈ {x=0, y=1, z=2}`:

```
cell   = (2 * half0) / (1 << n_bits)            # float64 scalar, isotropic
q_a    = floor((p_a - cube_lo_a) / cell)        # p_a upcast to float64
q_a    = clip(q_a, 0, (1 << n_bits) - 1)        # uint lattice index, fits n_bits (<=21)
```

The build multiplies by `inv_cell = 1.0 / cell` rather than dividing; for
n_bits ≤ 21 the two agree for every in-range index, but a bit-exact producer
should use `* (1.0/cell)` to match the last ULP.

Interleave (Morton "Part1By2" / Z-order), **x at bit 0 of each triple, y at bit
1, z at bit 2**:

```
code = Σ_{i=0..n_bits-1}  (x_i << (3i + 0)) | (y_i << (3i + 1)) | (z_i << (3i + 2))
```

i.e. `code = spread(x) | (spread(y) << 1) | (spread(z) << 2)` where `spread`
scatters bit `i` to position `3i`. The uint64 `spread` (`_spread_bits_u64`,
operating on the low 21 bits):

```
x &= 0x1FFFFF
x = (x | (x << 32)) & 0x1F00000000FFFF
x = (x | (x << 16)) & 0x1F0000FF0000FF
x = (x | (x <<  8)) & 0x100F00F00F00F00F
x = (x | (x <<  4)) & 0x10C30C30C30C30C3
x = (x | (x <<  2)) & 0x1249249249249249
```

Total occupied bits = `3 * n_bits` ≤ 63, so the code always fits `uint64`.
Higher code ⇒ higher Morton (Z-order) rank.

---

## 4. `n_bits` (Morton resolution) derivation

`n_bits` is the number of lattice bits **per axis** (`_grid_n_bits`):

```
pitch     = min(spacing)                                    # smallest component
cube_edge = 2 * half0
need_bits = ceil(log2(max(2.0, cube_edge / pitch)))         # enough that one voxel ⊂ one Morton cell
n_bits    = clip(max(max_depth, need_bits), max_depth, 21)  # >= max_depth (one bit/level), <= 21 (uint64)
```

The `max(max_depth, …)` floor guarantees every octree level has a bit-triple to
extract (§5). The `21` ceiling is the uint64 budget (`3 * 21 = 63`). Because a
grid voxel is finer than one Morton cell, **distinct voxels get distinct
codes** — which is what makes the sort order unique (§6).

---

## 5. Level → octant bit-slice (how the sorted range becomes a tree)

The octant of a point at octree level `L` (root = 0) is the level's bit-triple
of its code:

```
octant(L) = (code >> (3 * (n_bits - 1 - L))) & 0b111        # in {0..7}
          = x_bit | (y_bit << 1) | (z_bit << 2)
```

This matches the float builder's octant convention
`(p.x>cx) | (p.y>cy)<<1 | (p.z>cz)<<2`, and because the split planes are
power-of-two subdivisions of the root cube, the bit-slice reproduces the float
`>centre` test **exactly** for points at cell centres. Since the codes are
sorted, the eight octant groups within any node's range are contiguous and are
found with a single `searchsorted` — no per-node gather.

The producer does **not** need to emit the tree; it only supplies `codes` and
optionally `order`. This section is the contract that makes a globally
Morton-sorted array sliceable into the same tree the numpy build produces.

---

## 6. Sort / tie contract

`order` must be a **permutation of `range(N)`** such that `codes[order]` is
**non-decreasing**. The builder's internal path uses `np.argsort(codes)`
(quicksort).

- For a genuine grid cloud every voxel has a distinct code (§4), so the sorting
  permutation is **unique** and the whole octree — `order`, `centers`,
  `offsets`, `counts`, `levels`, `first_child`, `child_mask` — is
  **byte-identical** no matter who sorted (numpy quicksort, a Rust radix sort,
  etc.). A stable sort is **not** required.
- If codes collide (multiple points in one lattice cell — this **violates** the
  grid promise), tie order is unspecified. Node **membership** still matches
  (equal codes ⇒ same octant at every level), but which coincident point lands
  in which intra-node emit slot may differ. Producers that want fully
  reproducible output under collisions should break ties by ascending original
  index (a stable sort), which is what `np.argsort(kind="stable")` does.

The emit order **within** a node is a seeded `VERTEX_BATCH`-block shuffle of the
already-Morton-sorted range (`_batch_shuffle`, Schütz 2021); it depends only on
`seed` and the node's point count, not on the producer.

---

## 7. Array dtype / shape / contiguity requirements

| param   | dtype                         | shape  | contiguity | validated |
|---------|-------------------------------|--------|------------|-----------|
| `positions` | float32 (coerced)         | (N, 3) | any (reshaped) | length > 0 |
| `codes` | **uint64 exactly**            | (N,)   | C-contiguous 1-D | dtype, ndim, length, ≤ 3·n_bits budget |
| `order` | any signed/unsigned integer   | (N,)   | 1-D        | dtype kind, ndim, length, is-permutation, monotonicity of `codes[order]` |
| `n_bits`| Python int                    | scalar | —          | `max_depth ≤ n_bits ≤ 21` |

`codes` dtype must be **exactly `np.uint64`** (no silent cast — a signed or
32-bit array raises). `order` may be any integer dtype; it is cast to int64
internally. The builder does not require `order` to be C-contiguous, but a
contiguous array avoids a copy on the fancy-index gather.

---

## 8. Worked example

Four voxel centres on a **pitch = 1.0** lattice, `max_depth = 12`:

```
positions = [[0,0,0], [1,0,0], [0,1,0], [3,2,1]]   (float32)
```

Bounds and cube (§2):

```
lo = [0,0,0]  hi = [3,2,1]
center0 = [1.5, 1.0, 0.5]
half0   = 1.5001500010000002          # = 1.5 * 1.0001 + 1e-9 (max extent = 3)
cube_lo = [-0.000150001, -0.500150001, -1.000150001]
```

Resolution (§4): `cube_edge = 3.0003…`, `cube_edge/pitch ≈ 3.0`,
`need_bits = ceil(log2(3)) = 2`, floored to `max_depth` ⇒ **`n_bits = 12`**,
`cell = 2·half0 / 4096 ≈ 7.3250e-4`.

Quantise + interleave (§3) — note the cube is sized by the largest (x) extent,
so the finer lattice puts each point at a distinct fine index:

| point        | q = (x, y, z)        | code (dec)      | code (hex)     |
|--------------|----------------------|-----------------|----------------|
| [0, 0, 0]    | (0, 682, 1365)       | 4 635 837 716   | `0x114514514`  |
| [1, 0, 0]    | (1365, 682, 1365)    | 5 726 623 061   | `0x155555555`  |
| [0, 1, 0]    | (0, 2048, 1365)      | 21 543 010 564  | `0x504104104`  |
| [3, 2, 1]    | (4095, 3413, 2730)   | 64 083 639 019  | `0xeebaebaeb`  |

Codes are distinct and already ascending here, so the Morton `order` is
`[0, 1, 2, 3]` and `codes[order] = [4635837716, 5726623061, 21543010564,
64083639019]`.

Reproduce and check against the reference:

```python
from threejs_viewer.points_lod import grid_morton_codes, build_points_octree_grid
codes, n_bits = grid_morton_codes(positions, spacing=1.0)   # the Rust producer target
order = argsort(codes)                                        # any correct Morton sort
oct = build_points_octree_grid(positions, spacing=1.0,
                               codes=codes, order=order, n_bits=n_bits)
# identical to build_points_octree_grid(positions, spacing=1.0)
```

---

## 9. Validation (all raise `ValueError`, loudly)

- `spacing` not a positive scalar / 3-vector.
- `n_bits` outside `[max_depth, 21]`.
- `codes` not `uint64`, not 1-D, or length ≠ N.
- `codes` with any value ≥ `2**(3·n_bits)` (producer's `n_bits`/geometry
  disagrees with this build).
- `order` given without `codes`.
- `order` not an integer array, not 1-D, or length ≠ N.
- `order` not a permutation of `range(N)` (missing/duplicate indices).
- `codes[order]` not non-decreasing (order does not sort the codes).

---

## 10. Payoff (measured; 8-core, 16 GB, numpy)

Build time for `build_points_octree_grid` on a thin-shell grid cloud, and the
share removed by supplying precomputed input:

| N     | full build | + codes (skip quantise) | + codes & order (skip quantise+sort) |
|-------|-----------:|------------------------:|-------------------------------------:|
| 10M   | 1.62s      | 1.10s  (−32%)           | 0.65s  (−60%)                        |
| 30M   | 5.76s      | 4.14s  (−28%)           | 2.58s  (−55%)                        |
| 73M   | 17.5s      | 11.1s  (−37%)           | 7.3s   (−58%)                        |

(73M full build varies ~15–17.5s run to run on a 16 GB box under memory
pressure; the % savings are stable.)

So a producer that ships **codes + order** removes roughly the quantise (~30% of
build) and the sort (~30%), leaving the builder doing only the O(N) tree
range-slicing and the emit shuffle. The sort floor itself is ~3.8s at 73M
(numpy quicksort argsort); a native radix argsort can go below that, but since
sort is only ~30% of the build the end-to-end ceiling on the Python side is the
tree + emit passes.
