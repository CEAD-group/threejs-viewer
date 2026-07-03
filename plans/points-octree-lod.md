# add_points: octree LOD + per-point time-window visibility (issue #79)

> **Status: Phases 1–2 implemented (branch `points-octree-lod-time`);
> Phase 3 open.** Decision-making artifact for issue #79 (mill-sim animated
> voxel fields: up to ~1B source points, ~1M drawn). Synthesized 2026-07-03
> from five prior-art surveys (Potree ecosystem, formats & Python tooling,
> LOD techniques, encoding/compression, GPU efficiency) plus discussion
> with Thijs. Implementation notes: the builder lives in `points_lod.py`;
> node serving = lazy callable values in the existing blob store (the
> Phase 3 `node_provider` hook falls out of the same mechanism);
> **deviation from D8**: timestamps ship as f32, not u16 — u16 needs an
> "unbounded" sentinel the shader can read (an immortal flag); deferred.
> Current per-point payload is 6 (i16 xyz) + 3 (u8 rgb) + 8 (f32 times)
> = 17 B. CLAUDE.md carries the as-built behavior; this doc remains the
> rationale + Phase 3 blueprint.

## The problem

mill-sim wants to push a dense voxel field (machining stock) as a point cloud
and scrub material removal over time. Two gaps in `add_points`:

1. **Distance-dependent LOD.** One flat `THREE.Points` draw call tops out
   around ~10M points; the target is up to ~1B source points with ~1M on
   screen. Needs an octree that renders a camera-adaptive subset.
2. **Per-point time-window visibility.** Removal is not a monotonic prefix of
   the buffer, so `set_draw_range` can't express it. Each point needs a
   lifetime `[birth_time, removal_time)` and a global scrubbable time `t`
   that hides points outside their window, in the shader.

Both are needed together: scrubbing time on a 100M-point cloud only works if
LOD keeps the drawn set bounded.

## Decisions

### D1 — Rollups are built by the data provider (Python side), never the viewer

At 1B points the data can never fully reach the browser (~24 GB at
xyz+rgb+2×f32 time), so in-browser octree building (the parametric-tube LOD
approach) is structurally impossible at the top of the range. The library
ships a generic numpy builder (Morton-sort based, see D5) for providers with
plain point arrays; providers with structured data (mill-sim's voxel grid)
may bypass it and synthesize node payloads directly — grids make sampling
and Morton ordering nearly free. The viewer only traverses, fetches, and
draws.

### D2 — The render/wire structure is a Potree-style sampled point octree, not voxel bricks

The 8³-dense-brick (NanoVDB-style) layout was considered and rejected *as
the wire format* — it survives only as a provider-side build detail (D1).
Two reasons, the first decisive:

- **Aggregation corrupts the time attribute.** A coarse-level "rollup voxel"
  representing 8 children must merge their lifetimes; any merge
  (min-birth/max-removal, averages) fabricates an interval that belongs to
  no real point, so the time scrubber would reveal/hide invented points at
  coarse zoom. In a sampled octree every rendered point at every LOD is a
  *real* point carrying its *true* `[birth, removal)` — time filtering stays
  exactly correct at all distances. This is the same reason Potree/MNO
  subsample rather than average (classification/intensity/GPS-time survive),
  and our per-point scalars make it binding.
- **No grid assumption.** `add_points` is a general API (LiDAR, metrology
  clouds). Potree-style nodes make none; bricks bake in mill-sim's grid.

Nuance added after the encoding deep-dive: the rejection is of bricks as the
*render/aggregation semantics*, not of grid-derived encodings. Implicit
(occupancy-bitmask) coordinates are fine as a wire encoding for grid
providers — positions are a function of the cell, not of point identity —
and coarse-level samples may be grid-*snapped*, provided each sample is one
real point carrying its true timestamps (sample-picking, never averaging).
See D8. Schütz's 2023/SimLOD hybrid (voxels in inner nodes, points in
leaves) is the precedent, minus its color-averaging variant.

Structure (Potree "modifiable nested octree", additive refinement): every
node stores a *sample* of the points in its cube; the root is a coarse
sample of everything; children add detail, never replace it. Rendering a
region = union of all loaded ancestor samples covering it. No point is
stored twice.

### D3 — Roll our own node protocol, modeled on Potree 2.0; adopt no external format or renderer

Surveyed: Potree 2.0 (+ potree-core v2.0.15 / @pnext/three-loader v0.2.5
renderers), COPC (+ copclib/laspy/laz-perf), EPT (+ PDAL/untwine),
3D Tiles 1.1 (+ py3dtiles/3DTilesRendererJS), NanoVDB. Verdict:

- **No renderer embeds.** Every Potree-lineage renderer is a multi-module
  npm package with workers (and WASM for LAZ); none drops into our
  concat-only, single-`<script>`, CDN-three viewer. And none supports a
  birth/removal filter without forking its material (Potree's stock filter
  is a single GPS-time clip range — a proof of mechanism, not a reusable
  knob).
- **Every format charges for a constraint we don't have.** COPC/EPT/3D Tiles
  optimize for portable persisted files served by dumb hosts to unknown
  clients, paid for with browser-side codecs (LAZ WASM, Draco) and
  file→file converters that can't ingest ephemeral in-process numpy. We own
  both ends of a localhost link; portability is worthless and any codec is
  pure overhead.
- **Potree 2.0's *internal design* is the right blueprint** (and the only
  format whose attribute model swallows two custom floats with zero
  ceremony): uncompressed structure-of-arrays node blobs in one buffer,
  fixed-size binary hierarchy records (22 B/node: type, childMask,
  numPoints, byteOffset, byteSize), chunked hierarchy with proxy nodes so
  the tree itself streams, int32+scale/offset positions, implicit node
  bounds derived from root bbox + octant path. It has no mature Python
  writer anyway, so "adopting" it would still mean writing the encoder —
  we write a simpler in-house variant instead and keep Potree 2.0 as the
  escape hatch if on-disk interchange is ever needed.

### D4 — Time filter is a vertex-shader cull; it also lands on *plain* (non-LOD) clouds

Per-point `birthTime` / `removalTime` float32 attributes + a `uTime` uniform,
patched into `THREE.PointsMaterial` via `onBeforeCompile`. Hidden points get
`gl_Position` shoved outside the clip volume (Potree's GPS-time-filter
trick) — clipped before rasterization, zero fragment cost, composes with
size attenuation and EDL. Driven by:

- `set_points_time(id, t)` (WS message, streaming mode), and
- a new `point_times` binary animation channel (stride 1, lerp/hold like
  `clip_times`) so the existing animation slider scrubs it.

This half of #79 is independent of LOD and ships first (Phase 1) — it
unblocks mill-sim at the 1–5M scale the flat path already handles.

### D5 — Generic builder: Morton sort + per-node pseudo-random, time-stratified sampling

- **Skeleton:** quantize positions to the root cube, compute Morton
  (Z-order) codes, one `np.argsort`, and every octree node at every level is
  a contiguous range of the sorted array (`code >> 3*(maxLevel-level)`).
  Same trick that makes PotreeConverter 2.0 fast (~6–9 M pts/s). ~100 lines
  of numpy. In-RAM comfort zone: 10M ≈ seconds, 100M ≈ tens of seconds /
  few GB; 1B needs out-of-core chunking by top Morton bits — that tier is
  explicitly a provider-side concern (D1, hook in Phase 3).
- **Sampling: pseudo-random (seeded), stratified over the time axis.**
  Decision per Thijs: random subsampling is perfectly valid for coarse
  LODs — wherever detail matters the user is zoomed into max LOD anyway.
  Blue-noise/Poisson-disk (Potree's default) buys evenly-spaced coarse
  levels, which matters mainly for hole-free *adaptive point size*; it's an
  optional later upgrade, not v1. **Time stratification is the non-optional
  part** (the literature gap): each node's sample must be spread across the
  node's lifetime span so that a time filter thins every LOD uniformly
  instead of punching patchy holes. Cheap in numpy: order candidates by a
  time-binned shuffle before taking the per-node quota.
- **Vertex order inside a node:** Morton-sorted, then shuffled in ~128-point
  batches — up to 5× faster raw `GL_POINTS` throughput (Schütz 2021, the
  one portable nugget from the compute-shader literature). Free at build
  time.

### D6 — Viewer runtime: budgeted priority traversal over per-node `THREE.Points`

Straight from the Potree playbook, in-file (no new deps), WebGL2:

- One `THREE.Points` per loaded node, all sharing one patched material.
  Explicit per-node bounding spheres (frustum culling per node for free).
- Traversal each frame (cheap, ~100s of nodes): visit nodes in descending
  projected-screen-size order (`px ≈ screenH · r / (d · tan(fov/2))`);
  refine while projected size ≥ threshold (~a few px … 16 px, tunable) and
  the **point budget** (default ~1–1.5M, tunable) is not exhausted.
- **Per-node time bounds** `[tMin, tMax)` in the hierarchy record: a node
  whose interval misses the current `uTime` is skipped entirely — makes
  material-removal scrubbing cheap and stops the budget being spent on
  fully-dead nodes.
- Nodes not yet loaded are fetched async (render what's loaded meanwhile —
  additive refinement means the coarse picture is always valid), stale
  fetches cancelled, evicted LRU past a memory cap.
- Adaptive point size: scale `gl_PointSize` by node spacing so coarse nodes
  draw fatter points (clamped), hiding density steps at LOD boundaries.
- Out of scope for the LOD path: picking (spine-based controller doesn't
  apply), `draw_range` (buffer order is Morton, a prefix is spatially
  meaningless — the time window replaces it; the `draw_ranges` applier
  no-ops on LOD clouds).

### D7 — Transport: extend the existing HTTP sidecar to on-demand node serving

Today the sidecar serves pre-registered whole blobs. LOD needs per-node
range fetches, and the ~1B tier needs *lazy synthesis*:

- `add_points(..., lod=True)` sends one WS message with metadata (root bbox,
  scale/offset, spacing, totals, material opts) + a URL for the binary
  hierarchy blob; node payloads are fetched as
  `GET <sidecar>/points/<id>/<node>` (or equivalent registered handler).
- For the ≤100M in-RAM tier the handler slices the builder's already-Morton-
  sorted arrays at request time (zero-copy views) — no 2× materialization.
- Phase 3 exposes the same handler as a public hook (`node_provider`
  callback) so a provider with a compact representation (mill-sim's grid /
  8³ bricks) can synthesize `{xyz f32, rgb u8, birth f32, removal f32}`
  blobs on demand without ever materializing full point arrays. This is
  what makes "1B source, 1M drawn" reachable: the full cloud never exists
  as arrays on either side.

Node payload encoding is specified in D8.

### D8 — Encoding: quantize everything, skip entropy codecs, keep ints on the GPU

From the encoding + GPU deep-dives (2026-07-03). The naive
`{f32 xyz, u8 rgb, f32 birth, f32 removal}` = 24 B/point is ~2× heavier than
it needs to be, and the localhost link inverts the usual calculus: wire
bytes are nearly free (loopback), so quantization is about **VRAM capacity**
(a deeper LRU node cache under the fixed draw budget) and parse cost — fps
at a 1M budget is governed by fill/overdraw and vertex order, not attribute
bytes (~1.7 GB/s fetched vs 200–1000 GB/s available).

- **Positions: normalized `Int16`, node-local** (6 B), with scale/offset
  baked into each node's `Object3D` matrix — the fixed-function stage
  dequantizes for free, zero shader changes, stock three.js
  (`Int16BufferAttribute` + `normalized: true`; the KHR_mesh_quantization
  pattern). 16 bits across a node cube is sub-voxel exact for any sane node
  size. Note this *improves on Potree 2.0*, which decodes int32→f32 in a CPU
  worker and uploads floats; we keep ints GPU-resident. (three.js caveats
  found: integer — non-normalized — attributes only exist at 32-bit; and
  `BatchedMesh`/`WEBGL_multi_draw` is mesh-only, no Points — so per-node
  `THREE.Points` stays the submission model, which is fine at ~100 calls.)
- **Colors: u8 RGB normalized** (3 B), as today.
- **Timestamps: u16 each, quantized over the cloud's `[t_min, t_max]`**
  (2+2 B; 65k distinct times — plenty for frame-indexed sim data; fall back
  to f32 via an opt-out if a provider needs more). Dequant via two uniforms
  in the patched shader. Never aggregated (D2) — quantized, not merged.
- **Total: 13 B/point GPU-resident** vs 24 naive (~1.8×), ~200 KB per
  15k-point node on the wire.
- **Vertex order inside each node: Morton-sorted, shuffled in 128-point
  batches** (Schütz 2021, verified: up to 4× peak / ~2.25× average raster
  throughput on hardware `GL_POINTS`, and it flattens pathological
  slow-viewpoint frames, 153 ms → 3.4 ms in their Retz overview). One-time
  numpy reorder at build.
- **Grid providers (Phase 3): occupancy-bitmask implicit coords on the
  wire.** When the source is a dense voxel grid, a node's geometry can ship
  as `brick origin + occupancy bitmask` (1 bit per cell — e.g. a 64-byte
  mask covers an 8³ brick, ~1–2 bits/point vs 48 with f32), with parallel
  SoA attribute streams in bitmask order. This is SimLOD's child-mask /
  SVO trick and it composes with D2 because **positions are a function of
  the grid cell, not of point identity** — only *attribute aggregation*
  corrupts the time scrubber, so coarse-level samples must be
  **sample-picked** (one real point snapped to its cell, carrying its true
  timestamps), never color/time-averaged. Decode on node load: expand the
  bitmask to the Int16 attribute layout above in JS (or later a worker) —
  in-shader vertex pulling of packed data is a capability, not a throughput
  win, and is deferred with the format that needs it.
- **Compression: none by default over loopback.** If/when nodes ever
  travel a real network: gzip per node via browser-native
  `DecompressionStream` (zero-dependency, cross-browser; Brotli in
  DecompressionStream is Chromium-only in 2026, zstd absent). Evidence says
  quantization + Morton/SoA ordering captures most of the win and the
  entropy coder is only the last ~2× (Draco vs gzip-on-quantized ≈ 2.4×);
  LAZ/Draco WASM decode CPU is the wrong trade on localhost. Rejected:
  sparse voxel DAGs / NanoVDB — their dedup/field tricks assume voxels have
  no per-point identity, which our timestamps are.

### Parameter defaults (from the surveys, all tunable via `lod={...}`)

| knob | default | notes |
|---|---|---|
| node sample size | ~15k points | Potree range 1k–50k; → ~70–100 draw calls at 1M budget |
| point budget | 1.5M | Potree ships 1–2M |
| refine threshold | ~10–16 px projected node size | Cesium SSE default is 16 px |
| hierarchy record | 32 B/node | Potree's 22 B + tMin/tMax f32; bounds implicit from octant path |
| LRU cap | ~300 MB GPU | eviction only past cap, LRU by last-rendered frame; at 13 B/pt ≈ 23M resident points |
| point payload | 13 B/point GPU (D8) | i16n xyz + u8 rgb + 2×u16 time; bitmask wire encoding for grid providers |

## Phasing

1. **Phase 1 — time window on plain `add_points`** (no LOD; independent,
   small): `birth_times=` / `removal_times=` kwargs (NaN/±inf = unbounded),
   shader patch, `set_points_time`, `point_times` animation channel +
   `Animation.set_point_time_data`, `Frame.point_times` JSON path, tests.
   Unblocks mill-sim at ≤~5M today.
2. **Phase 2 — octree LOD for the in-RAM tier (≤~100M)**: numpy Morton
   builder with time-stratified sampling, hierarchy blob + node endpoint on
   the sidecar, viewer traversal/budget/LRU/adaptive size, per-node time
   culling, D8 node payload (normalized Int16/u8/u16 attributes, shuffled-
   Morton vertex order). Example with tunable N showing LOD + slider-driven
   appear/disappear on one cloud.
3. **Phase 3 — the 1B tier + polish (as needed)**: public `node_provider`
   hook for lazy provider-side node synthesis, occupancy-bitmask wire
   encoding for grid providers (D8), optional blue-noise sampler, hierarchy
   chunking with proxy nodes (only matters past ~1M nodes), per-node gzip
   via `DecompressionStream` if nodes ever cross a real network.
   **WebGPU compute rasterization stays out of scope until WGSL gets 64-bit
   atomics** (gpuweb #5071, unshipped 2026) — the desktop technique's ~10×
   depends on 64-bit atomicMin; the browser-feasible 32-bit two-pass
   variant (TU Wien 2025 thesis: 136M flat points at 145 fps on RTX 3090)
   is a from-scratch WGSL build for a tier the octree already covers.
   WebGL2 + quantized attributes (Phase 2, D8) is the 2026 target and the
   buffer layout stays compute-friendly for the eventual drop-in.

## Open questions (resolved ones marked)

- ~~Node-endpoint shape~~ → resolved: per-node **callable** values in the
  existing blob store (`/points_lod_<uuid>/<i>` → lazy quantize+pack of a
  numpy slice); `_BlobHandler` calls callables on GET. Phase 3's
  `node_provider` hook is the same mechanism with user-supplied callables.
- ~~Builder location~~ → resolved: `src/threejs_viewer/points_lod.py`
  (matches `animation.py`/`toolpath.py` precedent).
- ~~`point_times` on `unload_animation`~~ → resolved: scrub time left
  untouched (documented).
- ~~EDL pre-pass~~ → resolved: same material instance (shared uniform by
  reference), filter applies in the pre-pass too.
- u16 timestamp quantization: needs an immortal-flag encoding (see status
  note). Phase 3.
- Builder throughput: ~1s per 1M points (BFS numpy masking + per-node
  lexsort). Fine to ~20M; a Morton-radix build is the known upgrade path
  for the 100M in-RAM ceiling.

## Prior-art pointers (from the 2026-07-03 surveys)

- Potree 2.0 format + PotreeConverter: Schütz, Ohrhallinger, Wimmer 2020,
  "Fast Out-of-Core Octree Generation for Massive Point Clouds", CGF 39(7),
  DOI 10.1111/cgf.14134; `OctreeLoader.js` in github.com/potree/potree
  (22-byte hierarchy records, proxy-node chunking, SoA node blobs).
- MNO: Scheiblauer & Wimmer 2011 (CAG), the additive-refinement structure.
- GPS-time shader cull precedent: `potree/src/materials/shaders/pointcloud.vs`
  (`clip_gps_enabled` → off-screen `gl_Position`).
- Vertex-order 5× win: Schütz, Kerbl, Wimmer 2021, arXiv 2104.07526.
- Continuous LOD (popping-free, VR): Schütz, Krösl, Wimmer 2019 — compute
  shader only, not WebGL2; borrowable idea: stochastic acceptance at node
  edges.
- SimLOD (GPU-built LOD while streaming, CUDA): arXiv 2310.03567 — UX
  pattern reference only.
- Renderers evaluated and rejected as deps: potree-core v2.0.15 (MIT,
  active, not no-build-embeddable), @pnext/three-loader v0.2.5 (stale
  since 2021), copc.js v0.0.7 + laz-perf WASM, 3DTilesRendererJS 0.4.x,
  CesiumJS.
- Python writers evaluated: copclib 2.6.3 (COPC only, LAZ required),
  py3dtiles v12 (file-oriented, 1.1 incomplete), tdamsma's
  python_potree_converter PoC. None fit ephemeral numpy → localhost.
- Time-varying/4D point-cloud LOD: no direct prior art found (2011–2026);
  time-stratified node sampling + per-node time bounds is our own synthesis.
- Encoding deep-dive (D8 sources): Potree 2.0 BROTLI mode = per-node
  Brotli over SoA + Morton-sorted int32 (decoded CPU-side in a worker —
  the part we deviate from); Schütz 2023 CGF 14877 (voxel inner nodes,
  64³ build bitmask, color-filtered vs sample-picked variants); SimLOD's
  hierarchical child-mask voxel encoding (~2 bits/voxel geometry +
  ~8 bits BC color); SVDAGs (Kämpe 2013) and NanoVDB rejected — dedup
  assumes no per-point identity; Draco quantization ladder (Draco ≈ 2.4×
  smaller than 16-bit-quantized+gzip — entropy coding is only the last
  2×); LASzip = per-chunk arithmetic coding, serial decode, wrong trade on
  loopback; 3D Tiles pnts `POSITION_QUANTIZED` (u16 + volume offset/scale)
  = the same node-local 16-bit scheme we adopt.
- GPU deep-dive sources: three.js normalized Int16/Uint8 attributes
  (KHR_mesh_quantization pattern; integer attrs 32-bit only, issue #21595;
  BatchedMesh has no Points support, issue #29018); Schütz 2021 vertex-order
  numbers; Bauer 2025 TU Wien thesis (WebGPU 32-bit two-pass compute
  rasterizer in-browser); gpuweb #5071 (64-bit atomics, unshipped).
