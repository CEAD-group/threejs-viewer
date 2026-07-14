# Parametric tube corner quality — consolidated findings

Status: 2026-07-14. This document consolidates the full history of the
tube-corner / `strand_collapse` investigation (issues #41–#56, #107, #111,
#117–#119, and the ongoing curvature-adaptive-resampler work) into one place.
It supersedes `tmp/tube_lab/REPORT.md` and `tmp/tube_lab/REPORT_119.md`
(kept as scratch, safe to delete — see `tmp/tube_lab/HANDOVER.md`). For how
to run the live bench, see `README.md` in this directory.

Where an earlier report's recommendation was later overridden by review or
further probing, this doc calls that out explicitly rather than silently
picking a side — see "Chronology" and the box in "Current shipped behavior".

## 1. Current shipped behavior (verified against `src/threejs_viewer/viewer/viewer.js`, 2026-07-14)

Detector / snap constants inside `collapseTubeStrandFolds` (viewer.js:1490)
and its module-level header (viewer.js:1470–1477, 1580):

| constant | value | file:line | meaning |
|---|---|---|---|
| `TUBE_STRAND_COLLAPSE_MIN_GAP` | 4 | viewer.js:1476 | fold pair minimum span j−i |
| `TUBE_STRAND_COLLAPSE_WIN` | 50 | viewer.js:1475 | fold pair maximum span j−i |
| `TUBE_STRAND_COLLAPSE_TOL_FRAC` | 0.04 | viewer.js:1477 | un-mitered strand seg-seg self-touch tolerance, as a fraction of tube-global `max(W,H)` |
| `FOLD_SEP_FACTOR` | 0.5 | viewer.js:1556 | spine fold-back reject cap: `‖spine[j]−spine[i]‖ ≤ FOLD_SEP_FACTOR·min(dimI,dimJ)` |
| `MIN_FOLD_PEAK_TURN` | 8° (`(8·π)/180`) | viewer.js:1611 | minimum per-vertex turn inside the fold span to accept it as a real corner (not bead-width swelling) |
| `MAX_SNAP_FACTOR_DEFAULT` | **0.5** | viewer.js:1580 | per-ring snap cap, in bead-widths, applied against the *mitered* baseline |
| `LARGE_SEG_FACTOR_DEFAULT` | 1.0 | viewer.js:1504 | exempt rings whose shorter adjacent spine segment ≥ this many bead-widths (open-straight exemption, #119) |
| `TUBE_MITER_LIMIT` | **2** | viewer.js:1051 | turns sharper than ~120° bevel instead of growing an ever-taller miter blade |
| `MAX_TUBE_INDICES_PER_DRAW` | 24,000,000 | viewer.js:3399 | per-`drawElements` index cap (conservative vs. the ~30M ANGLE/Firefox ceiling); tubes over this are split via `geometry.addGroup` chunks (`applyTubeDrawCap`) |

**Note on `MAX_SNAP_FACTOR_DEFAULT`:** `tmp/tube_lab/REPORT_119.md` (written
2026-07-14 morning, commit `6cfe896`) explicitly recommended *keeping*
`max_snap_factor = 1.0` ("do NOT lower the default") based on the #119
harness (`hairpin_dense` at 0.5 showed "crease weakening", at 0.25 "under-
collapse"). That recommendation was **superseded the same day** by the
review commit `640aed0`, which found 1.0 over-snaps tight/messy folds
(spiky-cusp + z-fighting seam, over-snap spikes shooting along the adjacent
straight on jittered reversals) and lowered the default to 0.5 after
validating against a wider probe set (tight hairpin, the real #50
ribweaver-bulb apexes at 0.00% pixel diff vs. the old 1.0-tuned baseline, a
sharp jittered ±120° reversal, a rounded jittered S-curve). **0.5 is what
ships and what the code contains today** — treat REPORT_119.md's "keep 1.0"
line as historical, not current guidance.

Two PR branches produced this state:

- **`fix/tube-frame-collapse`** (#117/#118/#119, tip `640aed0`): degenerate
  spine-segment tangent fix, LOD zoom-gate rework, strand_collapse
  large-segment exemption, and the 1.0→0.5 snap-default review fix.
- **`fix/tube-index-cap`** (#113/#114, tip `d7c4dcd`): `applyTubeDrawCap` /
  `MAX_TUBE_INDICES_PER_DRAW`, layered on top of the above (its tip commit
  `d7c4dcd` is one ahead of `640aed0`).

## 2. Chronology of tuning tweaks

| date | commit(s) | change | why |
|---|---|---|---|
| 2026-04-30 | `f38a2fd` (#43) | `strand_collapse=True` introduced: per-cross-section-vertex seg-seg local-min fold detection (tol 4% of `max(W,H)`) + snap to mitered midpoint | inner offset surface self-intersects where curvature κ exceeds 1/half-width |
| 2026-05-19 | `dcce922` (#49) | Reject cross-link fold targets: spine separation cap `‖spine[j]−spine[i]‖ ≤ 0.5·min(W,H)` + `MIN_FOLD_PEAK_TURN` (8°) peak-turn gate; also fixed sepSq guard ordering (ran after, not before, the local-min sweep) | long-span strand pairs from *different* corners were coincidentally pairing up and zipping a flat diamond membrane across multi-corner features |
| 2026-05-19 | `a9b07ac` (#50) | 4-phase clustering (union-find across cross-strand targets, range-adjacency merge, range extension, cross-section neighbour expansion) + count-weighted averaged apex snap; 2-strand wedge guard | wide beads (W/H ≈ 7+) left a "cube + spokes" cluster of distinct apex vertices at a V-crease instead of one clean apex |
| 2026-05-20 | `9cecc9b` (#51) | Snap-distance cap `max_snap_factor` (default **1.0** at introduction), two-granularity guard (cluster fast-reject + per-ring vs. mitered baseline); `S`-key global toggle | densely-packed real toolpaths could pair offset strands whose seg-seg midpoint landed tens of bead-widths from the spine (16–26% of mesh vertices yanked, peak 138 mm on `tube_c34f9e39`) |
| 2026-06-10 | `6101efe` (#56) | Directional miter (stride-3 `[scale, mu, mv]`, stretches along actual turn direction instead of always U), cone-run frame freeze (`|T·up| > 0.99` inherits neighbour frame), `TUBE_MITER_LIMIT` 4→2, deposition-order bias (`TUBE_DEPOSITION_BIAS = 1e-3`) for retraces, LOD-worker ring-code dedup via `toString()` injection | fixed the corner/reversal artifact family from #41–#51: 1.40× lateral wing flare on vertical elbows, crumpled risers, 3.86× miter blades past 120°, exact-retrace z-fighting |
| 2026-07-08 | `3b1c898` (#107/#108) | `break_before` mask + flat fan caps, byte-identical wire format when unused | interior travel hops rendered as a stray bridging cone instead of two disconnected strips |
| 2026-07-09 | `5503fb7`/`9b77127` (#111/#112) | rAF hover coalescing + `max_pick_points` decimation for picking | O(N) per-move pick scan dominated frame budget on multi-million-point spines |
| 2026-07-14 | `6cfe896` (#117/#118/#119) | (#117) degenerate zero-length spine segments no longer snap tangent to +X — seed scans forward, rolling update carries previous direction through the degenerate segment; (#118) LOD skip-gate uses camera-to-bounding-sphere-**surface** distance instead of an absolute model-size threshold; (#119) `large_seg_factor` exemption (default 1.0) added, plus fixed a latent bug where the non-LOD path dropped `largeSegFactor` from the reconstructed config | duplicate-vertex bends rendered as fold-fan shards; close-in zoom never re-triggered LOD refinement (stale coarse tube); false snaps fired on long/clean segments near unrelated micro-segments |
| 2026-07-14 | `640aed0` | `max_snap_factor` default **1.0 → 0.5**; LOD gate scales bounding-sphere radius by max world-axis scale + reuses scratch vectors; skip per-segment sqrt when exemption disabled; dropped `examples/29_messy_toolpath_winddown.py` from the PR | 1.0 over-snapped tight/messy folds (spiky cusp, z-fighting seam, spikes along adjacent straights); 0.5 validated as the sweet spot across 4 scenes |
| 2026-07-14 | `d7c4dcd` (#113/#114) | `applyTubeDrawCap`, `MAX_TUBE_INDICES_PER_DRAW = 24_000_000`, `geometry.addGroup` chunking on both non-LOD and LOD-worker paths | GPU silently truncates index buffers past the WebGL per-draw cap (~30M on Firefox/ANGLE) with no error, only a console warning — tails of >833k-ring-pair tubes vanished |

## 3. Bench findings

### 3.1 Where `strand_collapse` fires vs. abstains (root scenario sweep, `tmp/tube_lab/` probes)

Verified across `strand119.py` (7 scenes × 6 variants) and the earlier
`REPORT.md` 8-scenario capture (~290 screenshots, z-fight + shimmer probes):

- **Fires**: tight wipe loops / dense hairpins (genuine self-touch, short
  segments — `hairpin_dense` in `strand119.py`), and the realistic
  decimated+jittered zigzag (`zigzag_rounded.py` in the current bench —
  collapse moves ~34 rings).
- **Abstains (correctly)**: sawtooth, retrace, arc sweeps — none fired in
  the `corner_sweep.py`/`zigzag_jitter.py`/original `REPORT.md` corner
  sweeps. Clean geometric corners (sharp V, smooth U, uniform tight arc)
  never hit the 0.04·`max(W,H)` self-touch tolerance, so they're handled by
  the miter alone, not collapse.
- **Structural no-op sites** (from `REPORT.md`, pre-#117/#119 fixes,
  hypothesis-level at the time): layer-up hop corners, exact A→B→A
  retraces, and turns ≥150° were all byte-identical collapse on/off. The
  hypothesized cause was the fold-target separation cap (`FOLD_SEP_FACTOR`,
  then 0.5·min(W,H)) rejecting layer-hop-scale separations, and the old
  snap cap being too tight for a heavily-mitered vertex. The #117 frame-
  tangent fix and #56 directional-miter/bevel changes address the
  underlying geometry at these sites directly (rather than making collapse
  engage there) — layer hops and retraces are now handled by frame
  continuity + deposition bias, not by strand_collapse.

### 3.2 `strand_collapse` snap-factor sweep (`strand119.py`, #119 harness)

Metric: per-vertex displacement of the collapsed mesh from its mitered
(pre-collapse) baseline, on `hairpin_dense` (tight cusp, short segments):

| variant | rings moved >2mm | max mm | note |
|---|---|---|---|
| exemption OFF, snap 1.0 | 168 | 7.93 | old (pre-#119) behaviour |
| exemption ON (default), snap 1.0 | 168 | 7.93 | **identical — exemption inert on a genuine dense fold** (all segments < 1 bead-width, nothing qualifies for exemption) |
| snap 0.5 | 85 | 3.98 | crease weakening |
| snap 0.25 | 2 | 2.00 | fold barely creased — under-collapse |

`large_seg_factor` exemption-path confirmation: probing with
`large_seg_factor = 0.02` (threshold 0.16mm, below the cusp's 0.33mm
segments) on `hairpin_dense` → 0 rings moved, confirming the exemption
genuinely skips snaps once a ring's segments exceed the threshold.

All other #119 scenes (`wipe_loop`, `parallel_passes`, `accordion`,
`raster_turns`, the real `winddown` G-code snippet) showed **0 deviation
for every variant** — `FOLD_SEP`/peak-turn guards plus the #117 tangent fix
already keep collapse off long/clean segments; the exemption is
defense-in-depth, not the primary mechanism, on this data.

As noted in §1, the #119 harness's own conclusion ("keep 1.0") was
superseded by the broader post-review probe set in `640aed0` that also
covered non-#119 scenes (the #50 ribweaver bulb, jittered ±120° reversals)
where 1.0 over-snapped. 0.5 is a compromise across both problem sets, not
a re-run of the #119 numbers above at a different value.

### 3.3 Corner-density / straight-bleed sweep (`corner_sweep.py`, `zigzag_jitter.py`)

Later same-day probes (after `REPORT_119.md`, before the bench move) asking
whether collapse's moved-ring range bleeds from a dense corner onto an
adjacent long straight, at the shipped default (`max_snap_factor=0.5`,
exemption ON) vs. exemption OFF:

- `corner_sweep.py`: sweeps arc sampling density (`n_arc` = 6..30, i.e.
  90°/(n_arc−1) per-vertex turn from 18° down to 3.1°) on a smooth 90°
  quarter-arc corner between two long straights. Reports `moved` (rings
  displaced anywhere) vs. `straight` (moved rings **outside** the arc's own
  ring range — the bleed signal) for exemption ON vs. OFF.
- `zigzag_jitter.py`: long sparse straight → dense irregularly-sampled
  jittered ±120°/−120° zigzag → long straight. Same `moved`/`straight`
  probe, plus a screenshot at the dense region. Directly the harness the
  current bench's `zigzag_rounded.py` scene generalizes.

(Numeric output from these two scripts is printed to stdout only, not
persisted to a file in `tmp/tube_lab/` — re-run them if exact bleed counts
are needed; they are read-only-safe scratch scripts per their docstrings.)

### 3.4 Current `bench/tube_lab` (`zr_metric.py`) quality-measure results

Three measures (intersecting-line dark pixels, IoU vs. OFF baseline,
triangle count) across three scenes (zigzag, sharpV, hairpin) and several
approaches (spine identity / spline resample / linear densify, crossed with
`strand_collapse` off / 0.25 / 0.5 / 1.0):

- **zigzag** (dense, decimated, jittered ±120° turns, +40% Hann-windowed
  bead bump): `strand_collapse` at 0.5 cuts intersecting-line pixels to
  **76%** of the OFF baseline at **IoU 0.998**, same triangle count.
  `sc0.5 == sc1.0` everywhere on this scene → **shipping 0.5 is already at
  its plateau** for this quality measure (raising it further buys nothing
  here, consistent with §3.2's per-vertex-displacement numbers showing 0.5
  and 1.0 diverge only on `hairpin_dense`, not this scene).
- **Corner resampling (Catmull-Rom spline) is a triangle AND line win on
  curved corners**: zigzag 924→552 triangles, lines to 51% at IoU 0.991;
  hairpin 360→264 triangles, 41% lines. Linear (collinear) densification
  does nothing on either measure.
- **Corner resampling is catastrophic on a genuine sharp vertex**: sharpV
  scene IoU drops to **0.60** when splined — rounding a real corner
  destroys the intended silhouette. This is the scene-design rationale for
  keeping spline and collapse scoped to curved/dense regions only.
- **sharpV** interior angle was changed from 35° to **90°** during scene
  design: 35° was rejected as an invalid baseline because the extreme apex
  miter at that angle mangles the effective leg width before any
  resampling/collapse treatment is even applied, confounding the
  measurement. 90° keeps constant-width legs and sits under the
  `TUBE_MITER_LIMIT = 2` (≈120°) bevel threshold, so miter behavior itself
  isn't the thing under test. `strand_collapse` correctly abstains on
  sharpV.
- **hairpin** (tight 180° U, R=2, W=12): `strand_collapse` correctly
  abstains — smooth arc, offset crosses wide, no cusp self-touch.

(These numbers are quoted as reported to this audit; `zr_metric.py` /
`zigzag_rounded.py` are owned by a concurrent bench-design effort and were
not re-run for this document — see README.md for how to reproduce.)

## 4. Rejected / dead-end approaches

- **Blind spline resampling on sharp vertices** — rounds away the intended
  silhouette (sharpV IoU→0.60, §3.4). Any curvature-adaptive resampler must
  detect and pin genuine sharp vertices rather than splining uniformly.
- **Linear (collinear) densification** — adds points without changing the
  rendered picture; no triangle, line, or IoU benefit on any scene tested.
  Ruled out as a mechanism for either quality axis.
- **`max_snap_factor` below 0.5** (0.25) — under-shoots real wide-bead
  corners: strands stop short of the apex, leaving a protruding wedge
  (measured against the #50 ribweaver-bulb tuned baseline). Also under-
  collapses `hairpin_dense` (168→2 rings moved, §3.2).
- **`max_snap_factor = 1.0`** (the original #51/#119 default) — over-snaps
  tight/messy folds: spiky-cusp + z-fighting seam, and over-snap spikes
  shooting out along the adjacent straight on jittered reversals (found in
  the `640aed0` review, after `REPORT_119.md` had recommended keeping it).
- **Subdivide-long-segments before collapse** (`large_seg_factor=0` +
  pre-subdivision, `strand119.py`'s `subdiv_noexempt` variant) — no better
  than the exemption on genuine folds (hairpin 153 vs. 168 rings, still a
  weaker crease than the shipped default), no benefit on long-segment
  scenes (nothing to fix there either), and inflates point count. The
  exemption (skip snapping, no geometry change) was judged cleaner and
  cheaper.
- **Sawtooth / retrace / arc corner sweeps as collapse triggers** — swept
  across configurations in the original `REPORT.md` capture and again in
  `corner_sweep.py`; none fire collapse. Confirms collapse is scoped to
  genuine dense self-touching corners by design, not a general corner
  smoother — the miter/bevel/frame-continuity fixes (#56, #117, #118) are
  what handle the rest of the corner space.
- **Depth-bias / polygonOffset for stacked or adjacent beads** — `REPORT.md`
  measured these as non-issues (0 flipped pixels under near-plane
  perturbation everywhere except exact retraces and dense interior
  self-overlap), so no fix was invested here. Retraces are handled instead
  by the deposition-order bias (#56), not a depth-bias trick (admitted
  in-code as a topology problem, not a z-fighting one — DoubleSide, single
  geometry, exact overlap).

## 5. Open questions / next steps

- **Curvature-adaptive resampler** — PROTOTYPED (2026-07-14, bench-only:
  `adaptive(step=3.0, pin_deg=35°, dense_factor=0.75)` in `zr_metric.py`).
  Pins vertices with turn ≥ 35° (plus region endpoints) as hard span
  boundaries, rounds only dense spans (seg < 0.75×median bead width) with
  clamped Catmull-Rom, passes sparse spans through byte-for-byte. Results:
  zigzag 564 tris / darkPx 53% of off / IoU 0.9911 (≈ spline3's win);
  sharpV **byte-for-byte identical to off** (spline3: IoU 0.60); hairpin
  276 tris / darkPx 70% / IoU 0.9833 (beats spline3's 0.9645; absolute
  0.99 unreachable at step 3 on an R=2/W=12 U — inherent, needs a
  curvature-scaled step); mixed IoU 0.9979 / tris halved / darkPx 57% —
  **strictly dominates spline3** (0.9784 / 456 / 71%). `adaptive3+sc0.5`
  == `adaptive3` on every scene: collapse abstains on the resampled spine,
  so the two tools don't interfere.
- **Mixed / combined scene** — ADDED to `SCENES` (2026-07-14):
  `scene_mixed()`, one tube with a sparse 90° corner + a dense 120°
  rounded turn. `corner_sweep.py` / `zigzag_jitter.py` still carry a
  numeric `straight` bleed-count probe that could be ported to the
  three-measure framework if bleed containment needs a number.
- **Promotion path to `viewer.js`** — none of the resampling/spline work is
  wired into the shipped `add_parametric_tube` path yet; it lives only in
  the bench harness (`zigzag_rounded.py`'s spine transforms). Shipping it
  would need: (1) a decision on whether it runs client-side (like
  `strand_collapse`, on the LOD worker) or is a pre-processing step in
  `client.py`; (2) the sharp-vertex-pinning logic promoted from a bench
  scene-design constraint into an actual detector; (3) a regression scene
  set covering the "rejected" cases in §4 so a shipped version can't
  regress into blind splining.
- **`corner_sweep.py` / `zigzag_jitter.py` numeric results are not
  persisted** (§3.3) — worth re-running and capturing output if they're
  going to inform the resampler's bleed-containment design, rather than
  relying on this document's qualitative summary of what they probe.
