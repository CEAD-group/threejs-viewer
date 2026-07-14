# tube_lab — tube-corner quality bench

Objectively scores tube-corner rendering approaches (spine transforms ×
`strand_collapse` settings) on **three quality measures**, so alternatives can be
compared against the shipping `strand_collapse` without eyeballing. Grew out of
the "rounded ±120 zig-zag" investigation (issues #117/#118/#119, PRs
`fix/tube-frame-collapse` / `fix/tube-index-cap`); formerly lived in
`tmp/tube_lab/`.

## The three measures

All read straight off the WebGL buffer via `gl.readPixels` (no PNG-decode
dependency — PIL/imageio/cv2 are not in the env):

1. **intersecting lines → `darkPx`**: count of interior dark EDGE pixels in the
   wireframe-overlay render. Fold-fans pile overlapping edges ⇒ lower = cleaner.
2. **contour fidelity → `IoU`**: intersection-over-union of the blue footprint
   (solid render, non-background pixels) vs the scene's OFF baseline. ~1.0 =
   "same pixels coloured blue".
3. **triangle count → `tris`**: fewer triangles at equal (1)&(2) is itself a win
   (lighter mesh, same picture). Read from `geometry.index.count/3`.

## Running

```bash
uv sync --group browser-test && uv run playwright install chromium   # once
uv run python bench/tube_lab/zr_metric.py
```

One headless browser session; prints a table per scene and dumps
`out/zrm_<scene>_<approach>.png` (wireframe overlay) plus
`out/zrm_<scene>_LINE.png` (the raw input spine as a polyline). `out/` is
gitignored.

The run ends with a **success-gate summary**: per scene, an approach passes
iff (vs that scene's `off` baseline) `IoU >= 0.99` **and** `tris <= off`
**and** `darkPx <= off` — i.e. it holds the contour without adding triangles
or intersecting lines. This is the acceptance bar a candidate tube-corner
approach must clear on **every** scene before promotion to `viewer.js` is
worth discussing. The bar is deliberately strict: an approach can beat the
reference on a scene and still fail the absolute IoU line (e.g. `adaptive3`
on hairpin, IoU 0.983 — better than `spline3`'s 0.965, still a fail; closing
it needs a curvature-scaled step).

## Files

- `zr_metric.py` — THE bench. Scene registry × approach list, per-scene
  top-view camera.
- `zigzag_rounded.py` — the zigzag scene builder + `widths_for` (imported by
  the bench). Standalone run produces the `out/zr_R6_*` panels
  (input line / off / 0.5 / 1.0). 80%-decimation + uneven gaps + smooth
  Hann-windowed +40% width bump in the turn.
- `FINDINGS.md` — consolidated findings/history from the investigation.

## Scenes (each a distinct regime)

- **zigzag** — dense, decimated, jittered ±120° turns with a +40% bead bump.
  **collapse FIRES** (moves ~34 rings). The realistic messy case; the only one
  where `strand_collapse` does anything. Primary positive.
- **sharpV** — 3-point corner, interior angle **90°** (35° was rejected: the
  extreme apex miter mangles the effective leg width — 90° keeps constant-W
  legs, turn 90° < 120° miter/bevel limit). **collapse ABSTAINS** correctly.
  Guard scene: rounding it (spline) wrecks all three measures (IoU→0.60).
- **hairpin** — tight 180° U (R=2, W=12). **collapse ABSTAINS** (smooth arc,
  offset crosses wide, no cusp self-touch). Correct.
- **mixed** — ONE tube with both a sparse 90° sharp corner AND a dense 120°
  rounded turn, so an adaptive treatment must discriminate within a single
  object (pin the corner, round the arc).

## How to extend

- New scene: add a `scene_*()` returning
  `dict(sp, w, ht, lo, hi, cx, cy, ext)` and append to `SCENES`.
- New approach: add `(label, spine_transform, strand_collapse_arg)` to
  `APPROACHES`. A spine transform is
  `fn(sp, w, ht, lo, hi) -> (sp, w, ht, lo, hi)`; `spline(step)` /
  `lindens(step)` / `adaptive(step, pin_deg, dense_factor)` / `_identity`
  exist. `adaptive` is the curvature-adaptive resampler: pins vertices whose
  turn ≥ `pin_deg` (and region endpoints) as hard boundaries, splits the
  region into uniformly dense/sparse spans (dense = seg < `dense_factor`
  × median bead width), rounds dense spans with clamped Catmull-Rom at
  `step` spacing, passes sparse spans through byte-for-byte.
