# Shadow lab

Run `python3 -m http.server 8765 --bind 127.0.0.1` from the repository root,
then open http://127.0.0.1:8765/bench/shadow_lab/ in Chrome.
Rebuild after source changes with
`.venv/bin/python src/threejs_viewer/viewer/build.py`.

The scene uses the bundled viewer and its environment/shadow-camera fitting.
It includes an open double-sided panel, angled closed panels, thin shelves,
metal cylinders, a sphere, and a box touching a shelf. A large floor spreads
shadow-map coverage beyond the close-up. Lighting matches the reported artifact:
sun 5.7, azimuth -46°, elevation 45°, ambient 0.5, environment 1.1, ACES exposure 1.9.

The initial `baseline` uses the original radius-4 stock filter. Select
`production` for the soft 5×5 tent with receiver-plane depth correction,
or `narrow` for the earlier radius-1 fix. Other presets compare stock radius 2,
stronger bias, a 4096² map, and receiving disabled while retaining sunlight.
Orbit/zoom normally. After changing the sun, reselect a preset to restore its
experimental bias. The production kernel has a fixed support; its radius value
is descriptive, not a variable blur control.

## Before / after

Direct Chrome screenshots, identical scene and camera. Images are release assets,
not source files.

| Original surface stippling | Current soft filter |
| --- | --- |
| ![Before](https://github.com/CEAD-group/threejs-viewer/releases/download/pr-245-images/before-detail.png) | ![After](https://github.com/CEAD-group/threejs-viewer/releases/download/pr-245-images/soft-after-detail.png) |

| Earlier radius-1 fix: harsh edge | Current 5×5 tent: softer edge |
| --- | --- |
| ![Hard](https://github.com/CEAD-group/threejs-viewer/releases/download/pr-245-images/hard-edge.png) | ![Soft](https://github.com/CEAD-group/threejs-viewer/releases/download/pr-245-images/soft-edge.png) |

Full scene: [original](https://github.com/CEAD-group/threejs-viewer/releases/download/pr-245-images/before.png) · [current](https://github.com/CEAD-group/threejs-viewer/releases/download/pr-245-images/soft-after.png).

## Current approach — 2026-09-29

Use a separable **5×5 tent filter**, with receiver-plane depth correction at each
sample. Pairing adjacent texels into hardware bilinear comparisons requires
**9 texture reads**, versus the stock filter's 5. Fractional weights follow the
texel grid, avoiding sparse sampling gaps and randomized per-pixel rotations.
Depth derivatives estimate the receiving surface's depth at each offset; the
comparison therefore follows a sloping panel instead of shadowing it against
its own neighbouring depths. Degenerate projections fall back to zero slope,
and correction is capped to 1% of normalized shadow depth.

The 2048² map, existing depth/normal bias, and dirty-only refresh policy remain.
No extra render pass, render target, or dependency. A material-local compile hook
preserves existing callbacks/program keys and patches directional-light reads
only. Global Three.js shader chunks and point/spot filters are untouched.
Material clones and clay/debug switches receive the patch independently.
Non-PCF map types retain the built-in fallback.

This is fixed-width soft filtering, not contact-hardening area-light shadows.
It is less broad than the old noisy radius-4 disk but smoother and visibly softer
than radius 1. Custom shader integration is additional maintenance when upgrading
Three.js; the pixel and material-compatibility regressions cover that boundary.

## Quality

`acneScore(name, azimuth, elevation)` hides other geometry, retaining full-scene
map coverage, and compares 2,548 interior panel samples to a shadow-free reference.
A falsely dark sample loses more than 3 mean RGB levels out of 255.

- Original radius 4 at -46°/45°: **79.1%** falsely dark samples.
- Current filter: **0%** at all 30 combinations of azimuth -130/-90/-46/0/45°
  and elevation 5/15/30/45/60/80°. Maximum mean loss is 0.061 RGB levels.
- `edgeScore(name)` measures the 20–80% transition across 17 scanlines on the
  back panel. Mean width grows from **0.0176 to 0.0330 scene units** versus
  radius 1: **1.88× wider**. The original noisy filter measured 0.0654.

These checks use a synthetic reproduction. The original CAD model was not
available; mesh-normal/faceting defects in it remain outside scope.

## GPU cost

Chrome 154 / Apple M1 Pro / ANGLE Metal, DPR 1, 1320×890 render area.
`EXT_disjoint_timer_query_webgl2`, five renders per sample, 15 warmup and 50
measured samples per pass; three passes alternate preset order. Entries are
medians of the three pass medians, in GPU milliseconds per render.

| Filter | Cached shadows | Refresh every render |
| --- | ---: | ---: |
| Original radius 4 | 1.406 | 1.634 |
| Earlier radius 1 | 1.407 | 1.616 |
| Current soft tent | 1.441 | 1.631 |

The soft filter adds about **0.035 ms / 2.5%** with cached shadows in this scene.
Refresh timings overlap. This is GPU render time, not whole-app FPS; large CAD
models and other GPUs can differ. There is no added shadow-map storage. Raw
current timings, quality results, and edge widths are in `soft-results.json`.

Alternatives tested:

- A 7×7 tent (16 reads) looked broader but cost about 0.32 ms more than stock
  in a single paired prototype run, versus about 0.04 ms for the 5×5 tent.
- Corrected sparse disk filters remove acne but retain uneven edge transitions.
- Stronger normal bias reduces acne but risks detaching contacts.
- VSM softens edges but introduced visible marks around contacts in this scene.
- 4096² alone does not solve the filter/depth mismatch, uses 4× map texel storage,
  and cost about 0.53 ms more when refreshing in the earlier experiment.
- `PCFSoftShadowMap` in Three.js 0.183.2 falls back to PCF with a deprecation warning.

`results.json` preserves the earlier radius-only experiment. Its absolute timings
were recorded in a different session; compare paired settings within each run,
not absolute times across sessions.

## Reproduce measurements and tests

The UI measures cached and refreshed GPU time. Console:

```js
await shadowLab.benchmark('production', false, 50)
await shadowLab.benchmark('production', true, 50)
shadowLab.acneScore('production', -46, 45)
shadowLab.edgeScore('production')
```

```sh
.venv/bin/python -m pytest tests/test_shadow_lab.py tests/test_browser.py tests/test_build.py \
  -k 'shadow or sun_casts or build or display_quality or n_key_cycles_shading or tone_mapping_change' \
  --browser-channel chrome -q
```

17 tests passed. The lab regression proves the original artifact, verifies the
soft filter's clean surface and wider transition, and checks existing material
callbacks, custom cache keys, clones, global chunk isolation, and non-PCF fallback.
