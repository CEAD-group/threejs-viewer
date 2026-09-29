# Shadow lab

Run `python3 -m http.server 8765 --bind 127.0.0.1` from the repository root,
then open http://127.0.0.1:8765/bench/shadow_lab/ in Chrome.
Rebuild the bundled viewer after source changes with
`.venv/bin/python src/threejs_viewer/viewer/build.py`.

The scene uses the real bundled viewer, including its environment and shadow
camera fitting. It contains an open double-sided panel, angled closed panels,
thin shelves, smooth metal cylinders, a sphere, and a box touching a shelf.
A large floor deliberately spreads the shadow map over more than the close-up.
Lighting matches the reported artifact: sun 5.7, azimuth -46°, elevation 45°,
ambient 0.5, environment 1.1, ACES exposure 1.9.

Choose `baseline` for the old radius 4 and `production` for the current build.
Other presets compare radius 1, radius 2, stronger normal bias, a 4096² map,
and shadow receiving disabled while retaining sunlight. Orbit/zoom normally.
After changing the sun, reselect a preset to reapply its experimental bias.

## Before / after

Identical camera, geometry, lighting, map size, and bias; only the PCF radius
changes. These are direct Chrome screenshots, with matching detail captures.

| Before: radius 4 | After: radius 1 |
| --- | --- |
| ![Stippled self-shadowing on the open panel](https://github.com/CEAD-group/threejs-viewer/releases/download/pr-245-images/before-detail.png) | ![Smooth panel with sharper cast-shadow edges](https://github.com/CEAD-group/threejs-viewer/releases/download/pr-245-images/after-detail.png) |

Full scene: [before](https://github.com/CEAD-group/threejs-viewer/releases/download/pr-245-images/before.png) · [after](https://github.com/CEAD-group/threejs-viewer/releases/download/pr-245-images/after.png).

## Findings — 2026-09-29

**Use PCF radius 1, retaining the 2048² map and existing bias.** This removes
the reproduced surface stippling without increasing map memory, filter sample
count, or shadow refresh frequency. Edges are sharper; this is a deliberate
quality tradeoff, not physically accurate soft sunlight.

`shadowLab.acneScore(name, azimuth, elevation)` hides other objects and compares
2,548 interior panel samples to the same panel with shadow receiving disabled.
The full-scene shadow coverage stays fixed. A sample is falsely dark if its mean
RGB loses more than 3 levels out of 255. This measures self-shadowing, not valid
occlusion from other objects. At azimuth -46°:

| Preset | Falsely dark at elevation 45° | Worst over elevations 5/15/30/45/60/80° |
| --- | ---: | ---: |
| Original: 2048², radius 4 | 79.1% | 97.2% |
| 2048², radius 1 | 0% | 0% |
| 2048², radius 2 | 1.1% | 35.9% |
| Radius 4, normal bias 4 texels | 0% | 0.6% |
| 4096², radius 4 | 55.7% | 95.1% |

Radius 1 also returned zero samples over threshold in a sweep of azimuths
-130/-90/-46/0/45° × elevations 5/30/60/80°. Some differences of up to 2 RGB
levels remain at grazing angles. This is a synthetic reproduction; the original
CAD model was not available, so mesh-normal/faceting issues remain outside scope.

Increasing bias preserves the wide filter's patterned edges and increases the
surface offset from about 0.021 to 0.056 scene units, risking detached contacts.
Increasing resolution alone does not solve the filter/bias mismatch.
`PCFSoftShadowMap` was also tried; Three.js 0.183.2 warns that it is deprecated
and falls back to PCF, so it is not an independent candidate.

## GPU measurements

Chrome 154, Apple M1 Pro, ANGLE Metal, DPR 1, 1320 × 890 render area.
`EXT_disjoint_timer_query_webgl2` measures GPU elapsed time for five renders per
sample, with 15 warmup samples and 50 measured samples. Three passes alternate
preset order. Table entries are medians of the three per-pass medians.

| Preset | Cached shadows, GPU ms/render | Refresh shadow map every render |
| --- | ---: | ---: |
| Original radius 4 / 2048² | 0.871 | 1.321 |
| Radius 1 / 2048² | 0.874 | 1.295 |
| Radius 4 / 4096² | 0.846 | 1.852 |

Radius 1 shows no meaningful performance penalty in this scene. The 4096²
refresh costs about 0.53 ms more (40%) and quadruples shadow-map texel storage.
These are isolated render timings on a small scene, not whole-app FPS or a
guarantee for large CAD assemblies. Individual samples were noisy; raw results
and per-pass p90s are in `results.json`.

The UI measures cached and refreshed shadows for the selected preset. Console:

```js
await shadowLab.benchmark('narrow', false, 50)
await shadowLab.benchmark('narrow', true, 50)
shadowLab.acneScore('narrow', -46, 45)
```

The regression test uses this same scene, verifies that radius 4 reproduces
the defect, and checks production settings against the shadow-free reference:

```sh
.venv/bin/python -m pytest tests/test_shadow_lab.py tests/test_browser.py \
  -k 'shadow or sun_casts' --browser-channel chrome -q
```
