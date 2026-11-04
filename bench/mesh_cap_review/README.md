# Mesh-cap review demo

Five reproducible mesh clip-cap cases, using the same geometry and GLB writer as
the browser regressions. All five small GLBs are included in `fixtures/`; their
textures and vertex colors are embedded. No external model download is needed.
The viewer retains its usual Three.js CDN dependency.

From the repository root:

```bash
uv sync --locked --group browser-test
uv run python bench/mesh_cap_review/demo.py
```

Open the printed localhost URL. The server binds an OS-assigned port by default;
use `--port 8080` to choose one. It serves only the demo and its known fixtures.
Ctrl-C stops it. Each browser tab runs independently, without a Python WebSocket
connection, using the public `handleMessage` embedder API. Reload resets the case.

Add your own GLBs with repeated `--glb /path/to/model.glb` arguments. They appear
as additional interactive cases, fitted to view with a whole-model/section
toggle. Only the selected files are served; they are not copied or committed.
Their cap measurements are observational, not analytic regression assertions.
`--capture` still captures the five deterministic regression fixtures.

Use the links to select a case, download its GLB, or force the cap worker. Drag
to orbit; press **C** for the normal clipping controls. The measured cap area is
per slice (the sum of both slab caps divided by two). The displayed check is a
snapshot after setup; it is not recalculated when you interact with the model.

| Case | Expected result |
| --- | --- |
| Replacement buffer | Cap grows to area 8 when position-attribute identity changes |
| Texture × vertex color | Embedded COLOR_0 = 0.5 multiplies the texture once |
| Material assignment | Changing materialIndex without changing group count refreshes the gold cap |
| TEXCOORD_1 | Samples the light half of the texture; compatibility check |
| GLB layouts | Mirrored, interleaved and quantized boxes all cap; combined area 12 |

The replacement-buffer and material-group cases perform their mutations in
`demo.js` after the initial caps settle; those runtime mutations are intentionally
not baked into the downloadable GLBs. Setup and capture wait for objects, animation
frames and cap completion, rather than arbitrary delays. Internal cap fields are
used only for the regression triggers and measurements in this development bench.

## Reproduce the screenshots

```bash
uv run playwright install chromium
git show origin/main:src/threejs_viewer/viewer.html > /tmp/mesh-caps-before.html
uv run python bench/mesh_cap_review/demo.py \
  --before /tmp/mesh-caps-before.html --capture /tmp/mesh-caps-main
uv run python bench/mesh_cap_review/demo.py \
  --before /tmp/mesh-caps-before.html --worker --capture /tmp/mesh-caps-worker
```

Each run writes 10 full-size PNGs, three comparison sheets (before left, after
right), and `results.json`. It exits nonzero on setup errors or failed branch
area/color checks. Baseline failures are expected and recorded. Both variants
use the exact same fixtures, camera and setup script. Screenshots are release
assets, not committed files. Pass `--before` without `--capture` for interactive
before/after links. `--viewer-html` selects a different candidate build.

To regenerate the committed models:

```bash
uv run python bench/mesh_cap_review/demo.py --generate
git diff --exit-code -- bench/mesh_cap_review/fixtures
```

The texture × vertex-color case is a repair that is already on main. The UV and
layout cases demonstrate preserved behavior and pass on the baseline. The
replacement-buffer and material-group cases exercise cache invalidation.

## Review stress cases and display checks

```bash
node bench/mesh_cap_review/slice_bench.mjs
# Optional comparison with another extracted viewer.js:
node bench/mesh_cap_review/slice_bench.mjs /tmp/pre-review-viewer.js

git show origin/main:src/threejs_viewer/viewer.html > /tmp/main-viewer.html
uv run python bench/mesh_cap_review/capture_display.py \
  --before /tmp/main-viewer.html --output /tmp/cap-display
```

The Node bench checks overlapping colored 1024/4096-segment cylinders, forty
nested rings, and 300 seeded scenes of overlapping prisms cut by a tilted plane.
Timings are informational; the checks assert areas, interpolated colors and
finite nonempty caps.
The display capture writes four PNGs showing cross-mesh cap priority and the
solid-plus-wire M mode. Browser regressions also check the winning pixels at
different camera angles and cap-overlay cleanup after rebuilding/disabling caps.

Overlapping solids merged into one mesh each keep their own filled region, so
the shared area is drawn twice. Regions with different colors can then show
either color there. Hole ownership within a merged mesh is inferred from contour
nesting. This is a rendering capper, not an exact solid-boolean export.
