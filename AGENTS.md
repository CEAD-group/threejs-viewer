# Repository instructions

## Working conventions

- Open every issue or PR comment and review an agent writes with an author line:
  `Comment by Claude (bot)` for Claude, `Review by **Codex (OpenAI)**` (or
  `**Codex (OpenAI)**` on an inline comment) for Codex. They post from Thijs's
  account, so the line is the only attribution a reader sees.
- Every PR with a visible effect (rendering, UI, gizmos, picking, colour,
  README media) gets before/after screenshots in its body, unasked. Capture
  "before" from `origin/main` and "after" from the branch with headless
  Playwright, and look at each image yourself. Host them on a
  `pr-<N>-images` pre-release (`gh release create pr-<N>-images --prerelease
  ...`) and link the release-asset URLs; never commit them to a branch.
- Preserve existing local changes and keep edits focused on the requested task.
- Keep this file focused on actionable guidance. Put API details in docstrings,
  usage in examples, and regression explanations beside the relevant code or tests.

## Project and source map

`threejs-viewer` is a lightweight Three.js viewer controlled from Python scripts
and Jupyter notebooks. Python runs the WebSocket server; the browser connects as
a client and survives Python restarts. Large payloads use an HTTP sidecar.
The coordinate system is Z-up, with the floor in the XY plane.

The viewer uses plain JavaScript, CSS, and HTML, with Three.js loaded from a CDN.
Its standalone HTML is assembled by a Python script. Node and TypeScript are
development tools for JavaScript type checking, not runtime or build dependencies.

| Location | Purpose |
| --- | --- |
| `src/threejs_viewer/client.py` | Python API, servers, protocol, and reconnect state |
| `src/threejs_viewer/animation.py` | Frames, binary animation channels, and markers |
| `src/threejs_viewer/kinematics.py` | Kinematic-chain schema, validation, and reference evaluator |
| `src/threejs_viewer/toolpath.py` | Toolpath preparation and coloring |
| `src/threejs_viewer/points_lod.py` | Point-cloud octree construction |
| `src/threejs_viewer/usd.py` | Optional USD BRep integration |
| `src/threejs_viewer/viewer/viewer.js` | Viewer, controllers, rendering, and message handling |
| `src/threejs_viewer/viewer/controls.js` | Camera interaction controls |
| `src/threejs_viewer/viewer/viewer.css` | Styles scoped under `.threejs-viewer` |
| `src/threejs_viewer/viewer/template.html` | Viewer markup and controls |
| `src/threejs_viewer/viewer/build.py` | Standalone HTML and cubemap index generation |
| `src/threejs_viewer/viewer.html` | Generated, committed viewer; never edit directly |
| `tests/` | Python tests and Playwright browser integration tests |
| `examples/` | Runnable demonstrations of the public API |
| `bench/` | Offline quality and performance tools; not shipped or run in CI |
| `tools/record_readme_media.py` | Re-records the README gallery in `docs/media` from the examples (`SHOTS`); not run in CI |
| `plans/` | Decision records with their own status; not a source of current architecture |

Start with `README.md` for usage and `DESIGN.md` for design context. Check the
implementation and relevant tests before relying on historical design notes.

## Development commands

Run commands from the repository root. Python 3.11 or newer is required.

```bash
# Install Python dependencies using the lockfile.
uv sync --locked

# Run tests that do not require a browser.
uv run pytest -m "not browser"

# Set up and run browser integration tests (includes a Firefox smoke test).
uv sync --locked --group browser-test
uv run playwright install chromium firefox
uv run pytest -m browser -v

# Lint and check Python formatting.
uv run ruff check .
uv run ruff format --check .

# Install development-only JS tooling, then type-check JSDoc-typed JavaScript.
npm install
npx tsc --noEmit -p jsconfig.json

# Regenerate the standalone viewer after changing viewer sources or assets.
uv run python src/threejs_viewer/viewer/build.py

# Open a model or run a demonstration.
uv run python -m threejs_viewer path/to/model
uv run python examples/01_primitives.py
```

Run checks appropriate to the change. For Python changes, run lint, formatting,
and relevant non-browser tests. For viewer or protocol changes, also rebuild the
viewer and run relevant browser tests; type-check changed JavaScript. Browser
tests auto-skip without `pytest-playwright`, so skipped tests do not validate
browser behavior. Documentation-only changes do not need the application suite.

CI checks lint, formatting, non-browser tests, browser tests, and byte-exact
freshness of `viewer.html` and `src/threejs_viewer/viewer/static/cubemaps/index.json`. Include generated
changes whenever their inputs change.

## Implementation constraints

### Viewer build and resources

- Edit sources under `src/threejs_viewer/viewer/`, then run `build.py`.
  The build concatenates and inlines sources; do not assume a module bundler.
  Controllers intentionally share `viewer.js` and the resulting script block.
- Keep `buildTubeArrays` dependency-free. The main thread and inline LOD worker
  share its definition through `toString()`; changes must work in both contexts.
- Preserve reuse of existing gzip/base64 blobs when their decoded bytes match.
  Recompressing unchanged assets can produce different bytes across zlib versions
  and fail CI's freshness check. See `tests/test_build.py`.
- Keep HDR cubemap assets in flat RGBE format. The viewer decodes them in-file;
  do not introduce an HDR loader dependency without accounting for embedders.
- The Draco decoder is vendored under `src/threejs_viewer/viewer/static/draco/`
  and inlined by the build. Keep its version aligned with the viewer's Three.js
  version. Never add a CDN fallback: production runs offline behind a firewall.
- The shadow map renders only when `_shadowDirty` is set (`shadowMap.autoUpdate`
  is off). A new path that moves, adds, hides or re-materials geometry must set
  it or call `requestShadowUpdate()`, or shadows go stale without any error.
- Dispose owned GPU resources, workers, and event handlers when replacing or
  destroying their owners. Guard asynchronous loads against stale results after
  replacement or destruction. Dispose the environment render target that owns
  the PMREM texture, as well as the raw environment cube.

### Protocol, embedding, and lifecycle

- Keep Python messages, JavaScript handlers, validation, and reconnect replay
  consistent. Persistent settings and objects must survive reconnection.
- Keep message dispatch in public `viewer.handleMessage(data)`. Embedders can
  use `autoConnect: false` and dispatch messages without a WebSocket.
- Prefer public viewer methods and hooks for embedder features over exposing
  private renderer, socket, or controller fields. Preserve unsubscribe behavior.
- Keep the WebSocket, sidecar, and viewer URLs on the same configured hostname.
  Rewriting `localhost` to `127.0.0.1` breaks Firefox sidecar fetches from file
  pages. Preserve IPv4/IPv6 binding and IPv6 URL bracketing.
- For test servers, bind port `0` and use the assigned port. Do not probe a free
  port and release it before binding; that creates a race.
- Fetch binary payloads through the checked fetch helpers. Report HTTP failures
  clearly and balance pending-fetch counters on both success and failure.
- Preserve initial visibility, deferred parenting, and object replacement
  behavior across all add paths, including asynchronously loaded models. Every
  add path registers through `_registerObject`, and `_applyInitialVisibility`
  runs before `_deleteObject`, which prunes the pending visibility it reads.
- Keep resize work coalesced to animation frames and skip rendering when the
  container has zero size. Avoid repeated framebuffer allocations for equal sizes.

### Animation and geometry

- Use binary channels and batch updates for large datasets. Preserve dtype,
  alignment, and precision contracts: draw-range, clip-time, and point-time
  convenience channels use float64; transforms and camera channels use float32.
- Binary channels override same-named JSON frame fields. Binary interpolation
  defaults to linear, with per-channel hold available; visibility always holds.
  JSON frames interpolate linearly except for visibility.
- Loading an animation for the first time starts at zero. Replacing one preserves
  the playhead, play state, and camera tracking unless `restart=True`.
  Unloading restores non-animation behavior; it is different from pausing.
- Reject non-finite transforms and retain the last valid pose. A NaN matrix can
  make an entire subtree disappear.
- Independent object-matrix interpolation does not preserve articulated-chain
  kinematics. Use joint-driven chains when that consistency matters, or densify
  keyframes/use hold interpolation. Follow-path tracks compute a tool's pose
  from the path on each render tick.
- Validate kinematic descriptions against `kinematics.py`; keep the JavaScript
  evaluator consistent with the Python reference and regression fixtures.
- Before promoting tube-corner changes, run the gate described in
  `bench/tube_lab/README.md` (`uv run python bench/tube_lab/zr_metric.py`). Every
  scene must meet `IoU >= 0.99`, triangle count no greater than the collapse-off
  baseline, and dark-pixel count no greater than that baseline. Consult
  `bench/tube_lab/FINDINGS.md` for tested and rejected approaches.

### Menus and interaction

- Use `MenuController` for built-in, embedded, and Python-defined menus. Keep
  styling, placement, persistence, visibility, and shortcuts in that shared path.
- Radio items share `options` and `value` semantics with select and segmented
  items, use `_fillOptions`, and persist through `storageKey`. Preserve their
  radio roles and selected-state animation when changing menu rendering.
- Move and clipping gizmos share `refineGizmoHandles`, hover styling, and the
  overlay `_gizmoScene`. Their helpers must stay outside scene framing and clipping.
  The clipping controls provide rotation and a local Z-only translation handle.
- Preserve platform-specific gizmo modifiers: Alt rotates and Shift toggles snap
  on macOS/Linux; Shift rotates and Ctrl toggles snap on Windows. A temporary
  rotate override must not replace the caller's configured mode.
- Keep move gizmos ahead of axis-control handles when hit areas overlap. Axis
  handles are picked in screen pixels, including under orthographic projection.

## Browser test synchronization

Use the helpers in `tests/conftest.py` instead of arbitrary delays:

- `settle(client)` performs an ordered WebSocket round trip, then waits for
  pending binary fetches to finish registering their objects.
- `frames(page, n=2)` waits for animation frames when checking render-loop work,
  camera changes, keyboard-driven display changes, or pixel output.

A Playwright `page.evaluate` call uses a separate channel and does not establish
ordering with WebSocket messages. Choose the appropriate barrier before reading
viewer state. Use timed waits only for behavior that actually depends on elapsed
time, such as playback, throttled LOD updates, or delayed streaming.

## Versioning and releases

- PRs describe their change in the PR body and do not edit `CHANGELOG.md`. The
  changelog is written at release time in its own PR from the merged PR bodies,
  and the release is that PR plus a pushed `v*` tag.
- Python package versions come from release tags through `hatch-vcs`; do not
  hard-code a package version. Build checkouts need tags (`fetch-depth: 0` in CI).
- Keep `VIEWER_VERSION = '0.0.0-dev'` in committed `viewer.js`. Release CI replaces
  only that assignment and rebuilds the HTML for the tagged release.
- Release CI sets `SETUPTOOLS_SCM_PRETEND_VERSION` because version substitution
  dirties the checkout. The distribution-specific environment variable is not
  effective with this hatch-vcs setup.
- Preserve handshake behavior: development versions may differ between Python
  and JavaScript without warning; mismatched release versions still warn.
