# Changelog

## 0.0.22

### Viewer display and controls

- **Default tone-mapping exposure lowered `1.5` → `1.0`.** Fixes highlight clipping / red-bleaching on saturated toolpath colors under the default ACES pipeline. **Visible behavior change**: scenes rendered before this version will look slightly darker after upgrading. Pass `ViewerClient(tone_mapping_exposure=1.5)` to restore the previous look.
- **Runtime lighting panel (`E` key or `☼ E` toolbar button).** Four live controls — tone-mapping mode (`none` / `linear` / `reinhard` / `cineon` / `aces` (default) / `agx` / `neutral`), tone-mapping exposure, environment intensity, ambient intensity. Values are applied live and persist across reloads in `localStorage` under the `tjsv.` namespace. A Reset button restores the page-load baseline (URL param > options > hard default, skipping localStorage) and clears the four persisted keys. Changing the tone-mapping mode flushes every material's shader (`material.needsUpdate = true`) because three.js bakes the tone-mapping constant into the compiled program — expect a one-frame recompile stutter on very large scenes.
- **`ViewerClient` accepts four lighting kwargs**: `tone_mapping`, `tone_mapping_exposure`, `environment_intensity`, `ambient_intensity`. When supplied, they're appended as snake-case query params on the viewer URL and act as authoritative initial values — they win over `localStorage` on reload. `tone_mapping` is validated case-insensitively against the seven modes and raises `ValueError` on anything else; the three float kwargs reject NaN/Inf so they never leak into the query string. Precedence for initial values: **URL param > `ThreeJSViewer` option > `localStorage` > hard default.**

### Parametric tube LOD

- **Attribute-aware RDP simplification.** Parametric-tube LOD previously measured perpendicular distance in 3D only, so attribute variation on a geometrically straight segment (a colormap gradient, a width bump on a flat line) was invisible to RDP and would collapse — `LOD_MAX_SKIP=100` was a blunt safety valve that inserted a midpoint every 100 collapsed points to hide the worst banding, without knowing how variable the attribute actually was. The worker now projects into an augmented space (xyz + weighted width, height, r, g, b) so a single camera-scaled epsilon bounds geometric *and* attribute error together: colormap gradients and width ramps survive simplification at the same sub-pixel threshold as geometric detail. Internal weights: width/height contribute 0.5 world units per unit of delta; a full per-channel color swing costs 5% of the tube's bounding radius (which now includes the max cross-section half-extent, so thick tubes with a compact spine don't under-weight color). `LOD_MAX_SKIP` is kept as a safety valve but should rarely fire. The sync fallback (used for the first render before the worker is ready) mirrors the worker's logic so the initial LOD matches subsequent worker-produced levels.
- **`update_parametric_tube_colors` invalidates the LOD chunk cache.** Chunk splits depend on color deltas under augmented-space RDP, so a color swap mid-animation now rebuilds affected chunks instead of reusing stale simplifications.

### Docs

- **Embedding contract documented.** The viewer issues a no-cors HTTP GET against `wsUrl` (scheme swapped to http/https, path+query preserved) before each WebSocket attempt to suppress the browser's unsilenceable "WebSocket failed" console warning. Embedders pointing at a non-`websockets` server need the host (or its proxy) to answer *something* on that GET (200/400/404/426 all count) or the probe never resolves and the WebSocket is never attempted. Previously only explained by a one-line code comment; now covered in `ThreeJSViewer` JSDoc, the probe-site comment, and a new "Embedding the Viewer" section in `DESIGN.md`.

### Internal

- Viewer JSDoc types tightened: typedefs for `ThreeJSViewerOptions`, `BinaryChannel`, `AnimationFrame`, `AnimationData`; binary-channel apply helpers and ring/color free functions fully annotated; `@type {any}` casts replaced with concrete types where practical. No runtime behavior change. Run `npx tsc --noEmit -p jsconfig.json` to type-check.

## 0.0.21

### Fixes

- **Camera-tracking no longer drifts across `load_animation` swaps.** The swap path (added in 0.0.20) preserved `_trackHasLastPos` across loads, causing the next tracking tick to compute a delta between the old trajectory's last position and the new trajectory's first-frame position — producing a small camera slide on every swap, which accumulated into visible framing drift when a caller swapped repeatedly (e.g. ribweaver nudging a spline-smoothness slider and rebuilding the tracked bead each time). The post-swap tick now snaps the orbit target and keeps the camera offset, as a fresh load does. Playhead, play state, track mode, target id, and interactive override are still preserved.

## 0.0.20

### Animation lifecycle

- **`load_animation` preserves playback state on subsequent loads.** The first call to `load_animation` (when no animation is loaded) still starts at t=0, with playback controlled by `autoplay` (default `True`; pass `autoplay=False` to load paused). But a *subsequent* call (one already loaded) now preserves the current playhead time (clamped to the new duration), play state, and camera-tracking — only the underlying frame data is swapped. This makes the "swap mid-playback" pattern (reconcilers, tab switches between related animations) flicker-free without any caller bookkeeping. Pass `load_animation(anim, restart=True)` to force the old "snap to t=0" behavior on a swap.
- **`load_animation(autoplay=False)`** — load the animation paused on first-load (or on a restart) instead of starting playback immediately. No effect on a swap without `restart`, where the prior play state is preserved.
- **`pause_animation()` / `resume_animation()`** — new Python API for pause/resume from a script. Previously pause/resume was browser-only (spacebar / play button).
- **Renamed `stop_animation()` / `clear_animation()` → `unload_animation(restore_visibility=True)`.** The old name implied "pause," but the method actually exits animation mode entirely (re-enables `matrixAutoUpdate`, resets draw ranges, optionally restores baseline visibility, hides controls). The rename makes the contract honest, and the two old methods collapse into one with a parameter (`restore_visibility=False` matches the old `clear_animation` semantic). **Breaking**: external callers of `stop_animation()` / `clear_animation()` must rename. No shim.

### Parametric tube improvements

- **Analytic normals** for tube body and caps — replaces computed-from-geometry normals, eliminates shading artifacts at the frontier ring during `draw_range` animation.
- **`anchor` parameter** — `"center"` (default) or `"top"` controls whether the bead is centered on the spine or extends downward from the top surface (useful when the spine is a nozzle-tip toolpath).
- **Smoother end caps** — `N_CAP_RINGS` 3 → 8 for a better top-down look on revolution caps.
- **Buffer-upload optimization** — `addUpdateRange()` on position/normal/color/index buffers instead of `needsUpdate = true`; ~1500× smaller per-frame GPU upload bandwidth.
- **LOD fixes** — preserve color resolution on straight segments via `LOD_MAX_SKIP`; color version tracking eliminates a race between color updates and worker geometry rebuilds; frontier-ring restore no longer overwrites a color swap mid-animation.

### Viewer display and controls

- **Replaced `OrbitControls` with a custom `ViewerControls`** — middle-click re-pivots without a camera jump (a long-standing `OrbitControls` annoyance). Press `R` to toggle turntable ⇄ free orbit mode.
- **Global wireframe cycle (`M` key)** — cycle scene display: normal → wireframe-only → combined (solid + black wireframe overlay) → normal. Removed the per-tube `wireframe_color` API in favor of this scene-wide toggle.
- **Shading debug cycle (`N` key)** — cycle off → normals-as-color → UV checker → vertex-normals helper. Independent of `M` — they compose. Vertex-normals helper size is camera-relative (~30 px regardless of zoom).
- **Stack-overflow fix in the `ViewHelper` `setViewport` shim** — a pre-existing bug re-wrapped the already-wrapped `setViewport` every frame, deepening the call chain by one level per frame until the stack blew. The shim now caches the true original once and restores inside `try/finally`.

### Internal

- Decompose `viewer.js` god-class into in-file controller classes (`ParametricTube`, `CameraController`, `ShadingDebugController`) plus shared ring/color helpers as free functions. Viewer now type-checked via JSDoc + `// @ts-check` (run `npx tsc --noEmit -p jsconfig.json`). No behavior change.

## 0.0.19

### New features

- **Animation interpolation** — animation playback now lerps/slerps between keyframes by default, so producers can sample at the signal's bandwidth (e.g. 10 Hz) and still get smooth 60 fps playback. Translations lerp, rotations slerp, float channels (`draw_ranges`, `opacity`, `clip_times`, scripted `camera_target`/`camera_position`) lerp element-wise, and the `colors` channel lerps hex values in 8-bit RGB space (works for direct hex and colormap-indexed `uint8`). Per-channel opt-out via `add_channel(interpolation="hold")` or the convenience setters (`set_transform_data`, `set_draw_range_data`, `set_clip_time_data`, `set_camera_target`, `set_camera_position`) for frame-accurate scientific/simulation replay where intermediate values would be meaningless. The `visibility` channel is boolean and always left-holds the floor keyframe regardless of the setting — a "linear bool" has no meaningful interpretation. JSON `Frame` objects don't carry per-field interpolation and always interpolate linearly; use a binary channel when you need hold behavior. See `examples/17_animation_interpolation.py` for a HOLD ⇄ LINEAR comparison on the same tumbling capsules.
- **Behavior change**: animations relying on exact frame-accurate playback (where each frame's transforms/colors/etc. were previously pinned to the nearest keyframe) must now pass `interpolation="hold"` explicitly to the relevant `add_channel` / `set_*_data` call. Examples 03/04/05 were downsampled to 10 Hz to demonstrate the smaller payloads.
- **`parametric_tube` primitive** — variable-cross-section extruded tube built on the client from per-spine-point parameter arrays (spine + widths + heights + optional colors). Chamfered hexagonal cross-section (6 vertices/ring). Supports `draw_range` with smooth frontier-ring morphing and the `draw_ranges` animation channel. Wire transfer is O(N) instead of O(N × nCs), ~6× smaller than a baked mesh. See `examples/18_parametric_tube.py`.
- **`update_parametric_tube_colors(id, colors)`** — replace a tube's per-ring color attribute without rebuilding geometry. Intended for interactive color-mode switching (layer → feed rate → curvature → …) in toolpath previews.
- **Automatic LOD for parametric tubes** — tubes with ≥25k spine points get distance-weighted RDP simplification running in a Web Worker. The full geometry build (tangents, frames, vertices, indices) also runs off the main thread, so the render loop is never blocked. Per-chunk results are cached and reused when the camera distance changes less than 50%. Handles 1M+ point toolpaths at 60 fps. See `examples/11_toolpath.py`.
- **`set_scene_visibility({id: bool, ...})`** — batch visibility updates that always persist to the animation baseline, so visibility set before `load_animation` is preserved across animation teardown.
- **`add_toolpath()` / `Toolpath` convenience wrapper** — builds a parametric tube from a `Toolpath` with optional `colorize()` (viridis/plasma/turbo/hex/RGB/per-point scalar). Zero-width segments collapse for natural taper transitions on travel moves.
- **Custom triangle meshes** — `add_mesh()` example with a procedural terrain heightmap, vertex colors, analytical normals, and draw_range row-by-row reveal. See `examples/19_custom_mesh.py`.

### Other

- Unknown WebSocket message types now log a warning instead of being silently ignored.
- Viewer source files excluded from the wheel (only the built `viewer.html` ships).

## 0.0.18

### New features

- **Grid API** — `show_grid(visible, size, divisions)` to toggle/resize the ground grid (hidden by default)
- **`add_mesh` transform** — `position`, `rotation`, `scale`, and `matrix` parameters for placing meshes directly
- **`query_scene` restructured** — returns `{"objects": {...}, "meta": {...}}` with per-object `drawRange` and viewer metadata (`animation.playing`, `grid.visible`, `pending_fetches`). **Breaking**: replaces flat dict.

### Bug fixes

- **Faster, quieter connect on FastAPI/non-HTTP WS routes** — fixes ~30s initial connection delay and 404 spam in the browser console when the viewer is served behind a FastAPI app or any route that doesn't answer plain HTTP GETs
- **`stop_animation()` fully resets mesh/polyline draw ranges** — previously a partial `draw_range` applied during playback could linger on screen after stopping
- **`clear()` wipes animation state** — previously a loaded animation could keep advancing (or reappear on replay) after `clear()`
- **No more ghost transforms after `clear_scene` / `load_animation` / reconnect** — in-flight binary fetches from a previous scene, animation, or connection used to land after the teardown and overwrite the current scene; they're now dropped
- **No more ghost updates after `delete_object` or ID reuse** — an animation channel could still apply its last-known transform/color to an object with a recycled ID; the channel now notices and re-resolves its targets
- **Pending-fetch counter no longer drifts negative after reconnect**

## 0.0.17

### New features

- **Camera tracking** — three modes for automatic camera control during animation playback:
  - `camera_follow="nozzle"` — orbit target pans with the object, user can rotate/zoom freely
  - `camera_lookat="nozzle"` — camera stays put but always points at the object
  - `set_camera_target()` / `set_camera_position()` — fully scripted per-frame camera via binary channels
- **Interactive tracking toggle** — `T` key cycles tracking modes; toolbar button shows current state
- **Shortcut labels in toolbar** — clip (✂ C) and track (⊚ T) buttons show their keyboard shortcut inline

### Bug fixes

- Fixed cubemap orientation for Z-up to Y-up conversion

## 0.0.16

### New features

- **`resize(width, height)`** — public method to force viewer resize (e.g. after container show/hide)
- **`frameAll()`** — resets orbit controls and positions camera to frame all scene objects
- **Dynamic near/far planes** — perspective camera near/far recomputed from scene bounding sphere, preventing geometry clipping on zoom-in
- **Version placeholder** — source uses `0.0.0-dev`, CI substitutes real version at build time

### Bug fixes

- Fixed `__version__` out of sync with package version (was stuck at 0.0.13)

## 0.0.15

Embeddable viewer: extract monolithic viewer.html into modular ES module.

### New features

- **Embeddable `ThreeJSViewer` class** — ES module in `viewer/viewer.js` that mounts into any container div with `new ThreeJSViewer(container, options)`. Supports `wsUrl`, `wsPort`, `htmlTemplate`, and `cubemapData` options.
- **`destroy()` method** — full cleanup of WebSocket, RAF, ResizeObserver, and event listeners
- **Scoped events** — keyboard shortcuts scoped to container (`tabindex`), resize via `ResizeObserver` instead of `window`
- **Build pipeline** — `viewer/build.py` regenerates the self-contained `viewer.html` from source files; CI verifies freshness
- **Source file separation** — CSS, HTML template, and cubemap images extracted into `viewer/` subfolder

### Bug fixes

- Multi-material meshes now correctly apply color changes (previously crashed on array materials)
- `destroy()` prevents reconnect attempts after teardown

## 0.0.14

Embedded cubemap environment and Three.js upgrade.

- **Embedded cubemap** — replaces external CDN dependency (polyhaven) with a 64x64 JPEG cubemap (~12KB) embedded as base64 data URIs for offline-capable PBR reflections
- **Three.js 0.170.0 → 0.183.2** upgrade
- **ACES filmic tone mapping** — replaces default tone mapping for better HDR rendering
- **Simplified lighting** — ambient + environment only (removed directional lights)
- **sRGB color space** on cubemap for correct PMREM filtering
- **Error handling** on cubemap face loading to prevent resource leaks

## 0.0.13

Auto-open the viewer in the default browser when no existing tab connects.

### New features

- **Auto-open browser** — `connect()` now waits 2.5 seconds for an existing browser tab to reconnect; if none does, it automatically opens the viewer in the default browser via `webbrowser.open()`. Controlled by the new `open_browser` parameter on `ViewerClient` constructor (default `True`). Set `open_browser=False` to disable (e.g. headless CI, remote servers). Also available via the `viewer()` convenience function. Gracefully falls back to printing the full viewer URL if the browser cannot be opened.

## 0.0.12

Interactive clipping plane with slab mode, orthographic camera, and slice view for inspecting toolpath layers.

### New features

- **Clipping plane panel** — press C to toggle an interactive cross-section plane with axis selection, position slider, and rotation gizmo
- **Slab mode** — dual-plane clipping shows a thin slice of geometry; toggle with S key or panel button
- **Orthographic camera** — O key toggles between perspective and orthographic projection
- **Slice view (V key)** — snaps the ortho camera to look along the clipping plane normal, ideal for layer-by-layer toolpath inspection
- **Normal XYZ inputs** — type an exact clipping plane normal in the panel; axis buttons, gizmo, and programmatic changes stay in sync
- **Auto-fit slider range** — position and thickness sliders derive their min/max from the scene bounding box projected onto the clip normal
- **Arrow key priority** — when clipping is active, arrow keys always control the clip plane, even with an animation loaded
- **`set_clipping_plane(normal, distance, show_helper)`** — programmatic single clipping plane
- **`set_clipping_slab(normal, center, thickness, show_helper)`** — programmatic dual-plane slab
- **`disable_clipping_plane()`** — remove clipping
- **`set_clipping_defaults(normal, distance)`** — pre-configure the default clip axis/position for when the user first opens the panel (C key); useful for toolpath workflows where the slice plane is known

### Keyboard shortcuts (clipping)

| Key | Action |
|-----|--------|
| C | Toggle clipping on/off |
| S | Toggle single/slab mode |
| V | Snap ortho view along clip normal |
| H | Toggle helper + panel visibility |
| O | Toggle ortho/perspective |
| ←→ | Nudge position |
| ↑↓ | Nudge thickness (slab mode) |

### New example

- `16_clipping_plane.py` — nested spheres + tilted toolpath with clipping along the slice plane normal, programmatic single/slab modes

## 0.0.11

Adds binary channel support for `clip_times`, enabling efficient transfer of embedded GLTF animation times alongside other binary channels.

### New features

- **`clip_times` binary channel** — `clip_times` can now be sent as a binary channel via `set_clip_time_data()` or `add_channel("clip_times", ...)`, avoiding per-frame JSON metadata for animations with many frames
- **`Animation.set_clip_time_data()`** — convenience method for creating a `clip_times` binary channel, matching `set_draw_range_data()` and `set_transform_data()`

## 0.0.10

Adds opacity support to `add_mesh()` and a `plane_normal` parameter to `Toolpath.to_mesh()` for non-horizontal layer planes.

### New features

- **`opacity` parameter on `add_mesh()`** — set initial material opacity (0.0–1.0) at mesh creation time, consistent with `add_primitive()` and `set_opacity()`
- **`plane_normal` parameter on `Toolpath.to_mesh()`** — the "up" direction for the bead cross-section. Defaults to `[0, 0, 1]` (world Z-up). Pass a transformed normal when slicing on a tilted plane so bead height is perpendicular to the layer surface rather than world-vertical.

## 0.0.9

Reverts the 0.0.8 Z-up primitive orientation change and makes GLB/GLTF Y-up correction opt-in.

### Breaking changes

- **Cylinder, cone, capsule are Y-aligned again** — the 0.0.8 change that baked `rotateX(90°)` into these geometries has been reverted. They now use the Three.js default (Y-axis aligned). Any code that added a rotation to compensate for the 0.0.8 Z-up change must remove that compensation. Code that relied on the old Y-up convention (pre-0.0.8) needs no changes.
- **GLB/GLTF Y-up correction is now opt-in** — `add_model()` and `add_model_binary()` no longer auto-apply `Rx(+90°)` to GLB/GLTF models. Pass `y_up=True` to enable correction for standard Blender/Sketchfab exports. Z-up CAD exports load correctly without this flag.

### New features

- **`y_up=True` on `add_model()` / `add_model_binary()`** — opt-in Y-up→Z-up correction for GLB/GLTF models that follow the glTF spec (Blender, Sketchfab). Default is `False`.

## 0.0.8

Refactors toolpath coloring and geometry into the `Toolpath` class, adds `wait_for_assets()` for clean script exit, auto-starts animation playback, and fixes Z-up orientation for all built-in primitives and GLTF models.

### Breaking changes

- **`add_bead()` removed** — use `v.add_mesh("id", **tp.to_mesh())` instead (build the mesh via `Toolpath.to_mesh()`)
- **Cylinder, cone, capsule are now Z-aligned** — previously these primitives used Three.js defaults (Y-up), so they appeared on their sides in the Z-up viewer. They are now upright. Any code that manually rotated these primitives to compensate must remove that rotation.
- **GLTF/GLB models are now auto-corrected to Z-up** — a `+90° X` rotation is baked in on load. Any code that manually applied a rotation matrix to a GLTF model to make it upright (e.g. a custom `matrix=` on `add_model_binary`) must remove that correction.
- **`wait_for_assets()` now raises `TimeoutError`** on timeout instead of silently disconnecting.
- **Animation auto-starts on `load_animation()`** — previously the animation started paused; it now begins playing immediately. Call `v.stop_animation()` before `v.load_animation()` if you want it to start paused.

### New features

- **`Toolpath.colorize()`** — colors stored on the toolpath directly; accepts a colormap name (`"viridis"`), hex int, RGB tuple, per-point scalar array (mapped through a colormap), or `(N,3)` RGB array; `travel_color` parameter grays out zero-width travel moves
- **`Toolpath.to_mesh()`** — builds bead mesh geometry (`positions`, `indices`, `normals`, `colors`) as a dict, usable with `v.add_mesh("id", **tp.to_mesh())`; supersedes `add_bead()` which has been removed from `ViewerClient`
- **`Toolpath.colors` property** — get/set vertex colors on a toolpath directly
- **`wait_for_assets()`** — blocks until the browser signals all HTTP binary assets have loaded, then disconnects; lets scripts exit cleanly instead of `while True: pass`
- **Auto-start animation** — `load_animation()` now starts playback immediately in the browser without a separate play call

### Bug fixes

- `from_gcode`: zero-length connectors within an extrusion segment no longer get `w=0` (were incorrectly treated as travel)
- `from_gcode` / `from_points`: raise `ValueError` with a clear message for bad input shape or fewer than 2 points
- `merge_animation_points`: fixed right-anchored interpolation at duplicate timestamps
- `Toolpath.merge()`: color interpolation now deduplicates timestamps before `np.interp`, matching geometry behavior at extrusion→travel transitions
- `_apply_colormap`: `frac==1.0` now maps exactly to the last colormap entry (was slightly short due to float clipping)
- `Toolpath.colors` setter: validates array shape `(N, 3)` matching `len(self)`, raises `ValueError` on mismatch
- `wait_for_assets()`: fixed race where early fetches could fire `assets_loaded` before later asset messages arrived; Python now sends an explicit `mark_assets_complete` message that the browser uses to gate its reply
- `wait_for_assets(disconnect=False)`: new parameter keeps the server alive after assets load, allowing subsequent streaming updates (`batch_update`) — fixes crash in example 07
- Cylinder, cone, and capsule geometry now baked with `rotateX(90°)` so they stand upright in the Z-up viewer without manual rotation workarounds
- GLTF/GLB models wrapped in a `+90° X` correction group on load, converting Y-up (glTF standard) to Z-up; set_matrix/animation targets the outer wrapper and the correction is permanent
- `14_grouping.py` robot arm: fixed joints to bend around Y-axis and arm segments to extend upward in Z

### New example

- `15_toolpath_interrupted.py` — G-code-derived pill/racetrack path with travel moves, per-point width array (`0` = travel), `merge_animation_points` for segment-aligned `draw_range` animation, and plasma colormap

### Updated examples

- `11_toolpath.py` — uses `tp.colorize("viridis")` + `v.add_mesh("path_tube", **tp.to_mesh())`; point count halved for faster loading
- All animation examples now call `v.wait_for_assets()` instead of blocking indefinitely
- `04_flying_teapots.py` — replaced with a teapot carousel: synchronized ring orbiting a golden sphere, ring breathes and tilts over time
- `07_stress_test.py` — calls `wait_for_assets(disconnect=False)` before streaming loop so assets finish loading before animation starts

### Other

- `run_examples.sh` — shell script that runs all examples sequentially

## 0.0.7

Adds object grouping (parent-child hierarchies) and version negotiation between Python client and browser viewer.

### New features

- **Object grouping** — `add_group(id, parent=...)` creates empty transform nodes; all `add_*` methods accept an optional `parent` parameter. Children inherit parent transforms automatically via Three.js scene graph. Deleting a group removes all children.
- **Version handshake** — viewer and Python client exchange versions on connect. Mismatches print a warning on both sides, catching stale browser cache issues immediately.

### New example

- `14_grouping.py` — animated robot arm with nested groups (base → shoulder → elbow → wrist), demonstrating how only joint transforms need animation while visual meshes follow automatically

### Updated examples

- `01_primitives.py` — pillars grouped under a common parent
- `03_animation_basics.py` — moon parented to earth_system group, uses local orbit offset instead of recomputing world position
- `04_flying_teapots.py` — reference pillars grouped

## 0.0.6

Enables environment map reflections on primitives and meshes, rebalances scene lighting.

### Changes

- **Environment reflections on primitives and meshes** — built-in shapes and `add_mesh()` objects now pick up the same cubemap reflections as GLB models; previously only GLB materials reflected the environment
- **Rebalanced environment intensity** — scene environment intensity halved to compensate for the new reflections on all objects
- Examples updated to use `roughness`/`metalness` material properties
- Fixed incorrect API references in README (`set_position` → `set_matrix`, `set_transforms` → `batch_update`)

## 0.0.5

Adds PBR material control (`roughness`/`metalness`) to primitive methods, matching the existing `add_mesh()` API.

### Changes

- **`roughness` and `metalness` on primitives** — `add_box()`, `add_sphere()`, `add_cylinder()`, `add_capsule()` accept optional `roughness` and `metalness` parameters (0.0–1.0); defaults unchanged when not specified
- Fixed `__version__` out of sync with package version (was stuck at 0.0.1)

### New example

- `13_material_properties.py` — 5×5 sphere grid varying roughness (columns) and metalness (rows), plus a metallic box and a rough cylinder

## 0.0.4

Adds transparency/opacity support across the entire API — primitives, runtime changes, batch updates, and binary animation channels. Also fixes a race condition with transforms on async-loaded models.

### New features

- **`opacity` parameter on primitives** — `add_box()`, `add_sphere()`, `add_cylinder()`, `add_capsule()` accept `opacity=0.0–1.0`
- **`set_opacity(id, opacity)`** — one-shot opacity change for any object
- **`set_color()` extended** — optional `opacity` argument: `set_color(id, color, opacity=0.5)`
- **Opacity in `batch_update()`** — include `"opacity": 0.5` in transform dicts for high-frequency updates
- **Binary `opacity` animation channel** — smooth opacity pulsing via `animation.add_channel("opacity", ...)`
- **Transform on `add_model_binary()`** — `position`, `rotation`, `scale`, and `matrix` parameters applied after async model load completes (fixes race condition)

- **Correct transparency rendering across all object types** — opacity now plays well with depth sorting on primitives, meshes, and GLB models (including GLBs with multi-material arrays, which previously crashed on material updates). Default `opacity=1.0` — no impact on existing code that doesn't use transparency.

### New example

- `12_transparency.py` — GLB helmet + primitives with pulsing opacity animation channel

## 0.0.3

Refactors animation to use generic binary channels, replacing the hardcoded transform/draw_range system with an extensible channel-based architecture. Any per-object-per-frame data can now be sent as a typed binary channel.

### Changes

- **Generic binary animation channels** — `animation.add_channel(name, object_ids, data, dtype, stride)` can carry any per-object-per-frame data as a typed binary channel. `set_transform_data()` / `set_draw_range_data()` are kept as convenience wrappers.
- **New channel types** — `colors` (uint32), `visibility` (uint8), and `opacity` (float32) channels join the existing `transforms` and `draw_ranges`
- **Clearer errors on bad channel data** — `add_channel()` raises on an unsupported dtype (allowed: float32, uint32, uint8) instead of letting it reach the browser; the viewer logs and skips unknown dtypes instead of crashing playback

### Updated examples

- Existing examples updated to use the new `add_channel()` API where appropriate

## 0.0.2

Adds GLB model support with PBR materials, a bead extrusion primitive for toolpath visualization, and binary animation channels for handling large animations efficiently. Binary data now transfers over a dedicated HTTP sidecar for better performance.

### New features

- **GLB/GLTF models with PBR** — load models with physically-based materials and play back embedded animations via `clip_times`
- **Bead extrusion** — `add_bead()` creates toolpath meshes with a bevelled rectangle cross-section, vectorized with numpy, and per-layer vertex colors
- **Pre-built meshes** — `add_mesh()` for custom triangle meshes with optional vertex colors and normals
- **Draw range** — `set_draw_range(id, 0.0–1.0)` controls the visible fraction of polylines and meshes, with a matching `draw_ranges` animation channel
- **Binary animation channels** — `set_frame_times()`, `set_transform_data()`, and `set_draw_range_data()` replace Python loops with numpy arrays for animations with 100k+ frames
- **HTTP binary transfer** — large data (models, polylines, animations) is served via an HTTP sidecar on port 5667 instead of over WebSocket

### Improvements

- Lighting overhaul for GLB/PBR materials
- Improved animation playback UI controls
- Animation now correctly applies to async-loaded models
- `disconnect()` properly shuts down the HTTP sidecar

### New examples

- `08_glb_models.py` — DamagedHelmet and Avocado with PBR materials
- `09_animated_glb.py` — embedded GLTF morph animation with orbiting model
- `10_animation_stress_test.py` — 520 objects, 2499 frames, vectorized numpy
- `11_toolpath.py` — spiral vase with bead mesh, draw_range animation, and binary channels (800k points)

## 0.0.1

Initial release of threejs-viewer — a lightweight Three.js viewer controlled from Python via WebSocket.

### Features

- **Python WebSocket server** — browser connects to Python on port 5666, survives script restarts with auto-reconnect
- **Primitives** — `add_box()`, `add_sphere()`, `add_cylinder()`, `add_capsule()`, `add_cone()`, `add_torus()`, `add_plane()` with color and transform
- **Polylines** — gradient-colored lines with built-in colormaps (viridis, plasma, turbo)
- **3D models** — load GLTF/GLB, STL, OBJ, FBX, DAE, PLY, 3DS from URL
- **Real-time streaming** — `batch_update()` and `set_matrix()` for 60fps transform updates
- **Looping animation** — `Animation` / `Frame` classes with interactive playback, timeline scrubbing, speed control, and frame stepping
- **Scene control** — `set_color()`, `set_visible()`, `delete()`, `clear()`

### Examples

- `01_primitives.py` — basic shapes with colors and positions
- `02_polylines.py` — gradient lines with different colormaps
- `03_animation_basics.py` — solar system with looping animation
- `04_flying_teapots.py` — Utah teapots with model loading + animation
- `05_lissajous_curves.py` — mathematical curves with tracer animation
- `06_realtime_streaming.py` — real-time bouncing spheres
- `07_stress_test.py` — torus knot tube with hundreds of followers
