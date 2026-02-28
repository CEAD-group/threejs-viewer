# Changelog

## 0.0.8

Refactors toolpath coloring and geometry into the `Toolpath` class, adds `wait_for_assets()` for clean script exit, auto-starts animation playback, and introduces an interrupted-toolpath example with travel moves.

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

- **Environment map on all objects** — removed `envMapIntensity: 0` from `createMaterial()`, so primitives and meshes now receive the same environment reflections as GLB models
- **Rebalanced environment intensity** — `scene.environmentIntensity` reduced from 2.0 to 1.0 to compensate
- Examples updated to use `roughness`/`metalness` material properties
- Fixed incorrect API references in README (`set_position` → `set_matrix`, `set_transforms` → `batch_update`)

## 0.0.5

Adds PBR material control (`roughness`/`metalness`) to primitive methods, matching the existing `add_mesh()` API.

### Changes

- **`roughness` and `metalness` on primitives** — `add_box()`, `add_sphere()`, `add_cylinder()`, `add_capsule()` accept optional `roughness` and `metalness` parameters (0.0–1.0)
- JS `createMaterial()` reads from params instead of hardcoding `0.7`/`0.3` — defaults unchanged when not specified
- Sync `__init__.py` version (was stuck at 0.0.1) with `pyproject.toml`

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

### Implementation details

- Shared `applyOpacity()` helper handles Three.js material quirks: `needsUpdate` for shader recompilation, `depthWrite` for correct transparency rendering, and multi-material arrays on GLB models
- Default `opacity=1.0` — no impact on existing code that doesn't use transparency

### New example

- `12_transparency.py` — GLB helmet + primitives with pulsing opacity animation channel

## 0.0.3

Refactors animation to use generic binary channels, replacing the hardcoded transform/draw_range system with an extensible channel-based architecture. Any per-object-per-frame data can now be sent as a typed binary channel.

### Changes

- **Generic binary animation channels** — `animation.add_channel(name, object_ids, data, dtype, stride)` replaces the fixed `set_transform_data()` / `set_draw_range_data()` methods (which are kept as convenience wrappers)
- **New channel types** — `colors` (uint32), `visibility` (uint8), and `opacity` (float32) channels join the existing `transforms` and `draw_ranges`
- **Extensible JS dispatcher** — `CHANNEL_APPLY` lookup table makes it trivial to add new channel types
- **dtype validation** — `add_channel()` validates dtype against allowed set (float32, uint32, uint8)
- **JS dtype guard** — unknown channel dtypes logged and skipped instead of crashing

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
