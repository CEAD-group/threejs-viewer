# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run tests (unit only, no browser needed)
uv run pytest

# Run tests including Playwright browser integration tests
uv sync --group browser-test
uv run playwright install chromium
uv run pytest -v

# Run linting
uv run ruff check . && uv run ruff format --check .

# Open viewer manually (browser auto-opens by default)
uv run python -m threejs_viewer path

# Run examples
uv run python examples/01_primitives.py
uv run python examples/04_flying_teapots.py
```

## Versioning

Source files use `0.0.0-dev` as a placeholder version. CI replaces it before build/publish:
```bash
sed -i "s/0\.0\.0-dev/$VERSION/g" pyproject.toml src/threejs_viewer/__init__.py src/threejs_viewer/viewer/viewer.js
uv run python src/threejs_viewer/viewer/build.py  # regenerate viewer.html with substituted version
```
The placeholder appears in three files: `pyproject.toml`, `src/threejs_viewer/__init__.py`, `src/threejs_viewer/viewer/viewer.js`. The build step propagates the version into `viewer.html`. Never commit a real version number — always keep `0.0.0-dev`.

## Project Overview

A lightweight Three.js viewer designed to be controlled from Python/Jupyter notebooks. Primary use case: visualizing 3D data and animations at 60fps. The browser viewer persists across Python script restarts.

## Architecture

### Tech Stack
- **Viewer**: Static HTML/JS (no build tools, no Node, no frameworks)
- **Three.js**: Loaded from CDN (unpkg)
- **Python**: WebSocket server via `websockets` library

### Core Files
- `src/threejs_viewer/client.py` - Python client that runs WebSocket server
- `src/threejs_viewer/animation.py` - Animation classes (Frame, Animation, AnimationChannel, Marker)
- `src/threejs_viewer/viewer.html` - Standalone viewer (**generated** — do not edit, run `uv run python src/threejs_viewer/viewer/build.py` to regenerate)
- `src/threejs_viewer/viewer/` - Viewer source files (edit these):
  - `viewer.js` - ES module exporting `ThreeJSViewer(container, options)` class
  - `viewer.css` - Scoped CSS under `.threejs-viewer` class
  - `template.html` - HTML template for toolbar, clipping panel, animation controls
  - `build.py` - Build script that inlines all sources into standalone `viewer.html`
  - `static/*.jpg` - Cubemap face images for PBR environment
- `examples/` - Demo scripts showcasing library capabilities
- `tests/` - Unit tests + Playwright browser integration tests (browser tests auto-skip without pytest-playwright)

### Communication Model
- **Direct connection**: Python runs WebSocket server on port 5666 (default), browser connects to it. Port is overridable via `?ws_port=` query param in viewer URL.
- **Browser survives restarts**: Viewer auto-reconnects when Python script restarts
- **Binary transfer**: Large data (models, polylines, animations) served via HTTP sidecar on port 5667, browser fetches with native `fetch()`
- **Batch updates**: `batch_update()` updates multiple objects in one message
- **60fps capable**: Minimal JSON payloads with 4x4 matrices

### Animation Modes
- **Streaming mode**: Real-time updates from Python (`batch_update()`, `set_matrix()`)
- **Looping mode**: Pre-computed frames with interactive playback (`load_animation()`)

### Supported Object Types
- **Groups**: `add_group(id, parent=...)` — empty transform nodes for parent-child hierarchies. Children inherit parent transforms. All `add_*` methods accept an optional `parent` parameter.
- Primitives: box, sphere, cylinder, capsule (with optional roughness/metalness)
- Polylines: gradient-colored with colormaps (viridis, plasma, turbo)
- Meshes: pre-built triangle meshes via `add_mesh()` with optional vertex colors and normals
- Beads: toolpath extrusion via `Toolpath.to_mesh()` + `add_mesh()` — 6-vertex bevelled rectangle cross-section, vectorized numpy, per-layer vertex colors
- **Parametric tubes**: `add_parametric_tube(id, spine, widths, heights, colors=..., n_cross_section_verts=..., corner_radius_frac=...)` — variable-cross-section extruded tube built on the client from per-spine-point parameter arrays. Rounded-rectangle cross-section with torsion-free parallel-transport frames (V anchored to global +Z). Supports `draw_range` with smooth frontier-ring morphing (the next ring's vertices are lerped to the interpolated spine position for pixel-smooth growth), the `draw_ranges` animation channel, and cheap color-mode swaps via `update_parametric_tube_colors(id, colors)`.
- 3D models: GLTF/GLB, STL, OBJ, FBX, DAE, PLY, 3DS
- **draw_range**: polylines, meshes, and parametric tubes support `set_draw_range(id, 0.0-1.0)` to control visible fraction, and `draw_ranges` channel in animation frames
- **Clipping plane**: interactive cross-section plane with GUI panel (press C to toggle). Supports single plane and slab (dual-plane) modes. Rotation gizmo for orienting the plane; arrow keys nudge position. V key snaps ortho camera to clip normal for slice inspection. Programmatic control via `set_clipping_plane(normal, distance, show_helper)` / `set_clipping_slab(normal, center, thickness, show_helper)` / `disable_clipping_plane()` / `set_clipping_defaults(normal, distance)`

### Animation: Two Approaches
**Frame-based (simple, familiar):** Build frames as Python dicts — good for small animations and prototyping.
**Binary channels (fast):** Use `add_channel()` / convenience wrappers for large animations (100+ objects × 1000+ frames). Data is packed as typed arrays, transferred via HTTP, and applied with zero-copy TypedArray views in JS.

Binary channel API:
- `animation.set_frame_times(times)` — numpy array of frame times
- `animation.set_transform_data(object_ids, data)` — (n_frames, n_objects, 16) float32
- `animation.set_draw_range_data(object_ids, data)` — (n_frames, n_objects) float32
- `animation.set_clip_time_data(object_ids, data)` — (n_frames, n_objects) float32
- `animation.add_channel(name, ids, data, dtype, stride, metadata)` — generic channel

Supported channel types: `transforms` (stride=16), `draw_ranges`, `colors`, `visibility`, `opacity`, `clip_times`
Supported dtypes: `float32`, `uint32`, `uint8`
Indexed colors: `dtype="uint8"` + `metadata={"colormap": [0x44AA44, 0xFF3333]}`

Binary channels and Frame-based JSON can coexist. A binary channel supersedes the same-named Frame field.

**Interpolation (per-channel):** Every *binary* animation channel carries its own interpolation mode — `"linear"` (lerp/slerp between keyframes) or `"hold"` (keep the previous keyframe's value until the next one hits). **Every channel defaults to `"linear"`** — opt out per channel with `add_channel(interpolation="hold")` or via the convenience setters (`set_transform_data`, `set_draw_range_data`, `set_clip_time_data`, `set_camera_target`, `set_camera_position`). Translations lerp, rotations slerp, float channels lerp element-wise, and `colors` lerps hex values in 8-bit RGB space (works for direct hex and colormap-indexed `uint8`). The `visibility` channel is a boolean and always left-holds regardless of the setting — a "linear bool" has no meaningful interpretation; the setting is accepted and validated but has no effect on that channel. **JSON `Frame` objects do not carry interpolation metadata and always interpolate linearly** between consecutive frames (visibility aside) — if you need hold behavior, use a binary channel. There is no Animation-wide knob — it was redundant with per-channel. Linear playback lets producers sample at the signal's bandwidth (e.g. 10 Hz) and still get smooth 60 fps. See `examples/17_animation_interpolation.py` for a HOLD ⇄ LINEAR comparison on the same tumbling capsules.

### Examples
- `01_primitives.py` - Basic shapes with colors and positions
- `02_polylines.py` - Gradient lines with different colormaps
- `03_animation_basics.py` - Solar system with looping animation
- `04_flying_teapots.py` - Flying Utah teapots (model loading + animation)
- `05_lissajous_curves.py` - Mathematical curves with tracer animation
- `06_realtime_streaming.py` - Real-time streaming mode (bouncing spheres)
- `07_stress_test.py` - Torus knot tube with hundreds of followers (performance test)
- `08_glb_models.py` - GLB models with PBR materials (DamagedHelmet, Avocado)
- `09_animated_glb.py` - Embedded GLTF animation via clip_times (AnimatedMorphCube + orbiting Avocado)
- `10_animation_stress_test.py` - Animation stress test (520 objects × 2499 frames, vectorized numpy)
- `11_toolpath.py` - Spiral vase toolpath with draw_range animation (polyline + bead mesh + nozzles, 800k points, alternating layer colors, binary animation channels)
- `12_transparency.py` - Transparency and opacity control (set_opacity, set_color with opacity)
- `13_material_properties.py` - PBR material properties (roughness/metalness grid on primitives)
- `14_grouping.py` - Object grouping with animated robot arm (nested parent-child hierarchy, local joint transforms)
- `15_toolpath_interrupted.py` - Interrupted extrusion with travel moves: per-point width array (0=travel), pill/racetrack path with varying point density, merge_animation_points for segment-aligned animation (frame times merged into mesh geometry → no partial triangle rings)
- `16_clipping_plane.py` - Interactive clipping plane with slab mode (nested spheres sliced to reveal internals, single plane + dual-plane slab, programmatic + GUI control)
- `17_animation_interpolation.py` - HOLD ⇄ LINEAR comparison on a sparse (3 Hz) tumbling-capsules animation. Auto-alternates the two modes on the same `transforms` channel so the lerp/slerp effect is obvious.
- `18_parametric_tube.py` - Variable-cross-section bead via `parametric_tube` with animated draw_range, nozzle-tip capsule indicator, and live color-mode swap through `update_parametric_tube_colors`.
