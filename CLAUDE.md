# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Open viewer in browser (get path with)
uv run python -m threejs_viewer path

# Run examples
uv run python examples/01_primitives.py
uv run python examples/04_flying_teapots.py
```

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
- `src/threejs_viewer/viewer.html` - Three.js viewer (self-contained)
- `examples/` - Demo scripts showcasing library capabilities

### Communication Model
- **Direct connection**: Python runs WebSocket server on port 5666, browser connects to it
- **Browser survives restarts**: Viewer auto-reconnects when Python script restarts
- **Binary transfer**: Large data (models, polylines, animations) served via HTTP sidecar on port 5667, browser fetches with native `fetch()`
- **Batch updates**: `batch_update()` updates multiple objects in one message
- **60fps capable**: Minimal JSON payloads with 4x4 matrices

### Animation Modes
- **Streaming mode**: Real-time updates from Python (`batch_update()`, `set_position()`)
- **Looping mode**: Pre-computed frames with interactive playback (`load_animation()`)

### Supported Object Types
- Primitives: box, sphere, cylinder, plane, cone, torus, capsule (with optional roughness/metalness)
- Polylines: gradient-colored with colormaps (viridis, plasma, turbo)
- Meshes: pre-built triangle meshes via `add_mesh()` with optional vertex colors and normals
- Beads: toolpath extrusion via `add_bead()` — 6-vertex bevelled rectangle cross-section, vectorized numpy, per-layer vertex colors
- 3D models: GLTF/GLB, STL, OBJ, FBX, DAE, PLY, 3DS
- **draw_range**: polylines and meshes support `set_draw_range(id, 0.0-1.0)` to control visible fraction, and `draw_ranges` channel in animation frames

### Animation: Two Approaches
**Frame-based (simple, familiar):** Build frames as Python dicts — good for small animations and prototyping.
**Binary channels (fast):** Use `add_channel()` / convenience wrappers for large animations (100+ objects × 1000+ frames). Data is packed as typed arrays, transferred via HTTP, and applied with zero-copy TypedArray views in JS.

Binary channel API:
- `animation.set_frame_times(times)` — numpy array of frame times
- `animation.set_transform_data(object_ids, data)` — (n_frames, n_objects, 16) float32
- `animation.set_draw_range_data(object_ids, data)` — (n_frames, n_objects) float32
- `animation.add_channel(name, ids, data, dtype, stride, metadata)` — generic channel

Supported channel types: `transforms` (stride=16), `draw_ranges`, `colors`, `visibility`, `opacity`
Supported dtypes: `float32`, `uint32`, `uint8`
Indexed colors: `dtype="uint8"` + `metadata={"colormap": [0x44AA44, 0xFF3333]}`

Binary channels and Frame-based JSON can coexist (e.g. binary transforms + JSON clip_times). A binary channel supersedes the same-named Frame field.

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
