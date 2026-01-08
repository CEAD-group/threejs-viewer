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
- `src/threejs_viewer/animation.py` - Animation classes (Frame, Animation, Marker)
- `src/threejs_viewer/viewer.html` - Three.js viewer (self-contained)
- `examples/` - Demo scripts showcasing library capabilities

### Communication Model
- **Direct connection**: Python runs WebSocket server on port 5666, browser connects to it
- **Browser survives restarts**: Viewer auto-reconnects when Python script restarts
- **Binary transfer**: Large polylines and models use binary protocol for efficiency
- **Batch updates**: `batch_update()` updates multiple objects in one message
- **60fps capable**: Minimal JSON payloads with 4x4 matrices

### Animation Modes
- **Streaming mode**: Real-time updates from Python (`batch_update()`, `set_position()`)
- **Looping mode**: Pre-computed frames with interactive playback (`load_animation()`)

### Supported Object Types
- Primitives: box, sphere, cylinder, plane, cone, torus
- Polylines: gradient-colored with colormaps (viridis, plasma, turbo)
- 3D models: GLTF/GLB, STL, OBJ, FBX, DAE, PLY, 3DS

### Examples
- `01_primitives.py` - Basic shapes with colors and positions
- `02_polylines.py` - Gradient lines with different colormaps
- `03_animation_basics.py` - Solar system with looping animation
- `04_flying_teapots.py` - Flying Utah teapots (model loading + animation)
- `05_lissajous_curves.py` - Mathematical curves with tracer animation
- `06_realtime_streaming.py` - Real-time streaming mode (bouncing spheres)
