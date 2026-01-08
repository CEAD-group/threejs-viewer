# threejs-viewer Design

## Overview

threejs-viewer is a lightweight 3D visualization tool that connects Python to a browser-based Three.js viewer via WebSocket. It's designed for robotics, scientific computing, and interactive 3D exploration.

## Architecture

```
┌─────────────────┐         WebSocket          ┌─────────────────┐
│  Python Client  │ ◄─────────────────────────► │  Browser Viewer │
│  (ViewerClient) │        localhost:5666       │   (viewer.html) │
└─────────────────┘                             └─────────────────┘
        │                                               │
        │ Runs WebSocket server                         │ Connects as client
        │ Sends commands (JSON + binary)                │ Renders with Three.js
        │ Stores state for reconnection                 │ Handles user interaction
        └───────────────────────────────────────────────┘
```

### Key Design Decisions

**1. Python as Server, Browser as Client**

Unlike typical web architectures, the Python code runs the WebSocket server. The browser viewer connects to it. This design:
- Avoids needing a separate server process
- Allows the viewer to reconnect automatically if refreshed
- Works naturally in Jupyter notebooks and scripts

The idea is that a user opens a single persistent viewer instance, and that while developing code interactively (e.g. in a REPL or notebook), they can repeatedly connect to the same viewer without restarting anything. Loaded objects are reaused, allowing for quick iterations without having to start a veiwer and a server each time.

**2. No Build Step Required**

The `viewer.html` file is self-contained:
- Uses ES modules with import maps pointing to CDN
- No npm, webpack, or bundler needed
- Just open the HTML file in any modern browser, or in a VS Code window with the built in simple webserver

**3. Binary Protocol for Large Data**

For efficiency with large meshes and polylines:
- Header is JSON (for metadata)
- Payload is raw binary (float32 positions, uint8 colors)
- 4-byte aligned for efficient parsing

```
┌──────────────┬──────────────────┬─────────────────┐
│ header_len   │ JSON header      │ binary payload  │
│ (4 bytes)    │ (padded to 4B)   │ (raw bytes)     │
└──────────────┴──────────────────┴─────────────────┘
```

**4. Animation State Persistence**

Animations are stored in Python for reconnection:
- If browser refreshes, Python re-sends the animation
- Animation playback state (time, speed) is client-side only
- Baseline visibility is captured when animation loads

**5. Z-Up Coordinate System**

The viewer uses Z-up (robotics convention):
- Camera up vector is (0, 0, 1)
- Grid is on the XY plane
- Matches ROS, URDF, and most robotics tools

## Components

### ViewerClient (Python)

The main interface for controlling the viewer:

```python
from threejs_viewer import ViewerClient

client = ViewerClient(port=5666)
client.connect()  # Starts server, waits for browser

# Add objects
client.add_sphere("ball", radius=0.5, position=[1, 0, 0])
client.add_model_binary("robot", "robot.stl")

# Update transforms
client.set_transforms({"ball": matrix_4x4})

# Load animation
client.load_animation(animation)
```

### Animation System

Pre-computed animations for interactive playback:

```python
from threejs_viewer import Animation, Frame

animation = Animation(loop=True)
for t in times:
    animation.add_frame(
        time=t,
        transforms={"obj1": matrix1, "obj2": matrix2},
        colors={"obj1": 0xFF0000},
        visibility={"obj2": False},
    )
animation.add_marker(5.0, "Collision!")
```

### Viewer (Browser)

The HTML viewer provides:
- Real-time 3D rendering with Three.js
- Orbit controls (pan, zoom, rotate)
- Animation playback controls (play/pause, scrub, speed)
- Keyboard shortcuts for frame stepping
- Automatic reconnection to Python

## Message Protocol

All messages are JSON (text) or binary with JSON header.

### Object Management

```json
{"type": "add_object", "id": "box1", "object": {"primitive": "box", "params": {"width": 1}}}
{"type": "delete_object", "id": "box1"}
{"type": "set_visibility", "id": "box1", "visible": false}
{"type": "set_color", "id": "box1", "color": 16711680}
{"type": "clear_scene"}
```

### Transform Updates

```json
{"type": "update_transform", "id": "box1", "transform": {"position": [1, 2, 3]}}
{"type": "update_transform", "id": "box1", "transform": {"matrix": [...]}}
{"type": "batch_update", "transforms": {"box1": {"matrix": [...]}, "box2": {...}}}
```

### Animation

```json
{"type": "load_animation", "animation": {"duration": 10, "frames": [...], "markers": [...]}}
{"type": "stop_animation"}
```

### Binary Messages

For `add_model_binary` and `add_polyline_binary`:
- First 4 bytes: header length (uint32 little-endian)
- Next N bytes: JSON header (null-padded to 4-byte boundary)
- Remaining bytes: raw binary data

## File Structure

```
threejs-viewer/
├── src/threejs_viewer/
│   ├── __init__.py      # Package exports
│   ├── __main__.py      # CLI entry point
│   ├── client.py        # ViewerClient class
│   ├── animation.py     # Animation, Frame, Marker classes
│   └── viewer.html      # Browser viewer (self-contained)
├── pyproject.toml
├── DESIGN.md
├── EXAMPLES.md
└── README.md
```

## Usage Patterns

### Script Mode

```python
from threejs_viewer import viewer

with viewer() as v:
    v.add_sphere("ball", radius=0.5)
    # ... interact with viewer
    input("Press Enter to exit")
```

### Interactive / Jupyter Mode

```python
from threejs_viewer import ViewerClient

client = ViewerClient()
client.connect()

# Keep running, make calls interactively
client.add_box("cube", width=1)
```

### Finding the Viewer

```bash
# CLI commands
threejs-viewer path    # Print path to viewer.html
threejs-viewer open    # Open in default browser
threejs-viewer code    # Open in VS Code
```

```python
# From Python
from threejs_viewer import ViewerClient
print(ViewerClient().viewer_path)
```

## Performance Considerations

- **Batch updates**: Use `set_transforms()` for multiple objects instead of individual calls
- **Binary transfer**: Use `add_model_binary()` and `add_polyline()` for large data
- **Animation pre-computation**: Build all frames upfront, let viewer handle playback
- **Object reuse**: Use `sync()` to avoid reloading unchanged objects

## Limitations

- Single viewer connection (one browser tab at a time)
- No server-side rendering (requires browser with WebGL)
- Animation state (playback position) not synced back to Python
- No persistence across Python restarts (objects must be re-added)
