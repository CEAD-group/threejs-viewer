# threejs-viewer

Lightweight Three.js viewer controlled from Python via WebSocket.

![Demo](docs/demo.gif)

A Python client runs a WebSocket server that a browser-based Three.js viewer connects to. Designed for robotics visualization, scientific computing, and interactive 3D exploration.

## Features

- **Simple API**: Add primitives, load models, update transforms
- **Animation support**: Pre-compute animations, scrub timeline, adjust playback speed
- **Binary transfer**: Efficient loading of large meshes and polylines
- **Auto-reconnect**: Browser reconnects automatically, animations persist
- **Z-up coordinates**: Robotics convention (matches ROS, URDF)
- **No build step**: Self-contained HTML viewer, just open in browser

## Installation

```bash
pip install threejs-viewer
```

## Quick Start

```python
from threejs_viewer import viewer

# Start server and wait for browser to connect
v = viewer()

# Add objects
v.add_sphere("ball", radius=0.3, color=0xFF0000, position=[0, 0, 0.5])
v.add_box("ground", width=5, height=5, depth=0.1, color=0x444444)

# Keep running
input("Press Enter to exit")
```

Open the viewer in your browser:

```bash
threejs-viewer open
# Or: threejs-viewer path  (prints path to viewer.html)
```

## Usage

### Objects

```python
# Primitives
client.add_box("box1", width=1, height=2, depth=0.5, color=0x4A90D9)
client.add_sphere("sphere1", radius=0.5, position=[2, 0, 0])
client.add_cylinder("cyl1", radius_top=0.3, radius_bottom=0.5, height=1)

# 3D models (binary transfer)
client.add_model_binary("robot", "robot.stl", format="stl")

# Polylines with colormaps
client.add_polyline("path", points, colors=z_values, colormap="viridis", line_width=3)
```

### Transforms

```python
# Single object
client.set_position("box1", 1.0, 2.0, 0.5)
client.set_matrix("box1", matrix_4x4.flatten().tolist())

# Batch update (efficient for 60fps)
client.set_transforms({
    "link1": matrix1.flatten().tolist(),
    "link2": matrix2.flatten().tolist(),
})
```

### Animations

```python
from threejs_viewer import Animation

animation = Animation(loop=True)
for t in times:
    animation.add_frame(
        time=t,
        transforms=compute_transforms(t),
        colors={"robot": 0xFF0000 if collision else 0x00FF00},
    )
animation.add_marker(3.5, "Collision detected")

client.load_animation(animation)
```

Viewer controls: Space (play/pause), Arrow keys (step frames), 1-5 (speed), L (loop)

## Documentation

- [DESIGN.md](DESIGN.md) - Architecture and protocol details
- [examples/](examples/) - Runnable demo scripts

## CLI

```bash
threejs-viewer path    # Print path to viewer.html
threejs-viewer open    # Open in default browser
threejs-viewer code    # Open in VS Code (use "Show Preview" for docked view)
```

## License

MIT
