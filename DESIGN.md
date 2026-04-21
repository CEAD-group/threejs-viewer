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

The viewer source files live in `src/threejs_viewer/viewer/` (JS module, CSS, HTML template, cubemap images). A build script (`viewer/build.py`) inlines everything into a self-contained `viewer.html`:
- Uses ES modules with import maps pointing to CDN
- No npm, webpack, or bundler needed
- Just open the HTML file in any modern browser, or in a VS Code window with the built in simple webserver
- The `ThreeJSViewer` class can also be embedded into any page as an ES module

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
client.set_matrix("ball", matrix_4x4)

# Load animation
client.load_animation(animation)
```

### Animation System

There are two ways to build animations, designed for different use cases:

**Frame-based (simple, familiar)** — build frames as Python dicts. Good for small
animations and prototyping:

```python
from threejs_viewer import Animation, Frame

animation = Animation(loop=True)
for t in times:
    animation.add_frame(
        time=t,
        transforms={"obj1": matrix1, "obj2": matrix2},
        colors={"obj1": 0xFF0000},
        visibility={"obj2": False},
        clip_times={"glb_model": t},
    )
animation.add_marker(5.0, "Collision!")
```

**Binary channels (fast)** — for large animations (100+ objects × 1000+ frames).
Data is packed as typed arrays and transferred via HTTP:

```python
animation = Animation(loop=True)
animation.set_frame_times(np.arange(n_frames) / 60.0)
animation.set_transform_data(object_ids, transforms)       # (n_frames, n_objects, 16) float32
animation.set_draw_range_data(draw_ids, draw_ranges)        # (n_frames, n_objects) float32
animation.add_channel("colors", ids, data, dtype="uint8",   # indexed colormap
                       metadata={"colormap": [0x44AA44, 0xFF3333]})
animation.add_channel("visibility", ids, data, dtype="uint8")
```

The two approaches can be mixed: binary channels supersede Frame fields with the
same name.

Supported channel types: `transforms` (stride=16), `draw_ranges`, `colors`,
`visibility`, `opacity`, `clip_times`. Supported dtypes: `float32`, `uint32`, `uint8`.

### Embedded GLTF Animations

GLB/GLTF models can contain embedded animations (skeletal, morph targets, etc.).
These are not auto-played — Python controls them deterministically via `clip_times`:

- On load, the viewer extracts animation clips and creates an `AnimationMixer`
- `set_clip_time(id, time)` seeks the mixer to a specific time
- In pre-computed animations, the `clip_times` field on each frame drives embedded animations
- This keeps all animation state controlled by Python, matching the design of transforms

### Viewer (Browser)

The HTML viewer provides:
- Real-time 3D rendering with Three.js
- Orbit controls (pan, zoom, rotate)
- Animation playback controls (play/pause, scrub, speed)
- Keyboard shortcuts for frame stepping
- Automatic reconnection to Python

### Embedding the Viewer

`viewer.js` exports `ThreeJSViewer` as an ES module so the viewer can be mounted
into any host page (a docs site, a custom panel, an Electron shell):

```js
import { ThreeJSViewer } from './viewer.js';

const viewer = new ThreeJSViewer(document.getElementById('viewer-root'), {
    wsUrl: 'ws://localhost:5666',     // or wsPort: 5666
    htmlTemplate: TEMPLATE_HTML,       // contents of viewer/template.html (required)
    cubemapData: { px, nx, py, ny, pz, nz }, // optional: base64 JPEGs for PBR env
    autoConnect: true,                 // default; pass false to call viewer.connect() yourself
});
```

The standalone `viewer.html` is the same code with `htmlTemplate` and `cubemapData`
inlined by `viewer/build.py` — embedders supply those themselves.

**WS host contract (the probe).** Before opening each WebSocket, `connect()`
issues a `mode: 'no-cors'` HTTP GET to the URL derived from `wsUrl` by swapping
only the scheme (`ws:` → `http:`, `wss:` → `https:`). Path and query are
preserved, so the probe targets the same host, port, **path, and query** as the
pending WebSocket — not just the same host:port. An embedder using
`ws://host:port/my-path` must answer plain HTTP on `http://host:port/my-path`,
not just on `/`. The browser logs `WebSocket connection to '...' failed`
whenever a `new WebSocket()` fails to upgrade and there's no API to silence it,
so the probe is what keeps devtools quiet while the Python server is down or
restarting. In `no-cors` mode, *any* HTTP response counts as "server is up" —
different servers and proxies may return 200, 400, 404, or 426 for a plain GET
to a WebSocket URL, and all of them satisfy the probe. Only a TCP-level failure
aborts the attempt and schedules a retry.

The standard Python `websockets` library satisfies this for free as part of the
upgrade handshake. **If you point `wsUrl` at a different WS host (custom server,
reverse proxy, etc.), that host must answer *something* on plain HTTP GET for
that same URL (same host, port, path, and query) — otherwise the probe fails
forever and the WebSocket is never tried.**

**HTTP sidecar.** Independent of the probe, `client.py` runs an HTTP blob server
on `port + 1` (default 5667) for binary transfers (meshes, polylines, animation
channels). The `blob_url` in each large-data message points to this port; the
browser fetches with native `fetch()`. Embedders that proxy the WebSocket must
also expose this sidecar (or rewrite `blob_url` to a reachable host) for binary
loads to work.

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
{"type": "unload_animation", "restore_visibility": true}
{"type": "pause_animation"}
{"type": "resume_animation"}
```

For binary-channel animations, the message is `load_animation_http` with a channel manifest:

```json
{
  "type": "load_animation_http",
  "blob_url": "http://localhost:5667/animation_XXXX",
  "frame_count": 2000,
  "frame_times": [0.0, 0.0167, ...],
  "duration": 33.3,
  "fps": 60,
  "loop": true,
  "markers": [],
  "channels": [
    {"name": "transforms",  "ids": ["obj1", ...], "dtype": "float32", "stride": 16},
    {"name": "colors",      "ids": ["s0", ...],   "dtype": "uint8",   "stride": 1, "colormap": [4489156, 16724787]},
    {"name": "visibility",  "ids": ["s0", ...],   "dtype": "uint8",   "stride": 1},
    {"name": "draw_ranges", "ids": ["obj1", ...], "dtype": "float32", "stride": 1}
  ],
  "frames_meta": [{"index": 42, "clip_times": {"model1": 1.5}}]
}
```

The binary blob (served via HTTP) is the concatenation of all channel data in manifest order.
Each channel's byte size = `n_frames × len(ids) × stride × sizeof(dtype)`.
Channels are sorted by dtype size descending (float32 first, uint8 last) to avoid alignment issues.

### Embedded GLTF Animations

```json
{"type": "set_clip_time", "id": "model1", "time": 1.5}
{"type": "batch_set_clip_time", "clip_times": {"model1": 1.5, "model2": 0.8}}
```

Animation frames can also include `clip_times` to drive embedded animations during playback.

### Binary Messages

For large data (meshes, polylines, animations), the Python client serves binary
payloads via an HTTP sidecar on port 5667. A JSON message over WebSocket carries
metadata and a `blob_url`; the browser fetches the binary blob with native `fetch()`.

For animation channels, the blob is a concatenation of all channel typed arrays
(see the channel manifest in the `load_animation_http` message above).

## File Structure

```
threejs-viewer/
├── src/threejs_viewer/
│   ├── __init__.py      # Package exports
│   ├── __main__.py      # CLI entry point
│   ├── client.py        # ViewerClient class
│   ├── animation.py     # Animation, AnimationChannel, Frame, Marker classes
│   ├── viewer.html      # Generated standalone viewer (do not edit)
│   └── viewer/          # Viewer source files (edit these)
│       ├── viewer.js    # ES module: ThreeJSViewer + ParametricTube, CameraController, ShadingDebugController
│       ├── viewer.css   # Scoped CSS (.threejs-viewer)
│       ├── template.html # UI controls template
│       ├── build.py     # Inlines sources → viewer.html
│       └── static/      # Cubemap JPEG images
├── plans/              # Decision-making artifacts for undecided future work (NOT landed architecture)
├── pyproject.toml
├── DESIGN.md
├── EXAMPLES.md
└── README.md
```

`plans/` holds working documents for refactors or features that have been analyzed but not committed to. Each file describes its own status. These are *not* architecture documentation — they describe paths-not-yet-taken. Delete each file once its decision is resolved (landed or formally declined). If you're an agent looking for landed architecture, skip this directory.

### Viewer internal structure

`viewer.js` is a single concat-built ES module (no bundler). It is organized as `ThreeJSViewer` plus a small set of in-file controller classes, each owning its own state and helpers:

- `ParametricTube` — per-tube geometry build, draw-range frontier morph, LOD worker dispatch, color-attribute updates.
- `CameraController` — perspective/ortho cameras, framing helpers (`_fitCameraToBox`), scene bounds, near/far updates.
- `ShadingDebugController` — `M`-key wireframe cycle and `N`-key shading-debug cycle, plus cached normal / UV materials.

Clipping and animation are intentionally kept as banner-grouped methods on `ThreeJSViewer` rather than classes (`// ========== Clipping ==========` / `// ========== Animation ==========`). Both subsystems cross-cut enough shared viewer state and DOM refs that a naive class extraction would be pure prefix churn; a proper extraction is tracked as a future decision in `plans/viewer-refactor-pass-two.md`.

Ring/color helpers used by parametric tubes (`writeRingVerts`, `writeCapRingVerts`, `fillRGBBlock`, `sampleChamferedRect`, `distanceWeightedRDP`) live as free functions at the top of the file. The LOD worker is embedded as a template-literal source string; shared helpers are duplicated into the worker string at build time because Web Workers can't access the main thread's module scope.

JSDoc + `// @ts-check` surface type errors in-editor. Type check: `npx tsc --noEmit -p jsconfig.json`.

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

client = ViewerClient(open_browser=False)  # disable auto-open for headless/remote
client.connect()

# Keep running, make calls interactively
client.add_box("cube", width=1)
```

### Finding the Viewer

The browser opens automatically by default. To open manually or find the path:

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

- **Batch updates**: Use `batch_update()` for multiple objects instead of individual calls
- **Binary transfer**: Use `add_model_binary()` and `add_polyline()` for large data
- **Binary animation channels**: Use `add_channel()` / `set_transform_data()` for large animations — avoids JSON serialization overhead on both Python and JS sides
- **Animation pre-computation**: Build all frames upfront, let viewer handle playback
- **Performance warnings**: Both Python (logging) and JS (console.warn) emit hints when JSON animation data is large enough that binary channels would help

## Limitations

- Single viewer connection (one browser tab at a time)
- No server-side rendering (requires browser with WebGL)
- Animation state (playback position) not synced back to Python
- No persistence across Python restarts (objects must be re-added)
