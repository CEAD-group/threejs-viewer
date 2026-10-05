# threejs-viewer

Lightweight Three.js viewer controlled from Python via WebSocket.

![A textured extrusion bead being printed layer by layer](docs/media/bead_texture_print.gif)

A Python client runs a WebSocket server that a browser-based Three.js viewer connects to. Designed for robotics visualization, additive manufacturing and machining toolpaths, scientific computing, and interactive 3D exploration. The browser tab survives Python restarts and reconnects on its own.

![VS Code integration](docs/demo.gif)

## Features

- **Simple API**: Add primitives, load models, update transforms
- **PBR materials**: Roughness, metalness, transparency, environment reflections and a shadow-casting sun on all objects
- **GLB/PBR support**: Load GLB models with PBR materials, embedded skeletal/morph animations, Draco compression
- **Animation support**: Pre-compute animations, scrub timeline, adjust playback speed
- **Binary channels**: Efficient transfer of large animations (100k+ objects × frames)
- **Toolpath visualization**: Extruded bead tubes with per-point colors, draw_range animation, and automatic LOD — handles 1M+ point toolpaths at 60 fps with smooth camera-distance-adaptive simplification via Web Worker
- **Bead textures**: A colour, normal and roughness map wrapped around the bead and repeated along the path, so a printed part reads like extruded plastic (die lines, sharkskin ripple, chopped fibres)
- **5-axis swept tool body**: `add_swept_tool` lofts an oriented shank/holder profile about the per-station tool axis (decoupled from the path tangent) into a swept surface — visualizes 5-axis reorientation rate and tool-body collisions, with per-station heatmap colours and draw_range reveal
- **GPU point clouds**: `add_points` renders millions of unconnected points (voxel fields, metrology/deviation clouds, LiDAR scans) in a single `THREE.Points` draw call, with per-point colormap colours, distance size-attenuation, and draw_range reveal
- **Live point streaming**: `append_points` grows an existing cloud in place — only the new points cross the wire and only they are uploaded to the GPU, so a producer streaming for hours (a laser tracker at 1 kHz) stays O(new points) per update instead of re-sending the whole cloud
- **Per-point time windows**: give each point a `[birth, removal)` lifetime and scrub a global time — points appear/disappear **out of buffer order** in the vertex shader (material-removal animations that a prefix reveal can't express), driven live or from the animation slider
- **Octree LOD streaming**: `add_points(lod=True)` builds a Potree-style sampled octree in Python and streams node payloads on demand — draw a ~1.5M-point budget of the biggest-on-screen nodes out of clouds far larger than one draw call, refining as you zoom; composes with the time windows at every LOD
- **Interaction from the browser back to Python**: pick a point along any line or bead, click objects, drag move/rotate gizmos, drag axis-control range widgets, press shortcuts on client-defined menus; every event lands in a Python callback
- **Annotations and section views**: linear dimensions and labelled points, an interactive clipping plane or slab with gizmos, a shader floor grid, fog and eye-dome lighting depth cues for dense line drawings
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

The viewer opens automatically in your default browser. To open it manually:

```bash
threejs-viewer open
# Or: threejs-viewer path  (prints path to viewer.html)
```

## Gallery

Every image comes from a script in [examples/](examples/); the file name is given under each one.

### Toolpaths and beads

| | |
|---|---|
| ![Textured bead, close-up of a corner](docs/media/bead_texture_corner.png) | ![Textured printed part](docs/media/bead_texture_hero.png) |
| Colour + normal + roughness maps follow the bead around every corner. `38_bead_texture.py` | The same part: one continuous 8 x 3 mm bead, 14 layers, overhanging round side. `38_bead_texture.py` |
| ![Spiral vase printed by a draw_range animation](docs/media/toolpath_vase.gif) | ![Interrupted toolpath with travel moves](docs/media/toolpath_interrupted.png) |
| A spiral vase bead revealed by a `draw_ranges` channel, viridis colours tinting the texture. `11_toolpath.py` | Travel moves taper the bead to zero width. `15_toolpath_interrupted.py` |
| ![Tube break mask](docs/media/tube_breaks.png) | ![5-axis swept tool body](docs/media/swept_tool.gif) |
| Bottom: one spine bridging three rings with stray cones. Top: the same spine with `break_before`, three capped strips. `28_tube_breaks.py` | A tilting tool lofted along a contact path, coloured by reorientation rate. `26_swept_tool.py` |

### Point clouds

| | |
|---|---|
| ![Deviation point cloud](docs/media/point_cloud.png) | ![draw_range sweep over a point cloud](docs/media/point_cloud_sweep.gif) |
| 160k-point machined surface coloured by signed deviation, one draw call. `25_point_cloud.py` | The same cloud revealed along the toolpath by `draw_ranges`. `25_point_cloud.py` |
| ![Octree LOD with birth and removal times](docs/media/points_octree_lod.gif) | ![Live point stream](docs/media/live_points_stream.gif) |
| Octree LOD streaming node detail on demand while a build front adds points and an erosion front removes them, out of buffer order. `27_points_octree_lod.py` | `append_points` growing a cloud a few thousand points at a time, only the new points cross the wire. `30_live_points_stream.py` |

### Lines, picking and depth cues

| | |
|---|---|
| ![Fog and eye-dome lighting on a dense polyline](docs/media/depth_cues.png) | ![Polyline picking](docs/media/polyline_picking.png) |
| A 12k-point native polyline under distance fog + eye-dome lighting (`D` / `Shift+D`). `20_line_depth_cues.py` | Hovering a line or bead glides a marker to the closest point and reads out the arc-length fraction; a click reaches Python. `22_polyline_picking.py` |

### Models and animation

| | |
|---|---|
| ![GLB models with PBR materials](docs/media/glb_models.png) | ![PBR material sampler](docs/media/materials_sampler.png) |
| DamagedHelmet and Avocado with the built-in HDR environment. `08_glb_models.py` | Thirty labelled PBR materials from one glTF (clearcoat, sheen, anisotropy, IOR). `31_watch_gltf.py` |
| ![Robot arm with nested groups](docs/media/robot_arm.gif) | ![Animation stress test](docs/media/animation_stress.gif) |
| Nested parent-child groups animated with local joint transforms. `14_grouping.py` | 520 followers on a torus-knot tube driven by binary `transforms` channels. `10_animation_stress_test.py` |
| ![Flying teapots](docs/media/flying_teapots.gif) | ![Billboards](docs/media/billboards.png) |
| Utah teapots on a looping animation. `04_flying_teapots.py` | Checkered donuts in every `set_billboard` mode, kept facing the camera. `34_billboard_modes.py` |

### Interaction and scene furniture

| | |
|---|---|
| ![Pinned move gizmos](docs/media/gizmos.png) | ![Axis-control range widgets](docs/media/axis_controls.png) |
| Six pinned gizmos: 1-, 2- and 3-DOF, in world and local space; drags report back to Python. `24_gizmo_dimensions.py` | Rotary and linear range widgets anchored to their target objects. `36_axis_control.py` |
| ![Dimension annotations](docs/media/annotations.png) | ![Clipping slab](docs/media/clipping_plane.png) |
| Linear dimensions, a diagonal and labelled points on a box. `add_dimension` / `add_point_annotation` | A dual-plane clipping slab through nested spheres, with the `C` panel and gizmos. `16_clipping_plane.py` |
| ![Shader floor grid](docs/media/floor_grid.png) | ![Client-defined menu](docs/media/client_menu.png) |
| An anti-aliased, distance-faded floor grid with screen-space-stable line width. `29_floor_grid.py` | A Python-defined menu on the right rail: eyes, toggles, selects and shortcuts round-trip to Python. `35_client_menu.py` |
| ![Lighting panel](docs/media/lighting_panel.png) | |
| The `E` panel: tone mapping, exposure, environment map, ambient and sun. | |

## Usage

### Objects

```python
from threejs_viewer import Toolpath, viewer

# Primitives with PBR materials
v.add_box("box1", width=1, height=2, depth=0.5, color=0x4A90D9, roughness=0.5, metalness=0.1)
v.add_sphere("ball", radius=0.5, position=[2, 0, 0], roughness=0.3, metalness=0.7)
v.add_cylinder("cyl1", radius_top=0.3, radius_bottom=0.5, height=1)

# 3D models (binary transfer)
v.add_model_binary("robot", "robot.stl", format="stl")

# Polylines with colormaps
v.add_polyline("path", points, colors=z_values, colormap="viridis", line_width=3)

# Bead toolpath (parametric tube, built client-side)
tp = Toolpath.from_points(points, bead_width=0.3, bead_height=0.08)
tp.colorize(per_point_rgb)
v.add_toolpath("bead", tp)

# Bead texture: colour + normal + roughness maps wrapped around the bead,
# repeated every `texture_length` units of path (examples/bead_texture_maker.py
# generates a tiling extruded-plastic set)
v.add_toolpath("bead", tp, texture="bead_color.png", normal_map="bead_normal.png",
               roughness_map="bead_roughness.png", texture_length=25.0, roughness=1.0)

# Transparency
v.set_opacity("box1", 0.5)
v.set_color("ball", 0xFF0000, opacity=0.3)
```

### Transforms

```python
# Single object
v.set_matrix("box1", matrix_4x4.flatten().tolist())

# Batch update (efficient for 60fps)
v.batch_update({
    "link1": {"position": [1, 2, 0.5]},
    "link2": {"position": [3, 0, 1], "rotation": [0, 0, 1.57]},
})
```

### Picking points on lines & beads

```python
# Click a point anywhere along any polyline or parametric tube; the pick
# travels back to Python. Hovering glides a marker to the closest point.
def on_pick(pick):
    print(f"{pick['kind']} {pick['id']}: {pick['fraction']:.1%} along  at {pick['point']}")

v.on_polyline_pick(on_pick)   # also enables picking
```

Each click sends `{id, kind, fraction, point, local_point, segment, t}` where
`kind` is `"line"` or `"tube"`. Movement is **continuous** — the marker never
snaps to vertices. Picking is **opt-out per object** — pass `pickable=False` to
`add_polyline` / `add_parametric_tube` to exclude one from picking (it stays
rendered); adding objects costs nothing when picking is never enabled. For a tube, `segment` indexes the full-resolution spine 1:1
with the per-point arrays you built it from, so it doubles as a lookup key for
other per-point data at the picked point. A browser embedder can subscribe in
JS instead — `viewer.onPolylinePick(cb)` (click) / `viewer.onPolylineHover(cb)`
(every hover move) — for a live readout with no Python round-trip. See
[examples/22_polyline_picking.py](examples/22_polyline_picking.py).

### Point clouds: live append, time windows & octree LOD

```python
# Live streaming: seed the cloud once, then append as data arrives. Both the
# wire payload and the GPU upload cover the new points only, so this holds up
# for hours at a few flushes per second (the browser grows the buffer with
# spare capacity, doubling when it runs out).
v.add_points("live", first_chunk, colors=first_values, colormap="turbo",
             cmin=-0.5, cmax=0.5)      # fix the colour range up front...
while streaming:
    v.append_points("live", chunk, colors=values)   # ...appends clamp to it
```

Appending to an id that was never added (or was deleted, cleared, or created
with `lod=` / `birth_times`) raises `ValueError` — a stream must not lose data
silently. A cloud is not replayed when the browser reloads, so re-seed with
`add_points` after a reconnect.

```python
# Per-point lifetimes: a point is visible while birth_time <= t < removal_time.
# NaN = unbounded (never born-late / never removed).
v.add_points("stock", positions, colors=depth, colormap="turbo",
             birth_times=birth, removal_times=removal)
v.set_points_time("stock", 3.0)   # scrub the time directly...

# ...or map the animation slider onto it (drag to scrub points in/out):
anim = Animation(loop=True)
anim.set_frame_times(times)
anim.set_point_time_data(["stock"], times.reshape(-1, 1))
v.load_animation(anim)

# Octree LOD for clouds beyond one draw call (10M+ points): nodes stream
# on demand from Python under a point budget as the camera moves. Keep the
# Python process running — it serves node payloads.
v.add_points("big", positions, colors=scalars, lod=True)
# tunable: lod={"node_capacity": 15000, "point_budget": 1_500_000,
#               "refine_pixels": 12, "seed": 0}
```

Both compose: octree node samples are stratified over the time values, so
scrubbing thins every LOD level uniformly. On LOD clouds use the time window
instead of `set_draw_range` (buffer order is spatial there). See
[examples/27_points_octree_lod.py](examples/27_points_octree_lod.py) — point
count tunable from the command line.

### Animations

```python
from threejs_viewer import Animation

animation = Animation(loop=True)
for t in times:
    animation.add_frame(
        time=t,
        transforms=compute_transforms(t),
        colors={"robot": 0xFF0000 if collision else 0x00FF00},
        clip_times={"glb_model": t},  # drive embedded GLTF animations
    )
animation.add_marker(3.5, "Collision detected")

client.load_animation(animation)
```

Viewer controls: Space (play/pause), Arrow keys (step frames), 1-5 (speed), L (loop)

### Binary Channels (Large Animations)

For animations with many objects or frames, use binary channels instead of Frame dicts for much faster serialization and transfer:

```python
import numpy as np
from threejs_viewer import Animation

animation = Animation(loop=True)
animation.set_frame_times(np.arange(n_frames) / 60.0)

# Transforms: (n_frames, n_objects, 16) float32
animation.set_transform_data(object_ids, transform_array)

# Draw ranges: (n_frames, n_objects) float32
animation.set_draw_range_data(object_ids, draw_range_array)

# Clip times for embedded GLTF animations: (n_frames, n_objects) float32
animation.set_clip_time_data(object_ids, clip_time_array)

# Colors with indexed colormap: (n_frames, n_objects) uint8
animation.add_channel("colors", object_ids, color_indices, dtype="uint8",
                       metadata={"colormap": [0x44AA44, 0xFF3333]})

# Visibility: (n_frames, n_objects) uint8 (0=hidden, 1=visible)
animation.add_channel("visibility", object_ids, vis_data, dtype="uint8")

client.load_animation(animation)
```

Binary channels and Frame-based JSON can be mixed. A binary channel supersedes the same-named Frame field.

### GLB Models with Embedded Animations

```python
# Load a GLB with embedded animations (skeletal, morph targets)
client.add_model_binary("fox", "fox.glb", format="glb")

# Seek embedded animation to a specific time (seconds)
client.set_clip_time("fox", 1.5)
```

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
