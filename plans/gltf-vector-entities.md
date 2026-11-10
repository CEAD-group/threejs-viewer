# glTF vector entities: DXF-style curves, layers and linetypes in glTF

> **Status: phases 1 and 2 implemented (branch `feat/vector-entities`).** Design record for carrying exact vector geometry (points, lines,
> polylines with bulges, circles, arcs, ellipses, NURBS splines) with
> DXF-style layers and linetypes in glTF, and for the viewer support that
> renders it with zoom-dependent refinement.
>
> **Scope:** the schema, generic vector objects in the viewer and its Python
> client, and loading the data from glTF. An external CAD exporter writes
> the files. Projection onto a view plane and DXF export are separate, later
> work; the schema keeps DXF semantics so that work is a direct mapping.

## Requirements

1. Exact geometry, stored the way DXF stores it.
2. Layers and linetypes with DXF semantics.
3. A loader without support still loads the model.
4. The data **must survive glTF-Transform, including Draco compression**
   (`gltf-transform draco` and `gltf-transform optimize --compress draco`).
5. The viewer renders the entities, also as standalone objects
   (`add_circle()` and friends), and refines curves as the camera gets closer.

## What glTF-Transform keeps (measured)

Tested with `@gltf-transform/cli` 4.5.1 on a GLB with two components, each
with a vector child node (fallback `LINES` + `POINTS` primitives), and the
payload stored on the root, the scene, nodes, primitives and materials, as
both extension and `extras`.

| Location | `draco` | `optimize --compress draco` | `optimize` (meshopt) | `quantize` |
| --- | --- | --- | --- | --- |
| Unregistered extension (anywhere) | **dropped** | **dropped** | **dropped** | **dropped** |
| Extension, registered via `--config` | kept | kept | — | — |
| Root `extras` | kept | kept | kept | kept |
| Asset / scene `extras` | kept | kept | kept | kept |
| Node `extras` | kept | kept only on nodes that survive | same | kept |
| Primitive `extras` | kept | **dropped** (join) | **dropped** | kept |
| Material `extras` | kept | kept | kept | kept |
| Node hierarchy and transforms | unchanged | children flattened; second component **joined** into the first | flattened, joined, and node T/S **rewritten** by quantization | mesh moved to a new child node; mesh nodes get a dequantization **scale** |
| `LINES` primitives | kept, not Draco-compressed | kept, merged | kept, merged | kept, quantized |
| `POINTS` primitives | kept | **dropped** | **dropped** | kept |

Conclusions:

- **Store the data in root `extras`.** They survive every command with no
  setup. Node-level data disappears when `join` merges nodes, and
  node-local coordinates break when quantization rewrites node transforms.
- **Store coordinates in scene (world) space**, for the same reason.
- **Draco itself is harmless.** It compresses only indexed triangle
  primitives; line and point primitives pass through. The damage comes from
  the other `optimize` steps (`join`, `flatten`, `quantize`, `instance`).
- The **fallback lines are best effort**: they survive, but get merged, and
  fallback points are dropped by `optimize`.
- **Material `extras` survive** everywhere, so they mark the fallback
  primitives reliably.

## Decision: extras only, plus a marker extension

The payload lives **only** in `extras.TJSV_vector_entities` on the glTF
root. A small marker extension at the root,
`extensions.TJSV_vector_entities = { "version": 1, "storage": "extras" }`,
listed in `extensionsUsed` and never in `extensionsRequired`, declares the
data to validators and tools. Tools that drop the marker lose nothing:
readers detect the data by the `extras` key alone.

What this gives up compared with storing the data in the extension:

- Binary buffers: `extras` cannot safely reference accessors (tools neither
  remap nor keep them), so arrays are inline JSON. If size becomes a
  problem, v1 can add base64 strings for large arrays
  (`{"base64": "...", "dtype": "f4"}`); strings survive like any JSON.
- Safe failure: a tool that transforms the whole scene (for example
  `gltf-transform center`) copies `extras` without updating them, so the
  vectors no longer line up. An extension would have been dropped instead.
  Pipelines should avoid scene-wide transforms after export, or re-export.

`TJSV` is a placeholder vendor prefix; rename it before the schema is frozen.

## Payload

The same entity format is used on the wire (`add_vector_entities`) and in
glTF `extras`. Layers and linetypes are referenced **by name**, as in DXF.

```jsonc
{
  "version": 1,
  "units": "m",                       // informational; glTF is metres
  "linetypes": [
    { "name": "DASHED", "description": "__ __ __", "pattern": [0.0127, -0.00635] },
    { "name": "CENTER", "pattern": [0.03175, -0.00635, 0.00635, -0.00635] }
  ],
  "layers": [
    { "name": "CUT", "color": 16732240, "linetype": "DASHED", "lineWidth": 2 },
    { "name": "HIDDEN", "color": { "aci": 8 }, "visible": false }
  ],
  "components": [
    { "id": "P-1042#3", "node": "Bracket", "layer": "CUT" }
  ],
  "entitySets": [
    { "component": "P-1042#3", "layer": "CUT", "entities": [ /* see Entities */ ] }
  ]
}
```

- **Linetypes:** DXF `LTYPE` semantics. In `pattern`, a positive number is
  a dash, a negative number a gap and `0` a dot (about one line width
  long), with lengths in scene units scaled by the entity's `linetypeScale`.
  Up to 12 elements. The viewer has the acadiso.lin set built in, in metres
  (`CONTINUOUS`, `DASHED`, `HIDDEN`, `CENTER`, `PHANTOM`, `DOT`, `DASHDOT`,
  `BORDER`, `DIVIDE`). Complex (text/shape) linetypes are out of scope.
- **Layers:** `color` is `0xRRGGBB` as a number, or `{ "aci": n }` (AutoCAD
  Color Index, mapped through the standard palette). `lineWidth` is in
  screen pixels. Open question: also accept DXF `lineweight` (1/100 mm) and
  map it to pixels. `visible` is the initial state.
- **Components** link entity sets to model parts without putting anything
  on the nodes. `node` is the glTF node name the exporter wrote; `id` is a
  stable part id. The exporter may also put `{ "id": ... }` in that node's
  `extras` as a second key for matching. A component's `layer` assigns the
  part's mesh to a layer, so layer toggles can hide bodies as well as curves.
- **Entity sets** carry a `layer` that entities without their own layer
  inherit (DXF's block-on-layer rule), kept at the root so it survives `join`.

### Entities

In glTF, coordinates are in **scene space** (world space of the default
scene as the exporter wrote it). On the wire they are in the object's local
frame.

Optional fields on every entity: `layer` (name, default `"0"`), `color`
(number; omitted = BYLAYER), `linetype` (name; omitted or `"BYLAYER"` =
the layer's), `lineWidth` (pixels), `linetypeScale`, `handle` (stable id).

| `type` | Fields | DXF |
| --- | --- | --- |
| `point` | `position` | POINT |
| `points` | `positions` (flat xyz), for large sets | POINT × n |
| `line` | `start`, `end` | LINE |
| `polyline` | `vertices` (flat xyz), `bulges?`, `closed?`, `normal?` | LWPOLYLINE (planar, bulges allowed) / 3D POLYLINE (no bulges) |
| `circle` | `center`, `radius`, `normal?` | CIRCLE |
| `arc` | `center`, `radius`, `normal?`, `startAngle`, `endAngle` (radians, counter-clockwise about `normal`) | ARC |
| `ellipse` | `center`, `majorAxis` (vector), `normal?`, `ratio`, `startParam?`, `endParam?` | ELLIPSE |
| `spline` | `degree`, `controlPoints` (flat xyz), `knots?`, `weights?`, `closed?` | SPLINE |
| `insert` | `block` (name), `position` or `positions` (flat xyz), `rotation?` / `rotations?`, `scale?` (number or xyz) / `scales?`, `normal?`, `rows?`, `columns?`, `rowSpacing?`, `columnSpacing?` | INSERT / MINSERT |

### Blocks

`blocks` (in the payload root, and on the wire beside `entities`) maps a
block name to a list of entities, which may include further inserts (no
cycles; at most 16 deep). Each placement is translate · plane basis of
`normal` · rotate about it · scale; grid spacing runs along the rotated
block axes and is not scaled.

Style inheritance: a block entity without a layer (or on layer `"0"`) takes
the insert's layer. Without its own colour, linetype or width (or with
`"BYBLOCK"`), it takes the insert's when the insert sets one, otherwise its
layer's. That is DXF's BYBLOCK, extended to unset values so a pattern of
plain circles can be coloured from the insert. Linetype patterns scale with
the insert, as in DXF.

The viewer expands each placement into an instance record that shares the
block entity's prepared curve and a per-level tessellation cache, so
refinement evaluates each block curve once per level and copies it.

- `normal` defaults to +Z. Arc angles and the in-plane x-axis follow DXF's
  **arbitrary axis algorithm**, so a DXF writer maps them unchanged.
- Bulge `b = tan(θ/4)` turns edge `i` into an arc, counter-clockwise about
  `normal` when positive; bulged polylines must be planar.
- Without `knots`, a spline gets a clamped uniform knot vector, or a
  periodic one when `closed` is true.
- Arrays are inline JSON numbers. Inline JSON is required, not just a
  convenience: glTF-Transform `prune` deletes accessors and buffer views
  that no standard property points to.

### Fallback geometry

One node named `TJSV_vector_fallback` at the scene root holds a mesh with
one `LINES` primitive per layer, plus `POINTS` where needed, tessellated at
a chordal tolerance the exporter picks. Positions are in scene space and
the node has the identity transform. Each layer gets its own material named
`TJSV layer:<name>`, with `extras.TJSV_vector_entities.fallback = true`.

A loader without support draws these lines. A supporting loader hides
every primitive whose material has the `fallback` flag. That flag survives
every command tested, including when `optimize` merges or moves the node.

## Viewer: generic vector objects (phase 1, implemented)

Python: `add_vector_entities(id, entities, ...)` with entity builders in
`threejs_viewer.vector`, the shortcuts `add_line`, `add_circle`, `add_arc`,
`add_ellipse` and `add_spline`, plus `set_layer` and `set_linetype`.
Objects are ordinary scene objects (`delete`, `set_visible`, parenting and
transforms work). Layers and linetypes are viewer-wide settings, replayed
on reconnect and kept by `clear()`.

JavaScript: `VectorEntityController` in `viewer.js`.

- **Batching:** one `LineSegments2` per (layer, linetype, line width) bucket
  per object, with per-vertex colour; one `THREE.Points` per (layer, size)
  for points. A layer toggle flips bucket visibility.
- **Refinement:** each entity has a bounding sphere. When the camera or
  canvas changes (and every 250 ms, for animated parents), the viewer
  computes world units per pixel at the entity (constant under orthographic
  projection) and a chord tolerance of 0.5 px, quantised to a power of two.
  It rebuilds a bucket only when an entity needs a finer level, or one at
  least four times coarser. Off-screen entities keep their level.
  - Arcs, circles, ellipses and bulges: `n = sweep / (2·acos(1 − tol/r))`,
    using the major radius for ellipses (where parametric sampling has its
    largest chord error).
  - Splines: de Boor on homogeneous coordinates; each knot span's count
    comes from a sampled bound on its second derivative,
    `n = sqrt(M / (8·tol))`.
  - At most 8192 segments per curve piece.
- **Linetypes:** a `LineMaterial` `onBeforeCompile` patch replaces the
  single dash/gap test with a walk over the pattern (uniform array of 12).
  Pattern distances restart at 0 for each entity and are divided by its
  `linetypeScale`, so dashes are in model units as in DXF.
- **Robustness:** invalid entities are skipped with a console warning
  (first five, then a count); the rest draw.
- Excluded from the M-key wireframe and clay/debug material swaps; respects
  clipping planes; never casts shadows.

Not yet done: a "Layers" menu (through `MenuController`), screen-space
linetypes, view-frustum trimming of very large arcs, and ACI colours on the
wire (only in the glTF payload, phase 2).

## Viewer: glTF loading (phase 2, implemented)

- `_parseModel` reads `extras.TJSV_vector_entities` from the parsed
  document (`result.parser.json`; the root extension is read too if it holds
  a payload rather than the marker) and keeps it on the model-cache
  template. `_instantiateModel` builds one vector object per instance under
  the glTF scene root, through the same `VectorEntityController`, so scene
  space lines up including the Y-up correction, and it moves with the model.
  Each set's `layer` becomes the default for its entities; `blocks` work as
  on the wire. Colours may be numbers, `{aci}` or `{rgb}`.
- Layers and linetypes from the file are merged into the viewer tables
  without overriding ones already set from Python.
- Primitives whose material has the fallback flag are hidden and excluded
  from picking and bounds. If the payload is missing or invalid, they stay
  visible and the viewer warns once.
- If no entity is valid, the fallback stays visible.
- Not yet: `components` (parenting entity sets to individual parts so they
  follow part animation, and layer toggles hiding part meshes), and a
  committed glTF-Transform-processed fixture. Tests build a GLB in Python;
  the exporter in `cell-setup-description/tools/add_vector_features.py`
  was checked by hand on a Draco GLB (76 circles).

## Open questions

- Final vendor prefix.
- DXF `lineweight` (1/100 mm) in the glTF payload, mapped to pixels?
- Text (`TEXT`/`MTEXT`), hatches and dimensions: v2?
- Layers menu in the viewer UI: built in, or left to embedders?
