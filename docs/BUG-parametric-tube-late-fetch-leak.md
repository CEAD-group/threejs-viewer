# Bug: `add_parametric_tube_binary` leaks a mesh on rapid re-add

## Symptom

Two toolpath tubes are visible in the scene, overlaid and z-fighting, after
a fast `add → delete → add` sequence on the same id. `query_scene` reports
a single `'bead'` object, but the scene graph actually contains two meshes
— one orphaned, one tracked in `this._objects`.

## Reproduction

Client (e.g. a parameter slider that re-runs the pipeline quickly):

```
dispatch({ type: 'add_parametric_tube_binary', id: 'bead', blob_url: A, ... })
dispatch({ type: 'delete_object', id: 'bead' })
dispatch({ type: 'add_parametric_tube_binary', id: 'bead', blob_url: B, ... })
```

All three messages are enqueued before fetch A resolves.

## Root cause

`src/threejs_viewer/viewer/viewer.js`, the `add_parametric_tube_binary`
handler (around line 4199):

```js
case 'add_parametric_tube_binary': {
    this._onFetchStart();
    const capturedScene = this._sceneGeneration;
    (async () => {
        const resp = await fetch(data.blob_url);
        const buffer = await resp.arrayBuffer();
        if (this._sceneGeneration !== capturedScene) {
            console.log('Discarding stale parametric tube fetch');
            return;
        }
        // ... build mesh ...
        this._addToParentOrScene(mesh, data.parent);   // ← attaches to scene
        this._objects.set(data.id, mesh);              // ← overwrites map entry
        // ... no cleanup of the previous object at this id ...
    })();
}
```

Timeline on the rapid sequence:

1. `add A` dispatched → fetch A begins
2. `delete 'bead'` dispatched → no-op, nothing under that id yet
3. `add B` dispatched → fetch B begins
4. Fetch A resolves → builds mesh A, `_addToParentOrScene(A)`, `_objects.set('bead', A)`
5. Fetch B resolves → builds mesh B, `_addToParentOrScene(B)`, `_objects.set('bead', B)`

After step 5 the map has `B`, but `A` is still attached to the scene and
rendered. `delete_object` didn't catch it because `A` didn't exist at
dispatch time, and `_sceneGeneration` isn't bumped by individual
`delete_object` calls — so the stale-fetch guard doesn't fire either.

## Impact

Happens for any async-fetch add handler that `_addToParentOrScene` +
`_objects.set` without first removing an existing same-id object:

- `add_parametric_tube_binary` (~L4199)
- `add_mesh_binary` (~L4126)
- `add_polyline_binary` (~L4059)
- possibly `add_object` with `model` fetches (~L4027)

In client-authoritative scene reconcilers (where the client emits
`delete → add` on every swap) this shows up as mesh leakage proportional
to the rate of parameter changes.

## Observability problem

The bug is invisible to `query_scene`: the orphan mesh is a direct child
of the scene root but isn't in `this._objects`, so the returned tree
contains one `'bead'` entry. Anything that walks `_objects` to check for
drift reports consistent state while the render buffer shows two meshes.

## Suggested fix

Before `_addToParentOrScene`, check for and remove any existing object at
`data.id`:

```js
const existing = this._objects.get(data.id);
if (existing) {
    this._deleteObject(data.id);
}
this._addToParentOrScene(mesh, data.parent);
this._objects.set(data.id, mesh);
```

Apply symmetrically across every `add_*_binary` handler and any `add_*`
handler that has an async fetch step.

Optional secondary improvement: bump `_sceneGeneration` (or a dedicated
`_objectGeneration[id]` counter) in `_deleteObject` so in-flight fetches
for the just-deleted id self-abort via their captured generation — that
would save wasted fetch + build work too, not just fix the visual leak.

## Workaround (client side, pre-fix)

Serialize bead swaps through a promise chain that awaits viewer
acknowledgement (e.g. via a `query_scene` round-trip) before dispatching
the next `add`. Slower and brittle — prefer the viewer-side fix.
