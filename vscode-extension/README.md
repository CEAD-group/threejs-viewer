# Three.js Viewer (VS Code webview)

A VS Code extension that opens `src/threejs_viewer/viewer.html` in an editor tab,
with the background following the editor theme. Run **Three.js: Open Three.js
Viewer** from the command palette, then connect from Python as usual.

## Models

- **Show in Viewer** (eye button in the editor title bar, or the explorer
  context menu) on a `.gltf` or `.glb` sends it to the viewer tab, opening one
  beside the editor if needed. The id is the file name, so showing it again
  replaces it.
- Shown models update in place, keeping the camera: when the file is saved
  or re-exported, when a `.gltf`'s `.bin` or textures change, and while you
  type in a `.gltf` (after a short pause; invalid JSON mid-edit is skipped).
- Opening a `.glb` previews it in its own tab, with no connection to Python.
  For a `.gltf`, use **Open With... → Three.js Model Preview**.
- Files opened while the viewer tab is focused open in another column, so
  the viewer keeps its own.
- Models are loaded as-is (Z-up). Set `threejsViewer.gltfYUp` for Y-up
  exports, like `add_model(y_up=True)`.

## API for other extensions

Other extensions can show a scene in the viewer tab without a Python
process. This API is a public contract of this repository:

```js
const api = await vscode.extensions.getExtension('local.threejs-viewer-webview').activate();
if (api.apiVersion !== 1) throw new Error(`unsupported threejs-viewer API ${api.apiVersion}`);
const sent = api.showScene('my-scene', { title: 'Cell', messages }, { frame: true });
api.hasScene('my-scene');   // whether the viewer tab shows it
```

- **`apiVersion`** is `1`. It goes up with any incompatible change to
  `showScene`/`hasScene`, so check it and refuse a version you don't know.
- **`showScene(key, { title, messages }, { frame })`** shows the scene `key`,
  opening the viewer tab beside the active editor if needed.
  - `messages` are the viewer's own wire messages, the JSON that
    `ViewerClient` sends, applied in order.
  - A binary message (`add_model_binary`, `add_mesh_binary`, ...) carries its
    payload as `bytes` (a `Uint8Array`) instead of a `blob_url`.
  - `frame` (default `true`) frames the whole scene once loaded.
  - Other scenes, and objects a Python session pushed, are left alone.
- **Showing the same `key` again sends only the difference.**
  - An object whose transform changed is moved by its own matrix.
  - An object whose add message, payload or per-object messages changed is
    deleted and added again, together with everything parented under it.
    Per-object messages are any message naming its `id`, such as `set_color`
    or `set_visibility`.
  - A removed object is deleted, and an unchanged one is left alone.
  - Any change to a message that names no object sends the scene whole.
- **The webview checks before it patches.** If an object of the scene that
  had loaded has disappeared since, the scene is sent whole instead. That
  happens when a Python client sent `clear_scene` or deleted one of its ids.
- **Returns** `false` when no viewer tab could be opened, otherwise
  `{ full, readded, removed, moved }`: whether the scene was sent whole, and
  the counts of a patch.
  - The call returns once the update is queued; the webview applies queued
    updates one at a time, in order.
  - Model load failures show up as a VS Code error notification, and the
    next update is then sent whole.
- **`hasScene(key)`** tells whether the viewer tab currently shows scene `key`.
  It is false after the tab is closed or cleared.

The diff is `scene_diff.js`, tested with `node --test vscode-extension/test`.

## Distribution and limits

- **An internal development tool, run from a checkout.** It has no
  marketplace listing and no `.vsix`: the extension loads
  `../src/threejs_viewer/viewer.html` from the checkout it is linked from,
  unless `threejsViewer.viewerHtml` names another.
- **No Remote or WSL windows yet.** The extension would run remotely while
  the webview's `ws://localhost:5666` resolves on the local machine.
- **Content-Security-Policy.** The webview allows only its inline scripts, the
  three.js build pinned in `viewer.html` (so it needs network access to
  unpkg), and WebSocket/HTTP connections to `localhost`. It reads no local
  files.
- **Untrusted workspaces:** a workspace's `threejsViewer.viewerHtml` is ignored.
- **Known costs, not fixed yet:**
  - The extension host keeps the last full copy of each scene, mesh payloads
    included, so it can resend it after the page reloads.
  - Typing in a shown `.gltf` re-reads and base64-inlines its buffers and
    textures on each pause.

## Development

```bash
npm install                                     # type packages, once
npx tsc --noEmit -p vscode-extension/jsconfig.json
node --test vscode-extension/test
```

CI runs both (the `vscode-extension` job).

## Install (linked to this checkout)

Link this folder into VS Code's extensions directory instead of installing a
`.vsix`. VS Code then runs the extension from the repo, so pulling the repo
updates it:

```powershell
# Windows (a junction needs no admin rights)
New-Item -ItemType Junction -Path "$env:USERPROFILE\.vscode\extensions\local.threejs-viewer-webview" -Target "$PWD\vscode-extension"
```

```bash
# macOS / Linux
ln -s "$PWD/vscode-extension" ~/.vscode/extensions/local.threejs-viewer-webview
```

Run the command from the repo root, then restart VS Code.

## Staying up to date

- **Viewer changes** (`viewer.html` after a pull or `build.py`): an open viewer tab
  reloads itself and reconnects. New tabs always read the current file.
- **Extension changes** (`extension.js`, `package.json`): run
  **Developer: Reload Window**.

To use a different `viewer.html`, set `threejsViewer.viewerHtml`.
