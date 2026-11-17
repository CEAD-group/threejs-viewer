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
