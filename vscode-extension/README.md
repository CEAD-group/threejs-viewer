# Three.js Viewer (VS Code webview)

A VS Code extension that opens `src/threejs_viewer/viewer.html` in an editor tab,
with the background following the editor theme. Run **Three.js: Open Three.js
Viewer** from the command palette, then connect from Python as usual.

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
