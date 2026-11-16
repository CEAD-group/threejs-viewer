const vscode = require('vscode');
const fs = require('fs');
const path = require('path');

// The extension runs from the repo checkout (vscode-extension/ is linked into
// ~/.vscode/extensions, see README.md), so the generated viewer sits next door.
// `threejsViewer.viewerHtml` overrides it, e.g. for an installed copy.
const DEFAULT_VIEWER_HTML = path.join(__dirname, '..', 'src', 'threejs_viewer', 'viewer.html');

// type="module" so it runs after the viewer's own module script; re-applies on theme
// change (VS Code rewrites the --vscode-* vars on <html> and the theme class on <body>).
const THEME_SCRIPT = `
<script type="module">
{
    const viewer = window.threejsViewer;
    const original = viewer.setBackground.bind(viewer);
    const themeBg = () => getComputedStyle(document.documentElement)
        .getPropertyValue('--vscode-editor-background').trim();
    // The theme wins over any background a client pushes.
    viewer.setBackground = (color) => original(themeBg() || color);
    const apply = () => viewer.setBackground();
    apply();
    const obs = new MutationObserver(apply);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['style'] });
    obs.observe(document.body, { attributes: true, attributeFilter: ['class'] });
    window.addEventListener('message', (e) => {
        if (e.data && e.data.type === 'clearScene') viewer._clearScene();
    });
}
</script>`;

let panel;
let watcher;
let reloadTimer;

function viewerHtmlPath() {
    const configured = vscode.workspace.getConfiguration('threejsViewer').get('viewerHtml');
    return configured || DEFAULT_VIEWER_HTML;
}

function render(file) {
    const html = fs.readFileSync(file, 'utf8');
    panel.webview.html = html.replace(/<\/body>/i, THEME_SCRIPT + '\n</body>');
}

// Reload the tab when viewer.html changes (git pull, build.py). The viewer
// reconnects to the Python server, which replays the scene.
function watch(file) {
    unwatch();
    try {
        watcher = fs.watch(file, () => {
            clearTimeout(reloadTimer);
            reloadTimer = setTimeout(() => {
                if (!panel) return;
                try {
                    render(file);
                } catch (err) {
                    // Mid-write or briefly deleted by git; the next event retries.
                }
            }, 300);
        });
    } catch (err) {
        // Watching is a convenience; the tab still works without it.
    }
}

function unwatch() {
    clearTimeout(reloadTimer);
    if (watcher) watcher.close();
    watcher = undefined;
}

function activate(context) {
    context.subscriptions.push(vscode.commands.registerCommand('threejsViewer.open', () => {
        if (panel) {
            panel.reveal();
            return;
        }
        const file = viewerHtmlPath();
        if (!fs.existsSync(file)) {
            vscode.window.showErrorMessage(`Cannot find viewer.html at ${file}. Set threejsViewer.viewerHtml.`);
            return;
        }
        panel = vscode.window.createWebviewPanel(
            'threejsViewer', 'Viewer', vscode.ViewColumn.Active,
            { enableScripts: true, retainContextWhenHidden: true },
        );
        try {
            render(file);
        } catch (err) {
            vscode.window.showErrorMessage(`Cannot read ${file}: ${err.message}`);
            panel.dispose();
            return;
        }
        watch(file);
        panel.onDidDispose(() => {
            unwatch();
            panel = undefined;
        }, null, context.subscriptions);
    }));
    context.subscriptions.push(vscode.commands.registerCommand('threejsViewer.clearScene', () => {
        if (panel) panel.webview.postMessage({ type: 'clearScene' });
    }));
}

function deactivate() {
    unwatch();
}

module.exports = { activate, deactivate };
