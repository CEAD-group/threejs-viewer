const vscode = require('vscode');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

// The extension runs from the repo checkout (vscode-extension/ is linked into
// ~/.vscode/extensions, see README.md), so the generated viewer sits next door.
// `threejsViewer.viewerHtml` overrides it, e.g. for an installed copy.
const DEFAULT_VIEWER_HTML = path.join(__dirname, '..', 'src', 'threejs_viewer', 'viewer.html');

const MODEL_FORMATS = { '.glb': 'glb', '.gltf': 'gltf' };

// type="module" so it runs after the viewer's own module script; re-applies on theme
// change (VS Code rewrites the --vscode-* vars on <html> and the theme class on <body>).
// It also loads models the extension posts: the bytes become a Blob URL fed to the
// viewer's public handleMessage, so no file:// or webview-resource access is needed.
const THEME_SCRIPT = `
<script type="module">
{
    const viewer = window.threejsViewer;
    const vscodeApi = acquireVsCodeApi();
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

    // The viewer reports a failed model load only on the console; keep the
    // last error so showModel can surface it as a VS Code notification.
    let lastError = '';
    const consoleError = console.error.bind(console);
    console.error = (...args) => {
        lastError = args.map((a) => (a && a.stack) || String(a)).join(' ');
        consoleError(...args);
    };

    async function showModel(msg) {
        const url = URL.createObjectURL(new Blob([msg.bytes]));
        lastError = '';
        try {
            await viewer.handleMessage({
                type: 'add_object', id: msg.id,
                object: { model: url, format: msg.format, yUp: msg.yUp, cache: false },
            });
        } finally {
            URL.revokeObjectURL(url);
        }
        if (!viewer._objects.has(msg.id)) {
            throw new Error(\`\${lastError || 'load failed'} (received \${msg.bytes.byteLength} of \${msg.size} bytes)\`);
        }
        if (msg.frame === 'all') viewer.frameAll();
        else if (msg.frame) await viewer.handleMessage({ type: 'frame_object', id: msg.id });
    }

    // Scenes another extension builds (see the exported showScene API):
    // key -> { roots: top-level ids it added, urls: id -> Blob URL it owns }.
    const scenes = new Map();
    let sceneChain = Promise.resolve();

    function dropScene(key) {
        const old = scenes.get(key);
        if (!old) return;
        scenes.delete(key);
        for (const id of old.roots) viewer.handleMessage({ type: 'delete_object', id });
        for (const url of old.urls.values()) URL.revokeObjectURL(url);
    }

    // Viewer protocol messages, applied in order as the WebSocket would. A
    // message carrying \`bytes\` is a binary one: its payload becomes a Blob
    // URL in \`blob_url\`, revoked once its object is gone (the fetch runs
    // after handleMessage returns, so it cannot be revoked straight away).
    // Returns the number of messages that threw.
    async function applyMessages(entry, messages) {
        lastError = '';
        let failed = 0;
        for (const m of messages) {
            let data = m;
            if (m.bytes) {
                const { bytes, ...header } = m;
                const url = URL.createObjectURL(new Blob([bytes]));
                if (entry.urls.has(m.id)) URL.revokeObjectURL(entry.urls.get(m.id));
                entry.urls.set(m.id, url);
                data = { ...header, blob_url: url };
            }
            try {
                await viewer.handleMessage(data);
            } catch (err) {
                failed++;
                lastError = String(err && err.message || err);
            }
        }
        // Binary models load after handleMessage returns; wait for them.
        while (viewer.getLoadState().pending) await new Promise((r) => setTimeout(r, 100));
        for (const [id, url] of entry.urls) {
            if (!viewer._objects.has(id)) { URL.revokeObjectURL(url); entry.urls.delete(id); }
        }
        return failed;
    }

    function reportScene(msg, failed) {
        const missing = msg.messages.filter((m) => m.type === 'add_model_binary'
            && !viewer._objects.has(m.id)).length;
        if (msg.frame) viewer.frameAll();
        if (failed || missing) {
            // The extension no longer knows what this scene shows, so its
            // next update is sent whole instead of as a patch.
            vscodeApi.postMessage({ type: 'sceneFailed', key: msg.key });
            vscodeApi.postMessage({ type: 'error', text:
                \`\${msg.title || msg.key}: \${missing} model(s) failed to load, \${failed} message(s) failed. Last error: \${lastError}\` });
        }
    }

    async function applyScene(msg) {
        dropScene(msg.key);
        const entry = { roots: [], urls: new Map() };
        for (const m of msg.messages) {
            if (String(m.type).startsWith('add_') && m.id && !m.parent) entry.roots.push(m.id);
        }
        scenes.set(msg.key, entry);
        reportScene(msg, await applyMessages(entry, msg.messages));
    }

    // Update a shown scene in place; the extension computed the difference
    // (diffScene): deletes, re-adds of changed objects, and one batch_update
    // carrying every object whose transform changed. The rest stays as is.
    async function patchScene(msg) {
        const entry = scenes.get(msg.key);
        if (!entry) {
            // Dropped meanwhile (cleared viewer): ask for the whole scene.
            vscodeApi.postMessage({ type: 'sceneResync', key: msg.key });
            return;
        }
        entry.roots = msg.roots;
        reportScene(msg, await applyMessages(entry, msg.messages));
    }

    const queueScene = (msg, apply, verb) => {
        // One at a time: a re-save mid-load must not interleave two updates.
        sceneChain = sceneChain.then(() => apply(msg)).catch((err) => {
            vscodeApi.postMessage({ type: 'sceneFailed', key: msg.key });
            vscodeApi.postMessage({
                type: 'error', text: \`Cannot \${verb} \${msg.title || msg.key}: \${err && err.message || err}\` });
        });
    };

    window.addEventListener('message', (e) => {
        const msg = e.data;
        if (!msg) return;
        if (msg.type === 'clearScene') {
            for (const key of [...scenes.keys()]) dropScene(key);
            viewer._clearScene();
        } else if (msg.type === 'applyScene') {
            queueScene(msg, applyScene, 'show');
        } else if (msg.type === 'patchScene') {
            queueScene(msg, patchScene, 'update');
        } else if (msg.type === 'showModel') {
            showModel(msg).catch((err) => msg.quiet || vscodeApi.postMessage({
                type: 'error', text: \`Cannot show \${msg.id}: \${err && err.message || err}\` }));
        }
    });
    vscodeApi.postMessage({ type: 'ready' });
}
</script>`;

function viewerHtmlPath() {
    const configured = vscode.workspace.getConfiguration('threejsViewer').get('viewerHtml');
    return configured || DEFAULT_VIEWER_HTML;
}

function yUp() {
    return vscode.workspace.getConfiguration('threejsViewer').get('gltfYUp') === true;
}

// `standalone` previews a single file: no WebSocket, so a running Python
// session cannot push its scene into (or clear) the preview.
function viewerHtml(file, standalone) {
    let html = fs.readFileSync(file, 'utf8');
    if (standalone) {
        html = html.replace('new ThreeJSViewer(container, {', 'new ThreeJSViewer(container, {\n    autoConnect: false,');
    }
    return html.replace(/<\/body>/i, THEME_SCRIPT + '\n</body>');
}

/**
 * Bytes the viewer can load from a Blob URL. A .gltf's relative buffer and
 * image URIs would not resolve against a Blob URL, so they are inlined as
 * data: URIs. `text` is the unsaved editor contents, if any.
 * @param {vscode.Uri} uri @param {string} [text] @returns {Promise<Uint8Array>}
 */
async function modelBytes(uri, text) {
    if (path.extname(uri.fsPath).toLowerCase() !== '.gltf') {
        // Copy to an exact-size buffer: readFile can return a view into a
        // larger pooled ArrayBuffer, and postMessage may send the whole pool.
        return new Uint8Array(await vscode.workspace.fs.readFile(uri));
    }
    if (text === undefined) text = new TextDecoder().decode(await vscode.workspace.fs.readFile(uri));
    const gltf = JSON.parse(text);
    const inline = async (item, fallbackMime) => {
        if (!item.uri || /^(data|https?):/i.test(item.uri)) return;
        const ref = vscode.Uri.joinPath(uri, '..', decodeURIComponent(item.uri));
        const data = await vscode.workspace.fs.readFile(ref);
        const mime = item.mimeType || fallbackMime(path.extname(ref.fsPath).toLowerCase());
        item.uri = `data:${mime};base64,${Buffer.from(data).toString('base64')}`;
    };
    const imageMime = (ext) => ({ '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.webp': 'image/webp', '.ktx2': 'image/ktx2' })[ext] || 'application/octet-stream';
    await Promise.all([
        ...(gltf.buffers || []).map((b) => inline(b, () => 'application/octet-stream')),
        ...(gltf.images || []).map((i) => inline(i, imageMime)),
    ]);
    return new TextEncoder().encode(JSON.stringify(gltf));
}

function modelFormat(uri) {
    return MODEL_FORMATS[path.extname(uri.fsPath).toLowerCase()];
}

/** Unsaved editor text for `uri`, if it is open and dirty. */
function dirtyText(uri) {
    const doc = vscode.workspace.textDocuments.find((d) => d.uri.toString() === uri.toString());
    return doc && doc.isDirty ? doc.getText() : undefined;
}

/** @type {WeakMap<Uint8Array, string>} payload -> sha1, so a payload shared by several messages hashes once */
const digests = new WeakMap();

function digestOf(bytes) {
    let d = digests.get(bytes);
    if (!d) digests.set(bytes, d = crypto.createHash('sha1').update(bytes).digest('hex'));
    return d;
}

/**
 * The final state a scene's messages leave, per object id: its add message
 * (`sig` compares header and payload), its transform (the last batch_update
 * or update_transform naming it), and its other per-object messages
 * (set_color, ...). Messages naming no object are kept in order in `other`.
 * @param {any[]} messages
 */
function sceneState(messages) {
    const defs = new Map();
    const transforms = new Map();
    const props = new Map();
    const other = [];
    for (const m of messages) {
        const type = String(m.type);
        if (type === 'batch_update') {
            for (const [id, tf] of Object.entries(m.transforms || {})) transforms.set(id, JSON.stringify(tf));
        } else if (type === 'update_transform' && m.id) {
            transforms.set(m.id, JSON.stringify(m.transform));
        } else if (type.startsWith('add_') && m.id) {
            const { bytes, ...header } = m;
            const sig = JSON.stringify(header) + (bytes ? `#${digestOf(bytes)}` : '');
            defs.set(m.id, { msg: m, sig, parent: m.parent });
            props.delete(m.id);
        } else if (m.id && type !== 'delete_object') {
            if (!props.has(m.id)) props.set(m.id, []);
            props.get(m.id).push(m);
        } else {
            other.push(JSON.stringify(m));
        }
    }
    const propSigs = new Map([...props].map(([id, list]) => [id, JSON.stringify(list)]));
    return { defs, transforms, props, propSigs, other: other.join('\n') };
}

/**
 * The messages that turn a shown scene `prev` into `next`, or null when only
 * a full replace can (their id-less messages differ).
 *
 * Every object whose transform changed gets its own entry in one
 * batch_update: each is moved individually, none relies on a parent carrying
 * it. An object whose add message or other per-object messages changed is
 * deleted and added again, and so is everything added under it (a delete
 * takes its children along). Unchanged objects are not touched.
 */
function diffScene(prev, next) {
    if (prev.other !== next.other) return null;
    const readd = new Set();
    for (const [id, def] of next.defs) {
        const old = prev.defs.get(id);
        if (!old || old.sig !== def.sig || prev.propSigs.get(id) !== next.propSigs.get(id)
            || (prev.transforms.has(id) && !next.transforms.has(id))) readd.add(id);
    }
    const children = new Map();
    for (const [id, def] of next.defs) {
        if (!def.parent) continue;
        if (!children.has(def.parent)) children.set(def.parent, []);
        children.get(def.parent).push(id);
    }
    const stack = [...readd];
    while (stack.length) {
        for (const child of children.get(stack.pop()) || []) {
            if (!readd.has(child)) { readd.add(child); stack.push(child); }
        }
    }
    const gone = new Set([...prev.defs.keys()].filter((id) => !next.defs.has(id) || readd.has(id)));
    const underGone = (id) => {
        const seen = new Set();
        for (let p = prev.defs.get(id).parent; p && !seen.has(p); p = prev.defs.get(p)?.parent) {
            if (gone.has(p)) return true;
            seen.add(p);
        }
        return false;
    };
    const messages = [...gone].filter((id) => !underGone(id)).map((id) => ({ type: 'delete_object', id }));
    for (const [id, def] of next.defs) if (readd.has(id)) messages.push(def.msg);
    for (const id of readd) messages.push(...(next.props.get(id) || []));
    const transforms = {};
    let moved = 0;
    for (const [id, tf] of next.transforms) {
        if (readd.has(id)) transforms[id] = JSON.parse(tf);
        else if (prev.transforms.get(id) !== tf) { transforms[id] = JSON.parse(tf); moved++; }
    }
    if (Object.keys(transforms).length) messages.push({ type: 'batch_update', transforms });
    const roots = [...next.defs].filter(([, def]) => !def.parent).map(([id]) => id);
    const removed = [...prev.defs.keys()].filter((id) => !next.defs.has(id)).length;
    return { messages, roots, readded: readd.size, removed, moved };
}

/** @type {Set<ViewerWebview>} every open viewer, main and previews */
const viewers = new Set();

/**
 * A webview running viewer.html. Messages posted before the page reports
 * ready are queued; models are re-sent when the page reloads (viewer.html
 * rebuilt), since only the Python server's scene replays on reconnect.
 * Shown models reload in place when their file, a .gltf's buffers or
 * textures, or their unsaved editor text change.
 */
class ViewerWebview {
    /** @param {vscode.WebviewPanel} panel @param {boolean} standalone */
    constructor(panel, standalone) {
        this.panel = panel;
        this.standalone = standalone;
        this.ready = false;
        this.loadedOnce = false;
        this.queue = [];
        /** @type {Map<string, vscode.Uri>} id -> model shown in this view */
        this.models = new Map();
        /** @type {Map<string, any>} scene key -> its last applyScene message */
        this.scenes = new Map();
        /** @type {Map<string, ReturnType<typeof sceneState>>} scene key -> what the webview shows of it */
        this.sceneStates = new Map();
        /** @type {Map<string, vscode.FileSystemWatcher>} directory -> watcher */
        this.dirWatchers = new Map();
        /** @type {Map<string, NodeJS.Timeout>} model uri -> pending reload */
        this.reloadTimers = new Map();
        this.file = viewerHtmlPath();
        this.disposables = [];
        viewers.add(this);
        panel.webview.options = { enableScripts: true };
        this.disposables.push(panel.webview.onDidReceiveMessage((msg) => {
            if (msg.type === 'ready') this._onReady();
            else if (msg.type === 'error') vscode.window.showErrorMessage(msg.text);
            // The webview's copy of the scene is unknown: stop patching it.
            else if (msg.type === 'sceneFailed') this.sceneStates.delete(msg.key);
            else if (msg.type === 'sceneResync' && this.scenes.has(msg.key)) this.post(this.scenes.get(msg.key));
        }));
        panel.onDidDispose(() => this.dispose());
    }

    /** @returns {boolean} whether the page could be rendered */
    render() {
        if (!fs.existsSync(this.file)) {
            vscode.window.showErrorMessage(`Cannot find viewer.html at ${this.file}. Set threejsViewer.viewerHtml.`);
            return false;
        }
        try {
            this.ready = false;
            this.panel.webview.html = viewerHtml(this.file, this.standalone);
            return true;
        } catch (err) {
            vscode.window.showErrorMessage(`Cannot read ${this.file}: ${err.message}`);
            return false;
        }
    }

    post(msg) {
        if (this.ready) this.panel.webview.postMessage(msg);
        else this.queue.push(msg);
    }

    /**
     * @param {vscode.Uri} uri
     * @param {{ frame?: string|boolean, quiet?: boolean }} [opts]
     *   `quiet`: a live reload; mid-edit JSON or load errors raise no notification.
     */
    async showModel(uri, opts = {}) {
        const format = modelFormat(uri);
        const id = path.basename(uri.fsPath);
        let bytes;
        try {
            // Unsaved .gltf edits are shown as they are in the editor.
            bytes = await modelBytes(uri, dirtyText(uri));
        } catch (err) {
            if (!opts.quiet) vscode.window.showErrorMessage(`Cannot read ${uri.fsPath}: ${err.message}`);
            return;
        }
        this.models.set(id, uri);
        this._watchDir(vscode.Uri.joinPath(uri, '..'));
        this.post({ type: 'showModel', id, format, bytes, size: bytes.byteLength,
            yUp: yUp(), frame: opts.frame ?? true, quiet: opts.quiet === true });
    }

    /** Reload `uri` shortly, keeping the camera, if this viewer shows it. */
    scheduleReload(uri) {
        const key = uri.toString();
        if (![...this.models.values()].some((m) => m.toString() === key)) return;
        clearTimeout(this.reloadTimers.get(key));
        this.reloadTimers.set(key, setTimeout(() => {
            this.reloadTimers.delete(key);
            this.showModel(uri, { frame: false, quiet: true });
        }, 300));
    }

    _watchDir(dir) {
        const key = dir.toString();
        if (this.dirWatchers.has(key)) return;
        const watcher = vscode.workspace.createFileSystemWatcher(new vscode.RelativePattern(dir, '*'));
        const changed = (file) => {
            for (const uri of this.models.values()) {
                if (uri.toString() === file.toString()) this.scheduleReload(uri);
                // A .gltf's .bin or texture beside it (other models are not its resources).
                else if (path.extname(uri.fsPath).toLowerCase() === '.gltf' && !modelFormat(file)
                    && vscode.Uri.joinPath(uri, '..').toString() === key) this.scheduleReload(uri);
            }
        };
        watcher.onDidChange(changed);
        watcher.onDidCreate(changed);
        this.dirWatchers.set(key, watcher);
    }

    /**
     * Show the scene `key` (see the exported API's showScene): sent whole the
     * first time, afterwards as a patch of what changed (diffScene).
     * @param {string} key @param {{ title?: string, messages: any[] }} scene @param {boolean} frame
     * @returns {{ full: boolean, readded: number, removed: number, moved: number }}
     */
    showScene(key, scene, frame) {
        const msg = { type: 'applyScene', key, title: scene.title, messages: scene.messages, frame };
        const next = sceneState(scene.messages);
        const prev = this.sceneStates.get(key);
        this.scenes.set(key, msg);
        this.sceneStates.set(key, next);
        const patch = prev && diffScene(prev, next);
        if (!patch) {
            this.post(msg);
            return { full: true, readded: next.defs.size, removed: 0, moved: 0 };
        }
        if (patch.messages.length || frame) {
            this.post({ type: 'patchScene', key, title: scene.title, messages: patch.messages,
                roots: patch.roots, frame });
        }
        return { full: false, readded: patch.readded, removed: patch.removed, moved: patch.moved };
    }

    clear() {
        this.models.clear();
        this.scenes.clear();
        this.sceneStates.clear();
        this.post({ type: 'clearScene' });
    }

    _onReady() {
        const reloaded = this.loadedOnce;
        this.loadedOnce = true;
        this.ready = true;
        let queued = this.queue.splice(0);
        if (reloaded) {
            // The reload lost every model and scene; send them again.
            queued = queued.filter((msg) => !['showModel', 'applyScene', 'patchScene'].includes(msg.type));
            for (const uri of this.models.values()) this.showModel(uri, { frame: false });
            for (const msg of this.scenes.values()) queued.push({ ...msg, frame: false });
        }
        for (const msg of queued) this.panel.webview.postMessage(msg);
    }

    // Reload when viewer.html changes (git pull, build.py). The viewer
    // reconnects to the Python server, which replays its scene.
    watch() {
        let timer;
        try {
            const watcher = fs.watch(this.file, () => {
                clearTimeout(timer);
                timer = setTimeout(() => {
                    // Mid-write or briefly deleted by git; the next event retries.
                    if (fs.existsSync(this.file)) this.render();
                }, 300);
            });
            this.disposables.push({ dispose: () => { clearTimeout(timer); watcher.close(); } });
        } catch (err) {
            // Watching is a convenience; the tab still works without it.
        }
    }

    dispose() {
        viewers.delete(this);
        for (const d of this.disposables.splice(0)) d.dispose();
        for (const w of this.dirWatchers.values()) w.dispose();
        this.dirWatchers.clear();
        for (const t of this.reloadTimers.values()) clearTimeout(t);
        this.reloadTimers.clear();
    }
}

/** @type {ViewerWebview | undefined} the shared, Python-connected viewer tab */
let mainViewer;

function openMainViewer(column = vscode.ViewColumn.Active) {
    if (mainViewer) {
        mainViewer.panel.reveal(column, true);
        return mainViewer;
    }
    const panel = vscode.window.createWebviewPanel(
        'threejsViewer', 'Viewer', { viewColumn: column, preserveFocus: true },
        { enableScripts: true, retainContextWhenHidden: true },
    );
    const viewer = new ViewerWebview(panel, false);
    if (!viewer.render()) {
        panel.dispose();
        return undefined;
    }
    viewer.watch();
    panel.onDidDispose(() => { if (mainViewer === viewer) mainViewer = undefined; });
    panel.onDidChangeViewState(() => { if (!panel.active) viewerBlurredAt = Date.now(); });
    mainViewer = viewer;
    return viewer;
}

/** When the main viewer last lost focus; see keepViewerColumnClear. */
let viewerBlurredAt = 0;

/**
 * Move a file opened while the viewer was focused into another column, so
 * the viewer keeps its own. Text and custom editors (e.g. .glb previews) are
 * reopened elsewhere and their tab in the viewer's group closed.
 * @param {vscode.TabChangeEvent} event
 */
async function keepViewerColumnClear(event) {
    if (!mainViewer) return;
    const panel = mainViewer.panel;
    const viewerColumn = panel.viewColumn;
    // The viewer loses focus as the new tab opens; allow for either event order.
    if (!panel.active && Date.now() - viewerBlurredAt > 500) return;
    for (const tab of event.opened) {
        if (tab.group.viewColumn !== viewerColumn) continue;
        const input = tab.input;
        const other = vscode.window.tabGroups.all.find((g) => g.viewColumn !== viewerColumn);
        const viewColumn = other ? other.viewColumn : vscode.ViewColumn.Beside;
        if (input instanceof vscode.TabInputText) {
            await vscode.window.showTextDocument(input.uri, { viewColumn, preview: tab.isPreview });
        } else if (input instanceof vscode.TabInputCustom) {
            await vscode.commands.executeCommand('vscode.openWith', input.uri, input.viewType,
                { viewColumn, preview: tab.isPreview });
        } else {
            continue;
        }
        await vscode.window.tabGroups.close(tab, true);
    }
}

/** The .gltf/.glb behind an explorer click, the active text editor, or the active preview. */
function targetModel(uri) {
    if (uri instanceof vscode.Uri) return uri;
    const editor = vscode.window.activeTextEditor;
    if (editor) return editor.document.uri;
    const tab = vscode.window.tabGroups.activeTabGroup.activeTab;
    return tab && tab.input && tab.input.uri;
}

/** Custom editor: opening a .glb (or "Open With..." on a .gltf) previews it alone. */
class ModelPreviewProvider {
    openCustomDocument(uri) {
        return { uri, dispose() {} };
    }

    resolveCustomEditor(document, panel) {
        const viewer = new ViewerWebview(panel, true);
        if (!viewer.render()) return;
        viewer.watch();
        // showModel watches the file, so a re-export reloads the preview.
        viewer.showModel(document.uri, { frame: 'all' });
    }
}

function activate(context) {
    context.subscriptions.push(vscode.commands.registerCommand('threejsViewer.open', () => {
        openMainViewer();
    }));
    context.subscriptions.push(vscode.commands.registerCommand('threejsViewer.clearScene', () => {
        if (mainViewer) mainViewer.clear();
    }));
    context.subscriptions.push(vscode.commands.registerCommand('threejsViewer.showModel', async (uri) => {
        const target = targetModel(uri);
        if (!target || !modelFormat(target)) {
            vscode.window.showErrorMessage('Three.js: open a .gltf or .glb file to show it in the viewer.');
            return;
        }
        // Open beside the model's editor rather than replacing it.
        const viewer = openMainViewer(mainViewer ? undefined : vscode.ViewColumn.Beside);
        if (viewer) await viewer.showModel(target);
    }));
    // Typing in a shown .gltf updates it live after a pause; revert and
    // save-with-formatting arrive here too (plain saves via the file watcher).
    context.subscriptions.push(vscode.workspace.onDidChangeTextDocument((e) => {
        if (!e.contentChanges.length) return;
        for (const viewer of viewers) viewer.scheduleReload(e.document.uri);
    }));
    context.subscriptions.push(vscode.window.tabGroups.onDidChangeTabs((e) => {
        keepViewerColumnClear(e).catch((err) => console.error('threejs-viewer:', err));
    }));
    // .glb opens in the preview by default; .gltf is JSON, so its preview is
    // an "Open With..." option beside the text editor (see package.json).
    for (const viewType of ['threejsViewer.modelPreview', 'threejsViewer.modelPreviewGltf']) {
        context.subscriptions.push(vscode.window.registerCustomEditorProvider(
            viewType, new ModelPreviewProvider(),
            { webviewOptions: { retainContextWhenHidden: true }, supportsMultipleEditorsPerDocument: true },
        ));
    }

    // API for other extensions: vscode.extensions.getExtension('local.threejs-viewer-webview').exports
    return {
        /**
         * Show a scene in the viewer tab, opening it beside the active editor
         * if needed. A scene is a list of viewer protocol messages (the JSON
         * ViewerClient sends); a binary message carries its payload as
         * `bytes` (Uint8Array) instead of a `blob_url`. Showing the same
         * `key` again replaces that scene and leaves everything else. It
         * only sends the difference: objects whose transform changed are
         * moved, each by its own matrix; objects whose add message, payload
         * or other per-object messages changed are deleted and added again,
         * together with anything parented under them; removed objects are
         * deleted; unchanged objects are left alone. A change to a message
         * that names no object replaces the scene whole.
         * @param {string} key
         * @param {{ title?: string, messages: any[] }} scene
         * @param {{ frame?: boolean }} [opts] frame the whole scene once loaded (default true)
         * @returns {false | { full: boolean, readded: number, removed: number, moved: number }}
         *   false when no viewer tab was available, else what was sent
         */
        showScene(key, scene, opts = {}) {
            const viewer = openMainViewer(mainViewer ? undefined : vscode.ViewColumn.Beside);
            if (!viewer) return false;
            return viewer.showScene(key, scene, opts.frame !== false);
        },
        /** Whether the viewer tab currently shows scene `key`. */
        hasScene(key) {
            return !!mainViewer && mainViewer.scenes.has(key);
        },
    };
}

function deactivate() {
    if (mainViewer) mainViewer.panel.dispose();
}

module.exports = { activate, deactivate };
