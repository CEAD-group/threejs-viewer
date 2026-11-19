// The scene diff behind the exported showScene API (see README.md, "API for
// other extensions"). Pure functions, no VS Code: test/scene_diff.test.js
// runs them under plain Node.
const crypto = require('crypto');

/** @type {WeakMap<Uint8Array, string>} payload -> sha1, so a payload shared by several messages hashes once */
const digests = new WeakMap();

/** @param {Uint8Array} bytes */
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
 * @param {ReturnType<typeof sceneState>} prev @param {ReturnType<typeof sceneState>} next
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
    /** @type {any[]} the viewer messages of the patch */
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

module.exports = { digestOf, sceneState, diffScene };
