// node --test vscode-extension/test
const test = require('node:test');
const assert = require('node:assert/strict');
const { sceneState, diffScene } = require('../scene_diff');

const tf = (x) => ({ matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 0, 0, 1] });

/**
 * A model of the viewer's object store: what applying `messages` leaves, as
 * id -> { def, transform, props }. A delete takes the children along, as the
 * viewer's does.
 */
function apply(store, messages) {
    for (const m of messages) {
        if (m.type === 'delete_object') {
            const gone = new Set([m.id]);
            let grew = true;
            while (grew) {
                grew = false;
                for (const [id, o] of store) if (!gone.has(id) && gone.has(o.def.parent)) { gone.add(id); grew = true; }
            }
            for (const id of gone) store.delete(id);
        } else if (m.type.startsWith('add_')) {
            const { bytes, ...def } = m;
            store.set(m.id, { def: JSON.stringify(def) + (bytes ? `#${[...bytes]}` : ''), transform: null, props: [] });
            store.get(m.id).def = { ...def, sig: store.get(m.id).def };
        } else if (m.type === 'batch_update') {
            for (const [id, t] of Object.entries(m.transforms)) if (store.has(id)) store.get(id).transform = JSON.stringify(t);
        } else if (m.id && store.has(m.id)) {
            store.get(m.id).props.push(JSON.stringify(m));
        }
    }
    return store;
}

const view = (store) => Object.fromEntries([...store].map(([id, o]) => [id, [o.def.sig, o.transform, o.props]]));

/** @returns {any[]} */
const base = () => [
    { type: 'add_group', id: 'root' },
    { type: 'add_group', id: 'arm', parent: 'root' },
    { type: 'add_model_binary', id: 'arm_mesh', parent: 'arm', format: 'glb', bytes: new Uint8Array([1, 2, 3]) },
    { type: 'add_group', id: 'table' },
    { type: 'set_visibility', id: 'table', visible: false },
    { type: 'batch_update', transforms: { root: tf(0), arm: tf(1), table: tf(2) } },
];

test('an unchanged scene patches to nothing', () => {
    const patch = diffScene(sceneState(base()), sceneState(base()));
    assert.deepEqual(patch.messages, []);
    assert.equal(patch.readded + patch.removed + patch.moved, 0);
});

test('a moved object is only moved', () => {
    const next = base();
    next[5] = { type: 'batch_update', transforms: { root: tf(0), arm: tf(5), table: tf(2) } };
    const patch = diffScene(sceneState(base()), sceneState(next));
    assert.deepEqual(patch.messages, [{ type: 'batch_update', transforms: { arm: tf(5) } }]);
    assert.equal(patch.moved, 1);
});

test('a changed payload re-adds the object and everything under it', () => {
    const next = base();
    next[1] = { type: 'add_group', id: 'arm', parent: 'root', name: 'renamed' };
    const patch = diffScene(sceneState(base()), sceneState(next));
    assert.deepEqual(patch.messages.filter((m) => m.type === 'delete_object'), [{ type: 'delete_object', id: 'arm' }]);
    assert.deepEqual(patch.messages.filter((m) => m.type.startsWith('add_')).map((m) => m.id), ['arm', 'arm_mesh']);
    assert.equal(patch.readded, 2);
});

test('an object whose per-object messages change is re-added with them', () => {
    const next = base();
    next[4] = { type: 'set_visibility', id: 'table', visible: true };
    const patch = diffScene(sceneState(base()), sceneState(next));
    assert.ok(patch.messages.some((m) => m.type === 'set_visibility' && m.id === 'table' && m.visible === true));
});

test('a change to a message naming no object needs a full send', () => {
    const prev = [...base(), { type: 'set_background', color: 1 }];
    const next = [...base(), { type: 'set_background', color: 2 }];
    assert.equal(diffScene(sceneState(prev), sceneState(next)), null);
});

test('full then patch leaves what a full send of the new scene leaves', () => {
    const next = [
        { type: 'add_group', id: 'root' },
        { type: 'add_group', id: 'arm', parent: 'root' },
        // new payload: re-added
        { type: 'add_model_binary', id: 'arm_mesh', parent: 'arm', format: 'glb', bytes: new Uint8Array([9]) },
        // `table` removed, `tool` new
        { type: 'add_group', id: 'tool', parent: 'arm' },
        { type: 'set_color', id: 'tool', color: 255 },
        { type: 'batch_update', transforms: { root: tf(3), arm: tf(1), tool: tf(4) } },
    ];
    const patch = diffScene(sceneState(base()), sceneState(next));
    assert.ok(patch);
    const patched = apply(apply(new Map(), base()), patch.messages);
    const fresh = apply(new Map(), next);
    assert.deepEqual(view(patched), view(fresh));
    assert.deepEqual(patch.roots, ['root']);
    assert.equal(patch.removed, 1);
});
