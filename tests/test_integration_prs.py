"""Cross-feature checks for the seven-PR integration branch."""

import numpy as np
import pytest

from conftest import frames, settle
from test_browser import (
    _PARENTED_OVERLAY_SETUP,
    _PARENTED_OVERLAY_WORLD,
    _hold_follow_path_blobs,
    _wait_for,
    _wait_for_animation_loaded,
)
from threejs_viewer import Animation, Frame


@pytest.mark.browser
@pytest.mark.parametrize("path_first", [True, False])
def test_follow_path_and_overlay_survive_same_id_replacement(
    viewer_client, viewer_page, path_first
):
    """PRs #268 and #269 must preserve both resources in either fetch order."""
    page = viewer_page
    viewer_client.add_box("link")
    settle(viewer_client)
    page.evaluate(
        _PARENTED_OVERLAY_SETUP,
        {"id": "triad", "parentId": "link", "size": 0.2, "x": 0, "y": 0, "z": 2},
    )
    path = dict(
        times=[0.0, 2.0],
        positions=[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        axes=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    )
    if not path_first:
        wait_held, release = _hold_follow_path_blobs(viewer_client, page)
    viewer_client.set_follow_path("link", **path)
    if path_first:
        settle(viewer_client)
    else:
        wait_held(1)
    viewer_client.add_mesh(
        "link",
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 2]], dtype=np.uint32),
    )
    _wait_for(
        page,
        "() => window.threejsViewer.getObject('link')?.userData.isMesh === true",
    )
    if not path_first:
        release(0)
    settle(viewer_client)
    assert page.evaluate("() => window.threejsViewer._followPaths.has('link')")
    assert page.evaluate(
        "() => window.__ov.parent === window.threejsViewer.getObject('link')"
        " && window.__disposed === false"
    )
    viewer_client.load_animation(
        Animation(
            frames=[Frame(time=0, transforms={}), Frame(time=2, transforms={})],
            loop=False,
        ),
        autoplay=False,
        initial_time=1.0,
    )
    _wait_for_animation_loaded(page)
    frames(page)
    world = page.evaluate(_PARENTED_OVERLAY_WORLD)
    assert [world["x"], world["y"], world["z"]] == pytest.approx([2, 0, 2])
    assert world["local"] == pytest.approx([0, 0, 2])
    viewer_client.delete("link")
    settle(viewer_client)
    assert not page.evaluate("() => window.threejsViewer._followPaths.has('link')")
    assert page.evaluate("() => window.__ov.parent === null && !window.__disposed")
    viewer_client.add_box("link")
    settle(viewer_client)
    assert page.evaluate(
        "() => window.__ov.parent === window.threejsViewer.getObject('link')"
    )


@pytest.mark.browser
@pytest.mark.parametrize("probe_succeeds", [True, False])
def test_destroy_during_probe_leaves_no_retry(viewer_page, probe_succeeds):
    result = viewer_page.evaluate(
        """async (succeeds) => {
            const proto = window.threejsViewer.constructor.prototype;
            const v = Object.create(proto);
            Object.assign(v, {_destroyed: false, _wsUrl: 'ws://localhost:1',
                              _connectStarted: false, _ws: null, _reconnectTimeout: null});
            const realFetch = window.fetch;
            let resolve, reject;
            window.fetch = () => new Promise((a, b) => { resolve = a; reject = b; });
            try {
                v.connect();
                v._destroyed = true;
                if (succeeds) resolve(new Response());
                else reject(new TypeError('offline'));
                await Promise.resolve();
                await Promise.resolve();
                return {socket: v._ws, retry: v._reconnectTimeout};
            } finally { window.fetch = realFetch; }
        }""",
        probe_succeeds,
    )
    assert result == {"socket": None, "retry": None}


@pytest.mark.browser
def test_stalled_socket_upgrade_retries_without_duplicate_loops(viewer_page):
    result = viewer_page.evaluate(
        """async () => {
            const v = Object.create(window.threejsViewer.constructor.prototype);
            Object.assign(v, {_destroyed: false, _wsUrl: 'ws://localhost:1',
                              _connectStarted: false, _ws: null});
            const realFetch = window.fetch, RealWS = window.WebSocket;
            const realSet = window.setTimeout, realClear = window.clearTimeout;
            const timers = new Map(), sockets = [];
            let next = 1, probes = 0;
            window.fetch = async () => { probes++; return new Response(); };
            window.WebSocket = class {
                static CONNECTING = 0;
                constructor() { this.readyState = 0; sockets.push(this); }
                close() { this.readyState = 3; }
            };
            window.setTimeout = (fn, ms) => { const id = next++; timers.set(id, {fn, ms}); return id; };
            window.clearTimeout = (id) => timers.delete(id);
            const tick = async () => { await Promise.resolve(); await Promise.resolve(); };
            try {
                v.connect(); v.connect();
                await tick();
                const initial = {probes, sockets: sockets.length};
                const timeout = timers.get(v._handshakeTimeout);
                if (!timeout) return {initial, timeout: null};
                timeout.fn();
                const closed = sockets[0].readyState === 3;
                const retry = timers.get(v._reconnectTimeout);
                retry.fn(); await tick();
                return {initial, timeout: timeout.ms, closed, retry: retry.ms,
                        probes, sockets: sockets.length};
            } finally {
                v._destroyed = true;
                window.fetch = realFetch; window.WebSocket = RealWS;
                window.setTimeout = realSet; window.clearTimeout = realClear;
            }
        }"""
    )
    assert result == {
        "initial": {"probes": 1, "sockets": 1},
        "timeout": 5000,
        "closed": True,
        "retry": 500,
        "probes": 2,
        "sockets": 2,
    }


@pytest.mark.browser
def test_slow_reachability_probe_can_establish_socket(viewer_page):
    result = viewer_page.evaluate(
        """async () => {
            const v = Object.create(window.threejsViewer.constructor.prototype);
            Object.assign(v, {_destroyed: false, _wsUrl: 'ws://localhost:1',
                              _connectStarted: false, _ws: null, _reconnectTimeout: null});
            const realFetch = window.fetch, RealWS = window.WebSocket;
            window.fetch = (url, options) => new Promise((resolve, reject) => {
                const timer = setTimeout(() => resolve(new Response()), 650);
                options.signal.addEventListener('abort', () => {
                    clearTimeout(timer); reject(options.signal.reason);
                }, {once: true});
            });
            window.WebSocket = class {
                static CONNECTING = 0;
                constructor() { this.readyState = 0; }
                close() { this.readyState = 3; }
            };
            try {
                v.connect();
                await new Promise((resolve) => setTimeout(resolve, 750));
                return {socketCreated: !!v._ws, retryScheduled: !!v._reconnectTimeout};
            } finally {
                v._destroyed = true;
                clearTimeout(v._handshakeTimeout); clearTimeout(v._reconnectTimeout);
                window.fetch = realFetch; window.WebSocket = RealWS;
            }
        }"""
    )
    assert result == {"socketCreated": True, "retryScheduled": False}
