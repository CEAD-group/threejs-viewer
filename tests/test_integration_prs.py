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
        "timeout": 65000,
        "closed": True,
        "retry": 500,
        "probes": 2,
        "sockets": 2,
    }


# A bare viewer instance with fake fetch, WebSocket and timers, for driving the
# connect() retry loop step by step. `probe` decides each probe's outcome.
_CONNECT_HARNESS = """(options) => {
    const v = Object.create(window.threejsViewer.constructor.prototype);
    Object.assign(v, {_destroyed: false, _wsUrl: 'ws://localhost:1', _connectStarted: false,
                      _ws: null, _handshakeTimeout: null, _reconnectTimeout: null,
                      _statusDot: {className: '', title: ''}, _statusText: {textContent: ''},
                      _connected: false, _connectionHooks: [], _sceneGeneration: 0,
                      _animGeneration: 0, _options: options, _stableOpenTimeout: null,
                      _onVisibilityChange: null, _unknownMessageHooks: [() => {}]});
    const real = {fetch: window.fetch, WebSocket: window.WebSocket,
                  setTimeout: window.setTimeout, clearTimeout: window.clearTimeout};
    const h = {v, timers: new Map(), sockets: [], next: 1, probe: () => new Response()};
    window.fetch = async () => h.probe();
    window.WebSocket = class {
        static CONNECTING = 0; static OPEN = 1;
        constructor() { this.readyState = 0; h.sockets.push(this); }
        send() {}
        close() { this.readyState = 3; }
    };
    window.setTimeout = (fn, ms) => { const id = h.next++; h.timers.set(id, {fn, ms}); return id; };
    window.clearTimeout = (id) => h.timers.delete(id);
    h.tick = async () => { for (let i = 0; i < 4; i++) await Promise.resolve(); };
    h.fire = async (id) => { const t = h.timers.get(id); h.timers.delete(id); t.fn(); await h.tick(); return t.ms; };
    h.error = () => {
        const e = v.lastConnectError();
        if (!e) return e;
        const {at, ...rest} = e;
        return typeof at === 'number' && at > 0 ? rest : {badAt: at};
    };
    h.restore = () => {
        v._destroyed = true;
        Object.assign(window, real);
        document.removeEventListener('visibilitychange', v._onVisibilityChange);
    };
    window.__connectHarness = h;
    return true;
}"""


@pytest.mark.browser
def test_reconnect_backoff_doubles_to_cap_and_resets_on_open(viewer_page):
    """Failed attempts back off 0.5, 1, 2, 4, 8, 10, 10 s; the first message on
    an open socket resets the count, so a server restart is retried after 500 ms
    again. Hooks and lastConnectError() hand out copies (issue #273)."""
    viewer_page.evaluate(_CONNECT_HARNESS, None)
    result = viewer_page.evaluate(
        """async () => {
            const h = window.__connectHarness, v = h.v;
            try {
                h.probe = () => { throw new TypeError('offline'); };
                v.connect(); await h.tick();
                const delays = [h.timers.get(v._reconnectTimeout).ms];
                for (let i = 0; i < 6; i++) {
                    await h.fire(v._reconnectTimeout);
                    delays.push(h.timers.get(v._reconnectTimeout).ms);
                }
                const probeError = h.error();
                h.probe = () => new Response();
                await h.fire(v._reconnectTimeout);
                const ws = h.sockets[0];
                const hooks = [];
                v.onConnectionChange((connected, error) => {
                    hooks.push([connected, error && {phase: error.phase, code: error.code}]);
                    if (error) error.code = 0;
                });
                ws.readyState = 1; ws.onopen();
                const afterOpen = {failures: v._reconnectFailures, error: v.lastConnectError()};
                ws.onmessage({data: '{"type": "app_ping"}'});
                const afterMessage = v._reconnectFailures;
                ws.readyState = 3; ws.onclose({code: 1006, reason: ''});
                v.lastConnectError().phase = 'changed by caller';
                return {delays, probeError, afterOpen, afterMessage, hooks,
                        afterClose: {delay: h.timers.get(v._reconnectTimeout).ms, error: h.error()}};
            } finally { h.restore(); }
        }"""
    )
    assert result == {
        "delays": [500, 1000, 2000, 4000, 8000, 10000, 10000],
        "probeError": {
            "phase": "probe",
            "message": "offline",
            "attempt": 7,
            "retryInMs": 10000,
        },
        "afterOpen": {"failures": 7, "error": None},
        "afterMessage": 0,
        "hooks": [[True, None], [False, {"phase": "closed", "code": 1006}]],
        "afterClose": {
            "delay": 500,
            "error": {
                "phase": "closed",
                "code": 1006,
                "reason": "",
                "attempt": 1,
                "retryInMs": 500,
            },
        },
    }


@pytest.mark.browser
def test_accept_then_close_keeps_backing_off(viewer_page):
    """A server that accepts and closes at once does not reset the backoff;
    a socket open for STABLE_OPEN_MS does (issue #273)."""
    viewer_page.evaluate(_CONNECT_HARNESS, None)
    result = viewer_page.evaluate(
        """async () => {
            const h = window.__connectHarness, v = h.v;
            try {
                v.connect(); await h.tick();
                const delays = [];
                for (let i = 0; i < 4; i++) {
                    const ws = h.sockets[i];
                    ws.readyState = 1; ws.onopen();
                    ws.readyState = 3; ws.onclose({code: 1000, reason: ''});
                    delays.push(h.timers.get(v._reconnectTimeout).ms);
                    await h.fire(v._reconnectTimeout);
                }
                const ws = h.sockets[4];
                ws.readyState = 1; ws.onopen();
                const stableMs = await h.fire(v._stableOpenTimeout);
                return {delays, stableMs, failures: v._reconnectFailures};
            } finally { h.restore(); }
        }"""
    )
    assert result == {
        "delays": [500, 1000, 2000, 4000],
        "stableMs": 5000,
        "failures": 0,
    }


@pytest.mark.browser
def test_tab_shown_again_retries_at_once(viewer_page):
    """A visibilitychange to visible while disconnected cancels the pending
    retry and connects now; hidden, or with no retry pending, it does nothing
    (issue #273)."""
    viewer_page.evaluate(_CONNECT_HARNESS, None)
    result = viewer_page.evaluate(
        """async () => {
            const h = window.__connectHarness, v = h.v;
            let state = 'hidden', probes = 0;
            Object.defineProperty(document, 'visibilityState', {configurable: true, get: () => state});
            const show = async (s) => {
                state = s; document.dispatchEvent(new Event('visibilitychange')); await h.tick();
            };
            try {
                h.probe = () => { probes++; throw new TypeError('offline'); };
                v.connect(); await h.tick();
                const pending = v._reconnectTimeout;
                await show('hidden');
                const hidden = {probes, same: v._reconnectTimeout === pending};
                await show('visible');
                const visible = {probes, oldGone: !h.timers.has(pending),
                                 next: h.timers.get(v._reconnectTimeout).ms};
                return {hidden, visible};
            } finally {
                delete document.visibilityState;
                h.restore();
            }
        }"""
    )
    assert result == {
        "hidden": {"probes": 1, "same": True},
        "visible": {"probes": 2, "oldGone": True, "next": 1000},
    }


@pytest.mark.browser
@pytest.mark.parametrize(
    ("option", "expected"),
    [
        (None, 65000),
        (1000, 5000),
        (20000, 20000),
        ("7000", 7000),
        ("not a number", 65000),
        ("Infinity", None),
    ],
)
def test_handshake_timeout_option_is_bounded(viewer_page, option, expected):
    """`handshakeTimeoutMs` defaults to 65 s, is raised to 5 s when lower, and
    `Infinity` arms no cutoff at all (issue #273)."""
    options = None if option is None else {"handshakeTimeoutMs": option}
    viewer_page.evaluate(_CONNECT_HARNESS, options)
    result = viewer_page.evaluate(
        """async () => {
            const h = window.__connectHarness, v = h.v;
            try {
                v.connect(); await h.tick();
                return {sockets: h.sockets.length,
                        timeout: v._handshakeTimeout ? h.timers.get(v._handshakeTimeout).ms : null};
            } finally { h.restore(); }
        }"""
    )
    assert result == {"sockets": 1, "timeout": expected}


@pytest.mark.browser
def test_last_connect_error_reports_probe_status_close_code_and_timeout(viewer_page):
    """A rejected upgrade is reported with its close code and the status of
    the probe sent to the same URL; a stalled one as a timeout (issue #273)."""
    viewer_page.evaluate(_CONNECT_HARNESS, None)
    result = viewer_page.evaluate(
        """async () => {
            const h = window.__connectHarness, v = h.v;
            try {
                h.probe = () => new Response(null, {status: 503});
                const before = v.lastConnectError();
                v.connect(); await h.tick();
                h.sockets[0].readyState = 3;
                h.sockets[0].onclose({code: 1006, reason: ''});
                const rejected = h.error();
                await h.fire(v._reconnectTimeout);
                await h.fire(v._handshakeTimeout);
                return {before, rejected, stalled: h.error(),
                        stalledClosed: h.sockets[1].readyState === 3};
            } finally { h.restore(); }
        }"""
    )
    assert result == {
        "before": None,
        "rejected": {
            "phase": "handshake",
            "code": 1006,
            "reason": "",
            "status": 503,
            "attempt": 1,
            "retryInMs": 500,
        },
        "stalled": {"phase": "timeout", "status": 503, "attempt": 2, "retryInMs": 1000},
        "stalledClosed": True,
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
