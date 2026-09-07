"""Shared fixtures for threejs-viewer tests."""

import socket
import threading
import time

import pytest

from threejs_viewer import ViewerClient

try:
    import pytest_playwright  # noqa: F401

    _has_playwright = True
except ImportError:
    _has_playwright = False


def pytest_collection_modifyitems(config, items):
    """Auto-skip browser-marked tests when pytest-playwright is not installed."""
    if _has_playwright:
        return
    skip = pytest.mark.skip(reason="pytest-playwright not installed")
    for item in items:
        if "browser" in item.keywords:
            item.add_marker(skip)


def _free_port():
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture()
def viewer_client():
    """Start a ViewerClient on a random port (does not wait for browser).

    The HTTP blob sidecar binds port 0 (the OS hands out a genuinely free
    port, no allocate-release-rebind window): under parallel/full-suite load
    the old `_free_port()`-then-bind dance intermittently lost the race and
    errored the whole test at setup with "Address already in use" (#95). The
    blob URLs embed the actual bound port, so it does not need to be ws+1.
    """
    port = _free_port()
    client = ViewerClient(port=port, open_browser=False)

    from http.server import HTTPServer

    from threejs_viewer.client import _BlobHandler

    http_server = HTTPServer((client.host, 0), _BlobHandler)
    client._http_port = http_server.server_address[1]
    http_server.blob_store = client._blob_store
    client._http_server = http_server
    threading.Thread(target=http_server.serve_forever, daemon=True).start()

    client._server_thread = threading.Thread(target=client._run_server, daemon=True)
    client._server_thread.start()

    yield client

    client.disconnect()


if _has_playwright:

    @pytest.fixture()
    def viewer_page(viewer_client, page):
        """Open the viewer in Playwright and wait for WebSocket connection.

        goto gets a generous timeout plus one retry: under full-suite machine
        load a file:// navigation occasionally exceeded the 30 s default and
        errored unrelated tests at setup (#95).
        """
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        viewer_path = viewer_client.viewer_path.resolve()
        url = f"{viewer_path.as_uri()}?ws_port={viewer_client.port}"
        try:
            page.goto(url, timeout=90_000)
        except PlaywrightTimeoutError:
            page.goto(url, timeout=90_000)

        # Wait for the browser to connect via WebSocket. The budget is
        # generous on purpose: this is a fixed deadline on a *loaded shared
        # runner*, not on the viewer doing anything slow — a cold Chromium
        # plus a file:// navigation plus the WS handshake normally lands in
        # well under a second, and 30 s once expired mid-release, failing an
        # unrelated test at setup and costing a re-run of the publish job.
        # Nothing is waiting on this number except a genuinely hung browser,
        # so trade a slower hard failure for not flaking.
        budget = 120
        connected = viewer_client._connected_event.wait(timeout=budget)
        assert connected, (
            f"Browser did not connect to the WebSocket server within {budget}s "
            f"(url={url})"
        )

        yield page


def settle(client, timeout: float = 15.0):
    """Block until the viewer has finished everything sent so far.

    Two guarantees, and the reason this exists instead of ``time.sleep``:

    1. **WS barrier.** ``query_scene`` is a request/response round-trip over
       the same socket the ``add_*``/``set_*`` messages went out on. Frames on
       one WebSocket are ordered and the viewer's ``handleMessage`` switch is
       synchronous, so by the time the reply lands every message sent before
       the query has been handled. A ``page.evaluate`` read, by contrast,
       travels over CDP — a *different* channel with no ordering relative to
       the socket — which is exactly the race a fixed sleep was papering over.
    2. **Fetch drain.** Binary payloads (meshes, polylines, tubes, models,
       animations) load over the HTTP sidecar *after* their message is
       handled. Every such case bumps ``_pendingFetches`` synchronously at the
       top of the case and drops it in a ``finally`` after the object is
       registered, and the counter rides along on ``query_scene``'s
       ``meta.pending_fetches`` — so polling it to zero means the objects are
       actually in the scene graph, not merely requested.

    What it deliberately does **not** cover: work driven by the render loop
    rather than by a message — octree-LOD node streaming (whose fetches are
    dispatched from the per-frame traversal, so ``pending_fetches`` reads 0
    before they even start), the 2 Hz tube-LOD throttle, rAF-coalesced resize
    and hover, and animation playback advancing the clock. Those are genuine
    wall-clock waits; keep the sleep (or poll for the end state).
    """
    deadline = time.monotonic() + timeout
    while True:
        meta = client.query_scene()["meta"]
        if meta.get("pending_fetches", 0) == 0:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Viewer still had {meta['pending_fetches']} fetch(es) in "
                f"flight after {timeout}s"
            )
        time.sleep(0.01)


_FRAME_WAIT_JS = """(n) => new Promise((resolve) => {
    let i = 0;
    const tick = () => (++i >= n ? resolve(null) : requestAnimationFrame(tick));
    requestAnimationFrame(tick);
})"""


def frames(page, n: int = 2):
    """Block until the viewer has rendered *n* animation frames.

    The companion to :func:`settle` for the other half of the waiting the
    browser tests do. Anything driven by the render loop rather than by a
    WebSocket message — a ``_seekToTime`` applied on the next tick, a key
    press cycling the M/N debug modes, a camera move, a pixel readback — is
    ready after a frame or two, and ``requestAnimationFrame`` says exactly
    when that is. A ``time.sleep`` guesses, and guesses low on a loaded
    runner.

    Two frames by default: one to run the handler that was queued, one to
    render its result.
    """
    page.evaluate(_FRAME_WAIT_JS, n)
