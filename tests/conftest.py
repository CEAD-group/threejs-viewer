"""Shared fixtures for threejs-viewer tests."""

import socket
import threading

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
        viewer_path = viewer_client.viewer_path.resolve()
        url = f"file://{viewer_path}?ws_port={viewer_client.port}"
        try:
            page.goto(url, timeout=90_000)
        except Exception:
            page.goto(url, timeout=90_000)

        # Wait for the browser to connect via WebSocket
        connected = viewer_client._connected_event.wait(timeout=30)
        assert connected, "Browser did not connect to WebSocket server within 30s"

        yield page
