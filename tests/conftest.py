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
    """Start a ViewerClient on a random port (does not wait for browser)."""
    port = _free_port()
    client = ViewerClient(port=port)

    # Start the server in the background without blocking for a browser connection
    client._http_port = port + 1
    from http.server import HTTPServer

    from threejs_viewer.client import _BlobHandler

    http_server = HTTPServer((client.host, client._http_port), _BlobHandler)
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
        """Open the viewer in Playwright and wait for WebSocket connection."""
        viewer_path = viewer_client.viewer_path.resolve()
        url = f"file://{viewer_path}?ws_port={viewer_client.port}"
        page.goto(url)

        # Wait for the browser to connect via WebSocket
        connected = viewer_client._connected_event.wait(timeout=10)
        assert connected, "Browser did not connect to WebSocket server within 10s"

        yield page
