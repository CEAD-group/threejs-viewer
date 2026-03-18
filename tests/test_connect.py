"""Tests for ViewerClient.connect() auto-open browser behavior."""

import threading
import time
from unittest.mock import patch

from threejs_viewer import ViewerClient

# Use high ports to avoid conflicts; each test gets its own port range.
_PORT_BASE = 15660


def _unique_port(offset):
    return _PORT_BASE + offset * 2  # *2 because HTTP sidecar uses port+1


def _connect_and_catch(client, timeout):
    """Call connect() and return None on success, the exception on failure."""
    try:
        client.connect(timeout=timeout)
        return None
    except TimeoutError as e:
        return e
    except OSError as e:
        return e


def test_open_browser_false_never_calls_webbrowser():
    """open_browser=False should never invoke webbrowser.open."""
    client = ViewerClient(port=_unique_port(0), open_browser=False)
    with patch("threejs_viewer.client.webbrowser.open") as mock_open:
        err = _connect_and_catch(client, timeout=0.2)
    mock_open.assert_not_called()
    assert isinstance(err, TimeoutError)


def test_open_browser_true_calls_open_method():
    """open_browser=True should call _open_viewer_in_browser when no tab connects."""
    client = ViewerClient(port=_unique_port(1), open_browser=True)
    with patch.object(client, "_open_viewer_in_browser") as mock_open:
        err = _connect_and_catch(client, timeout=0.5)
    mock_open.assert_called_once()
    assert isinstance(err, TimeoutError)


def test_open_browser_url_contains_ws_port():
    """Opened URL should contain ws_port query parameter matching the port."""
    port = _unique_port(2)
    client = ViewerClient(port=port, open_browser=True)
    with patch("threejs_viewer.client.webbrowser.open", return_value=True) as mock_wb:
        _connect_and_catch(client, timeout=0.5)
    mock_wb.assert_called_once()
    url = mock_wb.call_args[0][0]
    assert f"?ws_port={port}" in url


def test_open_browser_url_is_valid_file_uri():
    """Opened URL should be a valid file:// URI via Path.as_uri()."""
    client = ViewerClient(port=_unique_port(3), open_browser=True)
    with patch("threejs_viewer.client.webbrowser.open", return_value=True) as mock_wb:
        _connect_and_catch(client, timeout=0.5)
    url = mock_wb.call_args[0][0]
    assert url.startswith("file:///")
    assert "viewer.html" in url


def test_open_browser_skipped_when_tab_reconnects_during_grace():
    """If a connection arrives during grace period, browser should not open."""
    client = ViewerClient(port=_unique_port(4), open_browser=True)

    def simulate_reconnect():
        time.sleep(0.3)
        client._connected_event.set()

    t = threading.Thread(target=simulate_reconnect, daemon=True)
    t.start()

    with patch.object(client, "_open_viewer_in_browser") as mock_open:
        client.connect(timeout=5.0)
    mock_open.assert_not_called()
    client.disconnect()
    t.join(timeout=1)


def test_timeout_zero_does_not_open_browser():
    """timeout=0 should not attempt to open browser (no time to connect)."""
    client = ViewerClient(port=_unique_port(5), open_browser=True)
    with patch.object(client, "_open_viewer_in_browser") as mock_open:
        err = _connect_and_catch(client, timeout=0)
    mock_open.assert_not_called()
    assert isinstance(err, TimeoutError)


def test_timeout_cleans_up_servers():
    """On timeout, servers should be shut down so the port is released."""
    client = ViewerClient(port=_unique_port(6), open_browser=False)
    err = _connect_and_catch(client, timeout=0.2)
    assert isinstance(err, TimeoutError)
    assert client._http_server is None
    assert client._server is None


def test_timeout_respects_deadline():
    """Total wait should not significantly exceed the requested timeout."""
    client = ViewerClient(port=_unique_port(7), open_browser=False)
    start = time.monotonic()
    _connect_and_catch(client, timeout=0.3)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0


def test_constructor_open_browser_default():
    """open_browser should default to True."""
    client = ViewerClient()
    assert client.open_browser is True
