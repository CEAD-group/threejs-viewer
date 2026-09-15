"""Tests for ViewerClient."""

import json
import math
import socket
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
from websockets.sync.client import connect as ws_connect

from threejs_viewer import Animation, Frame, ViewerClient


def test_client_instantiation():
    """Test that ViewerClient can be instantiated."""
    client = ViewerClient()
    assert client.host == "localhost"
    assert client.port == 5666


def test_client_custom_host_port():
    """Test ViewerClient with custom host/port."""
    client = ViewerClient(host="127.0.0.1", port=8080)
    assert client.host == "127.0.0.1"
    assert client.port == 8080


def test_viewer_path():
    """Test that viewer_path points to existing file."""
    client = ViewerClient()
    path = client.viewer_path

    assert isinstance(path, Path)
    assert path.exists()
    assert path.name == "viewer.html"


def _params(url: str) -> dict[str, list[str]]:
    return parse_qs(urlparse(url).query)


def test_viewer_url_default_has_only_ws_port():
    """No lighting kwargs → only ws_port in the query string."""
    client = ViewerClient(port=1234)
    params = _params(client.viewer_url)
    assert params == {"ws_port": ["1234"]}


def test_viewer_url_with_tone_mapping_exposure_zero():
    """0.0 must still be emitted (falsy float slipped through the old check)."""
    client = ViewerClient(tone_mapping_exposure=0.0)
    params = _params(client.viewer_url)
    assert params["tone_mapping_exposure"] == ["0.0"]


def test_viewer_url_with_all_lighting_overrides():
    """All four lighting kwargs round-trip through the URL."""
    client = ViewerClient(
        tone_mapping="AgX",  # upper/mixed case accepted, normalized to lowercase
        tone_mapping_exposure=2.3,
        environment_intensity=0.5,
        ambient_intensity=0.7,
    )
    params = _params(client.viewer_url)
    assert params["ws_port"] == ["5666"]
    assert params["tone_mapping"] == ["agx"]
    assert params["tone_mapping_exposure"] == ["2.3"]
    assert params["environment_intensity"] == ["0.5"]
    assert params["ambient_intensity"] == ["0.7"]


def test_viewer_url_partial_overrides():
    """Only the kwargs the caller supplied appear in the URL."""
    client = ViewerClient(environment_intensity=1.25)
    params = _params(client.viewer_url)
    assert set(params) == {"ws_port", "environment_intensity"}
    assert params["environment_intensity"] == ["1.25"]


def test_viewer_url_environment_map_false():
    """environment_map=False emits environment_map=false (the perf toggle)."""
    client = ViewerClient(environment_map=False)
    params = _params(client.viewer_url)
    assert set(params) == {"ws_port", "environment_map"}
    assert params["environment_map"] == ["false"]


def test_viewer_url_environment_map_true():
    """environment_map=True emits environment_map=true."""
    client = ViewerClient(environment_map=True)
    params = _params(client.viewer_url)
    assert params["environment_map"] == ["true"]


def test_viewer_url_default_omits_environment_map():
    """No environment_map kwarg → no param (viewer/localStorage default)."""
    client = ViewerClient()
    assert "environment_map" not in _params(client.viewer_url)


def test_viewer_url_default_omits_fov():
    """No fov kwarg → no fov param (viewer uses its own default)."""
    client = ViewerClient()
    params = _params(client.viewer_url)
    assert "fov" not in params


def test_viewer_url_with_fov():
    """An explicit fov round-trips through the URL."""
    client = ViewerClient(fov=35)
    params = _params(client.viewer_url)
    assert params["fov"] == ["35.0"]


def test_viewer_client_rejects_invalid_tone_mapping():
    with pytest.raises(ValueError, match="tone_mapping must be one of"):
        ViewerClient(tone_mapping="bogus")


@pytest.mark.parametrize("bad", ["false", "true", 0, 1, "off"])
def test_viewer_client_rejects_non_bool_environment_map(bad):
    """A stray non-bool (e.g. "false") must raise, not silently coerce to
    environment_map=true via bool()."""
    with pytest.raises(ValueError, match="environment_map must be a bool or None"):
        ViewerClient(environment_map=bad)


@pytest.mark.parametrize("bad_fov", [0, 180, -10, 200])
def test_viewer_client_rejects_out_of_range_fov(bad_fov):
    with pytest.raises(ValueError, match=r"fov must be in the open interval"):
        ViewerClient(fov=bad_fov)


@pytest.mark.parametrize("bad_fov", [float("nan"), float("inf")])
def test_viewer_client_rejects_non_finite_fov(bad_fov):
    with pytest.raises(ValueError, match="fov must be a finite number"):
        ViewerClient(fov=bad_fov)


@pytest.mark.parametrize(
    "kwarg",
    ["tone_mapping_exposure", "environment_intensity", "ambient_intensity"],
)
def test_viewer_client_rejects_non_finite_floats(kwarg):
    with pytest.raises(ValueError, match="must be a finite number"):
        ViewerClient(**{kwarg: float("nan")})
    with pytest.raises(ValueError, match="must be a finite number"):
        ViewerClient(**{kwarg: float("inf")})


def test_enable_move_gizmo_payload():
    """enable_move_gizmo builds the wire payload (degrees → radians)."""
    client = ViewerClient()
    client.enable_move_gizmo(
        "box", mode="rotate", translate_snap=2.0, rotate_snap_deg=30, click_select=False
    )
    g = client._move_gizmo
    assert g["type"] == "set_move_gizmo"
    assert g["enabled"] is True
    assert g["id"] == "box"
    assert g["mode"] == "rotate"
    assert g["translateSnap"] == 2.0
    assert g["clickSelect"] is False
    assert g["rotateSnap"] == pytest.approx(math.radians(30))


def test_enable_move_gizmo_defaults():
    """No-arg enable uses translate / 1.0 grid / 15° snap / click-select on."""
    client = ViewerClient()
    client.enable_move_gizmo()
    g = client._move_gizmo
    assert g["id"] is None
    assert g["mode"] == "translate"
    assert g["translateSnap"] == 1.0
    assert g["translateSnapRelative"] is False
    assert g["clickSelect"] is True
    assert g["rotateSnap"] == pytest.approx(math.radians(15))


def test_enable_move_gizmo_relative_snap_flag():
    """translate_snap_relative is forwarded as translateSnapRelative."""
    client = ViewerClient()
    client.enable_move_gizmo("box", translate_snap=0.1, translate_snap_relative=True)
    assert client._move_gizmo["translateSnapRelative"] is True


def test_disable_move_gizmo_clears_state():
    client = ViewerClient()
    client.enable_move_gizmo("box")
    client.disable_move_gizmo()
    assert client._move_gizmo is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "scale"},
        {"translate_snap": 0},
        {"translate_snap": -1},
        {"translate_snap": float("inf")},
        {"rotate_snap_deg": 0},
        {"rotate_snap_deg": float("nan")},
    ],
)
def test_enable_move_gizmo_rejects_bad_args(kwargs):
    client = ViewerClient()
    with pytest.raises(ValueError):
        client.enable_move_gizmo(**kwargs)


def test_set_gizmo_axes_payload_and_defaults():
    """set_gizmo_axes builds the wire payload; no-arg restores all axes."""
    client = ViewerClient()
    client.set_gizmo_axes(x=False, y=False, z=True)
    a = client._gizmo_axes
    assert a == {"type": "set_gizmo_axes", "x": False, "y": False, "z": True}
    client.set_gizmo_axes()
    assert client._gizmo_axes == {
        "type": "set_gizmo_axes",
        "x": True,
        "y": True,
        "z": True,
    }


def test_disable_move_gizmo_clears_axis_constraint():
    """Disabling the gizmo drops any stored axis constraint (the viewer resets
    axes to all-true on detach, so the stale state must not replay)."""
    client = ViewerClient()
    client.enable_move_gizmo("box")
    client.set_gizmo_axes(x=True, y=False, z=False)
    client.disable_move_gizmo()
    assert client._gizmo_axes is None


def test_on_object_move_enables_and_dispatches():
    """Registering a callback enables the gizmo and receives moves."""
    client = ViewerClient()
    got = []
    client.on_object_move(got.append)
    assert client._move_gizmo is not None and client._move_gizmo["enabled"]
    client._dispatch_object_move(
        {
            "id": "box",
            "position": [1, 2, 3],
            "quaternion": [0, 0, 0, 1],
            "scale": [1, 1, 1],
            "matrix": list(range(16)),
            "positionStart": [0, 2, 3],
            "quaternionStart": [0, 0, 0, 1],
            "phase": "end",
        }
    )
    assert len(got) == 1
    assert got[0]["id"] == "box"
    assert got[0]["position"] == [1, 2, 3]
    assert got[0]["position_start"] == [0, 2, 3]
    assert got[0]["quaternion_start"] == [0, 0, 0, 1]
    assert got[0]["phase"] == "end"


def test_on_object_move_rejects_non_callable():
    client = ViewerClient()
    with pytest.raises(TypeError):
        client.on_object_move(42)


def test_add_gizmo_payload_and_accumulates():
    """add_gizmo builds a spec per call and accumulates them (multiple pinned
    gizmos), defaulting to all axes / translate / world space / free snap."""
    client = ViewerClient()
    client.add_gizmo("rail", x=False, y=False, z=True)
    client.add_gizmo("cube", space="local", mode="rotate", snap_default=True)
    assert client._gizmos == [
        {
            "type": "add_gizmo",
            "id": "rail",
            "x": False,
            "y": False,
            "z": True,
            "mode": "translate",
            "space": "world",
            "snapDefault": False,
        },
        {
            "type": "add_gizmo",
            "id": "cube",
            "x": True,
            "y": True,
            "z": True,
            "mode": "rotate",
            "space": "local",
            "snapDefault": True,
        },
    ]


def test_enable_move_gizmo_snap_default_flag():
    """enable_move_gizmo forwards snap_default into the wire payload."""
    client = ViewerClient()
    client.enable_move_gizmo("box", snap_default=True)
    assert client._move_gizmo["snapDefault"] is True
    client.enable_move_gizmo("box")
    assert client._move_gizmo["snapDefault"] is False


@pytest.mark.parametrize(
    "kwargs", [{"mode": "spin"}, {"space": "object"}, {"space": "World"}]
)
def test_add_gizmo_rejects_bad_args(kwargs):
    client = ViewerClient()
    with pytest.raises(ValueError):
        client.add_gizmo("box", **kwargs)


def test_clear_gizmos_and_disable_reset_pinned_state():
    """clear_gizmos empties the pinned list; disable_move_gizmo also clears it
    (the viewer's disable() removes pinned gizmos too)."""
    client = ViewerClient()
    client.add_gizmo("a")
    client.add_gizmo("b")
    client.clear_gizmos()
    assert client._gizmos == []
    client.add_gizmo("c")
    client.disable_move_gizmo()
    assert client._gizmos == []


def test_on_object_move_skips_primary_when_pinned_present():
    """With a pinned gizmo already present, registering a move callback does not
    also turn on the click-select interactive gizmo (which would draw an extra)."""
    client = ViewerClient()
    client.add_gizmo("box", x=False, y=False, z=True)
    client.on_object_move(lambda m: None)
    assert client._move_gizmo is None  # primary not auto-enabled
    assert len(client._gizmos) == 1


def test_clear_scene_drops_pinned_gizmos():
    """A scene clear forgets pinned gizmos so a reconnect can't re-pin them to
    ids that no longer exist."""

    class _StubWS:
        def send(self, _data):
            pass

    client = ViewerClient()
    client.add_gizmo("box")
    client._ws = _StubWS()  # clear() always sends; give it a no-op socket
    client.clear()
    assert client._gizmos == []


def _mini_animation():
    """Two-frame animation, just enough for load_animation's validation path."""
    return Animation(
        frames=[Frame(time=0, transforms={}), Frame(time=1, transforms={})],
        loop=True,
    )


@pytest.mark.parametrize("bad_loop", ["yes", "true", 1, 0, []])
def test_load_animation_rejects_non_bool_loop(bad_loop):
    """loop must be a real bool — strings/ints/etc. are not coerced silently."""
    client = ViewerClient()
    with pytest.raises(ValueError, match="loop must be a bool or None"):
        client.load_animation(_mini_animation(), loop=bad_loop)


@pytest.mark.parametrize(
    "bad_time",
    [float("nan"), float("inf"), "start", "bogus", [], True],
)
def test_load_animation_rejects_bad_initial_time(bad_time):
    """initial_time must be a finite number or the literal 'end'."""
    client = ViewerClient()
    with pytest.raises(ValueError, match="initial_time must be"):
        client.load_animation(_mini_animation(), initial_time=bad_time)


class TestVersion:
    """Version derivation (issue #183: hatch-vcs replaces the 0.0.0-dev placeholder)."""

    def test_version_is_pep440_and_not_the_old_placeholder(self):
        import threejs_viewer

        version = threejs_viewer.__version__
        assert isinstance(version, str) and version
        # The whole point of #183: an installed checkout no longer reports the
        # unsatisfiable 0.0.0 floor.
        assert not version.startswith("0.0.0")

    @pytest.mark.parametrize(
        "version",
        [
            "0.0.0-dev",
            "0.0.0.dev0",
            "0.0.51.dev3+ge8da667",
            "0.0.51+ge8da667",
            "unknown",
        ],
    )
    def test_dev_versions_suppress_mismatch_warning(self, version):
        from threejs_viewer.client import _is_dev_version

        assert _is_dev_version(version)

    @pytest.mark.parametrize("version", ["0.0.50", "1.2.3", "0.0.51a1"])
    def test_released_versions_still_compared(self, version):
        from threejs_viewer.client import _is_dev_version

        assert not _is_dev_version(version)


# --- Sidecar reachability over both loopback families (issue #187) ---


def _ipv6_loopback_available() -> bool:
    if not socket.has_ipv6:
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
        return True
    except OSError:
        return False


@pytest.fixture()
def bound_client():
    """A client with its servers bound on OS-chosen ports, no browser."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        ws_port = s.getsockname()[1]
    client = ViewerClient(port=ws_port, open_browser=False)
    client._start_servers(http_port=0)
    try:
        yield client
    finally:
        client.disconnect()


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=5) as resp:
        assert resp.status == 200
        return resp.read()


@pytest.mark.parametrize("address", ["127.0.0.1", "[::1]"])
def test_servers_reachable_on_both_loopback_families(bound_client, address):
    """``localhost`` may resolve to ::1 in the browser; both families must serve.

    The WebSocket half opens a real connection with the ``websockets`` sync
    client, which runs the same handshake the browser does.
    """
    if address == "[::1]" and not _ipv6_loopback_available():
        pytest.skip("no IPv6 loopback on this machine")
    payload = b"\x01\x02\x03\x04"
    bound_client._blob_store["/blob_test"] = payload
    assert _fetch(f"http://{address}:{bound_client._http_port}/blob_test") == payload
    # A real WebSocket handshake, so a regression in serve(sock=...) shows up
    # here and not only in the browser suite.
    with ws_connect(f"ws://{address}:{bound_client.port}", open_timeout=5):
        assert bound_client._connected_event.wait(timeout=5)


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "viewer.example", "::1"])
def test_blob_url_host_is_the_configured_host(host):
    """Blob URLs advertise ``host`` verbatim: one hostname in the page, or a
    file:// viewer page in Firefox refuses the sidecar fetch as cross-origin."""
    client = ViewerClient(host=host, port=5666, open_browser=False)
    client._ws = _CaptureWS()
    client.add_mesh(
        "m",
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 2]], dtype=np.uint32),
    )
    url = client._ws.messages[-1]["blob_url"]
    assert urlparse(url).hostname == host
    assert urlparse(url).port == 5667


def test_viewer_url_carries_ws_host_when_not_localhost():
    """The viewer defaults to ws://localhost; any other host rides along as
    ``ws_host`` so the WebSocket and the blob sidecar share one hostname."""
    assert "ws_host" not in _params(ViewerClient(port=1234).viewer_url)
    params = _params(ViewerClient(host="127.0.0.1", port=1234).viewer_url)
    assert params["ws_host"] == ["127.0.0.1"]
    assert _params(ViewerClient(host="::1", port=1).viewer_url)["ws_host"] == ["[::1]"]


class _CaptureWS:
    def __init__(self):
        self.messages = []

    def send(self, text):
        self.messages.append(json.loads(text))


def test_listen_sockets_raises_when_ipv6_port_is_taken():
    """A port in use on ::1 is a real failure, not a reason to go IPv4-only:
    the browser may still resolve localhost to ::1 and get nothing."""
    if not _ipv6_loopback_available():
        pytest.skip("no IPv6 loopback on this machine")
    from threejs_viewer.client import _listen_sockets

    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as taken:
        taken.bind(("::1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        with pytest.raises(OSError):
            _listen_sockets("localhost", port)
    # Nothing is left bound on the IPv4 side either.
    socks = _listen_sockets("localhost", port)
    assert {s.family for s in socks} == {socket.AF_INET, socket.AF_INET6}
    for s in socks:
        s.close()


def test_enable_object_click_payload():
    """enable_object_click stores the replayable enable message (issue #178)."""
    client = ViewerClient()
    assert client._object_click is None
    client.enable_object_click()
    assert client._object_click == {"type": "set_object_click", "enabled": True}
    client.disable_object_click()
    assert client._object_click is None


def test_on_object_click_enables_and_dispatches():
    """Registering a callback enables the event and receives clicks, including
    the null-id empty-space click and a payload without modifiers."""
    client = ViewerClient()
    got = []
    client.on_object_click(got.append)
    assert client._object_click is not None and client._object_click["enabled"]
    client._dispatch_object_click(
        {
            "type": "object_clicked",
            "id": "box",
            "point": [1.0, 2.0, 3.0],
            "button": 2,
            "modifiers": {"shift": True, "ctrl": False, "alt": False, "meta": False},
        }
    )
    client._dispatch_object_click({"type": "object_clicked", "id": None, "point": None})
    assert got == [
        {
            "id": "box",
            "point": [1.0, 2.0, 3.0],
            "button": 2,
            "modifiers": {"shift": True, "ctrl": False, "alt": False, "meta": False},
        },
        {
            "id": None,
            "point": None,
            "button": 0,
            "modifiers": {"shift": False, "ctrl": False, "alt": False, "meta": False},
        },
    ]


def test_on_object_click_rejects_non_callable():
    client = ViewerClient()
    with pytest.raises(TypeError):
        client.on_object_click(42)
    assert client._object_click is None


def test_object_click_callback_error_does_not_break_dispatch():
    """A raising callback is logged and the remaining callbacks still run."""
    client = ViewerClient()
    got = []

    def bad(_click):
        raise RuntimeError("boom")

    client.on_object_click(bad)
    client.on_object_click(got.append)
    client._dispatch_object_click({"id": "a", "point": [0, 0, 0], "button": 0})
    assert [c["id"] for c in got] == ["a"]
