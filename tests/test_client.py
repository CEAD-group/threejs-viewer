"""Tests for ViewerClient."""

import math
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

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


def test_viewer_client_rejects_invalid_tone_mapping():
    with pytest.raises(ValueError, match="tone_mapping must be one of"):
        ViewerClient(tone_mapping="bogus")


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
    assert g["clickSelect"] is True
    assert g["rotateSnap"] == pytest.approx(math.radians(15))


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
            "phase": "end",
        }
    )
    assert len(got) == 1
    assert got[0]["id"] == "box"
    assert got[0]["position"] == [1, 2, 3]
    assert got[0]["phase"] == "end"


def test_on_object_move_rejects_non_callable():
    client = ViewerClient()
    with pytest.raises(TypeError):
        client.on_object_move(42)


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
