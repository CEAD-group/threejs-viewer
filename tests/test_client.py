"""Tests for ViewerClient."""

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from threejs_viewer import ViewerClient


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
