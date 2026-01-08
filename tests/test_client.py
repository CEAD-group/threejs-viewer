"""Tests for ViewerClient."""

from pathlib import Path

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
