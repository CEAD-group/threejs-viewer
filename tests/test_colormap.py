"""Tests for colormap functionality."""

import numpy as np

from threejs_viewer import ViewerClient


def test_colormap_viridis():
    """Test viridis colormap application."""
    client = ViewerClient()

    values = np.array([0.0, 0.5, 1.0])
    result = client._apply_colormap(values, "viridis", 0.0, 1.0)

    assert result.shape == (3, 3)
    assert result.dtype == np.float32
    # Values should be in [0, 1]
    assert np.all(result >= 0) and np.all(result <= 1)


def test_colormap_plasma():
    """Test plasma colormap application."""
    client = ViewerClient()

    values = np.linspace(0, 1, 10)
    result = client._apply_colormap(values, "plasma", 0.0, 1.0)

    assert result.shape == (10, 3)


def test_colormap_turbo():
    """Test turbo colormap application."""
    client = ViewerClient()

    values = np.linspace(0, 1, 10)
    result = client._apply_colormap(values, "turbo", 0.0, 1.0)

    assert result.shape == (10, 3)


def test_colormap_normalization():
    """Test that values are normalized correctly."""
    client = ViewerClient()

    # Values outside [cmin, cmax] should be clamped
    values = np.array([-10.0, 5.0, 20.0])
    result = client._apply_colormap(values, "viridis", 0.0, 10.0)

    assert result.shape == (3, 3)


def test_colormap_same_min_max():
    """Test colormap when cmin == cmax."""
    client = ViewerClient()

    values = np.array([5.0, 5.0, 5.0])
    result = client._apply_colormap(values, "viridis", 5.0, 5.0)

    # Should not crash, all values should be same color
    assert result.shape == (3, 3)


def test_unknown_colormap_defaults_to_viridis():
    """Test that unknown colormap falls back to viridis."""
    client = ViewerClient()

    values = np.array([0.0, 0.5, 1.0])
    result = client._apply_colormap(values, "unknown_colormap", 0.0, 1.0)

    # Should use viridis instead
    assert result.shape == (3, 3)
