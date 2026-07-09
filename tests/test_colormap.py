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


def test_colormap_tables_are_exact_reference_data():
    """The lookup tables are the real 256-entry reference colormaps (matplotlib
    color data), not stop approximations: pin first/mid/last entries."""
    client = ViewerClient()
    # 0, 128/255 and 1 hit table nodes exactly (0.5 would land between nodes)
    ends = client._apply_colormap(np.array([0.0, 128 / 255, 1.0]), "viridis", 0.0, 1.0)
    np.testing.assert_allclose(ends[0], [0.267004, 0.004874, 0.329415], atol=1e-6)
    np.testing.assert_allclose(ends[2], [0.993248, 0.906157, 0.143936], atol=1e-6)
    # node 128 is the teal ~#21918c (the old approximation had no exact entry here)
    np.testing.assert_allclose(ends[1], [0.127568, 0.566949, 0.550556], atol=1e-6)
    plasma = client._apply_colormap(np.array([0.0, 1.0]), "plasma", 0.0, 1.0)
    np.testing.assert_allclose(plasma[0], [0.050383, 0.029803, 0.527975], atol=1e-6)
    np.testing.assert_allclose(plasma[1], [0.940015, 0.975158, 0.131326], atol=1e-6)
    # turbo (Apache-2.0 reference data): pin first/mid/last so an accidental
    # edit to the embedded table is caught.
    turbo = client._apply_colormap(np.array([0.0, 128 / 255, 1.0]), "turbo", 0.0, 1.0)
    np.testing.assert_allclose(turbo[0], [0.18995, 0.07176, 0.23217], atol=1e-6)
    np.testing.assert_allclose(turbo[1], [0.64362, 0.98999, 0.23356], atol=1e-6)
    np.testing.assert_allclose(turbo[2], [0.47960, 0.01583, 0.01055], atol=1e-6)
