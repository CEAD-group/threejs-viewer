"""Tests for per-channel interpolation (linear vs hold)."""

import time

import numpy as np
import pytest

from threejs_viewer import Animation, Frame


# --- Python API unit tests ---


def test_add_channel_default_interpolation_is_linear():
    """Every channel defaults to linear regardless of name."""
    anim = Animation()
    anim.set_frame_times(np.linspace(0, 1, 2))
    anim.add_channel(
        "draw_ranges",
        ["m1"],
        np.array([[0.0], [1.0]], dtype=np.float32),
        dtype="float32",
        stride=1,
    )
    assert anim._channels[0].interpolation == "linear"


def test_add_channel_hold_override():
    """add_channel accepts a per-channel 'hold' override."""
    anim = Animation()
    anim.set_frame_times(np.linspace(0, 1, 2))
    anim.add_channel(
        "draw_ranges",
        ["m1"],
        np.array([[0.0], [1.0]], dtype=np.float32),
        dtype="float32",
        stride=1,
        interpolation="hold",
    )
    assert anim._channels[0].interpolation == "hold"


def test_add_channel_invalid_interpolation_raises():
    """Unknown interpolation modes are rejected."""
    anim = Animation()
    with pytest.raises(ValueError, match="interpolation"):
        anim.add_channel(
            "draw_ranges",
            ["m1"],
            np.array([[0.0], [1.0]], dtype=np.float32),
            interpolation="cubic",
        )


def test_convenience_setters_default_to_linear():
    """All set_*_data convenience wrappers default to linear."""
    anim = Animation()
    mats = np.tile(np.eye(4, dtype=np.float32).flatten(), (2, 1, 1)).reshape(2, 1, 16)
    anim.set_transform_data(["b1"], mats)
    anim.set_draw_range_data(["b1"], np.array([[0.0], [1.0]], dtype=np.float32))
    anim.set_clip_time_data(["b1"], np.array([[0.0], [1.0]], dtype=np.float32))
    anim.set_camera_position(np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32))
    anim.set_camera_target(np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32))
    for ch in anim._channels:
        assert ch.interpolation == "linear", f"{ch.name} should default to linear"


def test_set_clip_time_data_hold_override():
    """set_clip_time_data accepts an explicit 'hold' for frame-accurate seeks."""
    anim = Animation()
    anim.set_clip_time_data(
        ["g1"],
        np.array([[0.0], [1.0]], dtype=np.float32),
        interpolation="hold",
    )
    assert anim._channels[0].interpolation == "hold"


# --- Browser integration: linear draw_range interpolates at midpoint ---


def _pause_and_seek_to_midpoint(page):
    """Pause playback and seek to exactly half the animation duration, then
    apply that frame. Returns the t value used (should be ~0.5 for uniform dt).
    """
    return page.evaluate(
        """() => {
            const v = window.threejsViewer;
            if (!v || !v._animation) return null;
            v._animationPlaying = false;
            v._animationTime = 0.5 * v._animation.duration;
            const { index, t } = v._getFrameAtTime(v._animationTime);
            v._applyFrame(index, t);
            return { index, t };
        }"""
    )


def _make_grid_mesh():
    """Return (positions, indices) for a 100-triangle flat grid.

    Used so that draw_range midpoint (0.5) snaps cleanly to a valid integer
    index count (50 out of 100 triangles → 150/300 indices = 0.5 exactly).
    """
    n_tris = 100
    positions = np.zeros((n_tris * 3, 3), dtype=np.float32)
    for i in range(n_tris):
        positions[i * 3 + 0] = [i, 0, 0]
        positions[i * 3 + 1] = [i + 1, 0, 0]
        positions[i * 3 + 2] = [i, 1, 0]
    indices = np.arange(n_tris * 3, dtype=np.uint32)
    return positions, indices


@pytest.mark.browser
def test_linear_interp_draw_range_midpoint(viewer_client, viewer_page):
    """draw_range lerps at the midpoint with its default (linear) interpolation."""
    positions, indices = _make_grid_mesh()
    viewer_client.add_mesh("m1", positions, indices)
    time.sleep(0.3)

    anim = Animation(
        frames=[
            Frame(time=0.0, transforms={}, draw_ranges={"m1": 0.0}),
            Frame(time=1.0, transforms={}, draw_ranges={"m1": 1.0}),
        ],
        loop=False,
    )
    viewer_client.load_animation(anim)

    for _ in range(20):
        time.sleep(0.1)
        if viewer_client.query_scene()["meta"]["animation"]["playing"]:
            break
    assert viewer_client.query_scene()["meta"]["animation"]["playing"], (
        "animation did not start playing within timeout"
    )

    seek = _pause_and_seek_to_midpoint(viewer_page)
    assert seek is not None
    time.sleep(0.05)

    result = viewer_client.query_scene()
    dr = result["objects"]["m1"]["drawRange"]
    # 100 triangles × 0.5 → 50 tris → 150 indices → 150/300 = 0.5 exactly
    assert abs(dr - 0.5) < 0.02, f"expected ~0.5, got {dr}"


@pytest.mark.browser
def test_hold_interp_draw_range_midpoint(viewer_client, viewer_page):
    """draw_range holds the floor keyframe when the channel is hold (explicit)."""
    positions, indices = _make_grid_mesh()
    viewer_client.add_mesh("m2", positions, indices)
    time.sleep(0.3)

    anim = Animation(loop=False)
    anim.set_frame_times(np.array([0.0, 1.0]))
    anim.add_channel(
        "draw_ranges",
        ["m2"],
        np.array([[0.0], [1.0]], dtype=np.float32),
        dtype="float32",
        stride=1,
        interpolation="hold",
    )
    viewer_client.load_animation(anim)

    for _ in range(20):
        time.sleep(0.1)
        if viewer_client.query_scene()["meta"]["animation"]["playing"]:
            break
    assert viewer_client.query_scene()["meta"]["animation"]["playing"], (
        "animation did not start playing within timeout"
    )

    _pause_and_seek_to_midpoint(viewer_page)
    time.sleep(0.05)

    result = viewer_client.query_scene()
    dr = result["objects"]["m2"]["drawRange"]
    # hold at floor → draw_range equals frame[0] value = 0.0
    assert dr < 0.05, f"expected ~0.0 under hold mode, got {dr}"


@pytest.mark.browser
def test_linear_interp_transforms_midpoint(viewer_client, viewer_page):
    """Transforms are slerped/lerped at midpoint by default (linear)."""
    viewer_client.add_box("b1")
    time.sleep(0.1)

    identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    translated = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 0, 0, 1]
    anim = Animation(
        frames=[
            Frame(time=0.0, transforms={"b1": identity}),
            Frame(time=1.0, transforms={"b1": translated}),
        ],
        loop=False,
    )
    viewer_client.load_animation(anim)

    for _ in range(20):
        time.sleep(0.1)
        if viewer_client.query_scene()["meta"]["animation"]["playing"]:
            break
    assert viewer_client.query_scene()["meta"]["animation"]["playing"], (
        "animation did not start playing within timeout"
    )

    _pause_and_seek_to_midpoint(viewer_page)
    time.sleep(0.05)

    x = viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            const obj = v._objects.get('b1');
            obj.updateMatrixWorld(true);
            return obj.matrixWorld.elements[12];
        }"""
    )
    assert abs(x - 5.0) < 0.1, f"expected ~5.0 at midpoint, got {x}"
