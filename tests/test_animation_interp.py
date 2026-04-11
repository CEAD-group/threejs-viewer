"""Tests for the animation interpolation field and client-side linear interp."""

import time

import numpy as np
import pytest

from threejs_viewer import Animation, Frame


# --- Python API unit tests ---


def test_animation_interpolation_default_step():
    """Animation defaults to step interpolation for back-compat."""
    anim = Animation()
    assert anim.interpolation == "step"


def test_animation_interpolation_linear_opt_in():
    """interpolation='linear' is accepted via constructor."""
    anim = Animation(interpolation="linear")
    assert anim.interpolation == "linear"


def test_animation_interpolation_invalid_raises():
    """Constructor rejects unknown interpolation modes."""
    with pytest.raises(ValueError, match="interpolation"):
        Animation(interpolation="cubic")


def test_animation_to_dict_includes_interpolation():
    """to_dict serializes interpolation for the non-HTTP reconnect path."""
    step_dict = Animation().to_dict()
    assert step_dict["interpolation"] == "step"

    linear_dict = Animation(interpolation="linear").to_dict()
    assert linear_dict["interpolation"] == "linear"


def test_channel_metadata_interpolation_override():
    """add_channel accepts a per-channel interpolation override via metadata."""
    anim = Animation(interpolation="linear")
    anim.set_frame_times(np.linspace(0, 1, 2))
    # Global linear; this channel explicitly steps.
    anim.add_channel(
        "draw_ranges",
        ["m1"],
        np.array([[0.0], [1.0]], dtype=np.float32),
        dtype="float32",
        stride=1,
        metadata={"interpolation": "step"},
    )
    ch = anim._channels[0]
    assert ch.metadata == {"interpolation": "step"}


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
    # 10x10 vertex grid → 9x9 quads → 162 triangles → 486 indices (too odd)
    # Use a simpler triangle-fan: 100 independent triangles = 300 indices.
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
    """draw_range interpolates linearly at the midpoint when mode is linear."""
    positions, indices = _make_grid_mesh()
    viewer_client.add_mesh("m1", positions, indices)
    time.sleep(0.3)

    anim = Animation(
        frames=[
            Frame(time=0.0, transforms={}, draw_ranges={"m1": 0.0}),
            Frame(time=1.0, transforms={}, draw_ranges={"m1": 1.0}),
        ],
        loop=False,
        interpolation="linear",
    )
    viewer_client.load_animation(anim)

    for _ in range(20):
        time.sleep(0.1)
        if viewer_client.query_scene()["meta"]["animation"]["playing"]:
            break

    seek = _pause_and_seek_to_midpoint(viewer_page)
    assert seek is not None
    time.sleep(0.05)

    result = viewer_client.query_scene()
    dr = result["objects"]["m1"]["drawRange"]
    # 100 triangles × 0.5 → 50 tris → 150 indices → 150/300 = 0.5 exactly
    assert abs(dr - 0.5) < 0.02, f"expected ~0.5, got {dr}"


@pytest.mark.browser
def test_step_interp_draw_range_midpoint(viewer_client, viewer_page):
    """draw_range stays at the floor keyframe when mode is step (default)."""
    positions, indices = _make_grid_mesh()
    viewer_client.add_mesh("m2", positions, indices)
    time.sleep(0.3)

    anim = Animation(
        frames=[
            Frame(time=0.0, transforms={}, draw_ranges={"m2": 0.0}),
            Frame(time=1.0, transforms={}, draw_ranges={"m2": 1.0}),
        ],
        loop=False,
        # interpolation defaults to "step"
    )
    viewer_client.load_animation(anim)

    for _ in range(20):
        time.sleep(0.1)
        if viewer_client.query_scene()["meta"]["animation"]["playing"]:
            break

    _pause_and_seek_to_midpoint(viewer_page)
    time.sleep(0.05)

    result = viewer_client.query_scene()
    dr = result["objects"]["m2"]["drawRange"]
    # Step at floor → draw_range equals frame[0] value = 0.0
    assert dr < 0.05, f"expected ~0.0 under step mode, got {dr}"


@pytest.mark.browser
def test_linear_interp_transforms_midpoint(viewer_client, viewer_page):
    """Transforms are slerped/lerped at midpoint when mode is linear."""
    viewer_client.add_box("b1")
    time.sleep(0.1)

    # Identity at t=0, translated by (10, 0, 0) at t=1.
    identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    translated = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 0, 0, 1]
    anim = Animation(
        frames=[
            Frame(time=0.0, transforms={"b1": identity}),
            Frame(time=1.0, transforms={"b1": translated}),
        ],
        loop=False,
        interpolation="linear",
    )
    viewer_client.load_animation(anim)

    for _ in range(20):
        time.sleep(0.1)
        if viewer_client.query_scene()["meta"]["animation"]["playing"]:
            break

    _pause_and_seek_to_midpoint(viewer_page)
    time.sleep(0.05)

    # Read the box's x position from the browser directly.
    x = viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            const obj = v._objects.get('b1');
            obj.updateMatrixWorld(true);
            return obj.matrixWorld.elements[12];
        }"""
    )
    assert abs(x - 5.0) < 0.1, f"expected ~5.0 at midpoint, got {x}"
