"""Tests for Animation classes."""

import numpy as np
import pytest

from threejs_viewer import Animation, AnimationChannel, Frame, Marker
from threejs_viewer.animation import merge_animation_points, toolpath_frame_times


def test_frame_creation():
    """Test Frame dataclass creation."""
    frame = Frame(
        time=1.0,
        transforms={"obj1": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]},
        colors={"obj1": 0xFF0000},
        visibility={"obj1": True},
        opacity={"obj1": 0.5},
    )
    assert frame.time == 1.0
    assert "obj1" in frame.transforms
    assert frame.colors["obj1"] == 0xFF0000
    assert frame.visibility["obj1"] is True
    assert frame.opacity["obj1"] == 0.5


def test_marker_creation():
    """Test Marker dataclass creation."""
    marker = Marker(time=2.5, label="Test marker", color=0x00FF00)
    assert marker.time == 2.5
    assert marker.label == "Test marker"
    assert marker.color == 0x00FF00


def test_animation_creation():
    """Test Animation creation and properties."""
    animation = Animation(loop=True)
    assert animation.loop is True
    assert animation.n_frames == 0
    assert animation.duration == 0.0


def test_animation_add_frame():
    """Test adding frames to animation."""
    animation = Animation()

    for i in range(10):
        animation.add_frame(
            time=i * 0.1,
            transforms={"obj": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, i, 0, 0, 1]},
        )

    assert animation.n_frames == 10
    assert animation.duration == 0.9
    assert animation.fps > 0


def test_animation_add_marker():
    """Test adding markers to animation."""
    animation = Animation()
    animation.add_marker(1.0, "Start")
    animation.add_marker(2.0, "Middle", color=0xFFFF00)

    assert len(animation.markers) == 2
    assert animation.markers[0].label == "Start"
    assert animation.markers[1].color == 0xFFFF00


def test_animation_to_dict():
    """Test animation serialization."""
    animation = Animation(loop=False)
    animation.add_frame(time=0.0, transforms={"a": list(range(16))})
    animation.add_frame(time=1.0, transforms={"a": list(range(16))})
    animation.add_marker(0.5, "Halfway")

    data = animation.to_dict()

    assert data["loop"] is False
    assert data["duration"] == 1.0
    assert len(data["frames"]) == 2
    assert len(data["markers"]) == 1
    assert data["markers"][0]["label"] == "Halfway"


# --- Binary channel tests ---


def test_set_transform_data():
    """Test set_transform_data convenience wrapper creates a transforms channel."""
    animation = Animation(loop=True)
    animation.set_frame_times(np.linspace(0, 10, 100))

    transforms = np.zeros((100, 2, 16), dtype=np.float32)
    transforms[:, :, [0, 5, 10, 15]] = 1.0
    animation.set_transform_data(["obj_a", "obj_b"], transforms)

    assert animation.n_frames == 100
    assert animation.duration == 10.0
    assert len(animation._channels) == 1
    assert animation._channels[0].name == "transforms"
    assert animation._channels[0].ids == ["obj_a", "obj_b"]
    assert animation._channels[0].dtype == "float32"
    assert animation._channels[0].stride == 16


def test_set_draw_range_data():
    """Test set_draw_range_data convenience wrapper creates a draw_ranges channel."""
    animation = Animation(loop=True)
    animation.set_frame_times(np.linspace(0, 5, 50))

    draw_ranges = np.linspace(0, 1, 50).reshape(50, 1).astype(np.float32)
    draw_ranges = np.column_stack([draw_ranges, draw_ranges])
    animation.set_draw_range_data(["obj_a", "obj_b"], draw_ranges)

    assert len(animation._channels) == 1
    assert animation._channels[0].name == "draw_ranges"
    assert animation._channels[0].dtype == "float32"
    assert animation._channels[0].stride == 1


def test_set_clip_time_data():
    """Test set_clip_time_data convenience wrapper creates a clip_times channel."""
    animation = Animation(loop=True)
    animation.set_frame_times(np.linspace(0, 5, 50))

    clip_times = np.linspace(0, 2, 50).reshape(50, 1).astype(np.float32)
    clip_times = np.column_stack([clip_times, clip_times * 0.5])
    animation.set_clip_time_data(["model_a", "model_b"], clip_times)

    assert len(animation._channels) == 1
    assert animation._channels[0].name == "clip_times"
    assert animation._channels[0].dtype == "float32"
    assert animation._channels[0].stride == 1


def test_add_channel_generic():
    """Test add_channel with custom channel types."""
    animation = Animation(loop=True)
    animation.set_frame_times(np.arange(10) / 30.0)

    # Colors with colormap
    color_data = np.zeros((10, 3), dtype=np.uint8)
    animation.add_channel(
        "colors",
        ["s0", "s1", "s2"],
        color_data,
        dtype="uint8",
        metadata={"colormap": [0x44AA44, 0xFF3333]},
    )

    # Visibility
    vis_data = np.ones((10, 3), dtype=np.uint8)
    animation.add_channel("visibility", ["s0", "s1", "s2"], vis_data, dtype="uint8")

    # Opacity
    opacity_data = np.ones((10, 2), dtype=np.float32)
    animation.add_channel("opacity", ["s0", "s1"], opacity_data, dtype="float32")

    assert len(animation._channels) == 3
    assert animation._channels[0].name == "colors"
    assert animation._channels[0].metadata == {"colormap": [0x44AA44, 0xFF3333]}
    assert animation._channels[1].name == "visibility"
    assert animation._channels[2].name == "opacity"


def test_add_channel_duplicate_replaces():
    """Test that adding a channel with the same name replaces the existing one."""
    animation = Animation(loop=True)
    animation.set_frame_times(np.arange(5) / 30.0)

    animation.set_transform_data(["a", "b"], np.zeros((5, 2, 16), dtype=np.float32))
    assert len(animation._channels) == 1
    assert len(animation._channels[0].ids) == 2

    # Replace with a 3-object version
    animation.set_transform_data(["a", "b", "c"], np.ones((5, 3, 16), dtype=np.float32))
    assert len(animation._channels) == 1
    assert len(animation._channels[0].ids) == 3


def test_animation_channel_dataclass():
    """Test AnimationChannel dataclass."""
    ch = AnimationChannel(
        name="test",
        ids=["a", "b"],
        data=np.zeros((10, 2), dtype=np.float32),
        dtype="float32",
        stride=1,
        metadata=None,
    )
    assert ch.name == "test"
    assert ch.ids == ["a", "b"]
    assert ch.dtype == "float32"
    assert ch.stride == 1
    assert ch.metadata is None


def test_multiple_channels_coexist():
    """Test that transforms, draw_ranges, colors, and visibility can coexist."""
    animation = Animation(loop=True)
    n_frames = 20
    animation.set_frame_times(np.arange(n_frames) / 60.0)

    ids = ["obj1", "obj2"]
    animation.set_transform_data(ids, np.zeros((n_frames, 2, 16), dtype=np.float32))
    animation.set_draw_range_data(ids, np.ones((n_frames, 2), dtype=np.float32))
    animation.add_channel(
        "colors",
        ids,
        np.zeros((n_frames, 2), dtype=np.uint8),
        dtype="uint8",
        metadata={"colormap": [0x00FF00, 0xFF0000]},
    )
    animation.add_channel(
        "visibility", ids, np.ones((n_frames, 2), dtype=np.uint8), dtype="uint8"
    )

    assert len(animation._channels) == 4
    names = [ch.name for ch in animation._channels]
    assert "transforms" in names
    assert "draw_ranges" in names
    assert "colors" in names
    assert "visibility" in names


def test_binary_channels_with_frame_objects():
    """Test that binary channels and Frame objects can coexist (mixed mode)."""
    animation = Animation(loop=True)

    # Frames with clip_times as JSON
    for i in range(5):
        animation.add_frame(
            time=i / 30.0,
            transforms={"obj1": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]},
            clip_times={"model1": i * 0.1},
        )

    # Binary colors channel alongside Frame-based clip_times
    animation.add_channel(
        "colors",
        ["obj1"],
        np.zeros((5, 1), dtype=np.uint8),
        dtype="uint8",
        metadata={"colormap": [0x44AA44, 0xFF3333]},
    )

    assert animation.n_frames == 5
    assert len(animation._channels) == 1
    assert animation.frames[0].clip_times == {"model1": 0.0}


# --- merge_animation_points tests ---


def test_merge_animation_points_basic():
    """User's example: times [0,3,5,7,12,20], frames [0,10,20] → 7 pts, fracs [0, 4/6, 1]."""
    # toolpath with 6 points: (N, 2) — column 0 is time, column 1 is dummy value
    tp = np.array([[0, 0], [3, 1], [5, 2], [7, 3], [12, 4], [20, 5]], dtype=np.float32)
    frame_times = np.array([0.0, 10.0, 20.0])

    combined, frame_indices = merge_animation_points(tp, frame_times)

    # Combined should have 7 points: original 6 + inserted t=10
    assert len(combined) == 7
    np.testing.assert_allclose(combined[:, 0], [0, 3, 5, 7, 10, 12, 20], atol=1e-6)

    # frame_indices: 0→0, 10→4, 20→6
    assert frame_indices[0] == 0
    assert frame_indices[1] == 4
    assert frame_indices[2] == 6

    # draw_fracs: [0/6, 4/6, 6/6]
    draw_fracs = frame_indices / (len(combined) - 1)
    np.testing.assert_allclose(draw_fracs, [0.0, 4 / 6, 1.0], atol=1e-6)


def test_merge_animation_points_no_duplicates():
    """Frame time coinciding with existing point should not create a duplicate."""
    tp = np.array([[0, 0], [5, 1], [10, 2]], dtype=np.float32)
    frame_times = np.array([0.0, 5.0, 10.0])

    combined, frame_indices = merge_animation_points(tp, frame_times)

    # All frame_times already in toolpath — no new points
    assert len(combined) == 3
    assert frame_indices[0] == 0
    assert frame_indices[1] == 1
    assert frame_indices[2] == 2


def test_merge_animation_points_interpolation():
    """Inserted point should be linearly interpolated across all columns."""
    # 2-segment toolpath: t=[0,10], x=[0,10], y=[0,20]
    tp = np.array([[0, 0, 0], [10, 10, 20]], dtype=np.float32)
    frame_times = np.array([4.0])

    combined, frame_indices = merge_animation_points(tp, frame_times)

    assert len(combined) == 3
    # t=4 → frac=0.4 → x=4, y=8
    np.testing.assert_allclose(combined[1, 0], 4.0, atol=1e-5)
    np.testing.assert_allclose(combined[1, 1], 4.0, atol=1e-5)
    np.testing.assert_allclose(combined[1, 2], 8.0, atol=1e-5)
    assert frame_indices[0] == 1


def test_merge_animation_points_output_dtype():
    """Output combined array should be float32, frame_indices should be int64."""
    tp = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
    frame_times = np.array([0.0, 0.5, 1.0, 2.0])

    combined, frame_indices = merge_animation_points(tp, frame_times)

    assert combined.dtype == np.float32
    assert frame_indices.dtype == np.int64


# --- toolpath_frame_times tests ---


def test_toolpath_frame_times_uniform_timing():
    """Uniform point spacing → draw_fracs should be linear (matching frame_times)."""
    N = 100
    point_times = np.linspace(0.0, 10.0, N)
    frame_times, draw_fracs = toolpath_frame_times(point_times, n_frames=50)

    assert len(frame_times) == 50
    assert len(draw_fracs) == 50
    assert frame_times[0] == pytest.approx(0.0)
    assert frame_times[-1] == pytest.approx(10.0)
    assert draw_fracs[0] == pytest.approx(0.0)
    assert draw_fracs[-1] == pytest.approx(1.0)
    # With uniform spacing, draw_fracs should be linear
    expected = np.linspace(0.0, 1.0, 50)
    np.testing.assert_allclose(draw_fracs, expected, atol=1e-6)


def test_toolpath_frame_times_nonuniform_timing():
    """Non-uniform timing: dense slow region → draw_fracs should advance slowly there."""
    # First half of path takes 9/10 of the time (slow extrusion)
    # Second half takes 1/10 of the time (fast travel)
    N = 100
    path_fracs = np.arange(N) / (N - 1)  # 0..1
    # Extrusion (first half of path): t goes 0..9, Travel (second half): t goes 9..10
    extrusion_mask = path_fracs <= 0.5
    point_times = np.where(
        extrusion_mask, path_fracs * 18.0, 9.0 + (path_fracs - 0.5) * 2.0
    )

    frame_times, draw_fracs = toolpath_frame_times(point_times, n_frames=100)

    # At t=5 (halfway through time), draw_frac should be well below 0.5
    # because the slow extrusion region covers most of the time
    midpoint_frac = np.interp(5.0, frame_times, draw_fracs)
    assert midpoint_frac < 0.35, (
        f"At t=5, draw_frac={midpoint_frac:.3f} should be < 0.35 (slow extrusion dominates)"
    )

    # draw_fracs should be monotonically non-decreasing
    assert np.all(np.diff(draw_fracs) >= -1e-10)
    assert draw_fracs[0] == pytest.approx(0.0)
    assert draw_fracs[-1] == pytest.approx(1.0)


def test_toolpath_frame_times_two_points():
    """Minimum path length (2 points) should work without errors."""
    point_times = np.array([0.0, 5.0])
    frame_times, draw_fracs = toolpath_frame_times(point_times, n_frames=10)

    assert len(frame_times) == 10
    assert len(draw_fracs) == 10
    assert draw_fracs[0] == pytest.approx(0.0)
    assert draw_fracs[-1] == pytest.approx(1.0)


def test_toolpath_frame_times_output_shape():
    """Output arrays have length n_frames."""
    point_times = np.linspace(0.0, 3.0, 50)
    for n in [1, 5, 100]:
        ft, df = toolpath_frame_times(point_times, n_frames=n)
        assert len(ft) == n
        assert len(df) == n


def test_merge_animation_points_duplicate_timestamps():
    """Co-located transition points (same t, width bw then 0) must stay ordered.

    Regression: a frame_time just after a duplicate timestamp used to get
    width=bw via np.interp (leftmost duplicate), creating a spurious full-width
    ring in the travel gap.  With the right-anchored interpolation it gets
    width=0 (the LAST value at that timestamp).
    """
    bw = 2.0
    # Toolpath with an extrusion→travel transition at t=5:
    #   t=0  w=bw  (extruding)
    #   t=5  w=bw  (last extrusion point, co-located with cap)
    #   t=5  w=0   (zero-width cap / travel start, same location)
    #   t=10 w=0   (travel continues)
    tp = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, bw],
            [5.0, 1.0, 0.0, 0.0, bw],
            [5.0, 1.0, 0.0, 0.0, 0.0],
            [10.0, 2.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    # Frame just after the transition should get width=0, not width=bw
    frame_times = np.array([5.5])
    combined, frame_indices = merge_animation_points(tp, frame_times)

    # The new point at t=5.5 should have width ≈ 0 (travel), not bw
    new_pt = combined[frame_indices[0]]
    assert new_pt[4] == pytest.approx(0.0, abs=1e-5), (
        f"Width at t=5.5 should be 0 (travel), got {new_pt[4]:.4f}"
    )

    # The zero-width cap (t=5, w=0) must come AFTER the full-width ring (t=5, w=bw)
    t5_indices = np.where(np.abs(combined[:, 0] - 5.0) < 1e-5)[0]
    assert len(t5_indices) == 2
    assert combined[t5_indices[0], 4] == pytest.approx(bw)  # full-width first
    assert combined[t5_indices[1], 4] == pytest.approx(0.0)  # cap second


# --- Camera tracking tests ---


def test_camera_follow():
    """Test camera_follow metadata on Animation."""
    anim = Animation(loop=True, camera_follow="nozzle")
    assert anim.camera_follow == "nozzle"
    assert anim.camera_lookat is None
    d = anim.to_dict()
    assert d["camera_follow"] == "nozzle"
    assert "camera_lookat" not in d


def test_camera_lookat():
    """Test camera_lookat metadata on Animation."""
    anim = Animation(loop=True, camera_lookat="nozzle")
    assert anim.camera_lookat == "nozzle"
    assert anim.camera_follow is None
    d = anim.to_dict()
    assert d["camera_lookat"] == "nozzle"
    assert "camera_follow" not in d


def test_camera_follow_lookat_mutually_exclusive():
    """Setting both camera_follow and camera_lookat raises ValueError."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        Animation(camera_follow="a", camera_lookat="b")


def test_camera_no_tracking_to_dict():
    """to_dict omits camera keys when not set."""
    anim = Animation()
    d = anim.to_dict()
    assert "camera_follow" not in d
    assert "camera_lookat" not in d


def test_set_camera_target():
    """Test set_camera_target creates a camera_target channel."""
    anim = Animation()
    anim.set_frame_times(np.linspace(0, 1, 10))
    data = np.random.rand(10, 3).astype(np.float32)
    anim.set_camera_target(data)

    assert len(anim._channels) == 1
    ch = anim._channels[0]
    assert ch.name == "camera_target"
    assert ch.ids == ["__camera__"]
    assert ch.stride == 3
    assert ch.dtype == "float32"


def test_set_camera_position():
    """Test set_camera_position creates a camera_position channel."""
    anim = Animation()
    anim.set_frame_times(np.linspace(0, 1, 10))
    data = np.random.rand(10, 3).astype(np.float32)
    anim.set_camera_position(data)

    assert len(anim._channels) == 1
    ch = anim._channels[0]
    assert ch.name == "camera_position"
    assert ch.ids == ["__camera__"]
    assert ch.stride == 3
    assert ch.dtype == "float32"
