"""Tests for Animation classes."""

import numpy as np

from threejs_viewer import Animation, AnimationChannel, Frame, Marker


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

    # Frames with clip_times (no binary channel for this)
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
