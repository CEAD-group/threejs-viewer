"""Tests for Animation classes."""

import numpy as np

from threejs_viewer import Animation, AnimationRecorder, Frame, Marker


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


def test_animation_from_function():
    """Test creating animation from a function."""

    def simulate(t):
        return {
            "transforms": {"obj": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, t, 0, 0, 1]},
            "colors": {"obj": 0xFF0000 if t < 0.5 else 0x00FF00},
        }

    animation = Animation.from_function(simulate, duration=1.0, fps=10, loop=True)

    assert animation.n_frames == 10
    assert animation.loop is True


def test_animation_recorder():
    """Test AnimationRecorder context manager."""
    with Animation.record(duration=1.0, fps=10) as rec:
        for t in rec.times:
            rec.add_frame(
                transforms={"obj": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, t, 0, 0, 1]}
            )

    animation = rec.animation
    assert animation.n_frames == 10


def test_recorder_times():
    """Test that recorder generates correct time array."""
    rec = AnimationRecorder(duration=2.0, fps=10)
    times = rec.times

    assert len(times) == 20
    assert times[0] == 0.0
    assert np.isclose(times[-1], 1.9)
