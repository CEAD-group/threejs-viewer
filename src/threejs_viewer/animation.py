"""
Animation class for batch visualization.

Allows pre-computing entire simulations and sending them to the viewer
for interactive playback with timeline scrubbing, speed control, and frame stepping.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Marker:
    """A labeled point on the animation timeline."""

    time: float
    label: str
    color: int = 0xFF0000


@dataclass
class Frame:
    """A single animation frame."""

    time: float
    transforms: dict[str, list[float]]  # object_id -> column-major 4x4 matrix
    colors: dict[str, int] | None = None  # object_id -> hex color (optional)
    visibility: dict[str, bool] | None = None  # object_id -> visible (optional)
    opacity: dict[str, float] | None = None  # object_id -> opacity 0-1 (optional)


@dataclass
class Animation:
    """
    Animation data for batch visualization.

    Instead of sending transforms frame-by-frame in real-time, an Animation
    contains all frames pre-computed. The viewer can then play back with
    full control: play/pause, speed, scrubbing, frame stepping.

    Example:
        frames = []
        for t in np.linspace(0, 10, 300):
            joints = compute_joints(t)
            frames.append(Frame(
                time=t,
                transforms=model.get_transforms(joints),
                colors=compute_colors(t),
            ))

        animation = Animation(frames=frames, loop=True)
        animation.add_marker(3.5, "Collision detected")
        viewer.load_animation(animation)
    """

    frames: list[Frame] = field(default_factory=list)
    loop: bool = True
    markers: list[Marker] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Animation duration in seconds."""
        if not self.frames:
            return 0.0
        return self.frames[-1].time

    @property
    def fps(self) -> float:
        """Approximate frames per second."""
        if len(self.frames) < 2:
            return 0.0
        return len(self.frames) / self.duration

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return len(self.frames)

    def add_frame(
        self,
        time: float,
        transforms: dict[str, list[float]],
        colors: dict[str, int] | None = None,
        visibility: dict[str, bool] | None = None,
        opacity: dict[str, float] | None = None,
    ) -> None:
        """Add a frame to the animation."""
        self.frames.append(
            Frame(
                time=time,
                transforms=transforms,
                colors=colors,
                visibility=visibility,
                opacity=opacity,
            )
        )

    def add_marker(self, time: float, label: str, color: int = 0xFF0000) -> None:
        """Add a labeled marker on the timeline."""
        self.markers.append(Marker(time=time, label=label, color=color))

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "duration": self.duration,
            "fps": self.fps,
            "loop": self.loop,
            "frames": [
                {
                    "time": f.time,
                    "transforms": f.transforms,
                    **({"colors": f.colors} if f.colors else {}),
                    **({"visibility": f.visibility} if f.visibility else {}),
                    **({"opacity": f.opacity} if f.opacity else {}),
                }
                for f in self.frames
            ],
            "markers": [
                {"time": m.time, "label": m.label, "color": m.color}
                for m in self.markers
            ],
        }

    @classmethod
    def from_function(
        cls,
        fn: Callable[[float], dict],
        duration: float,
        fps: float = 30.0,
        loop: bool = True,
    ) -> "Animation":
        """
        Create animation by sampling a function.

        Args:
            fn: Function that takes time (seconds) and returns
                {"transforms": {...}, "colors": {...}}
            duration: Total animation duration in seconds
            fps: Frames per second
            loop: Whether animation should loop

        Example:
            def simulate(t):
                joints = compute_joints(t)
                return {
                    "transforms": model.get_transforms(joints),
                    "colors": compute_colors(t),
                }

            animation = Animation.from_function(simulate, duration=10.0, fps=30)
        """
        n_frames = int(duration * fps)
        frames = []

        for i in range(n_frames):
            t = i / fps
            result = fn(t)
            frames.append(
                Frame(
                    time=t,
                    transforms=result.get("transforms", {}),
                    colors=result.get("colors"),
                )
            )

        return cls(frames=frames, loop=loop)

    @classmethod
    def record(
        cls, duration: float, fps: float = 30.0, loop: bool = True
    ) -> "AnimationRecorder":
        """
        Create a recorder context manager.

        Example:
            with Animation.record(duration=10.0, fps=30) as rec:
                for t in rec.times:
                    rec.add_frame(
                        transforms=model.get_transforms(compute_joints(t)),
                        colors=compute_colors(t),
                    )
            viewer.load_animation(rec.animation)
        """
        return AnimationRecorder(duration=duration, fps=fps, loop=loop)


class AnimationRecorder:
    """Context manager for recording animations."""

    def __init__(self, duration: float, fps: float = 30.0, loop: bool = True):
        self.duration = duration
        self.fps = fps
        self.loop = loop
        self._frames: list[Frame] = []
        self._current_time = 0.0
        self._time_step = 1.0 / fps

    @property
    def times(self) -> np.ndarray:
        """Array of frame times to iterate over."""
        n_frames = int(self.duration * self.fps)
        return np.linspace(0, self.duration, n_frames, endpoint=False)

    @property
    def animation(self) -> Animation:
        """Get the recorded animation."""
        return Animation(frames=self._frames, loop=self.loop)

    def add_frame(
        self,
        transforms: dict[str, list[float]],
        colors: dict[str, int] | None = None,
        visibility: dict[str, bool] | None = None,
        opacity: dict[str, float] | None = None,
    ) -> None:
        """Add a frame at the current time."""
        self._frames.append(
            Frame(
                time=self._current_time,
                transforms=transforms,
                colors=colors,
                visibility=visibility,
                opacity=opacity,
            )
        )
        self._current_time += self._time_step

    def __enter__(self) -> "AnimationRecorder":
        return self

    def __exit__(self, *args) -> None:
        pass
