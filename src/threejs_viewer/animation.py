"""
Animation class for batch visualization.

Allows pre-computing entire simulations and sending them to the viewer
for interactive playback with timeline scrubbing, speed control, and frame stepping.
"""

from dataclasses import dataclass, field
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
    clip_times: dict[str, float] | None = (
        None  # object_id -> embedded animation time (optional)
    )
    draw_ranges: dict[str, float] | None = (
        None  # object_id -> 0.0-1.0 fraction visible (optional)
    )


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

    # Optional pre-built binary data for fast transfer.
    # Set via set_transform_data() / set_draw_range_data() / set_frame_times().
    _transform_data: np.ndarray | None = field(default=None, repr=False)
    _object_ids: list[str] | None = field(default=None, repr=False)
    _frame_times: np.ndarray | None = field(default=None, repr=False)
    _draw_range_data: np.ndarray | None = field(default=None, repr=False)
    _draw_range_ids: list[str] | None = field(default=None, repr=False)

    @property
    def duration(self) -> float:
        """Animation duration in seconds."""
        if self._frame_times is not None and len(self._frame_times) > 0:
            return float(self._frame_times[-1])
        if not self.frames:
            return 0.0
        return self.frames[-1].time

    @property
    def fps(self) -> float:
        """Approximate frames per second."""
        n = self.n_frames
        d = self.duration
        if n < 2 or d <= 0:
            return 0.0
        return n / d

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        if self._frame_times is not None:
            return len(self._frame_times)
        return len(self.frames)

    def add_frame(
        self,
        time: float,
        transforms: dict[str, list[float]],
        colors: dict[str, int] | None = None,
        visibility: dict[str, bool] | None = None,
        opacity: dict[str, float] | None = None,
        clip_times: dict[str, float] | None = None,
        draw_ranges: dict[str, float] | None = None,
    ) -> None:
        """Add a frame to the animation."""
        self.frames.append(
            Frame(
                time=time,
                transforms=transforms,
                colors=colors,
                visibility=visibility,
                opacity=opacity,
                clip_times=clip_times,
                draw_ranges=draw_ranges,
            )
        )

    def set_transform_data(self, object_ids: list[str], data: np.ndarray) -> None:
        """
        Set pre-built transform data for fast binary transfer.

        This bypasses the per-frame dict-to-numpy conversion in load_animation,
        which is the main bottleneck for large animations.

        Args:
            object_ids: Ordered list of object IDs matching axis 1 of data
            data: numpy array of shape (n_frames, n_objects, 16), dtype float32
                  Column-major 4x4 matrices.
        """
        self._object_ids = object_ids
        self._transform_data = np.ascontiguousarray(data, dtype=np.float32)

    def set_frame_times(self, times) -> None:
        """Set frame times from array, bypassing Frame object creation."""
        self._frame_times = np.asarray(times, dtype=np.float64)

    def set_draw_range_data(self, object_ids: list[str], data: np.ndarray) -> None:
        """
        Set pre-built draw_range data for fast binary transfer.

        Args:
            object_ids: Ordered list of object IDs matching axis 1 of data
            data: numpy array of shape (n_frames, n_objects), dtype float32
                  Values in 0.0-1.0 range.
        """
        self._draw_range_ids = list(object_ids)
        self._draw_range_data = np.ascontiguousarray(data, dtype=np.float32)

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
                    **({"clip_times": f.clip_times} if f.clip_times else {}),
                    **({"draw_ranges": f.draw_ranges} if f.draw_ranges else {}),
                }
                for f in self.frames
            ],
            "markers": [
                {"time": m.time, "label": m.label, "color": m.color}
                for m in self.markers
            ],
        }
