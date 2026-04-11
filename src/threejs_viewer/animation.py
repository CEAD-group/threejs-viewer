"""
Animation class for batch visualization.

Allows pre-computing entire simulations and sending them to the viewer
for interactive playback with timeline scrubbing, speed control, and frame stepping.
"""

from dataclasses import dataclass, field

import numpy as np

ALLOWED_DTYPES = {"float32", "uint32", "uint8"}


@dataclass
class Marker:
    """A labeled point on the animation timeline."""

    time: float
    label: str
    color: int = 0xFF0000


@dataclass
class AnimationChannel:
    """A named binary data channel for animation."""

    name: str
    ids: list[str]
    data: np.ndarray  # (n_frames, n_ids, stride) or (n_frames, n_ids) for stride=1
    dtype: str  # "float32", "uint32", "uint8"
    stride: int  # elements per object per frame (16 for 4x4 matrices, 1 for scalars)
    metadata: dict | None  # e.g. {"colormap": [0x44AA44, 0xFF3333]}


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
    camera_follow: str | None = None
    camera_lookat: str | None = None
    interpolation: str = "step"

    def __post_init__(self):
        if self.camera_follow is not None and self.camera_lookat is not None:
            raise ValueError("camera_follow and camera_lookat are mutually exclusive")
        if self.interpolation not in ("step", "linear"):
            raise ValueError(
                f"interpolation must be 'step' or 'linear', got {self.interpolation!r}"
            )

    # Generic binary channels for fast transfer.
    _channels: list[AnimationChannel] = field(default_factory=list, repr=False)

    # Optional frame times (set via set_frame_times() for binary-only animations).
    _frame_times: np.ndarray | None = field(default=None, repr=False)

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

    def add_channel(
        self,
        name: str,
        object_ids: list[str],
        data: np.ndarray,
        dtype: str = "float32",
        stride: int = 1,
        metadata: dict | None = None,
    ) -> None:
        """Add a binary data channel.

        Replaces any existing channel with the same name.

        Args:
            name: Channel name (e.g. "transforms", "colors", "visibility").
            object_ids: Ordered list of object IDs matching axis 1 of data.
            data: Array of shape (n_frames, n_objects * stride). Will be cast to dtype.
            dtype: Element type — "float32", "uint32", or "uint8".
            stride: Elements per object per frame (16 for 4x4 matrices, 1 for scalars).
            metadata: Extra info sent in the header (e.g. colormap for indexed colors).
        """
        if dtype not in ALLOWED_DTYPES:
            raise ValueError(
                f"Unsupported dtype {dtype!r}. Allowed: {sorted(ALLOWED_DTYPES)}"
            )
        # Replace existing channel with same name (prevents duplicates)
        self._channels = [ch for ch in self._channels if ch.name != name]
        self._channels.append(
            AnimationChannel(
                name=name,
                ids=list(object_ids),
                data=data,
                dtype=dtype,
                stride=stride,
                metadata=metadata,
            )
        )

    def set_transform_data(self, object_ids: list[str], data: np.ndarray) -> None:
        """Convenience: add a 'transforms' channel with stride=16."""
        self.add_channel("transforms", object_ids, data, dtype="float32", stride=16)

    def set_frame_times(self, times) -> None:
        """Set frame times from array, bypassing Frame object creation."""
        self._frame_times = np.asarray(times, dtype=np.float64)

    def set_draw_range_data(self, object_ids: list[str], data: np.ndarray) -> None:
        """Convenience: add a 'draw_ranges' channel."""
        self.add_channel("draw_ranges", object_ids, data, dtype="float32", stride=1)

    def set_clip_time_data(self, object_ids: list[str], data: np.ndarray) -> None:
        """Convenience: add a 'clip_times' channel."""
        self.add_channel("clip_times", object_ids, data, dtype="float32", stride=1)

    def set_camera_target(self, data: np.ndarray) -> None:
        """Convenience: add a 'camera_target' channel with stride=3."""
        self.add_channel(
            "camera_target", ["__camera__"], data, dtype="float32", stride=3
        )

    def set_camera_position(self, data: np.ndarray) -> None:
        """Convenience: add a 'camera_position' channel with stride=3."""
        self.add_channel(
            "camera_position", ["__camera__"], data, dtype="float32", stride=3
        )

    def add_marker(self, time: float, label: str, color: int = 0xFF0000) -> None:
        """Add a labeled marker on the timeline."""
        self.markers.append(Marker(time=time, label=label, color=color))

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "duration": self.duration,
            "fps": self.fps,
            "loop": self.loop,
            "interpolation": self.interpolation,
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
        if self.camera_follow is not None:
            result["camera_follow"] = self.camera_follow
        if self.camera_lookat is not None:
            result["camera_lookat"] = self.camera_lookat
        return result


def merge_animation_points(
    toolpath: np.ndarray,
    frame_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge animation frame times into a toolpath for segment-aligned draw_ranges.

    Inserts new interpolated points into the toolpath at each frame_time not
    already present. The returned combined array guarantees that each animation
    frame maps exactly to a toolpath point, so draw_range always shows only
    complete mesh segments (no partial triangle rings).

    Args:
        toolpath: (M, C) array with timestamps in column 0, monotonically
                  non-decreasing. Typically the output of process_toolpath:
                  columns [t, x, y, z, width, height].
        frame_times: (F,) array of desired animation frame times.

    Returns:
        combined: (K, C) float32 array — original toolpath with new points
                  interpolated and inserted at each new frame_time. Sorted by time.
        frame_indices: (F,) int64 array — index in combined for each frame_time.

    Example:
        combined, frame_indices = merge_animation_points(toolpath, frame_times)
        draw_fracs = frame_indices / (len(combined) - 1)
    """
    toolpath = np.asarray(toolpath, dtype=np.float64)
    frame_times = np.asarray(frame_times, dtype=np.float64)
    t = toolpath[:, 0]

    if len(t) < 2:
        raise ValueError("toolpath must have at least 2 points")

    # Clip frame_times to the toolpath domain up-front so that all downstream
    # operations (skip detection, interpolation, frame_indices) are consistent.
    # Out-of-range frames map to the first/last point rather than extrapolating.
    frame_times = np.clip(frame_times, t[0], t[-1])

    # Find which frame_times are not already present in t (within tolerance)
    ins_pos = np.searchsorted(t, frame_times)
    right_pos = np.minimum(ins_pos, len(t) - 1)
    left_pos = np.maximum(ins_pos - 1, 0)

    at_right = (ins_pos < len(t)) & (np.abs(t[right_pos] - frame_times) < 1e-9)
    at_left = (ins_pos > 0) & (np.abs(t[left_pos] - frame_times) < 1e-9)
    skip = at_right | at_left

    new_times = frame_times[~skip]
    if len(new_times) > 0:
        # Interpolate all columns for new_times.
        # Use right-side anchoring (searchsorted side='right' - 1) so that a
        # new point just after a duplicate timestamp (e.g. an extrusion→travel
        # transition where two co-located points share the same t) inherits the
        # LAST value at that timestamp rather than the first.  With plain
        # np.interp the leftmost duplicate is used, which can spuriously assign
        # full bead width to a new frame point that should be in the travel gap.
        lo = np.searchsorted(t, new_times, side="right") - 1
        lo = np.clip(lo, 0, len(t) - 2)
        hi = lo + 1
        dt = t[hi] - t[lo]
        alpha = np.where(dt > 0, (new_times - t[lo]) / dt, 0.0)[:, None]
        new_rows = toolpath[lo] * (1.0 - alpha) + toolpath[hi] * alpha
        new_rows[:, 0] = new_times  # keep exact frame time in column 0
        combined = np.concatenate([toolpath, new_rows], axis=0)
        order = np.argsort(combined[:, 0], kind="stable")
        combined = combined[order]
    else:
        combined = toolpath

    # side='right' - 1 returns the index of the LAST duplicate at each
    # frame_time, so a frame that lands exactly on a transition timestamp
    # points to the zero-width cap row rather than the preceding full-width row.
    frame_indices = np.searchsorted(combined[:, 0], frame_times, side="right") - 1
    frame_indices = np.clip(frame_indices, 0, len(combined) - 1)
    return combined.astype(np.float32), frame_indices.astype(np.int64)


def toolpath_frame_times(
    point_times: np.ndarray,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (frame_times, draw_fracs) for equi-time-distant sampling of a toolpath.

    Produces animation frames that are evenly spaced in *time* rather than in
    path points. This ensures smooth playback even when point density varies
    (e.g. dense arcs vs. sparse straight segments).

    During fast travel moves (few seconds, many path points) the draw_range
    fraction advances quickly, revealing no geometry. During slow extrusion
    (many seconds, few points) it advances slowly, growing the bead gradually.

    Args:
        point_times: (N,) per-point timestamps, monotonically non-decreasing.
        n_frames: Desired number of animation frames.

    Returns:
        frame_times: (n_frames,) array, evenly spaced from t0 to t_end.
        draw_fracs: (n_frames,) array of draw_range fractions in [0, 1].
    """
    point_times = np.asarray(point_times, dtype=np.float64)
    N = len(point_times)
    if N == 0:
        raise ValueError("point_times must not be empty")
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    t0, t_end = float(point_times[0]), float(point_times[-1])
    frame_times = np.linspace(t0, t_end, n_frames)
    path_fracs = np.arange(N) / max(N - 1, 1)
    # np.interp requires strictly increasing xp.  point_times may contain
    # duplicate timestamps (zero-length segments), so deduplicate by keeping
    # the last (highest) path_frac at each timestamp before interpolating.
    _, last_idx = np.unique(point_times[::-1], return_index=True)
    last_idx = np.sort(N - 1 - last_idx)
    draw_fracs = np.interp(frame_times, point_times[last_idx], path_fracs[last_idx])
    return frame_times, draw_fracs
