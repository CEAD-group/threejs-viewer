"""Toolpath processing: G-code → bead geometry with animation support.

Typical usage::

    raw = make_my_toolpath()  # (N,5) [x,y,z,E_cc,F_mm_per_min]
    tp = Toolpath.from_gcode(raw, bead_width=2.0, bead_height=0.9)

    frame_times, _ = tp.frame_times(n_frames=1000)
    merged, frame_indices = tp.merge(frame_times)
    draw_fracs = (frame_indices / max(len(merged) - 1, 1)).reshape(-1, 1)

    v.add_bead("bead", merged.points, width=merged.widths, height=merged.heights,
               colors=merged.gradient_colors("plasma"))
"""

from __future__ import annotations

import numpy as np

from .animation import merge_animation_points, toolpath_frame_times

# ---------------------------------------------------------------------------
# Perceptual colormap tables — 11 key colours at t=0, 0.1, …, 1.0
# Values from matplotlib (CC0 / public domain).
# ---------------------------------------------------------------------------
_COLORMAPS: dict[str, np.ndarray] = {
    "viridis": np.array(
        [
            [0.267004, 0.004874, 0.329415],
            [0.282623, 0.140926, 0.457517],
            [0.253935, 0.265254, 0.529983],
            [0.206756, 0.371758, 0.553117],
            [0.163625, 0.471133, 0.558148],
            [0.127568, 0.566949, 0.550556],
            [0.134692, 0.658636, 0.517649],
            [0.266941, 0.748751, 0.440573],
            [0.477504, 0.821444, 0.318195],
            [0.741388, 0.873449, 0.149561],
            [0.993248, 0.906157, 0.143936],
        ],
        dtype=np.float32,
    ),
    "plasma": np.array(
        [
            [0.050383, 0.029803, 0.527975],
            [0.254627, 0.013882, 0.615419],
            [0.387998, 0.001370, 0.658636],
            [0.514579, 0.025955, 0.659574],
            [0.634229, 0.108818, 0.626296],
            [0.741388, 0.214982, 0.538982],
            [0.826588, 0.329527, 0.434444],
            [0.893606, 0.451294, 0.331176],
            [0.943759, 0.577154, 0.225299],
            [0.973416, 0.708818, 0.119595],
            [0.940015, 0.975158, 0.131326],
        ],
        dtype=np.float32,
    ),
    "turbo": np.array(
        [
            [0.18995, 0.07176, 0.23217],
            [0.25105, 0.42930, 0.89988],
            [0.13610, 0.68323, 0.86543],
            [0.03830, 0.85959, 0.64671],
            [0.17299, 0.96162, 0.34963],
            [0.52886, 0.99480, 0.00629],
            [0.85967, 0.87953, 0.02330],
            [0.99643, 0.65731, 0.02234],
            [0.97554, 0.38488, 0.04246],
            [0.82109, 0.14657, 0.08578],
            [0.47960, 0.01583, 0.01055],
        ],
        dtype=np.float32,
    ),
}


def _apply_colormap(frac: np.ndarray, name: str) -> np.ndarray:
    """Map (N,) fractions in [0, 1] through a named colormap → (N, 3) float32."""
    table = _COLORMAPS.get(name)
    if table is None:
        raise ValueError(f"Unknown colormap {name!r}. Choose: {list(_COLORMAPS)}")
    n = len(table) - 1
    idx = np.clip(frac * n, 0, n - 1e-9)
    lo = idx.astype(int)
    hi = np.minimum(lo + 1, n)
    t = (idx - lo).astype(np.float32)[:, None]
    return table[lo] * (1.0 - t) + table[hi] * t


# ---------------------------------------------------------------------------


class Toolpath:
    """Processed bead toolpath — columns: [t_s, x, y, z, width, height].

    Construct via:

    - ``Toolpath.from_gcode(raw, bead_width, bead_height)``
      for a G-code-style raw array ``[x, y, z, E_cc, F_mm_per_min]``.
    - ``Toolpath.from_points(points, bead_width, bead_height, duration)``
      for a continuous path with no travel/extrusion distinction.

    Then use the instance to drive ``add_bead`` + animation::

        colors = tp.gradient_colors("plasma")
        frame_times, _ = tp.frame_times(n_frames)
        merged, frame_indices = tp.merge(frame_times)
        draw_fracs = (frame_indices / max(len(merged) - 1, 1)).reshape(-1, 1)
    """

    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data, dtype=np.float32)

    # ── constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_gcode(
        cls,
        raw: np.ndarray,
        bead_width: float,
        bead_height: float,
    ) -> Toolpath:
        """Convert a G-code-style toolpath to a Toolpath.

        Detects extruding segments from ``dE > 0`` and computes per-point
        time from the feedrate column.  Transition points (extrusion start/end)
        become zero-width rings, which ``add_bead`` renders as tapered caps.

        Args:
            raw: (N, 5) float32 ``[x_mm, y_mm, z_mm, E_cc, F_mm_per_min]``.
                 ``E_cc``: cumulative extrusion volume in cc — constant on
                 travel moves, increasing on extrusion.
                 ``F_mm_per_min``: feedrate for the move *arriving* at this
                 point.
            bead_width: cross-section width (mm).
            bead_height: cross-section height (mm).
        """
        xyz, E_cc, F = raw[:, :3], raw[:, 3], raw[:, 4]
        seg_len = np.linalg.norm(np.diff(xyz, axis=0, prepend=xyz[0:1]), axis=1)
        t = np.cumsum(60.0 * seg_len / np.maximum(F, 1e-10)).astype(np.float32)
        # A point is extruding when its arriving segment deposits material.
        # Exception: zero-length connector segments (e.g. arc→straight join
        # points) have dE=0 by construction even when extrusion is continuous.
        # For those, look at the *departing* segment instead.
        # Index 0 is always zero-length by the prepend convention — keep it as
        # a start cap (w=0) regardless of what follows.
        ext = np.diff(E_cc, prepend=E_cc[0]) > 1e-10
        zero_len = seg_len < 1e-10
        zero_len[0] = False  # preserve start cap
        dE_depart = np.diff(E_cc, append=E_cc[-1]) > 1e-10
        ext = ext | (zero_len & dE_depart)
        widths = np.where(ext, bead_width, 0.0).astype(np.float32)
        heights = np.where(ext, bead_height, 0.0).astype(np.float32)
        return cls(np.column_stack([t, xyz, widths, heights]))

    @classmethod
    def from_points(
        cls,
        points: np.ndarray,
        bead_width: float | np.ndarray,
        bead_height: float | np.ndarray,
        duration: float = 1.0,
    ) -> Toolpath:
        """Create a uniform-time Toolpath from xyz points (no travel/extrusion).

        Assigns synthetic timestamps via ``linspace(0, duration, N)``.  Use
        this for continuous paths (spiral vases, trajectories) where physical
        print time is not available.

        Args:
            points: (N, 3) float32 path points.
            bead_width: scalar or (N,) per-point width.
            bead_height: scalar or (N,) per-point height.
            duration: total animation duration in seconds.
        """
        points = np.asarray(points, dtype=np.float32)
        N = len(points)
        t = np.linspace(0.0, duration, N, dtype=np.float32)
        W = np.broadcast_to(np.asarray(bead_width, dtype=np.float32), (N,)).copy()
        H = np.broadcast_to(np.asarray(bead_height, dtype=np.float32), (N,)).copy()
        return cls(np.column_stack([t, points, W, H]))

    # ── data accessors ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._data)

    @property
    def data(self) -> np.ndarray:
        """Underlying (N, 6) float32 array: [t_s, x, y, z, width, height]."""
        return self._data

    @property
    def times(self) -> np.ndarray:
        """(N,) per-point timestamps in seconds."""
        return self._data[:, 0]

    @property
    def points(self) -> np.ndarray:
        """(N, 3) xyz positions."""
        return self._data[:, 1:4]

    @property
    def widths(self) -> np.ndarray:
        """(N,) per-point bead widths (0 = travel / cap)."""
        return self._data[:, 4]

    @property
    def heights(self) -> np.ndarray:
        """(N,) per-point bead heights (0 = travel / cap)."""
        return self._data[:, 5]

    @property
    def duration(self) -> float:
        """Total path duration in seconds."""
        return float(self._data[-1, 0] - self._data[0, 0])

    # ── animation helpers ─────────────────────────────────────────────────────

    def frame_times(self, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(frame_times, draw_fracs)`` evenly spaced in time.

        Wraps :func:`toolpath_frame_times`: frames advance proportionally to
        print time, so fast travel moves pass quickly and slow extrusion is
        gradual.  Use the returned ``frame_times`` with :meth:`merge` for
        segment-aligned animation.
        """
        return toolpath_frame_times(self.times, n_frames)

    def merge(self, frame_times: np.ndarray) -> tuple[Toolpath, np.ndarray]:
        """Insert ``frame_times`` into geometry for segment-aligned draw_range.

        Wraps :func:`merge_animation_points`.  Each frame time gets an exact
        mesh vertex so ``draw_range`` never cuts through a triangle ring.

        Returns:
            merged: new :class:`Toolpath` with interpolated points added.
            frame_indices: (n_frames,) index into *merged* for each frame.
        """
        combined, frame_indices = merge_animation_points(self._data, frame_times)
        return Toolpath(combined), frame_indices

    # ── color ─────────────────────────────────────────────────────────────────

    def gradient_colors(
        self,
        colormap: str = "viridis",
        travel_color: tuple[float, float, float] | None = (0.25, 0.25, 0.25),
    ) -> np.ndarray:
        """Per-point RGB colors using a perceptual colormap along arc-length.

        Args:
            colormap: ``"viridis"``, ``"plasma"``, or ``"turbo"``.
            travel_color: RGB for zero-width (travel/cap) points, or ``None``
                          to leave travel/cap points with their normal
                          colormap-derived gradient colour (no override).

        Returns:
            (N, 3) float32 RGB in [0, 1].
        """
        seg_len = np.linalg.norm(
            np.diff(self.points, axis=0, prepend=self.points[0:1]), axis=1
        )
        arc = np.cumsum(seg_len)
        frac = (arc / max(float(arc[-1]), 1e-10)).astype(np.float32)
        colors = _apply_colormap(frac, colormap)
        if travel_color is not None:
            is_travel = (self.widths == 0) & (self.heights == 0)
            colors[is_travel] = travel_color
        return colors
