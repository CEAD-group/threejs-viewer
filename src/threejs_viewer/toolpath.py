"""Toolpath processing: G-code → parametric tube visualization.

Typical usage::

    raw = make_my_toolpath()  # (N,5) [x,y,z,E_cc,F_mm_per_min]
    tp = Toolpath.from_gcode(raw, bead_width=2.0, bead_height=0.9)
    tp.colorize("plasma")
    v.add_toolpath("bead", tp, roughness=0.4)
"""

from __future__ import annotations

import numpy as np

# Colormaps come from the exact 256-entry reference tables in `_colormap_data`
# (the same tables `ViewerClient._apply_colormap` uses), so bead colouring here
# matches scalar colouring elsewhere instead of drifting on a coarser stop
# approximation.
from threejs_viewer._colormap_data import TABLES as _COLORMAPS


def _apply_colormap(frac: np.ndarray, name: str) -> np.ndarray:
    """Map (N,) fractions in [0, 1] through a named colormap → (N, 3) float32."""
    table = _COLORMAPS.get(name)
    if table is None:
        raise ValueError(f"Unknown colormap {name!r}. Choose: {list(_COLORMAPS)}")
    n = len(table) - 1
    idx = np.clip(frac * n, 0, n)
    lo = np.minimum(idx.astype(int), n - 1)
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

    Then visualize as a parametric tube::

        tp.colorize("plasma")
        v.add_toolpath("bead", tp, roughness=0.4)
    """

    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data, dtype=np.float32)
        self._colors: np.ndarray | None = None

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
        become zero-width rings, which ``to_mesh()`` renders as tapered caps.

        Args:
            raw: (N, 5) float32 ``[x_mm, y_mm, z_mm, E_cc, F_mm_per_min]``.
                 ``E_cc``: cumulative extrusion volume in cc — constant on
                 travel moves, increasing on extrusion.
                 ``F_mm_per_min``: feedrate for the move *arriving* at this
                 point.
            bead_width: cross-section width (mm).
            bead_height: cross-section height (mm).
        """
        raw = np.asarray(raw)
        if raw.ndim != 2 or raw.shape[1] != 5:
            raise ValueError(
                f"from_gcode expected a (N, 5) array [x,y,z,E,F], got shape {raw.shape!r}"
            )
        if raw.shape[0] < 2:
            raise ValueError(
                f"from_gcode requires at least 2 points, got {raw.shape[0]}"
            )
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
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"from_points expected a (N, 3) array of xyz points, got shape {points.shape!r}"
            )
        if len(points) < 2:
            raise ValueError(
                f"from_points requires at least 2 points, got {len(points)}"
            )
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

    # ── color ─────────────────────────────────────────────────────────────────

    @property
    def colors(self) -> np.ndarray | None:
        """(N, 3) float32 per-point RGB colors, or None if not set."""
        return self._colors

    @colors.setter
    def colors(self, value: np.ndarray | None) -> None:
        if value is None:
            self._colors = None
            return
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(self) or arr.shape[1] != 3:
            raise ValueError(
                f"colors must have shape ({len(self)}, 3), got {arr.shape!r}"
            )
        self._colors = arr

    @property
    def packed_colors(self) -> np.ndarray | None:
        """(N,) uint32 packed 0x00RRGGBB, or None if colors not set.

        Suitable for passing to ``add_parametric_tube(colors=...)``.
        """
        if self._colors is None:
            return None
        c = np.clip(self._colors, 0.0, 1.0)
        r = (c[:, 0] * 255).astype(np.uint32)
        g = (c[:, 1] * 255).astype(np.uint32)
        b = (c[:, 2] * 255).astype(np.uint32)
        return (r << 16) | (g << 8) | b

    def colorize(
        self,
        color_or_values,
        colormap: str = "viridis",
        travel_color: tuple[float, float, float] | None = (0.25, 0.25, 0.25),
    ) -> "Toolpath":
        """Set per-point colors and return self for chaining.

        Args:
            color_or_values: Dispatch by type:

                - ``str`` — arc-length gradient with that colormap name
                  (``"viridis"``, ``"plasma"``, ``"turbo"``).
                - ``int`` — solid hex color (e.g. ``0xFF0000``).
                - ``tuple`` — solid RGB float triple (e.g. ``(0.8, 0.2, 0.1)``).
                - ``(3,) ndarray`` — solid RGB float triple.
                - ``(N,) ndarray`` — per-point scalar values mapped through
                  ``colormap``.

            colormap: Colormap used when ``color_or_values`` is a string or
                ``(N,)`` array.
            travel_color: RGB applied to zero-width (travel/cap) points when
                ``color_or_values`` is a string. Pass ``None`` to disable.

        Returns:
            ``self`` for chaining.
        """
        N = len(self)
        v = color_or_values

        if isinstance(v, str):
            # Arc-length gradient
            seg_len = np.linalg.norm(
                np.diff(self.points, axis=0, prepend=self.points[0:1]), axis=1
            )
            arc = np.cumsum(seg_len)
            frac = (arc / max(float(arc[-1]), 1e-10)).astype(np.float32)
            result = _apply_colormap(frac, v)
            if travel_color is not None:
                is_travel = (self.widths == 0) & (self.heights == 0)
                result[is_travel] = travel_color
        elif isinstance(v, int):
            # Solid hex color
            r = ((v >> 16) & 0xFF) / 255.0
            g = ((v >> 8) & 0xFF) / 255.0
            b = (v & 0xFF) / 255.0
            result = np.tile(np.array([r, g, b], dtype=np.float32), (N, 1))
        elif isinstance(v, tuple) or (
            isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == 3
        ):
            # Solid RGB float triple
            result = np.tile(np.asarray(v, dtype=np.float32).reshape(1, 3), (N, 1))
        else:
            arr = np.asarray(v, dtype=np.float32)
            if arr.ndim == 2 and arr.shape == (N, 3):
                # (N, 3) per-point RGB passed directly
                result = arr
            elif arr.ndim == 1 and len(arr) == N:
                # (N,) per-point scalar values mapped through colormap
                vmin, vmax = float(arr.min()), float(arr.max())
                span = vmax - vmin if vmax != vmin else 1.0
                frac = ((arr - vmin) / span).astype(np.float32)
                result = _apply_colormap(frac, colormap)
            else:
                raise ValueError(
                    f"color_or_values array has shape {arr.shape}; "
                    f"expected ({N},) scalars or ({N}, 3) RGB"
                )

        self._colors = result
        return self
