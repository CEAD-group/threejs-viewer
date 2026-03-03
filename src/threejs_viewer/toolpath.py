"""Toolpath processing: G-code → bead geometry with animation support.

Typical usage::

    raw = make_my_toolpath()  # (N,5) [x,y,z,E_cc,F_mm_per_min]
    tp = Toolpath.from_gcode(raw, bead_width=2.0, bead_height=0.9)

    frame_times, _ = tp.frame_times(n_frames=1000)
    merged, frame_indices = tp.merge(frame_times)
    draw_fracs = (frame_indices / max(len(merged) - 1, 1)).reshape(-1, 1)

    merged.colorize("plasma")
    v.add_mesh("bead", **merged.to_mesh())
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

    Then use the instance to drive mesh building + animation::

        frame_times, _ = tp.frame_times(n_frames)
        merged, frame_indices = tp.merge(frame_times)
        merged.colorize("plasma")
        draw_fracs = (frame_indices / max(len(merged) - 1, 1)).reshape(-1, 1)
        v.add_mesh("bead", **merged.to_mesh())
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

        If ``self.colors`` is set, colors are interpolated for the new points.

        Returns:
            merged: new :class:`Toolpath` with interpolated points added.
            frame_indices: (n_frames,) index into *merged* for each frame.
        """
        combined, frame_indices = merge_animation_points(self._data, frame_times)
        merged = Toolpath(combined)
        if self._colors is not None:
            merged_times = combined[:, 0]
            orig_times = self.times
            # np.interp requires strictly increasing xp; deduplicate by keeping
            # the last occurrence at each duplicate timestamp (matches geometry
            # behavior where the last point at a duplicate time wins).
            N_orig = len(orig_times)
            _, last_idx = np.unique(orig_times[::-1], return_index=True)
            last_idx = np.sort(N_orig - 1 - last_idx)
            merged_colors = np.stack(
                [
                    np.interp(
                        merged_times, orig_times[last_idx], self._colors[last_idx, c]
                    )
                    for c in range(3)
                ],
                axis=1,
            ).astype(np.float32)
            merged._colors = merged_colors
        return merged, frame_indices

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

    # ── geometry ──────────────────────────────────────────────────────────────

    def to_mesh(self, plane_normal: "np.ndarray | None" = None) -> dict:
        """Build bead mesh geometry.

        Generates a 6-vertex bevelled rectangle cross-section extruded along
        the path with analytical normals.  Supports ``draw_range`` for
        progressive reveal animation.

        Args:
            plane_normal: The "up" direction for the bead cross-section, i.e.
                the slice-plane normal in viewer space.  Defaults to
                ``[0, 0, 1]`` (world Z-up).  Pass the transformed plane normal
                when slicing on a tilted plane so the bead height is
                perpendicular to the layer surface rather than world-vertical.

        Returns:
            Dict with keys ``positions``, ``indices``, ``normals``, ``colors``.
            Use with ``v.add_mesh(id, **tp.to_mesh())``.
        """
        points = self.points  # (N, 3)
        N = len(points)
        if N < 2:
            raise ValueError("to_mesh() requires at least 2 path points")
        P = 6  # vertices per ring

        W = self.widths.copy()
        H = self.heights.copy()
        hw = W / 2
        hh = H / 2
        ft = np.maximum(0.0, W - H) / 2

        # Per-point profiles: (N, P, 2) — [binormal_offset, z_offset]
        zeros = np.zeros(N, dtype=np.float32)
        pb = np.column_stack([ft, hw, ft, -ft, -hw, -ft])  # (N, P) binormal offsets
        pz = -np.column_stack([zeros, hh, H, H, hh, zeros])  # (N, P) z offsets

        # Profile vertex normals (vectorized, average of adjacent edge outward normals)
        profile = np.stack([pb, pz], axis=-1)  # (N, P, 2)
        edges = np.roll(profile, -1, axis=1) - profile  # (N, P, 2)
        edge_n = np.stack([edges[:, :, 1], -edges[:, :, 0]], axis=-1)  # (N, P, 2)
        edge_n /= np.maximum(np.linalg.norm(edge_n, axis=-1, keepdims=True), 1e-10)
        prof_n = edge_n + np.roll(edge_n, 1, axis=1)  # (N, P, 2)
        prof_n /= np.maximum(np.linalg.norm(prof_n, axis=-1, keepdims=True), 1e-10)

        # Plane normal ("up" direction for cross-section height)
        if plane_normal is None:
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            up = np.asarray(plane_normal, dtype=np.float32)
            up = up / np.maximum(float(np.linalg.norm(up)), 1e-10)

        # Central-difference tangents (3D), projected perpendicular to plane normal
        tangents = np.empty((N, 3), dtype=np.float32)
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
        tangents[1:-1] = points[2:] - points[:-2]
        tangents -= (tangents * up).sum(axis=1, keepdims=True) * up
        t_len = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents /= np.maximum(t_len, 1e-10)

        # Binormals: tangent × up  (perpendicular to both, lies in the layer plane)
        binormals = np.cross(tangents, up).astype(np.float32)  # (N, 3)
        b_len = np.linalg.norm(binormals, axis=1, keepdims=True)
        binormals /= np.maximum(b_len, 1e-10)

        # Positions: (N, P, 3)
        positions = (
            points[:, np.newaxis, :]  # (N, 1, 3)
            + pb[:, :, np.newaxis] * binormals[:, np.newaxis, :]  # (N, P, 3)
            + pz[:, :, np.newaxis] * up  # (N, P, 3)
        )

        # Normals: (N, P, 3) — profile normals rotated into world frame
        nb = prof_n[:, :, 0]  # (N, P) binormal component
        nz = prof_n[:, :, 1]  # (N, P) z (plane-normal) component
        normals = (
            nb[:, :, np.newaxis] * binormals[:, np.newaxis, :]  # (N, P, 3)
            + nz[:, :, np.newaxis] * up  # (N, P, 3)
        )

        # Step-major indices for draw_range reveal along path
        i_range = np.arange(N - 1, dtype=np.uint32)
        j_range = np.arange(P, dtype=np.uint32)
        ig, jg = np.meshgrid(i_range, j_range, indexing="ij")
        j1 = (jg + 1) % P
        va = ig * P + jg
        vb = ig * P + j1
        vc = (ig + 1) * P + jg
        vd = (ig + 1) * P + j1
        # Two tris per quad: (va,vc,vb) and (vb,vc,vd)
        tris = np.stack([va, vc, vb, vb, vc, vd], axis=-1)  # (N-1, P, 6)
        indices = tris.reshape(-1).astype(np.uint32)

        # Broadcast per-path-point colors to per-vertex (N, P, 3)
        vertex_colors = None
        if self._colors is not None:
            c_arr = np.asarray(self._colors, dtype=np.float32)
            if c_arr.ndim == 2 and c_arr.shape == (N, 3):
                vertex_colors = np.broadcast_to(c_arr[:, None, :], (N, P, 3)).reshape(
                    -1, 3
                )

        return {
            "positions": positions.reshape(-1, 3),
            "indices": indices,
            "normals": normals.reshape(-1, 3),
            "colors": vertex_colors,
        }
