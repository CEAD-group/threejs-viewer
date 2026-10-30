"""
Zigzag toolpaths with 180 degree hairpin turns (frame-continuity regression scene).

Two beads whose tangent reverses abruptly: ``retrace`` goes forward, turns through a
C-shaped end and comes back above itself, and ``infill`` is a six-pass serpentine.
Orbit around the hairpins: the bead should stay a continuous, flat surface instead of
pinching and flipping inside out. Colours ramp along the path to show ring order.

Run: uv run python examples/21_toolpath_zigzag.py
"""

import numpy as np

from threejs_viewer import Toolpath, viewer


def make_retrace(n=60, length=6.0, dz=0.35, x_over=1.0):
    """Forward at z=0, backward at z=dz, joined by a C-shape turnaround.

    The loop has two midway points at (length+x_over, 0, dz/4) and
    (length+x_over, 0, 3*dz/4) — between them the tangent flips from
    +X-ish to -X-ish in a single step, the canonical 180 deg reversal.
    The extension past the end point keeps the tangent horizontal (no
    near-vertical instability in the constant-up frame).
    """
    fwd_x = np.linspace(0.0, length, n)
    fwd_z = np.zeros(n)
    back_x = np.linspace(length, 0.0, n)
    back_z = np.full(n, dz)
    loop_x = np.array([length + x_over, length + x_over])
    loop_z = np.array([dz / 4, 3 * dz / 4])
    x = np.concatenate([fwd_x, loop_x, back_x])
    z = np.concatenate([fwd_z, loop_z, back_z])
    y = np.zeros_like(x)
    return np.column_stack([x, y, z]).astype(np.float32)


def make_infill(n_passes=6, pass_length=8.0, spacing=0.9, n_per_pass=40):
    """Serpentine infill: alternating +Y / -Y passes stepping in +X.

    Each pass is connected to the next by a single mid-point, so the
    tangent flips ~180 deg abruptly (worst case for the constant-up frame).
    """
    points = []
    for i in range(n_passes):
        x = i * spacing
        y = np.linspace(-pass_length / 2, pass_length / 2, n_per_pass)
        if i % 2 == 1:
            y = y[::-1]
        pass_pts = np.column_stack([np.full_like(y, x), y, np.zeros_like(y)])
        points.append(pass_pts)
        if i < n_passes - 1:
            x_mid = x + spacing / 2
            y_end = pass_pts[-1, 1]
            points.append(np.array([[x_mid, y_end, 0.0]], dtype=np.float32))
    return np.concatenate(points, axis=0).astype(np.float32)


v = viewer()
v.clear()

v.add_grid(
    "floor", cell_size=1.0, extent=40.0, center_color=0x888888, position=[0, 0, -0.3]
)

# --- Case 1: retrace on top of itself ---
retrace = make_retrace()
retrace[:, 1] -= 6.0  # shift so it sits below the infill
w_r = np.full(len(retrace), 0.45, dtype=np.float32)
h_r = np.full(len(retrace), 0.25, dtype=np.float32)
v.add_parametric_tube(
    "retrace",
    spine=retrace,
    widths=w_r,
    heights=h_r,
    colors=Toolpath.from_points(retrace, w_r, h_r).colorize("plasma").packed_colors,
    roughness=0.4,
    metalness=0.1,
)

# --- Case 2: serpentine infill ---
infill = make_infill()
w_i = np.full(len(infill), 0.5, dtype=np.float32)
h_i = np.full(len(infill), 0.3, dtype=np.float32)
v.add_parametric_tube(
    "infill",
    spine=infill,
    widths=w_i,
    heights=h_i,
    colors=Toolpath.from_points(infill, w_i, h_i).colorize("plasma").packed_colors,
    roughness=0.4,
    metalness=0.1,
)

v.set_camera(position=[3, -10, 18], target=[3, -2, 0])
v.wait_for_assets()
