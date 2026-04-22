"""
Zigzag toolpath — twist artifact on hairpin turns (regression test)

A serpentine infill path (parallel lines connected by 180° hairpins) is
pathological for the current constant-up frame derivation in
``add_parametric_tube``: when the tangent reverses at a hairpin, the
width axis ``U = V x T`` also reverses, so adjacent rings point their
cross-section vertices at opposite physical points. The quad between
them twists into a figure-eight and you get self-intersecting faces at
every hairpin.

Two beads are drawn:

1. ``retrace`` — two horizontally-stacked passes (z=0 forward, z=dz back)
   joined by a short C-shape turnaround that extends past the end point
   so the tangent stays horizontal through the hairpin. Two intermediate
   "return loop" points sit side-by-side with opposite tangents, giving
   a clean 180 deg reversal at a single ring pair.
2. ``infill`` — a classic zigzag pattern with six parallel passes.

Rotate around the hairpin: with the bug present, the bead pinches and
flips inside-out where the tangent reverses. After the frame-continuity
fix, quads stay flat and the tube surface is continuous.

Run: uv run python examples/20_toolpath_zigzag.py
"""

import time

import numpy as np

from threejs_viewer import viewer


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


def ramp_colors(n, start=0xFF5533, end=0x3377FF):
    """Linear RGB ramp so you can see ring order along the path."""
    sr, sg, sb = (start >> 16) & 0xFF, (start >> 8) & 0xFF, start & 0xFF
    er, eg, eb = (end >> 16) & 0xFF, (end >> 8) & 0xFF, end & 0xFF
    t = np.linspace(0, 1, n)
    r = (sr + (er - sr) * t).astype(np.uint32)
    g = (sg + (eg - sg) * t).astype(np.uint32)
    b = (sb + (eb - sb) * t).astype(np.uint32)
    return (r << 16) | (g << 8) | b


v = viewer()
v.clear()

v.add_box(
    "ground",
    width=14,
    height=12,
    depth=0.04,
    color=0x222222,
    position=[3.0, 0, -0.05],
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
    colors=ramp_colors(len(retrace)),
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
    colors=ramp_colors(len(infill)),
    roughness=0.4,
    metalness=0.1,
)

print(f"retrace: {len(retrace)} pts (one hairpin at x=end)")
print(f"infill:  {len(infill)} pts ({6 - 1} hairpins between passes)")
print("Rotate around the hairpin turns — with the bug, each turn shows")
print("a self-crossing twist where the bead collapses and flips inside-out.")
print("Press Ctrl+C to exit.")

try:
    while True:
        time.sleep(1.0)
except KeyboardInterrupt:
    pass
