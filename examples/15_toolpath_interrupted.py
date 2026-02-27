"""
Toolpath with interrupted extrusion — variable width + travel moves

Demonstrates:
- make_pill_toolpath: builds [t, x, y, z, width, height] directly (time + width/height
  assignment + zero-width endcap rows, all vectorized numpy)
- add_bead with per-point width/height (travel=W0H0, extrude=WfullHfull)
- merge_animation_points: inserts frame times into mesh geometry so each frame shows
  only complete segments (no partial triangle rings)

Run: uv run python examples/15_toolpath_interrupted.py
"""

import numpy as np

from threejs_viewer import Animation, merge_animation_points, viewer


def make_pill_toolpath(
    n_arc: int = 32,
    n_layers: int = 4,
    radius: float = 1.2,
    half_length: float = 1.5,
    layer_dz: float = 0.09,
    print_speed: float = 2.0,
    travel_factor: float = 8.0,
    bead_width: float = 0.20,
    bead_height: float = 0.09,
) -> np.ndarray:
    """Pill/racetrack toolpath ready for add_bead.

    Path per layer: right_arc → top_straight → left_arc → bottom_straight(travel)

    Returns:
        (N, 6) float32: [t, x, y, z, width, height].
        width/height = 0 on travel segments and at the zero-width endcap rows.
    """
    z_layers = (np.arange(n_layers, dtype=np.float32) + 1) * layer_dz
    right_angles = np.linspace(-np.pi / 2, np.pi / 2, n_arc)
    left_angles = np.linspace(np.pi / 2, 3 * np.pi / 2, n_arc)

    # Right arc — extrusion
    ra = np.empty((n_layers, n_arc, 3), dtype=np.float32)
    ra[:, :, 0] = half_length + radius * np.cos(right_angles)
    ra[:, :, 1] = radius * np.sin(right_angles)
    ra[:, :, 2] = z_layers[:, None]

    # Top straight — extrusion
    ts = np.empty((n_layers, 2, 3), dtype=np.float32)
    ts[:, 0, :2] = [half_length, radius]
    ts[:, 1, :2] = [-half_length, radius]
    ts[:, :, 2] = z_layers[:, None]

    # Left arc — extrusion
    la = np.empty((n_layers, n_arc, 3), dtype=np.float32)
    la[:, :, 0] = -half_length + radius * np.cos(left_angles)
    la[:, :, 1] = radius * np.sin(left_angles)
    la[:, :, 2] = z_layers[:, None]

    # Bottom straight — travel (no extrusion, 4× faster)
    bs = np.empty((n_layers, 2, 3), dtype=np.float32)
    bs[:, 0, :2] = [-half_length, -radius]
    bs[:, 1, :2] = [half_length, -radius]
    bs[:, :, 2] = z_layers[:, None]

    xyz = np.concatenate([ra, ts, la, bs], axis=1).reshape(-1, 3)

    n_ext = 2 * n_arc + 2  # ra + ts + la points per layer
    ext = np.tile([True] * n_ext + [False] * 2, n_layers)
    velocity = np.tile(
        [print_speed] * n_ext + [print_speed * travel_factor] * 2, n_layers
    ).astype(np.float32)

    seg_len = np.linalg.norm(np.diff(xyz, axis=0, prepend=xyz[0:1]), axis=1)
    t = np.cumsum(seg_len / np.maximum(velocity, 1e-10)).astype(np.float32)

    widths = np.where(ext, bead_width, 0.0).astype(np.float32)
    heights = np.where(ext, bead_height, 0.0).astype(np.float32)

    out = np.column_stack([t, xyz, widths, heights])
    # Zero-width endcap rows prepended/appended
    start_cap = out[0:1].copy()
    start_cap[0, 4:] = 0.0
    end_cap = out[-1:].copy()
    end_cap[0, 4:] = 0.0
    return np.vstack([start_cap, out, end_cap])


# --- Generate toolpath ---
N_FRAMES = 1000

toolpath = make_pill_toolpath()

# Merge N_FRAMES evenly-spaced animation times into the toolpath geometry.
# This inserts new interpolated mesh points at each frame time so that every
# animation frame corresponds exactly to a mesh vertex — no partial rings.
frame_times = np.linspace(toolpath[0, 0], toolpath[-1, 0], N_FRAMES)
combined, frame_indices = merge_animation_points(toolpath, frame_times)
draw_fracs = (
    (frame_indices / max(len(combined) - 1, 1)).reshape(-1, 1).astype(np.float32)
)

points = combined[:, 1:4]
widths = combined[:, 4]
heights = combined[:, 5]

# Per-layer alternating colors
LAYER_DZ = 0.09
LAYER_COLORS = np.array(
    [
        [0.30, 0.65, 0.80],  # steel blue
        [0.85, 0.55, 0.25],  # orange
        [0.40, 0.78, 0.45],  # green
        [0.75, 0.40, 0.80],  # purple
    ],
    dtype=np.float32,
)
layer_idx = np.clip(np.round(points[:, 2] / LAYER_DZ).astype(int), 0, 99)
bead_colors = LAYER_COLORS[layer_idx % len(LAYER_COLORS)]

print(f"Toolpath:  {len(toolpath)} points")
print(f"Combined:  {len(combined)} points (after merging {N_FRAMES} frame times)")
print(f"Duration:  {frame_times[-1]:.1f}s")

# --- Scene ---
v = viewer()
v.clear()

v.add_box(
    "ground", width=8, height=6, depth=0.02, color=0x222222, position=[0, 0, -0.01]
)

v.add_bead(
    "bead",
    points,
    width=widths,
    height=heights,
    colors=bead_colors,
    roughness=0.4,
    metalness=0.1,
)

nozzle_h = 0.5
v.add_cylinder(
    "nozzle",
    radius_top=0.12,
    radius_bottom=0.04,
    height=nozzle_h,
    color=0xCC8844,
    roughness=0.3,
    metalness=0.8,
)

frame_nozzle_xyz = points[frame_indices]

transforms = np.zeros((N_FRAMES, 2, 16), dtype=np.float32)
transforms[:, 0, [0, 5, 10, 15]] = 1.0  # bead: identity
m = transforms[:, 1]  # nozzle: Rot(+90° about X) + translation
m[:, 0] = 1.0
m[:, 6] = 1.0
m[:, 9] = -1.0
m[:, 15] = 1.0
m[:, 12] = frame_nozzle_xyz[:, 0]
m[:, 13] = frame_nozzle_xyz[:, 1]
m[:, 14] = frame_nozzle_xyz[:, 2] + nozzle_h / 2

animation = Animation(loop=True)
animation.set_frame_times(frame_times)
animation.set_transform_data(["bead", "nozzle"], transforms)
animation.set_draw_range_data(["bead"], draw_fracs)
animation.add_marker(0.0, "Start", color=0x00FF00)
animation.add_marker(float(frame_times[-1]) / 2, "50%", color=0xFFFF00)

v.load_animation(animation)

print(f"Animation: {N_FRAMES} frames, {animation.duration:.1f}s")
print("Each frame = one complete ring of triangles (no partial rings).")
print("Bottom straight is travel (no bead); all other segments extrude.")
print("Press Ctrl+C to exit.")

try:
    while True:
        pass
except KeyboardInterrupt:
    v.disconnect()
