"""
Toolpath with interrupted extrusion — variable width + travel moves

Demonstrates:
- make_pill_toolpath: G-code-style [x_mm, y_mm, z_mm, E_cc, F_mm_per_min]
  (cumulative extrusion in cc, feedrate in mm/min)
- toolpath_to_bead: detects extruding segments from dE > 0, computes time
  from F, returns [t_s, x, y, z, width, height] — transition points naturally
  become zero-width rings (caps) without explicit insertion
- add_bead with per-point width/height (travel=W0H0, extrude=WfullHfull)
- merge_animation_points: inserts frame times into mesh geometry so each frame shows
  only complete segments (no partial triangle rings)

Run: uv run python examples/15_toolpath_interrupted.py
"""

import numpy as np

from threejs_viewer import Animation, merge_animation_points, viewer

BEAD_WIDTH = 2.0  # mm
BEAD_HEIGHT = 0.9  # mm


def make_pill_toolpath(
    n_arc: int = 32,
    n_layers: int = 4,
    radius: float = 12.0,  # mm
    half_length: float = 15.0,  # mm
    layer_dz: float = 0.9,  # mm
    print_speed: float = 3000.0,  # mm/min
    travel_factor: float = 3.0,
    bead_width: float = BEAD_WIDTH,
    bead_height: float = BEAD_HEIGHT,
) -> np.ndarray:
    """Pill/racetrack toolpath in G-code-style columns.

    Path per layer: right_arc → top_straight → left_arc → bottom_straight(travel)

    Returns:
        (N, 5) float32: [x_mm, y_mm, z_mm, E_cc, F_mm_per_min].
        E_cc: cumulative extrusion volume in cc (constant on travel moves).
        F_mm_per_min: feedrate for the move arriving at this point.
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

    # Bottom straight — travel (no extrusion, travel_factor× faster)
    bs = np.empty((n_layers, 2, 3), dtype=np.float32)
    bs[:, 0, :2] = [-half_length, -radius]
    bs[:, 1, :2] = [half_length, -radius]
    bs[:, :, 2] = z_layers[:, None]

    xyz = np.concatenate([ra, ts, la, bs], axis=1).reshape(-1, 3)

    # is_ext_next[i]: is the segment from point i to point i+1 extruding?
    n_ext = 2 * n_arc + 2  # ra + ts + la points per layer
    is_ext_next = np.tile([True] * n_ext + [False] * 2, n_layers)

    # F arriving at each point (feedrate of the move from previous point)
    speed_next = np.where(is_ext_next, print_speed, print_speed * travel_factor)
    F = np.concatenate([[print_speed], speed_next[:-1]]).astype(np.float32)

    # Cumulative extrusion E (cc): cross_section * seg_length / 1000 per extruding segment
    seg_len_next = np.linalg.norm(np.diff(xyz, axis=0, append=xyz[-1:]), axis=1)
    dE_next = np.where(
        is_ext_next, bead_width * bead_height * seg_len_next / 1000.0, 0.0
    )
    E_cc = np.concatenate([[0.0], np.cumsum(dE_next[:-1])]).astype(np.float32)

    return np.column_stack([xyz, E_cc, F]).astype(np.float32)


def toolpath_to_bead(
    raw: np.ndarray,
    bead_width: float,
    bead_height: float,
) -> np.ndarray:
    """Convert G-code-style toolpath to bead geometry array for add_bead.

    Detects extruding segments from dE > 0 and computes time from feedrate.
    Transition points (extrusion start/end) naturally become zero-width rings,
    which add_bead renders as tapered caps.

    Args:
        raw: (N, 5) float32 [x_mm, y_mm, z_mm, E_cc, F_mm_per_min]
        bead_width: bead cross-section width (mm)
        bead_height: bead cross-section height (mm)

    Returns:
        (N, 6) float32: [t_s, x_mm, y_mm, z_mm, width_mm, height_mm].
    """
    xyz, E_cc, F = raw[:, :3], raw[:, 3], raw[:, 4]

    # dE > 0 at point i means the move arriving at i was extruding
    ext = np.diff(E_cc, prepend=E_cc[0]) > 1e-10

    # dt = 60 * seg_len / F  (mm / (mm/min) × 60 = s)
    seg_len = np.linalg.norm(np.diff(xyz, axis=0, prepend=xyz[0:1]), axis=1)
    t = np.cumsum(60.0 * seg_len / np.maximum(F, 1e-10)).astype(np.float32)

    widths = np.where(ext, bead_width, 0.0).astype(np.float32)
    heights = np.where(ext, bead_height, 0.0).astype(np.float32)

    return np.column_stack([t, xyz, widths, heights]).astype(np.float32)


# --- Generate and process toolpath ---
N_FRAMES = 1000

raw = make_pill_toolpath()
toolpath = toolpath_to_bead(raw, BEAD_WIDTH, BEAD_HEIGHT)

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
LAYER_DZ = 0.9  # mm
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

print(f"Raw toolpath: {len(raw)} points, {raw[-1, 3]:.4f} cc total extrusion")
print(f"Toolpath:     {len(toolpath)} points")
print(f"Combined:     {len(combined)} points (after merging {N_FRAMES} frame times)")
print(f"Duration:     {frame_times[-1]:.1f} s")

# --- Scene ---
v = viewer()
v.clear()

v.add_box(
    "ground", width=80, height=50, depth=0.2, color=0x222222, position=[0, 0, -0.1]
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

nozzle_h = 5.0  # mm
v.add_cylinder(
    "nozzle",
    radius_top=1.2,
    radius_bottom=0.4,
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

print(f"Animation: {N_FRAMES} frames, {animation.duration:.1f} s")
print("Each frame = one complete ring of triangles (no partial rings).")
print("Bottom straight is travel (no bead); all other segments extrude.")
print("Press Ctrl+C to exit.")

try:
    while True:
        pass
except KeyboardInterrupt:
    v.disconnect()
