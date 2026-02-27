"""
Toolpath with interrupted extrusion — variable width + travel moves

Demonstrates:
- Raw toolpath as [x, y, z, extruding, speed] (extruding describes segment to next point)
- process_toolpath: transition duplicates, time computation, Z offset
- add_bead with per-point width/height (0 = travel = degenerate triangles)
- merge_animation_points: inserts frame times into mesh geometry so each frame shows
  only complete segments (no partial triangle rings)
- All operations vectorized numpy — no Python hot loops

Run: uv run python examples/15_toolpath_interrupted.py
"""

import numpy as np

from threejs_viewer import Animation, merge_animation_points, viewer


def make_pill_toolpath(
    n_arc: int = 50,
    n_layers: int = 4,
    radius: float = 1.2,
    half_length: float = 1.5,
    layer_dz: float = 0.09,
    print_speed: float = 0.06,
    travel_factor: float = 4.0,
) -> np.ndarray:
    """Raw pill/racetrack toolpath.

    Path per layer: right_arc → top_straight → left_arc → bottom_straight(travel)
    XYZ is the top of the bead when extruding.

    Returns:
        (N, 5) float32: [x, y, z, extruding, speed].
        extruding: 1.0 if depositing towards next point, 0.0 if travel.
        speed: m/s for the segment towards next point.
    """
    cx = half_length
    spd_ext = print_speed
    spd_trv = print_speed * travel_factor
    z_layers = (np.arange(n_layers, dtype=np.float32) + 1) * layer_dz

    right_angles = np.linspace(-np.pi / 2, np.pi / 2, n_arc)
    left_angles = np.linspace(np.pi / 2, 3 * np.pi / 2, n_arc)

    # Right arc — extrusion
    ra = np.empty((n_layers, n_arc, 5), dtype=np.float32)
    ra[:, :, 0] = cx + radius * np.cos(right_angles)
    ra[:, :, 1] = radius * np.sin(right_angles)
    ra[:, :, 2] = z_layers[:, None]
    ra[:, :, 3] = 1.0
    ra[:, :, 4] = spd_ext

    # Top straight — extrusion
    ts = np.empty((n_layers, 2, 5), dtype=np.float32)
    ts[:, 0, :3] = [cx, radius, 0.0]
    ts[:, 1, :3] = [-cx, radius, 0.0]
    ts[:, :, 2] = z_layers[:, None]
    ts[:, :, 3] = 1.0
    ts[:, :, 4] = spd_ext

    # Left arc — extrusion
    la = np.empty((n_layers, n_arc, 5), dtype=np.float32)
    la[:, :, 0] = -cx + radius * np.cos(left_angles)
    la[:, :, 1] = radius * np.sin(left_angles)
    la[:, :, 2] = z_layers[:, None]
    la[:, :, 3] = 1.0
    la[:, :, 4] = spd_ext

    # Bottom straight — travel (no extrusion, 4× faster)
    bs = np.empty((n_layers, 2, 5), dtype=np.float32)
    bs[:, 0, :3] = [-cx, -radius, 0.0]
    bs[:, 1, :3] = [cx, -radius, 0.0]
    bs[:, :, 2] = z_layers[:, None]
    bs[:, :, 3] = 0.0
    bs[:, :, 4] = spd_trv

    return np.concatenate([ra, ts, la, bs], axis=1).reshape(-1, 5)


def process_toolpath(
    raw: np.ndarray,
    bead_width: float = 0.20,
    bead_height: float = 0.09,
) -> np.ndarray:
    """Process raw toolpath into bead-ready array.

    Steps:
      1. Insert duplicate points at extrusion on/off transitions
         (half-point offset: extruding[i] describes segment i → i+1).
      2. Compute cumulative time from arc-length and speed.
      3. Set bead width/height from extrusion flag.
      4. Offset Z: raw xyz = top of bead → bead path z = raw_z − bead_height.

    Returns:
        (M, 6) float32: [t, x, y, z, bead_width, bead_height].
    """
    # --- Step 1: transition duplicates ---
    ext = raw[:, 3] > 0.5
    trans_idx = np.where(ext[:-1] != ext[1:])[0] + 1  # points where state changes

    n_ins = len(trans_idx)
    expanded = np.empty((len(raw) + n_ins, 5), dtype=np.float32)

    # Output positions for original points (shifted right by prior insertions)
    bump = np.zeros(len(raw) + 1, dtype=np.int64)
    np.add.at(bump, trans_idx, 1)
    offsets = np.cumsum(bump[: len(raw)])
    orig_pos = np.arange(len(raw)) + offsets

    expanded[orig_pos] = raw

    # Duplicate: same xyz as transition point, ext/speed from previous segment
    ins_pos = orig_pos[trans_idx] - 1
    expanded[ins_pos] = raw[trans_idx]
    expanded[ins_pos, 3] = raw[trans_idx - 1, 3]
    expanded[ins_pos, 4] = raw[trans_idx - 1, 4]

    # --- Step 2: cumulative time ---
    xyz = expanded[:, :3]
    seg_len = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    dt_seg = seg_len / np.maximum(expanded[:-1, 4], 1e-10)
    times = np.empty(len(expanded), dtype=np.float32)
    times[0] = 0.0
    times[1:] = np.cumsum(dt_seg)

    # --- Step 3: build output with bead width/height ---
    ext_exp = expanded[:, 3] > 0.5
    out = np.empty((len(expanded), 6), dtype=np.float32)
    out[:, 0] = times
    out[:, 1:4] = xyz
    out[:, 4] = np.where(ext_exp, bead_width, 0.0)
    out[:, 5] = np.where(ext_exp, bead_height, 0.0)

    # --- Step 4: offset Z (raw = top of bead → bead path = bottom) ---
    out[ext_exp, 3] -= bead_height

    # Flat caps with ε spacing for deterministic triangle winding.
    # Zero-width ring placed ε along the path tangent away from the full-width ring,
    # giving a non-degenerate "disc" face at each extrusion start and end.
    if n_ins > 0:
        EPS = 1e-4
        is_to_ext = ~ext[trans_idx - 1] & ext[trans_idx]
        is_to_travel = ext[trans_idx - 1] & ~ext[trans_idx]

        # Start caps (travel→ext): zero-width ring placed ε before the extrusion start
        sc = ins_pos[is_to_ext]  # zero-width duplicate indices
        if len(sc) > 0:
            nxt = np.minimum(sc + 2, len(out) - 1)
            d = out[nxt, 1:4] - out[sc + 1, 1:4]
            dlen = np.linalg.norm(d, axis=1, keepdims=True)
            d /= np.where(dlen > 1e-10, dlen, 1.0)
            out[sc, 1:4] = out[sc + 1, 1:4] - EPS * d

        # End caps (ext→travel): zero-width ring placed ε past the extrusion end
        ec = ins_pos[is_to_travel]  # full-width duplicate indices
        if len(ec) > 0:
            prv = np.maximum(ec - 1, 0)
            d = out[ec, 1:4] - out[prv, 1:4]
            dlen = np.linalg.norm(d, axis=1, keepdims=True)
            d /= np.where(dlen > 1e-10, dlen, 1.0)
            ec_orig = orig_pos[trans_idx[is_to_travel]]
            out[ec_orig, 1:4] = out[ec, 1:4] + EPS * d

    return out


# --- Generate and process toolpath ---
N_FRAMES = 30

raw = make_pill_toolpath()
toolpath = process_toolpath(raw)

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
K = len(combined)

# Nozzle XYZ = top of bead = processed_z + bead_height (undoes the Z offset)
nozzle_xyz = points.copy()
nozzle_xyz[:, 2] += heights

# Per-layer alternating colors (based on original Z = nozzle Z)
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
layer_idx = np.clip(np.round(nozzle_xyz[:, 2] / LAYER_DZ).astype(int), 0, 99)
bead_colors = LAYER_COLORS[layer_idx % len(LAYER_COLORS)]

print(f"Raw toolpath: {len(raw)} points")
print(f"Processed:    {len(toolpath)} points (transition duplicates)")
print(f"Combined:     {K} points (after merging {N_FRAMES} frame times)")
print(f"Duration:     {frame_times[-1]:.1f}s")

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

# --- Animation: N_FRAMES frames, each aligned to an exact mesh vertex ---
# frame_nozzle_xyz: nozzle position at each animation frame
frame_nozzle_xyz = nozzle_xyz[frame_indices]

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
