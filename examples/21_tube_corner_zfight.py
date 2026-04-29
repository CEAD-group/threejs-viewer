"""
Three-tube comparison of the strand-collapse fix on an open 3D path.

Spine is the same blobby periodic Catmull-Rom that the earlier closed-
loop demo used, trimmed to ~70% of the arc so it's open with multiple
sharp inward lobes, plus a linear Z ramp totalling ≈ 3.7× the bead
height (per-sample dZ stays well below FOLD_THRESHOLD so it doesn't
suppress folds — the path is genuinely 3D, not just XY). Width
modulates sinusoidally so each corner hits a different bead size.

LEFT   — `add_parametric_tube` (no collapse). Sphere markers visualize
         every fold target detected by the Python prototype: cells
         (i, j) of the per-strand 3D seg-seg shortest-line-distance
         grid that are local minima below FOLD_THRESHOLD.
MIDDLE — `add_parametric_tube(strand_collapse=True)`. Same call site,
         the viewer runs the snap pass on the client (un-mitered fold
         detection, mitered-mesh midpoint snap).
RIGHT  — Python reference: a chamfered-hex tube mesh built locally
         with the same snaps applied, uploaded via `add_mesh`. Acts as
         a visual oracle for the JS implementation in the middle.

The middle and right tubes should match at every collapsed corner.

Run: uv run python examples/21_tube_corner_zfight.py
"""

import time

import numpy as np

from threejs_viewer import viewer


def blobby_control_points(n, base_r, jitter, seed):
    """Random polar control points on a closed loop: base radius + jitter.

    The Catmull-Rom that interpolates these naturally produces tight
    inward lobes wherever the radial jitter dips below the base — the
    same generator that drove the earlier closed-loop example.
    """
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    theta = theta + rng.uniform(-0.35, 0.35, size=n) * (2.0 * np.pi / n)
    r = base_r * (1.0 + rng.uniform(-jitter, jitter, size=n))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def catmull_rom_periodic(points, samples_per_segment, alpha=0.5):
    """Centripetal Catmull-Rom through a periodic loop of 2D control points."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)

    def knot(ti, a, b):
        d = float(np.linalg.norm(b - a))
        return ti + max(d, 1e-9) ** alpha

    out = []
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        t0 = 0.0
        t1 = knot(t0, p0, p1)
        t2 = knot(t1, p1, p2)
        t3 = knot(t2, p2, p3)
        ts = np.linspace(t1, t2, samples_per_segment, endpoint=False)
        for t in ts:
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            c = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
            out.append(c)
    return np.asarray(out, dtype=np.float64)


def hue_ramp(n, offset=0.0):
    """HSV sweep so ring order along the spine is visible."""
    h = (np.linspace(0, 1, n) + offset) % 1.0
    s, v = 0.9, 0.95
    hp = h * 6.0
    c = v * s
    x = c * (1 - np.abs(np.mod(hp, 2) - 1))
    m = v - c
    r = np.zeros(n)
    g = np.zeros(n)
    b = np.zeros(n)
    for i, hi in enumerate(hp):
        ri, gi, bi = {
            0: (c, x[i], 0),
            1: (x[i], c, 0),
            2: (0, c, x[i]),
            3: (0, x[i], c),
            4: (x[i], 0, c),
            5: (c, 0, x[i]),
        }[int(hi) % 6]
        r[i], g[i], b[i] = ri + m, gi + m, bi + m
    r8 = (r * 255).astype(np.uint32)
    g8 = (g * 255).astype(np.uint32)
    b8 = (b * 255).astype(np.uint32)
    return (r8 << 16) | (g8 << 8) | b8


v = viewer()
v.clear()

BEAD_W_MIN = 0.40
BEAD_W_MAX = 0.95
BEAD_H = 0.30
BASE_R = 1.10
SAMPLES_PER_SEG = 10  # sparse on purpose — realistic toolpath density
Z_TOTAL = 0.60  # total Z rise across the open path (≈ 2 × BEAD_H)

# Build a periodic blobby loop, then trim to an open arc of ~70% of the
# loop. The trim breaks the periodicity but keeps the Catmull-Rom-driven
# sharp inward lobes that the closed-loop generator produces naturally.
ctrl = blobby_control_points(n=13, base_r=BASE_R, jitter=0.45, seed=7)
spine_xy_full = catmull_rom_periodic(ctrl, samples_per_segment=SAMPLES_PER_SEG)
n_full = len(spine_xy_full)
trim_lo = int(round(n_full * 0.10))
trim_hi = int(round(n_full * 0.85))
spine_xy = spine_xy_full[trim_lo:trim_hi]
n = len(spine_xy)

# Linear Z ramp along arc length — total range Z_TOTAL, smooth per-sample
# step (~Z_TOTAL/n) well below FOLD_THRESHOLD so it doesn't interfere with
# detection but the path is genuinely 3D.
seg_len = np.linalg.norm(np.diff(spine_xy, axis=0), axis=1)
arc = np.concatenate([[0.0], np.cumsum(seg_len)])
arc_total = arc[-1]
phase = arc / arc_total  # 0..1 along the open path
spine_z = phase * Z_TOTAL
spine = np.column_stack([spine_xy, spine_z]).astype(np.float32)

# Width modulates sinusoidally so each corner hits a different bead size.
widths = (
    BEAD_W_MIN
    + 0.5 * (BEAD_W_MAX - BEAD_W_MIN) * (1.0 + np.sin(2.0 * np.pi * 2.5 * phase + 0.7))
).astype(np.float32)
heights = np.full(n, BEAD_H, dtype=np.float32)
colors = hue_ramp(n)


GLOBAL_UP = np.array([0.0, 0.0, 1.0])
FOLD_THRESHOLD = 0.04  # world units; strands closer than this count as folded.
# Looser than the JS-side per-bead-fraction tolerance because the open path
# has a constant linear Z ramp, which puts a floor on 3D seg-seg distance
# at corners (≈ ΔZ across the fold span). 0.04 lets typical 4-6 sample
# folds register without triggering spurious snaps on smooth bends.
MIN_SEG_GAP = 4  # require |j - i| >= MIN_SEG_GAP — adjacent/near segments share
# vertices and have ~0 distance trivially (not a fold)


def constant_up_frames(spine_3d):
    """Per-sample (U, V, T) frames matching the viewer's constant-up tube."""
    n_local = len(spine_3d)
    T = np.zeros((n_local, 3))
    for i in range(n_local):
        a = spine_3d[max(i - 1, 0)]
        b = spine_3d[min(i + 1, n_local - 1)]
        d = b - a
        nrm = np.linalg.norm(d)
        if nrm > 1e-12:
            T[i] = d / nrm
        else:
            T[i] = T[i - 1] if i > 0 else np.array([1.0, 0.0, 0.0])

    U = np.zeros_like(T)
    V = np.zeros_like(T)
    for i in range(n_local):
        t = T[i]
        seed = (
            GLOBAL_UP if abs(np.dot(t, GLOBAL_UP)) < 0.99 else np.array([1.0, 0.0, 0.0])
        )
        v_axis = seed - np.dot(seed, t) * t
        vn = np.linalg.norm(v_axis)
        v_axis = v_axis / vn if vn > 1e-12 else GLOBAL_UP.copy()
        u_axis = np.cross(v_axis, t)
        un = np.linalg.norm(u_axis)
        if un > 1e-12:
            u_axis /= un
        v_axis = np.cross(t, u_axis)
        U[i] = u_axis
        V[i] = v_axis
    return U, V, T


def seg_seg_closest_3d(p1, p2, p3, p4):
    """Closest-points pair on two 3D segments. Returns (pa, pb, distance)."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    if a <= 1e-12 and e <= 1e-12:
        return p1, p3, float(np.linalg.norm(p1 - p3))
    if a <= 1e-12:
        s, t = 0.0, float(np.clip(f / e, 0.0, 1.0))
    else:
        c = np.dot(d1, r)
        if e <= 1e-12:
            t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b
            s = (
                float(np.clip((b * f - c * e) / denom, 0.0, 1.0))
                if abs(denom) > 1e-12
                else 0.0
            )
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((b - c) / a, 0.0, 1.0))
    pa = p1 + s * d1
    pb = p3 + t * d2
    return pa, pb, float(np.linalg.norm(pa - pb))


def fold_targets(strand_3d, threshold, min_gap=MIN_SEG_GAP):
    """Local minima of the seg-seg distance grid below `threshold`.

    For each segment pair (i, j) with j - i >= min_gap, compute the 3D
    shortest-line distance. A target is a cell that's strictly less than
    its 8 valid neighbours in the (i, j) grid AND below the threshold.
    Returns a list of (i, j, midpoint, distance).
    """
    nseg = len(strand_3d) - 1
    INF = np.inf
    dist = np.full((nseg, nseg), INF)
    mids = {}
    for i in range(nseg):
        for j in range(i + min_gap, nseg):
            pa, pb, d_ij = seg_seg_closest_3d(
                strand_3d[i],
                strand_3d[i + 1],
                strand_3d[j],
                strand_3d[j + 1],
            )
            dist[i, j] = d_ij
            mids[(i, j)] = (pa + pb) * 0.5

    targets = []
    for i in range(nseg):
        for j in range(i + min_gap, nseg):
            d_ij = dist[i, j]
            if not (d_ij < threshold):
                continue
            is_min = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < nseg and 0 <= nj < nseg):
                        continue
                    a, b = (ni, nj) if ni < nj else (nj, ni)
                    if b - a < min_gap:
                        continue
                    if dist[a, b] <= d_ij:
                        is_min = False
                        break
                if not is_min:
                    break
            if is_min:
                targets.append((i, j, mids[(i, j)], d_ij))
    return targets


# Per-ring chamfered-hex (u, v) for each cross-section vertex, given the
# ring's W and H. Variable W means each strand's offset shifts ring-by-ring.
RED, BLUE, GREEN = 0xFF3355, 0x3399FF, 0x33CC55
VERT_NAMES = [
    "right_tip",
    "top_right",
    "top_left",
    "left_tip",
    "bottom_left",
    "bottom_right",
]
VERT_COLORS = [BLUE, RED, RED, BLUE, GREEN, GREEN]


def cross_section_uv(widths_arr, heights_arr):
    """Return (u, v) arrays of shape (n_rings, 6) for every CS vertex."""
    n_rings = len(widths_arr)
    hw = widths_arr * 0.5
    hh = heights_arr * 0.5
    c = np.minimum(hw, hh)
    u = np.zeros((n_rings, 6), dtype=np.float64)
    v_ = np.zeros((n_rings, 6), dtype=np.float64)
    u[:, 0] = +hw
    u[:, 1] = +(hw - c)
    u[:, 2] = -(hw - c)
    u[:, 3] = -hw
    u[:, 4] = -(hw - c)
    u[:, 5] = +(hw - c)
    v_[:, 0] = 0.0
    v_[:, 1] = +hh
    v_[:, 2] = +hh
    v_[:, 3] = 0.0
    v_[:, 4] = -hh
    v_[:, 5] = -hh
    return u, v_


U, V, T = constant_up_frames(spine)
CS_U, CS_V = cross_section_uv(widths, heights)

# Detect fold targets per strand. Each entry: (k, i, j, midpoint, color, name).
# Strand polylines are built from per-ring (u, v) so variable widths are
# handled — strand[i] = spine[i] + u[i, k]·U[i] + v[i, k]·V[i].
fold_pairs = []
for k in range(6):
    strand_3d = spine + CS_U[:, k : k + 1] * U + CS_V[:, k : k + 1] * V
    for i, j, mid, d in fold_targets(strand_3d, FOLD_THRESHOLD):
        fold_pairs.append((k, i, j, mid, VERT_COLORS[k], VERT_NAMES[k], d))


def build_collapsed_hex_tube(
    spine_3d, U_arr, V_arr, T_arr, widths_arr, heights_arr, ring_colors_uint32, snaps
):
    """Build a chamfered-hex tube mesh (with flat end caps) with snaps applied.

    snaps: iterable of (k, i, j, midpoint_xyz). For each entry, rings
    [i+1 .. j] inclusive on cross-section vertex k are overwritten to
    midpoint_xyz. Side index buffer is left intact; collapsed columns
    produce degenerate triangles that the GPU culls.

    End caps are flat triangle fans with their own vertex copies (so cap
    normals are axial, not the radial side normals).
    """
    n_rings = len(spine_3d)
    n_cs = 6

    side_pos = np.zeros((n_rings * n_cs, 3), dtype=np.float32)
    side_nrm = np.zeros((n_rings * n_cs, 3), dtype=np.float32)

    def hex_uv(hw, hh):
        c = min(hw, hh)
        return np.array(
            [
                [+hw, 0.0],
                [+hw - c, +hh],
                [-(hw - c), +hh],
                [-hw, 0.0],
                [-(hw - c), -hh],
                [+hw - c, -hh],
            ]
        )

    for i in range(n_rings):
        hw = float(widths_arr[i]) * 0.5
        hh = float(heights_arr[i]) * 0.5
        pts_uv = hex_uv(hw, hh)
        edges = np.roll(pts_uv, -1, axis=0) - pts_uv
        edge_n = np.column_stack([edges[:, 1], -edges[:, 0]])
        edge_n /= np.linalg.norm(edge_n, axis=1, keepdims=True) + 1e-12
        vn_uv = np.roll(edge_n, 1, axis=0) + edge_n
        vn_uv /= np.linalg.norm(vn_uv, axis=1, keepdims=True) + 1e-12

        Ui, Vi = U_arr[i], V_arr[i]
        for kk in range(n_cs):
            u, v_ = pts_uv[kk]
            side_pos[i * n_cs + kk] = spine_3d[i] + u * Ui + v_ * Vi
            nu, nv = vn_uv[kk]
            side_nrm[i * n_cs + kk] = nu * Ui + nv * Vi

    for k, i_lo, i_hi, mid in snaps:
        mid32 = np.asarray(mid, dtype=np.float32)
        for ring in range(i_lo + 1, i_hi + 1):
            side_pos[ring * n_cs + k] = mid32

    side_idx = np.zeros((n_rings - 1) * n_cs * 6, dtype=np.uint32)
    idx = 0
    for i in range(n_rings - 1):
        base_a = i * n_cs
        base_b = (i + 1) * n_cs
        for kk in range(n_cs):
            kn = (kk + 1) % n_cs
            v00 = base_a + kk
            v01 = base_a + kn
            v10 = base_b + kk
            v11 = base_b + kn
            side_idx[idx : idx + 6] = (v00, v10, v01, v01, v10, v11)
            idx += 6

    # End caps: 1 center + 6 rim duplicates, all sharing the cap normal.
    def cap_block(ring_i, axial_n, reverse_winding):
        rim_xyz = side_pos[ring_i * n_cs : (ring_i + 1) * n_cs].copy()
        center = spine_3d[ring_i].astype(np.float32)
        pos = np.vstack([center[None, :], rim_xyz])
        nrm = np.tile(axial_n.astype(np.float32), (n_cs + 1, 1))
        idx_local = np.zeros(n_cs * 3, dtype=np.uint32)
        for kk in range(n_cs):
            kn = (kk + 1) % n_cs
            if reverse_winding:
                idx_local[kk * 3 : kk * 3 + 3] = (0, 1 + kn, 1 + kk)
            else:
                idx_local[kk * 3 : kk * 3 + 3] = (0, 1 + kk, 1 + kn)
        return pos, nrm, idx_local

    cap0_pos, cap0_nrm, cap0_idx_local = cap_block(0, -T_arr[0], reverse_winding=True)
    cap1_pos, cap1_nrm, cap1_idx_local = cap_block(
        n_rings - 1, T_arr[-1], reverse_winding=False
    )

    n_side = side_pos.shape[0]
    cap0_offset = n_side
    cap1_offset = n_side + cap0_pos.shape[0]
    cap0_idx = cap0_idx_local + cap0_offset
    cap1_idx = cap1_idx_local + cap1_offset

    positions = np.vstack([side_pos, cap0_pos, cap1_pos])
    normals = np.vstack([side_nrm, cap0_nrm, cap1_nrm])
    indices = np.concatenate([side_idx, cap0_idx, cap1_idx])

    rgb = np.column_stack(
        [
            ((ring_colors_uint32 >> 16) & 0xFF) / 255.0,
            ((ring_colors_uint32 >> 8) & 0xFF) / 255.0,
            (ring_colors_uint32 & 0xFF) / 255.0,
        ]
    ).astype(np.float32)
    side_colors = np.repeat(rgb, n_cs, axis=0)
    cap0_colors = np.tile(rgb[0], (n_cs + 1, 1))
    cap1_colors = np.tile(rgb[-1], (n_cs + 1, 1))
    vert_colors = np.vstack([side_colors, cap0_colors, cap1_colors])
    return positions, normals, indices, vert_colors


# Three-tube layout. OFFSET picked from spine extent so the tubes don't kiss.
spine_x_extent = float(spine[:, 0].max() - spine[:, 0].min())
OFFSET = spine_x_extent + BEAD_W_MAX

v.add_group("baseline", position=[-OFFSET, 0.0, 0.0])
v.add_group("builtin", position=[0.0, 0.0, 0.0])
v.add_group("reference", position=[+OFFSET, 0.0, 0.0])

# LEFT: baseline parametric tube + fold-target markers.
v.add_parametric_tube(
    "tube_baseline",
    spine=spine,
    widths=widths,
    heights=heights,
    colors=colors,
    roughness=0.35,
    metalness=0.05,
    parent="baseline",
)

MARKER_R = 0.04
z_min, z_max = float(spine[:, 2].min()), float(spine[:, 2].max())
print(f"spine: {n} samples, open serpentine, Z range [{z_min:.2f}, {z_max:.2f}]")
print(f"width range: [{BEAD_W_MIN:.2f}, {BEAD_W_MAX:.2f}]   height: {BEAD_H:.2f}")
print(f"fold threshold: {FOLD_THRESHOLD}   detected folds: {len(fold_pairs)}")
for hit_idx, (k, i, j, mid, color, vert_name, d) in enumerate(fold_pairs):
    v.add_sphere(
        f"fold_{vert_name}_{hit_idx}",
        radius=MARKER_R,
        color=color,
        position=[float(mid[0]), float(mid[1]), float(mid[2])],
        parent="baseline",
    )
    print(
        f"  k={k} {vert_name:12s} segs=({i:2d},{j:2d})  d={d:.4f}  "
        f"snap rings [{i + 1}..{j}]  mid=({mid[0]:+.3f}, {mid[1]:+.3f}, {mid[2]:+.3f})"
    )

# MIDDLE: client-side strand_collapse=True — the viewer runs the snap pass.
v.add_parametric_tube(
    "tube_builtin",
    spine=spine,
    widths=widths,
    heights=heights,
    colors=colors,
    roughness=0.35,
    metalness=0.05,
    strand_collapse=True,
    parent="builtin",
)

# RIGHT: Python-built reference with the same snaps applied locally.
snaps = [(k, i, j, mid) for k, i, j, mid, _, _, _ in fold_pairs]
positions_c, normals_c, indices_c, colors_c = build_collapsed_hex_tube(
    spine, U, V, T, widths, heights, colors, snaps
)
v.add_mesh(
    "tube_reference",
    positions=positions_c,
    indices=indices_c,
    normals=normals_c,
    colors=colors_c,
    roughness=0.35,
    metalness=0.05,
    parent="reference",
)

print(
    "layout: left=baseline+markers, middle=client strand_collapse, right=python reference"
)
print(f"x offsets: {-OFFSET:+.3f} / 0.000 / {+OFFSET:+.3f}")
print("Ctrl+C to exit.")

try:
    while True:
        time.sleep(1.0)
except KeyboardInterrupt:
    pass
