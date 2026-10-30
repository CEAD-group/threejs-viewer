"""
Million-point toolpath: six twisted vases printed one after another.

One ``add_toolpath`` with about 1.2M spine points: each vase is a single
spiral-vase bead (tube LOD switches on above 25k points, round joins), the
travel hops between vases are drawn as one line, the bead is coloured by wall
lean (overhang), and a nozzle rides the frontier during playback.

Run: uv run python examples/40_million_point_toolpath.py [points_per_vase]
(default 200000, so 1.2M points; 500000 gives 3M)
"""

import sys
import time

import numpy as np

from threejs_viewer import Animation, Toolpath, viewer

PER_VASE = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
LAYER_H, BEAD_W = 0.6, 1.6  # mm
PLAYBACK_S = 60.0  # the whole print, compressed
KEY_STRIDE = 8  # animation and follow-path keys every 8th spine point

# (x, y, base radius, height, lobes, lobe depth, twist turns over the height)
VASES = [
    (-85, 40, 22, 140, 5, 0.12, 0.6),
    (0, 40, 18, 110, 7, 0.10, -0.8),
    (85, 40, 24, 160, 3, 0.18, 0.4),
    (-85, -40, 16, 90, 9, 0.08, 1.2),
    (0, -40, 25, 125, 6, 0.14, -0.5),
    (85, -40, 19, 100, 4, 0.16, 0.9),
]


def vase(cx, cy, r0, height, lobes, depth, twist, n):
    """Spiral-vase wall (a lobed section that bulges, necks and twists) and
    its outward lean in degrees from vertical, d(radius)/d(height) at fixed angle."""
    theta = np.linspace(0.0, height / LAYER_H * 2 * np.pi, n)
    u = theta / theta[-1]  # 0 at the bed, 1 at the rim
    prof = 1.0 + 0.35 * np.sin(np.pi * u) - 0.2 * np.sin(2.4 * np.pi * u) ** 2
    phase = lobes * (theta + 2 * np.pi * twist * u)
    r = r0 * prof * (1.0 + depth * np.cos(phase))
    dr_du = r0 * (
        np.gradient(prof, u) * (1.0 + depth * np.cos(phase))
        - prof * depth * np.sin(phase) * lobes * 2 * np.pi * twist
    )
    xyz = np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta), u * height])
    return xyz + [0, 0, LAYER_H], np.degrees(np.arctan(dr_du / height))


t0 = time.perf_counter()
pts, lean, ext = [], [], []
for i, spec in enumerate(VASES):
    p, a = vase(*spec, PER_VASE)
    if i:  # travel hop: lift clear of the tallest vase, cross over, drop in
        hop_z = max(v[3] for v in VASES) + 10
        pts.append(np.array([[*pts[-1][-1, :2], hop_z], [*p[0, :2], hop_z]]))
        lean.append(np.zeros(2))
        ext.append(np.zeros(2, bool))
    pts += [p]
    lean += [a]
    ext += [np.ones(len(p), bool)]
pts, lean, ext = np.concatenate(pts), np.concatenate(lean), np.concatenate(ext)

# Feed rate: 60 mm/s on a vertical wall, slowing to 20 on a 45 degree overhang;
# travel at 200 mm/s. The playback clock follows the resulting print time.
feed = np.where(ext, 60.0 - 40.0 * np.minimum(np.abs(lean) / 45.0, 1.0), 200.0)
seg = np.linalg.norm(np.diff(pts, axis=0, prepend=pts[:1]), axis=1)
print_s = np.cumsum(seg / feed)
times = print_s * (PLAYBACK_S / print_s[-1])
w = np.where(ext, BEAD_W, 0.0)
tp = Toolpath(np.column_stack([times, pts, w, np.where(ext, LAYER_H, 0.0)]))
tp.colorize(np.clip(lean, -45, 45), colormap="turbo")
t_build = time.perf_counter() - t0

v = viewer()
v.clear()
t0 = time.perf_counter()
v.add_grid("floor", cell_size=10, extent=400, color=0x444444, center_color=0x777777)
v.add_toolpath(
    "part",
    tp,
    travel="line",
    travel_color=0x4FC3F7,
    joins="round",
    anchor="top",
    roughness=0.8,
    metalness=0.0,
)

# Nozzle: a group the follow path aims along +Z, with a Y-up cone stood upright.
v.add_group("nozzle")
v.add_cylinder(
    "nozzle_body",
    radius_top=6,
    radius_bottom=1.2,
    height=14,
    color=0xC0C4CC,
    metalness=0.9,
    roughness=0.25,
    position=[0, 0, 7.5],
    rotation=[np.pi / 2, 0, 0],
    parent="nozzle",
)
keys = np.r_[np.arange(0, len(pts), KEY_STRIDE), len(pts) - 1]
v.set_follow_path(
    "nozzle", times[keys], pts[keys], np.tile([0.0, 0.0, 1.0], (len(keys), 1))
)

anim = Animation(loop=True)
anim.set_frame_times(times[keys])
anim.set_draw_range_data(["part"], (keys / (len(pts) - 1))[:, None])
v.load_animation(anim)

v.add_menu(
    "legend",
    mode="panel",
    label="Toolpath",
    items=[
        {"type": "label", "label": "Wall lean: -45 (in) to +45 deg (out)"},
        {"type": "eye", "id": "travel", "label": "Travel hops", "ids": ["part_travel"]},
        {"type": "eye", "id": "nozzle", "label": "Nozzle", "ids": ["nozzle"]},
    ],
)
v.set_camera(position=[230, -370, 220], target=[0, 0, 65], up=[0, 0, 1])
v.wait_for_assets()
t_load = time.perf_counter() - t0

hours = print_s[-1] / 3600
print(
    f"{len(pts):,} spine points, {seg.sum() / 1000:.0f} m of bead, {hours:.1f} h print"
)
print(f"build {t_build:.2f} s, transfer + browser load {t_load:.2f} s")
