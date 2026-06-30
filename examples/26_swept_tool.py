"""
Swept 5-axis tool body (shank/holder) — add_swept_tool

The cutting *tip* swept volume is covered by add_parametric_tube (a tube of the
tool radius along the contact path). What that can't show is the **tool body**:
the shank/holder extending from the cutting point **along the tool axis**, which
on a 5-axis machine is generally NOT the path tangent.

``add_swept_tool`` lofts a surface-of-revolution profile about the per-station
*tool axis* and sweeps it along the path — visualizing the swept shank as the
tool tilts. Here the body is coloured by **reorientation rate** (the 5-axis
"wrist speed": angle between consecutive tool axes per unit travel) — red where
the rotary axes swing fastest. A draw_range animation advances the tool along
the path.

Run: uv run python examples/26_swept_tool.py
"""

import numpy as np

from threejs_viewer import Animation, viewer


def heatmap(t):
    """Map t in [0,1] to packed 0x00RRGGBB: blue -> cyan -> green -> yellow -> red."""
    stops = np.array(
        [
            [0.00, 0x22, 0x44, 0xFF],
            [0.25, 0x22, 0xCC, 0xCC],
            [0.50, 0x44, 0xCC, 0x44],
            [0.75, 0xFF, 0xCC, 0x22],
            [1.00, 0xFF, 0x33, 0x33],
        ]
    )
    r = np.interp(t, stops[:, 0], stops[:, 1]).astype(np.uint32)
    g = np.interp(t, stops[:, 0], stops[:, 2]).astype(np.uint32)
    b = np.interp(t, stops[:, 0], stops[:, 3]).astype(np.uint32)
    return (r << 16) | (g << 8) | b


v = viewer()
v.clear()
v.unload_animation()

# --- A 5-axis contact path with a tilting tool axis ---
N = 240
t = np.linspace(0, 1, N)
x = np.linspace(-7, 7, N)
y = 1.5 * np.sin(t * 2 * np.pi * 1.5)
z = 0.4 * np.cos(t * 2 * np.pi * 2.0)
positions = np.column_stack([x, y, z]).astype(np.float32)

# Tool axis (tip -> holder) leans back and forth and rolls — uneven wrist speed.
lean = 0.7 * np.sin(t * 2 * np.pi * 2.0)
ax = np.sin(lean)
ay = 0.35 * np.sin(t * 2 * np.pi * 3.0)
az = np.cos(lean)
axes = np.column_stack([ax, ay, az]).astype(np.float32)

# --- Tool profile: ball nose (radius R) + shank (radius rs, length L) ---
R, rs, L = 0.6, 0.5, 6.0
ball_h = np.linspace(0.0, R, 7)
ball_r = np.sqrt(np.maximum(R**2 - (R - ball_h) ** 2, 0.0))
profile = np.array(
    [*zip(ball_h, ball_r), (R, rs), (R + L, rs)],
    dtype=np.float32,
)

# --- Colour by reorientation rate (deg of tool-axis swing per unit travel) ---
unit = axes / np.linalg.norm(axes, axis=1, keepdims=True)
dot = np.clip(np.sum(unit[1:] * unit[:-1], axis=1), -1.0, 1.0)
dist = np.linalg.norm(np.diff(positions, axis=0), axis=1)
rate = np.zeros(N, dtype=np.float32)
rate[1:] = np.degrees(np.arccos(dot)) / np.maximum(dist, 1e-6)
rate_n = rate / (np.percentile(rate, 98) + 1e-9)
colors = heatmap(np.clip(rate_n, 0, 1))

v.add_swept_tool(
    "shank",
    positions,
    axes,
    profile,
    colors=colors,
    sections=20,
    opacity=0.9,
    metalness=0.4,
    roughness=0.5,
)

# A thin line marking the cutting-contact path for reference.
v.add_polyline("contact", positions, color=0x222222, line_width=2, fat=False)

# --- Reveal the swept body progressively along the path ---
n_frames = 200
anim = Animation(loop=True)
anim.set_frame_times(np.linspace(0, 6.0, n_frames, dtype=np.float32))
anim.set_draw_range_data(
    ["shank"], np.linspace(0, 1, n_frames, dtype=np.float32).reshape(n_frames, 1)
)
v.load_animation(anim)

print(f"Swept 5-axis tool body: {N} stations, ball+shank profile.")
print("Coloured by reorientation rate (blue calm -> red fast wrist motion).")
print("draw_range advances the tool along the path.")
v.wait_for_assets()
