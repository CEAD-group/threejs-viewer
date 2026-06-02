"""
Depth cues for a flat line drawing — fog + eye-dome lighting.

When a toolpath is too complex to extrude as a shaded bead you fall back to a
flat polyline, which loses every depth cue: a single line through a tangled,
self-overlapping path reads as a flat ball of yarn. This example draws exactly
that — a tall ripply spiral-vase toolpath as one polyline — and restores depth
with the combination that works best for dense bundles: distance **fog** plus
screen-space **eye-dome lighting** (EDL).

  * Fog dims distant windings toward black (classic CAD depth cueing), so the
    eye reads the front-to-back ordering of the bundle as a whole.
  * EDL darkens each fragment that sits behind its screen-space neighbours,
    sculpting local crossings into legible 3D.

Both cues are **scoped to polylines only**. The scene also has a solid build
platform + corner clamps (meshes) as spatial reference, and they render exactly
the same with the cues on or off — fog never dims them, EDL never darkens them,
and the dark background stays the same colour. Toggle the cues to confirm: only
the toolpath line changes. (This is what makes the cues usable in a real machine
view, where the cell and fixtures must stay legible while the line gets depth.)

Two implementation notes:

  * **Native line (`fat=False`)** — at a million points the fat-line renderer
    (`Line2`, one instanced quad per segment) gets heavy. ``add_polyline(...,
    fat=False)`` draws a native ``THREE.Line`` instead: one vertex per point,
    one draw call, ~1px wide. Fog and EDL need no line width, so the only cost
    is fixed 1px lines — a good trade for million-point toolpaths.
  * **Rainbow colour** — a ``turbo`` colormap runs along the path (start →
    end) so colour encodes print progression; the viewer interpolates it along
    each segment. (Per-vertex colour works the same on native and fat lines.)

The viewer opens with ``set_depth_cue(fog=True, edl=True)`` already applied — no
keys needed. Press **D** to toggle fog and **Shift+D** to toggle EDL, to compare
each cue on its own.

Run: uv run python examples/20_line_depth_cues.py
"""

import numpy as np

from threejs_viewer import viewer

N_POINTS = 1_000_000
N_TURNS = 1000  # number of stacked layers (windings)
HEIGHT = 8.0


def vase_toolpath(n_points=N_POINTS, n_turns=N_TURNS, height=HEIGHT):
    """A ripply spiral-vase toolpath: ``n_turns`` windings stacked into a vase
    silhouette, with multi-frequency angular ripples so the line still overlaps
    itself front-to-back. Bump ``N_TURNS`` for a denser ball-of-yarn worst case.

    Returns the (N, 3) points and the 0→1 progress parameter for colouring."""
    t = np.linspace(0.0, 1.0, n_points)
    angle = t * n_turns * 2.0 * np.pi

    # Vase silhouette: wide base → waist → belly → flared rim.
    profile = (
        2.6
        - 1.1 * np.sin(t * np.pi) ** 2  # pinch the waist
        + 1.4 * np.sin(t * np.pi * 0.5) ** 3  # swell the belly / rim
    )
    # Per-winding angular ripple at two frequencies so the single line
    # overlaps itself heavily front-to-back (extra crossings to disambiguate).
    radius = profile + 0.16 * np.sin(angle * 7.0) + 0.07 * np.sin(angle * 23.0)

    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = t * height - height / 2.0  # centre on the origin for nice rotation
    return np.column_stack([x, y, z]).astype(np.float32), t


v = viewer()
v.clear()
v.unload_animation()

points, progress = vase_toolpath()
# Rainbow along the toolpath (turbo colormap, start → end), drawn as a native
# THREE.Line so a million points stay fast. fat=False fixes the width at ~1px,
# which fog + EDL don't care about.
v.add_polyline("toolpath", points, colors=progress, colormap="turbo", fat=False)

# Solid mesh fixtures as spatial reference — these stay fully lit with the cues
# on (fog/EDL are scoped to polylines), so they prove the line is the only thing
# being depth-cued. A brushed-steel build platform just under the toolpath base,
# plus four corner clamps holding it down.
plate_z = -HEIGHT / 2.0 - 0.2
v.add_box(
    "build_plate",
    width=8.0,
    height=8.0,
    depth=0.3,
    color=0x9AA3AD,
    roughness=0.5,
    metalness=0.6,
    position=[0.0, 0.0, plate_z],
)
for sx in (-1, 1):
    for sy in (-1, 1):
        v.add_box(
            f"clamp_{sx}_{sy}",
            width=0.9,
            height=0.9,
            depth=0.7,
            color=0x33373D,
            roughness=0.7,
            metalness=0.3,
            position=[sx * 3.3, sy * 3.3, plate_z + 0.35],
        )

# Open with fog + eye-dome lighting already on (the programmatic equivalent of
# pressing D then Shift+D). The browser keeps this after the Python process exits.
v.set_depth_cue(fog=True, edl=True)

print(__doc__)
print(f"Loaded a {len(points):,}-point rainbow toolpath (native line, fog + EDL)")
print("plus a steel build platform + clamps that the cues leave untouched.")
print("Focus the viewer and press F to frame it. D toggles fog, Shift+D toggles EDL.")

# Block until the browser has fetched the polyline, then disconnect cleanly.
v.wait_for_assets()
