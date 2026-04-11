"""
Animation Interpolation Demo

Demonstrates the two Animation interpolation modes by comparing them directly.
A row of elongated capsules traces a Lissajous path while tumbling end-over-end,
sampled at only 3 Hz — low enough that step playback visibly snaps between
frames.

- "linear" (default): translations lerp, rotations slerp, float channels
  (draw_ranges, opacity) lerp — smooth 60 fps playback from sparse data.
  This is the mode used by every other example in this repo.
- "step": each keyframe holds until the next. Use it for frame-accurate
  scientific/simulation replay or when intermediate values would be
  physically meaningless.

The example cannot show both modes simultaneously (a single Animation has one
interpolation setting), so it auto-alternates: it loads the same sparse
animation with `interpolation="step"`, plays one full loop, then reloads with
`interpolation="linear"` and plays another loop — forever. The timeline
marker label tells you which mode is currently active.

Run: uv run python examples/17_animation_interpolation.py
"""

import math
import time

import numpy as np

from threejs_viewer import Animation, viewer

N_CAPSULES = 10
DURATION = 6.0
KEYFRAME_HZ = 3  # deliberately sparse — 19 keyframes over 6 seconds
N_FRAMES = int(DURATION * KEYFRAME_HZ) + 1


def trs_matrix(pos, axis, angle):
    """Build a column-major 4x4 TRS matrix from position + axis-angle rotation."""
    ax, ay, az = axis
    norm = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
    ax, ay, az = ax / norm, ay / norm, az / norm
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    r = np.array(
        [
            [c + ax * ax * C, ax * ay * C - az * s, ax * az * C + ay * s],
            [ay * ax * C + az * s, c + ay * ay * C, ay * az * C - ax * s],
            [az * ax * C - ay * s, az * ay * C + ax * s, c + az * az * C],
        ],
        dtype=np.float32,
    )
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = r
    m[:3, 3] = pos
    return m.T.flatten().tolist()  # column-major for Three.js


def build_animation(mode: str) -> Animation:
    anim = Animation(loop=True, interpolation=mode)
    for k in range(N_FRAMES):
        t = k / KEYFRAME_HZ
        transforms = {}
        for i in range(N_CAPSULES):
            phase = 2 * math.pi * i / N_CAPSULES
            x = 3.0 * math.sin(2 * math.pi * t / DURATION + phase)
            y = 2.0 * math.sin(4 * math.pi * t / DURATION + phase)
            z = 0.6 * math.sin(6 * math.pi * t / DURATION + phase) + 1.2
            # Tumble end-over-end around a tilted axis; each capsule offset in phase.
            spin = 2 * math.pi * t / DURATION * 2 + phase
            transforms[f"cap_{i}"] = trs_matrix([x, y, z], axis=(1, 1, 0), angle=spin)
        anim.add_frame(time=t, transforms=transforms)
    color = 0x33FF88 if mode == "linear" else 0xFF5544
    anim.add_marker(0.0, f"interpolation = {mode.upper()}", color=color)
    return anim


v = viewer()
v.clear()

colors = [
    0xFF3355,
    0xFF8833,
    0xFFDD33,
    0x88DD33,
    0x33DD88,
    0x33DDDD,
    0x3388FF,
    0x5533FF,
    0xAA33FF,
    0xFF33AA,
]
for i in range(N_CAPSULES):
    v.add_capsule(
        f"cap_{i}",
        radius=0.18,
        length=0.8,
        color=colors[i % len(colors)],
        roughness=0.35,
        metalness=0.2,
    )

print(
    f"Built sparse animation: {N_FRAMES} keyframes at {KEYFRAME_HZ} Hz "
    f"over {DURATION:.1f}s."
)
print("Auto-alternating: STEP (choppy) ⇄ LINEAR (smooth). Ctrl+C to stop.")

step_anim = build_animation("step")
linear_anim = build_animation("linear")

# Give the viewer a moment to create capsules before the first load.
# disconnect=False keeps the websocket open for the alternating loop below.
v.wait_for_assets(disconnect=False)

try:
    while True:
        print("  -> loading STEP animation")
        v.load_animation(step_anim)
        time.sleep(DURATION + 0.2)

        print("  -> loading LINEAR animation")
        v.load_animation(linear_anim)
        time.sleep(DURATION + 0.2)
except KeyboardInterrupt:
    print("\nStopped.")
