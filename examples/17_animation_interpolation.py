"""
Animation Interpolation Demo

Demonstrates the two per-channel interpolation modes by comparing them directly.
A row of elongated capsules traces a Lissajous path while tumbling end-over-end,
sampled at only 3 Hz — low enough that hold playback visibly snaps between
frames.

- "linear" (default for transforms/opacity/draw_ranges/camera channels):
  translations lerp, rotations slerp, float channels lerp — smooth 60 fps
  playback from sparse data. This is what every other example in this repo uses.
- "hold" (default for colors/visibility/clip_times): each keyframe holds until
  the next. Use it on continuous channels for frame-accurate scientific /
  simulation replay or when intermediate values would be physically meaningless.

Interpolation is set per channel via `add_channel(interpolation=...)` or the
convenience wrappers (`set_transform_data`, `set_draw_range_data`, ...). The
example auto-alternates between HOLD and LINEAR on the same transforms channel
so the lerp/slerp effect is obvious. The timeline marker tells you which mode
is currently active.

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
    """Build a sparse (3 Hz) tumbling capsules animation with the given
    interpolation mode set explicitly on the transforms channel."""
    ids = [f"cap_{i}" for i in range(N_CAPSULES)]
    frame_times = np.arange(N_FRAMES, dtype=np.float64) / KEYFRAME_HZ

    data = np.zeros((N_FRAMES, N_CAPSULES, 16), dtype=np.float32)
    for k, t in enumerate(frame_times):
        for i in range(N_CAPSULES):
            phase = 2 * math.pi * i / N_CAPSULES
            x = 3.0 * math.sin(2 * math.pi * t / DURATION + phase)
            y = 2.0 * math.sin(4 * math.pi * t / DURATION + phase)
            z = 0.6 * math.sin(6 * math.pi * t / DURATION + phase) + 1.2
            spin = 2 * math.pi * t / DURATION * 2 + phase
            data[k, i] = trs_matrix([x, y, z], axis=(1, 1, 0), angle=spin)

    anim = Animation(loop=True)
    anim.set_frame_times(frame_times)
    anim.set_transform_data(ids, data, interpolation=mode)
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
print("Auto-alternating: HOLD (choppy) ⇄ LINEAR (smooth). Ctrl+C to stop.")

hold_anim = build_animation("hold")
linear_anim = build_animation("linear")

# Give the viewer a moment to create capsules before the first load.
# disconnect=False keeps the websocket open for the alternating loop below.
v.wait_for_assets(disconnect=False)

try:
    while True:
        print("  -> loading HOLD animation")
        v.load_animation(hold_anim)
        time.sleep(DURATION + 0.2)

        print("  -> loading LINEAR animation")
        v.load_animation(linear_anim)
        time.sleep(DURATION + 0.2)
except KeyboardInterrupt:
    print("\nStopped.")
