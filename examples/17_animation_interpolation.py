"""
Animation interpolation

Compares the two per-channel interpolation modes on the same data: capsules
tumble along Lissajous paths, sampled at only 3 Hz.

- "linear" (the default for every binary channel): translations lerp,
  rotations slerp, so sparse keyframes play back smoothly.
- "hold": each keyframe is kept until the next, which snaps visibly here but
  suits stepwise or frame-accurate data.

The script alternates the two on the transforms channel, and the timeline
marker names the active mode. Press Ctrl+C to stop.

Run: uv run python examples/17_animation_interpolation.py
"""

import colorsys
import time

import numpy as np

from threejs_viewer import Animation, viewer

N, DURATION, HZ = 12, 6.0, 3  # 19 keyframes over 6 seconds

v = viewer()
v.clear()
v.add_grid("floor", cell_size=1.0, extent=30, center_color=0x888888)
ids = [f"cap_{i}" for i in range(N)]
for i, id in enumerate(ids):
    r, g, b = (int(c * 255) for c in colorsys.hsv_to_rgb(i / N, 0.8, 1.0))
    v.add_capsule(
        id, radius=0.18, length=0.8, color=(r << 16) | (g << 8) | b, roughness=0.35
    )

# Keyframes: positions on a Lissajous path, tumbling end over end (Euler X then Y)
times = np.arange(int(DURATION * HZ) + 1) / HZ
phase = 2 * np.pi * np.arange(N) / N
u = 2 * np.pi * times[:, None] / DURATION
x, y, z = (
    3 * np.sin(u + phase),
    2 * np.sin(2 * u + phase),
    0.6 * np.sin(3 * u + phase) + 1.2,
)
cx, sx = np.cos(2 * u + phase), np.sin(2 * u + phase)
cy, sy = np.cos(u + phase), np.sin(u + phase)
zero = np.zeros_like(cx)
data = np.stack(
    [cy, zero, -sy, zero, sx * sy, cx, sx * cy, zero, cx * sy, -sx, cx * cy, zero, x, y, z, zero + 1],
    axis=-1,
).astype(np.float32)  # fmt: skip


def build_animation(mode: str) -> Animation:
    anim = Animation(loop=True)
    anim.set_frame_times(times)
    anim.set_transform_data(ids, data, interpolation=mode)
    anim.add_marker(
        0.0,
        f"interpolation = {mode.upper()}",
        color=0x33FF88 if mode == "linear" else 0xFF5544,
    )
    return anim


animations = {mode: build_animation(mode) for mode in ("hold", "linear")}
v.set_camera(position=[0, -9, 6], target=[0, 0, 1.2])
v.wait_for_assets(disconnect=False)  # keep the socket open for the loop below

print("Alternating HOLD (choppy) and LINEAR (smooth). Ctrl+C to stop.")
try:
    while True:
        for mode, anim in animations.items():
            print(f"  -> {mode}")
            v.load_animation(anim, restart=True)
            time.sleep(DURATION + 0.2)
except KeyboardInterrupt:
    v.disconnect()
