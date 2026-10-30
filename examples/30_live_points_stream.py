"""
Live point streaming with ``append_points``.

A cloud is seeded once with ``add_points`` and then grown by ~2500 points four times a
second, the way a laser tracker would feed it. Only the new points cross the wire, so
an update costs O(new points) instead of O(total). Colour is per-point speed with
``cmin``/``cmax`` fixed at seed time; later chunks clamp to that range instead of
rescaling (which would recolour every existing point).

Run: uv run python examples/30_live_points_stream.py   (Ctrl+C to stop)
"""

import time

import numpy as np

from threejs_viewer import viewer

CHUNK = 2500  # points per flush
FLUSH_HZ = 4.0  # flushes per second
SPEED_MIN, SPEED_MAX = 0.0, 2.5  # colour range, fixed for the whole run

v = viewer()
v.clear()

rng = np.random.default_rng(0)


def produce(t0: float, n: int):
    """One chunk of a noisy Lissajous probe path starting at time t0."""
    t = t0 + np.arange(n, dtype=np.float32) * 0.002
    x = 4.0 * np.sin(1.1 * t)
    y = 4.0 * np.sin(1.7 * t + 0.6)
    z = 1.5 * np.sin(0.23 * t) + 0.4 * np.sin(5.0 * t)
    pts = np.column_stack([x, y, z]).astype(np.float32)
    pts += rng.normal(0.0, 0.02, pts.shape).astype(np.float32)
    # Per-point speed (mm/sample scaled): the scalar we colour by.
    speed = np.gradient(np.linalg.norm(pts, axis=1)) * 100.0
    return pts, np.abs(speed).astype(np.float32)


# --- Seed the cloud (this is the only full upload) ---
t = 0.0
pts, speed = produce(t, CHUNK)
t += CHUNK * 0.002
v.add_points(
    "live",
    pts,
    colors=speed,
    colormap="turbo",
    cmin=SPEED_MIN,
    cmax=SPEED_MAX,  # fixed here — appends clamp to it, never rescale
    size=0.04,
)

v.add_grid(
    "floor", cell_size=1.0, extent=30.0, center_color=0x888888, position=[0, 0, -2.5]
)
v.set_view("iso", animate=False)
v.frame_points(
    np.array(np.meshgrid(*zip([-4, -4, -2], [4, 4, 2]))).reshape(3, -1).T, fill=0.85
)
total = CHUNK

try:
    while True:
        time.sleep(1.0 / FLUSH_HZ)
        pts, speed = produce(t, CHUNK)
        t += CHUNK * 0.002
        v.append_points("live", pts, colors=speed)
        total += CHUNK
        print(f"\r{total:,} points", end="", flush=True)
except KeyboardInterrupt:
    print(f"\nStopped at {total:,} points.")
finally:
    v.disconnect()
