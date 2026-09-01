"""
Live point streaming — append_points on a growing cloud

Demonstrates ``append_points``: a cloud seeded once with ``add_points`` and
then grown chunk by chunk, the way a laser tracker or scanner feeds one. Only
the new points cross the wire, and the viewer writes them into spare capacity
at the tail of the GPU buffers — so an update costs O(new points), not
O(total). Re-sending the whole cloud with ``add_points`` instead is quadratic
in the total and stalls the browser well before a million points.

The simulated producer traces a slowly drifting Lissajous "probe path" with
sensor noise, flushing ~2500 points a few times a second. Colour is the
per-point speed through a turbo ramp, with **cmin/cmax fixed at seed time**:
appended chunks map through that same range (values past it clamp) rather than
rescaling, which would mean re-colouring every point already in the cloud.

Runs until interrupted. Ctrl-C to stop.

Run: uv run python examples/30_live_points_stream.py
"""

import time

import numpy as np

from threejs_viewer import viewer

CHUNK = 2500  # points per flush
FLUSH_HZ = 4.0  # flushes per second
SPEED_MIN, SPEED_MAX = 0.0, 2.5  # colour range, fixed for the whole run

v = viewer()
v.clear()
v.unload_animation()

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
    size=2.0,
)

total = CHUNK
print(
    f"Seeded 'live' with {total:,} points. Appending {CHUNK} every "
    f"{1 / FLUSH_HZ:.2f}s — Ctrl-C to stop."
)

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
