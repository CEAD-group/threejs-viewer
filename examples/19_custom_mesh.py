"""
Custom Mesh — server-side triangle mesh with draw_range animation

Demonstrates ``add_mesh``: building arbitrary triangle geometry in Python
and sending it to the viewer.  Unlike parametric tubes (which are built
client-side from a spine), ``add_mesh`` accepts raw positions, indices,
normals, and vertex colors — full control, no constraints on topology.

This example builds a procedural terrain heightmap, colors it by altitude,
and reveals it row-by-row with draw_range animation.

Run: uv run python examples/19_custom_mesh.py
"""

import numpy as np

from threejs_viewer import Animation, viewer


def make_terrain(nx=200, ny=200, size=10.0, height_scale=2.0):
    """Generate a procedural terrain mesh with vertex colors.

    Returns positions (N,3), indices (M,), normals (N,3), colors (N,3).
    """
    x = np.linspace(-size / 2, size / 2, nx, dtype=np.float32)
    y = np.linspace(-size / 2, size / 2, ny, dtype=np.float32)
    xg, yg = np.meshgrid(x, y)

    # Layered noise-like height from trig functions
    z = np.zeros_like(xg)
    for freq, amp in [(0.5, 1.0), (1.0, 0.5), (2.3, 0.25), (4.7, 0.12)]:
        z += amp * np.sin(freq * xg + 0.3) * np.cos(freq * yg * 1.1 + 0.7)
    z *= height_scale
    z += 0.4 * np.sin(xg * 0.7 + yg * 0.9) * np.cos(xg * 1.3 - yg * 0.6)

    positions = np.column_stack([xg.ravel(), yg.ravel(), z.ravel()]).astype(np.float32)

    # Triangle indices — two tris per quad, row-major for draw_range reveal
    indices = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = j * nx + i
            v10 = v00 + 1
            v01 = v00 + nx
            v11 = v01 + 1
            indices.extend([v00, v01, v10, v10, v01, v11])
    indices = np.array(indices, dtype=np.uint32)

    # Per-vertex normals from cross products of adjacent edges
    normals = np.zeros_like(positions)
    for tri_start in range(0, len(indices), 3):
        i0, i1, i2 = indices[tri_start : tri_start + 3]
        e1 = positions[i1] - positions[i0]
        e2 = positions[i2] - positions[i0]
        n = np.cross(e1, e2)
        normals[i0] += n
        normals[i1] += n
        normals[i2] += n
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norms, 1e-10)

    # Color by altitude — green valleys, brown midlands, white peaks
    z_flat = z.ravel()
    z_min, z_max = z_flat.min(), z_flat.max()
    t = (z_flat - z_min) / (z_max - z_min + 1e-10)

    colors = np.zeros((len(t), 3), dtype=np.float32)
    # Green (low) -> brown (mid) -> white (high)
    # 0..0.4: green to brown
    mask_low = t < 0.4
    t_low = t[mask_low] / 0.4
    colors[mask_low, 0] = 0.15 + 0.45 * t_low
    colors[mask_low, 1] = 0.5 - 0.2 * t_low
    colors[mask_low, 2] = 0.1

    # 0.4..0.7: brown to grey
    mask_mid = (t >= 0.4) & (t < 0.7)
    t_mid = (t[mask_mid] - 0.4) / 0.3
    colors[mask_mid, 0] = 0.6 - 0.1 * t_mid
    colors[mask_mid, 1] = 0.3 + 0.2 * t_mid
    colors[mask_mid, 2] = 0.1 + 0.3 * t_mid

    # 0.7..1.0: grey to white
    mask_high = t >= 0.7
    t_high = (t[mask_high] - 0.7) / 0.3
    colors[mask_high, 0] = 0.5 + 0.5 * t_high
    colors[mask_high, 1] = 0.5 + 0.5 * t_high
    colors[mask_high, 2] = 0.4 + 0.6 * t_high

    return positions, indices, normals, colors, ny


v = viewer()
v.clear()

positions, indices, normals, colors, ny = make_terrain()

v.add_mesh(
    "terrain",
    positions=positions,
    indices=indices,
    normals=normals,
    colors=colors,
    roughness=0.8,
    metalness=0.05,
)

# Draw-range animation: reveal row by row
n_frames = 300
frame_times = np.linspace(0, 8.0, n_frames, dtype=np.float32)
draw_fracs = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)

anim = Animation(loop=True)
anim.set_frame_times(frame_times)
anim.set_draw_range_data(["terrain"], draw_fracs.reshape(n_frames, 1))
v.load_animation(anim)

print(f"Terrain: {len(positions)} vertices, {len(indices) // 3} triangles")
print("Draw-range reveals the mesh row by row.")
print("Waiting for browser...")
v.wait_for_assets()
print("Done.")
