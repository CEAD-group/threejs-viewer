"""
Mesh Vertex Alpha — custom triangle mesh with per-vertex RGBA colors

Demonstrates ``add_mesh`` with per-vertex colors containing varying alpha values.
Pass an (N, 4) float32 array in [0, 1] to ``colors`` to give each vertex its own
RGB color and alpha transparency.

This example creates a plane grid where the RGB color forms a smooth gradient
and the alpha channel varies radially from fully opaque at the center to
completely transparent at the edges, placed over a background grid and box
so the transparency is clearly visible.

Run: uv run python examples/33_mesh_vertex_alpha.py
"""

import numpy as np

from threejs_viewer import viewer


def make_alpha_plane(nx=60, ny=60, size=8.0):
    """Generate a plane mesh with per-vertex RGBA colors.

    Returns positions (N,3), indices (M,), normals (N,3), colors (N,4).
    """
    x = np.linspace(-size / 2, size / 2, nx, dtype=np.float32)
    y = np.linspace(-size / 2, size / 2, ny, dtype=np.float32)
    xg, yg = np.meshgrid(x, y)

    # Place slightly above z=0 to avoid z-fighting with the floor grid
    z = np.full_like(xg, 0.05)
    positions = np.column_stack([xg.ravel(), yg.ravel(), z.ravel()]).astype(np.float32)

    # Triangle indices — CCW winding so normals point UP (+Z)
    indices = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = j * nx + i
            v10 = v00 + 1
            v01 = v00 + nx
            v11 = v01 + 1
            indices.extend([v00, v10, v01, v10, v11, v01])
    indices = np.array(indices, dtype=np.uint32)

    # Explicit +Z normals so lighting reflects upward toward the camera
    normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(positions), 1))

    # Per-vertex RGBA colors
    # R varies along X, G varies along Y, B is constant
    r = (xg.ravel() - x.min()) / (x.max() - x.min())
    g = (yg.ravel() - y.min()) / (y.max() - y.min())
    b = np.full_like(r, 0.8)

    # Alpha varies radially from center (1.0) to edge (0.0)
    dist_from_center = np.sqrt(xg.ravel() ** 2 + yg.ravel() ** 2)
    max_radius = size / 2.0
    alpha = np.clip(1.0 - (dist_from_center / max_radius) ** 1.5, 0.0, 1.0)

    colors = np.column_stack([r, g, b, alpha]).astype(np.float32)
    return positions, indices, normals, colors


if __name__ == "__main__":
    v = viewer()
    v.clear()

    v.add_box(
        "checker_box", width=4, height=4, depth=1, position=[0, 0, -0.6], color=0xFF5500
    )

    positions, indices, normals, colors = make_alpha_plane(nx=60, ny=60, size=8.0)

    v.add_mesh(
        "alpha_plane",
        positions=positions,
        indices=indices,
        normals=normals,
        colors=colors,
        roughness=0.3,
        metalness=0.1,
    )

    print("Created plane mesh with varying vertex alpha (RGBA).")
    print("Waiting for browser...")
    v.wait_for_assets()
