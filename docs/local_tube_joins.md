# Bounded local tube joins

Use `add_parametric_tube(..., surface="local_join")` to opt in. The default ring renderer is unchanged. Color, normal, and roughness maps are supported, including the existing wrapped path-distance UVs. This is a best-effort surface renderer, not a watertight solid-union operation across an entire print.

## Construction

- Smooth bends whose miter displacement is below the tolerance retain their original geometry. The default tolerance is local width / 100.
- An isolated corner trims its inner offsets and samples its outer arc. Its section is transported around the actual 3D bend axis. It replaces two ring pairs, reuses the exact boundary rings, and costs at most 96 triangles.
- A short almost-planar jog or reversal can become one bounded cluster, up to ten segments. Analytic transverse slices of the segment capsules give the outer and top outlines. Each sample retains the six-vertex chamfered section. Disconnected slices, incompatible interfaces, large section changes, and over-budget patches fall back to the original strip. There is no polygon clipping or general-purpose Boolean dependency.
- Cluster planarity is local and tolerant: residual rise after subtracting the endpoint slope may be up to 10% of bead height plus 0.2% of width. Actual elevations are interpolated back into the mesh. A spiral vase does not need to fit a single plane.
- Clusters have at most 192 triangles and may add at most 96 over the replaced strip. The whole object has an additional-triangle budget of `max(96, ceil(original_triangles * 0.15))`. Once exhausted, later candidates retain the original strip.

The isolated-arc tolerance is a chord tolerance. Cluster sample spacing is a geometric heuristic, **not a certified Hausdorff-error bound**. Other layers and distant self-intersections are not unioned. The spatial-jog example deliberately retains an unsupported tight bend; adjacent repairs are not allowed to overlap. Large changes of section or near-vertical incoming frames can also fall back.

## Playback, storage, and maps

Construction is a single linear scan with bounded work per candidate. Accepted patches are built once. Completed geometry uses a static index buffer and shares the original attribute buffers with the existing frontier renderer. A binary search selects the completed prefix during playback and backward scrubbing. No completed triangles are rebuilt each frame.

A patch becomes visible after its entire window is behind the frontier. Until then, that bounded region uses the existing animated strip and cap. This can produce a visible transition: the unresolved region is up to ten segments long, not necessarily completely hidden by the nozzle overlay. New joins are not progressively Boolean-unioned while the head is inside their window.

Outside patches, positions and UVs are unchanged. Within isolated joins, texture distance advances around the outer arc and the inner trim shares one coordinate. Within clusters, distance is interpolated monotonically across the window; this deliberately smooths texture flow through small reversals and may stretch the material. The original dual-wrapped UV representation is retained.

Objects with repairs add a second index buffer and a second mesh for the completed prefix, plus bounded patch vertices. Objects with no repairs use the original geometry directly and allocate no additional GPU geometry. LOD and strand collapse are disabled for this opt-in mode. Explicit orientations and per-station colors are currently unsupported; uniform color and all three texture maps work. Runtime per-station color updates are rejected.

## Captured results

All captures use the actual viewer. Left is the original strip with strand collapse off; right is local joins. Rows show solid shading, wire overlay, and color + normal + roughness maps.

| Case | Original triangles | Result | Comparison |
|---|---:|---:|---|
| #277 tight S-jog | 540 | 540 | [Screenshot](media/local_joins/issue_277_comparison.png) |
| #278 reversal | 540 | 504 | [Screenshot](media/local_joins/issue_278_comparison.png) |
| Planar 90° corner | 264 | 288 | [Screenshot](media/local_joins/planar_corner_comparison.png) |
| Tilted 90° corner | 264 | 288 | [Screenshot](media/local_joins/tilted_corner_comparison.png) |
| Vertical elbow | 264 | 280 | [Screenshot](media/local_joins/vertical_elbow_comparison.png) |
| Spatial jog | 300 | 336 | [Screenshot](media/local_joins/spatial_jog_comparison.png) |
| Spatial zigzag | 312 | 366 | [Screenshot](media/local_joins/spatial_zigzag_comparison.png) |
| 10,001-station spiral vase | 120,192 | 120,192 | [Screenshot](media/local_joins/spiral_vase_comparison.png) |
| Rising corner | 264 | 288 | [Screenshot](media/local_joins/rising_corner_comparison.png) |

![S-jog comparison](media/local_joins/issue_277_comparison.png)

![Reversal comparison](media/local_joins/issue_278_comparison.png)

Run `python bench/tube_lab/local_joins.py` to regenerate the gallery, metrics, and interactive `out/local_joins/demo.html`. The demo supports case selection, animation/scrubbing, original/repaired geometry, maps, and wire overlay. Serve that directory with a local HTTP server. Three.js still loads from the viewer's normal CDN.

## Scale check

`python bench/tube_lab/local_join_scale.py` compares original and local rendering on spirals with 1,001, 10,001, and 100,001 stations, with and without maps. Raw results are in `bench/tube_lab/local_join_scale_results.json`.

At 100,001 stations (1,200,192 triangles), both renderers used 28,804,752 geometry-buffer bytes without maps and 36,005,976 with maps. Local detection added roughly 12–21 ms once in this run and added no triangles. CPU draw-range updates were roughly 0.004–0.006 ms for both renderers at that size. These timings exclude GPU rendering, texture memory, upload time, and initial tube construction; they are not a frame-rate claim or a worst-case bound. A repair-heavy path also pays for the additional static index buffer and patch vertices.

Browser tests cover geometric seam closure, projected top-surface overlap in both reproductions, a slightly rising cluster, arbitrary plane orientation, an elbow, playback/backward seeks, all three maps, unchanged outside-patch attributes, stable completed buffers, smooth-spiral triangle count, and total triangle budget.

The repair-bearing stress case prepends a few corners to the large spiral: 100,007 stations, two accepted repairs, and only 48 extra triangles. It also exposes the current storage tradeoff: with maps, geometry buffers grew from 36,008,136 to 50,412,888 bytes (about 14.4 MB extra), mainly the second index buffer. Join construction took about 46–53 ms and draw-range updates about 0.010–0.012 ms in that run. Compact prefix tables add eight CPU bytes per original pair on repaired objects. The algorithm bounds triangle growth and per-frame work; it does not eliminate additional storage for repaired objects.
