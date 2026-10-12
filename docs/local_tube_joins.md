# Bounded local tube joins

Use `add_parametric_tube(..., surface="local_join")` to opt in. The default ring renderer is unchanged. Color, normal, and roughness maps are supported, including the existing wrapped path-distance UVs. This is a best-effort surface renderer, not a watertight solid-union operation across an entire print.

## Construction

- Smooth bends whose miter displacement is below the tolerance retain their original geometry. The default tolerance is local width / 100.
- An isolated corner trims its inner offsets and samples its outer arc. Its section is transported around the actual 3D bend axis. It replaces two ring pairs, reuses the exact boundary rings, and costs at most 96 triangles.
- A short almost-planar jog or reversal can become one bounded cluster, up to ten segments. Analytic transverse slices of the segment capsules give the outer and top outlines. Each sample retains the six-vertex chamfered section. Disconnected slices, incompatible interfaces, large section changes, and over-budget patches fall back to the original strip. There is no polygon clipping or general-purpose Boolean dependency.
- Cluster planarity is local and tolerant: residual rise after subtracting the endpoint slope may be up to 10% of bead height plus 0.2% of width. Actual elevations are interpolated back into the mesh. A spiral vase does not need to fit a single plane.
- Clusters have at most 192 triangles and may add at most 96 over the replaced strip. The whole object has an additional-triangle budget of `max(96, ceil(original_triangles * 0.15))`. Once exhausted, later candidates retain the original strip.

The isolated-arc tolerance is a chord tolerance. Cluster sample spacing is a geometric heuristic, **not a certified Hausdorff-error bound**. Other layers and distant self-intersections are not unioned. The spatial-jog example deliberately retains an unsupported tight bend; adjacent repairs are not allowed to overlap. Large changes of section, near-vertical incoming frames, or patch frames pointing opposite to the original interface frames also fall back.

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
| Gentle width taper, 8 → 9.2 mm | 264 | 288 | [Screenshot](media/local_joins/width_gentle_comparison.png) |
| Strong width changes, 4–14 mm | 264 | 264 (fallback) | [Screenshot](media/local_joins/width_abrupt_comparison.png) |
| Forward, up 10 mm, back | 288 | 304 (partial repair) | [Screenshot](media/local_joins/uturn_step_comparison.png) |
| Rounded out-of-plane U-turn | 336 | 352 (partial repair; frame fallback) | [Screenshot](media/local_joins/uturn_round_comparison.png) |
| Widening out-of-plane U-turn, 6 → 12 mm | 336 | 336 (fallback) | [Screenshot](media/local_joins/uturn_taper_comparison.png) |

### Variable widths and out-of-plane U-turns

The width examples supply real per-station width arrays to both renderers. Gentle taper permits the corner repair; the strong taper triggers the existing section-change guard. All U-turns go forward along X, rise along Z, and return along negative X with the default +Z up vector. The rounded versions rise 12 mm along a semicircle of radius 6 mm; the stepped version rises 10 mm through a vertical segment.

The rounded U-turn previously accepted three patches and introduced a pronounced twist near its upper return. Recomputing the patch frame from `up × tangent` gave the opposite orientation to the original transported frame. A new interface-frame guard rejects that patch; its upper-return triangles now remain exactly original. Two other patches remain (336 → 352 triangles). This removes the large added twist, but does **not** solve all bend seams or make the U-turn a clean swept surface. The stepped U-turn still retains its vertical-frame seam, and the widening U-turn falls back. The PR remains a draft.


### Reused earlier examples and synchronized overview


The first five tiles are the recovered, more complex historical paths. Their inputs are frozen in [`bench/tube_lab/fixtures`](../bench/tube_lab/fixtures/README.md), with provenance and all original stations, widths, heights, up vectors, and anchors retained.

| Earlier complex case | Stations | Original → local triangles | Repairs | Comparison |
|---|---:|---:|---:|---|
| #49/#50 Ribweaver rectangle and bulbs | 100 | 1,380 → 1,392 | 1 | [Screenshot](media/local_joins/pr49_bulbs_comparison.png) |
| #120 full tilted G-code winddown | 2,734 | 32,988 → 33,132 | 12 | [Screenshot](media/local_joins/pr120_winddown_comparison.png) |
| #43 blobby 3D path, variable width | 97 | 1,344 → 1,344 | 0 | [Screenshot](media/local_joins/pr43_blobby_comparison.png) |
| #56 eight-layer wall | 427 | 5,304 → 5,304 | 0 | [Screenshot](media/local_joins/pr56_wall_const_comparison.png) |
| #56 eight-layer wall, variable width | 427 | 5,304 → 5,304 | 0 | [Screenshot](media/local_joins/pr56_wall_vary_comparison.png) |

These cases reveal limited repair coverage: the blobby shape and both wall variants remain unchanged, and most difficult bulb sites fall back. The #49 input is the historical reduced reproduction, not the full source dump. The #120 case includes the entire retained G-code snippet. No paths were resampled to make the algorithm look better. Historical per-station colors are replaced by the shared mapped comparison material because local joins do not support those colors.

The gallery also uses the exact four scene generators from `bench/tube_lab/zr_metric.py`, previously used for PR #279/#280: rounded jittered zigzag with variable widths/heights, sharp V, tight hairpin, and mixed sparse corner/dense arc. The jog/retrace/loop path is copied from PR #281's `examples/39_planar_swept_union.py` at `05e85ef`. These compare the current original strip (collapse off) with local joins; they do not run the old PR renderers.

| Historical path | Original → local triangles | Comparison |
|---|---:|---|
| Rounded zigzag | 924 → 924 | [Screenshot](media/local_joins/prior_zigzag_comparison.png) |
| Sharp V | 216 → 240 | [Screenshot](media/local_joins/prior_sharpV_comparison.png) |
| Tight hairpin | 360 → 384 | [Screenshot](media/local_joins/prior_hairpin_comparison.png) |
| Mixed sparse/dense bends | 648 → 672 | [Screenshot](media/local_joins/prior_mixed_comparison.png) |
| PR #281 jog/retrace/loop | 336 → 396 | [Screenshot](media/local_joins/prior_union_demo_comparison.png) |

The historical examples expose remaining limits: the smoothness gate leaves the zigzag's inner folds untouched, the tight hairpin still overlaps, and the retraced/overlapping sections of the PR #281 path are not unioned. A repair count is not a visual-quality score.

`python bench/tube_lab/local_joins.py --demo-only` builds both interactive pages without capturing screenshots. Serve `bench/tube_lab/out/local_joins` and open `overview.html` for all **24 examples**, each with original and local joins side by side. All 48 objects share a 0–100% slider and looping playback (ten seconds printing, two seconds held at completion). Each pair shares a camera: drag either side to orbit, scroll to zoom, or select top/side/perspective. Click a title to enlarge that pair; Escape or “All examples” returns to the grid. Material-map and wireframe switches apply to every example. One WebGL context renders the whole grid with separate viewports.

![All examples at 100%](media/local_joins/animated_overview.png)


![S-jog comparison](media/local_joins/issue_277_comparison.png)

![Reversal comparison](media/local_joins/issue_278_comparison.png)

Run `python bench/tube_lab/local_joins.py --case uturn_round --case width_gentle` to refresh selected captures while retaining the existing gallery, or omit `--case` to regenerate the gallery, metrics, and interactive `out/local_joins/demo.html`. The demo supports case selection, animation/scrubbing, original/repaired geometry, maps, and wire overlay. Serve that directory with a local HTTP server. Three.js still loads from the viewer's normal CDN.

## Scale check

`python bench/tube_lab/local_join_scale.py` compares original and local rendering on spirals with 1,001, 10,001, and 100,001 stations, with and without maps. Raw results are in `bench/tube_lab/local_join_scale_results.json`.

At 100,001 stations (1,200,192 triangles), both renderers used 28,804,752 geometry-buffer bytes without maps and 36,005,976 with maps. Local detection added roughly 12–21 ms once in this run and added no triangles. CPU draw-range updates were roughly 0.004–0.006 ms for both renderers at that size. These timings exclude GPU rendering, texture memory, upload time, and initial tube construction; they are not a frame-rate claim or a worst-case bound. A repair-heavy path also pays for the additional static index buffer and patch vertices.

Browser tests cover geometric seam closure, projected top-surface overlap in both reproductions, a slightly rising cluster, arbitrary plane orientation, an elbow, playback/backward seeks, all three maps, unchanged outside-patch attributes, stable completed buffers, smooth-spiral triangle count, and total triangle budget.

The repair-bearing stress case prepends a few corners to the large spiral: 100,007 stations, two accepted repairs, and only 48 extra triangles. It also exposes the current storage tradeoff: with maps, geometry buffers grew from 36,008,136 to 50,412,888 bytes (about 14.4 MB extra), mainly the second index buffer. Join construction took about 46–53 ms and draw-range updates about 0.010–0.012 ms in that run. Compact prefix tables add eight CPU bytes per original pair on repaired objects. The algorithm bounds triangle growth and per-frame work; it does not eliminate additional storage for repaired objects.
