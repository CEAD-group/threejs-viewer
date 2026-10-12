# Historical toolpath fixtures

These are frozen inputs recovered from the older PR demonstrations, not new
synthetic replacements. `local_joins.py` uses every station and the original
width/height arrays. NPZ keys are `spine`, `widths`, `heights`, `up`, `anchor`,
and `breaks` (a break before that station).

- `pr49_bulbs`: exact 100-station Ribweaver reproduction from `repro_min.py`
  and `tools/capture_strand_collapse.py` used in PRs #49/#50. Original 100×
  display scale and top anchoring retained. This is the historical reduced
  reproduction, not the complete 13,165-station source dump.
- `pr120_winddown`: all 2,734 stations of the A45 G-code snippet in the retained
  `tmp/tube_lab/messy_toolpath_winddown.py`, used by `scene_winddown` in the
  #120 investigation (`tmp/tube_lab/strand119.py`). Original parser and
  extrusion-to-width calculation; 3 mm bead height, 45° up vector, top anchor,
  original recentering. Stored here so the ignored temporary script is no
  longer needed to reproduce the comparison.
- `pr43_blobby`: the 97-station open Catmull–Rom path from
  `examples/21_tube_corner_zfight.py` (introduced by commit `f38a2fd`, PR #43).
  Original seed, control points, sampling, trim, arc-length Z ramp, and widths.
- `pr56_wall_const` / `pr56_wall_vary`: the 427-station eight-layer walls from
  `scenario_wall` in the retained #56 capture harness `tmp/tube_lab/capture.py`.
  Original 0.1 sampling, 0.3 layer height, constant 0.8 width or sinusoidal
  0.4–0.9 width. Comparison-display group offsets are omitted.

Historical per-station colors are replaced by the shared comparison material,
including color/normal/roughness maps. `local_join` does not support per-station
colors. The before column uses the current original strip with collapse off;
it does not run a historical viewer build.
