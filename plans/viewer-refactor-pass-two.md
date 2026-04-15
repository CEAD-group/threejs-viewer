# viewer.js refactor — AnimationPlayer + ClippingController extraction

> **Status: undecided / future work.** This is a decision-making artifact for a pass-two refactor that has not been approved. Pass one (ParametricTube, CameraController, ShadingDebugController + JSDoc types) landed separately. Read the "Is it worth doing?" section below before acting on any of this.

Working doc for the possible second pass on viewer.js. Captures the plan, the honest cost/benefit analysis, and the caveats that could bite during execution.

## Context

`src/threejs_viewer/viewer/viewer.js` used to be a 4802-line god-class. The first refactor pass on branch `viewer-refactor` (7 commits on top of `main`) landed:

- `// @ts-check` + JSDoc on viewer sources
- Shared ring/color helpers as free functions
- `ParametricTube` extracted as an in-file class owning geometry + LOD + morph state (consolidated ~15 scattered `mesh.userData.tubeX` fields)
- `ShadingDebugController` extracted (owns `wireframeMode`, `shadingMode`, cached materials)
- `CameraController` extracted (owns perspective/ortho cameras, framing, scene bounds)
- `frameObject` / `frameAll` deduped via `_fitCameraToBox`
- Clipping-planes-or-empty consolidated via `_activeClippingPlanes()`

**What's left:** animation and clipping were deliberately *not* extracted in pass one. They were left as banner-grouped methods (`// ========== Animation ==========`) on `ThreeJSViewer` because a naïve extraction would just swap `this._X` for `this.v._X` and gain nothing — the state and DOM refs would still live on the viewer, with the class being a hollow method container.

This plan does the extraction *properly*: the controllers own their own fields; the viewer holds a handle. That's the shape the three already-extracted controllers have.

## Is it worth doing?

**Short answer:** borderline. Worth it *if* agentic/LLM-driven edits are a primary workflow going forward. Skip if most edits are human-driven.

### For human readability alone — probably not

The existing banner-grouped code works. The "smell" is cosmetic — names like `this._animationX` instead of `this._anim.X`, ctor field initializations scattered instead of co-located. No bug is being fixed, no feature unblocked. pytest catches maybe 30% of the surface area involved. `@ts-check` already prevents the typo class of bug (`_animPlaying` vs `_animationPlaying`) that motivated part of the original refactor.

### For agentic/LLM workflow — marginally yes

Concrete wins for agent-driven editing:

1. **Contiguous reads.** Animation surface today is scattered across ~5 regions (ctor fields ~L2150, DOM bind ~L2240, methods ~L3070-3503, render-loop tick elsewhere, WS dispatch elsewhere). A subagent asked "change scrub behavior" has to assemble from 5 reads. After: one ~500-line chunk, one `Read` with `offset`/`limit`.
2. **Narrower edit blast radius.** Class enclosure gives a clean "these are the fields I can touch" boundary. Flat viewer fields make cross-contamination easier.
3. **Better symbol discoverability.** `grep "class AnimationPlayer"` → 1 hit with the full API. `grep "_animation"` → 80+ hits across ctor, UI, dispatcher, render loop, tests.
4. **Stronger typing.** JSDoc on class fields lives at the declaration line, which `tsc` checks harder than scattered `@type` comments.
5. **Delegatable tasks.** "Subagent, modify AnimationPlayer to add reverse playback" is a cleaner prompt than "subagent, read these 5 regions of a 4800-line file."

What it doesn't fix:
- Cross-cutting seams (the `_CHANNEL_APPLY` → `_setClipTime` coupling) are unchanged — just relocated.
- Context-window pressure: modern Claude models handle 4800 lines fine. The benefit is about *task framing*, not raw comprehension.
- Dominant factors for agentic success on this codebase (good `CLAUDE.md`, good tests, fast feedback loops) dwarf internal structure.

### Calibrated verdict

| Scenario | Recommendation |
| --- | --- |
| Most future edits are human-driven | Skip. Ship the 7 existing commits. |
| Leaning heavily on subagents for per-subsystem edits in the next few months | Do it. |
| File keeps growing past ~6k lines or a new cross-cutting subsystem lands | Revisit regardless. |

## Internal test surface (migrate, don't shim)

Project is early; backward-compat on internal JS field names is not a goal. Two browser tests poke viewer internals:

- `tests/test_parametric_tube.py:575` — writes `v._animationPlaying`
- `tests/test_animation_interp.py:89-93` — reads `v._animation`, writes `v._animationPlaying` / `v._animationTime`, calls `v._getFrameAtTime()` and `v._applyFrame()`

**Migrate these in the same PR.** No getter/setter shim layer on the viewer.

| Old (delete) | New |
| --- | --- |
| `v._animation` | `v._anim.animation` |
| `v._animationPlaying` | `v._anim.playing` |
| `v._animationTime` | `v._anim.time` |
| `v._getFrameAtTime(t)` | `v._anim.getFrameAtTime(t)` |
| `v._applyFrame(i, t)` | `v._anim.applyFrame(i, t)` |

The WS-message API (`set_clipping_plane` / `set_clipping_slab` / `disable_clipping_plane` / `set_clipping_defaults` / `load_animation` / `stop_animation`) is external and MUST keep working — that's a dispatcher concern (`_onMessage` still routes those names to the right controller methods), not a field-name concern.

Also remove any compat getters left over from pass one (e.g., `_wireframeMode`) — grep confirms they're not used by tests.

## Target shape

Two new classes in `src/threejs_viewer/viewer/viewer.js` (same file — concat-only build stays trivial):

```js
class AnimationPlayer {
    constructor(viewer) {
        this.v = viewer;
        // State (moved off viewer)
        this.animation = null;
        this.playing = false;
        this.time = 0;
        this.speed = 1;
        this.loop = false;
        this.generation = 0;
        this.lastUpdate = 0;
        this.speedIndex = DEFAULT_SPEED_INDEX;
        this.baselineVisibility = new Map();
        this.scrubbing = false;
        this.wasPlayingBeforeScrub = false;
        this.frameId = 0;
        // Tracking sub-state
        this.track = {
            mode: 'off', targetId: null, lastPos: new THREE.Vector3(),
            hasLastPos: false, interactive: false,
            _tmpPos: new THREE.Vector3(), _tmpDelta: new THREE.Vector3(),
            _tmpTarget: null,
        };
        // DOM refs (queried once in bind())
        this.ui = { /* btnPlay, btnLoop, btnTrack, timelineContainer,
                      timelineProgress, timelineMarkers, currentTime,
                      totalTime, currentFrame, totalFrames, speedDisplay,
                      animControls */ };
        // Channel apply table — lives here now
        this._channelApply = makeChannelApply(viewer);
    }
    bind(rootEl) { /* query DOM, wire listeners */ }
    load(animData)                 // was _loadAnimation
    reset()                        // was _resetAnimationState
    stop(restoreVisibility = true) // was _stopAnimation
    applyFrame(frameIndex, t = 0)  // was _applyFrame
    getFrameAtTime(time)           // was _getFrameAtTime
    updateUI()                     // was _updateAnimationUI
    step(delta) seek(time) togglePlay() setSpeed(s) scrubFromEvent(e)
    cycleTrack() updateTrackingUI() applyTracking(i, iNext, t) guessTrackTarget()
    tick(now)                      // moved from render loop
}

class ClippingController {
    constructor(viewer) {
        this.v = viewer;
        // State
        this.enabled = false;
        this.plane = new THREE.Plane(...);
        this.plane2 = new THREE.Plane(...);
        this.planes = [this.plane];
        this.position = 0;
        this.axis = null;
        this.slabMode = false;
        this.slabThickness = 2;
        this.helperVisible = true;
        this.syncFromGizmo = false;
        this.defaults = null;
        // 3D objects (added to viewer._scene by bind())
        this.anchor = null; this.gizmo = null; this.gizmoHelper = null;
        this.disc = null; this.ring = null; this.disc2 = null; this.ring2 = null;
        // DOM refs
        this.ui = { /* panel, btnClip, distanceSlider, distanceValue,
                      thicknessSlider, thicknessValue, nx, ny, nz,
                      modeSingle, modeSlab, thicknessSection, close */ };
        // Scratch (moved off viewer — only clipping uses them)
        this._zAxis = new THREE.Vector3();
        this._localZ = new THREE.Vector3(0, 0, 1);
    }
    bind(rootEl)
    enable(normal, distance)
    enableSlab(normal, center, thickness)
    disable()
    setDefaults(defaults)
    setClipTime(id, v)             // moved off viewer; caller: AnimationPlayer.applyFrame
    activePlanes() { return this.enabled ? this.planes : []; }
    applyToMaterial(mat) { mat.clippingPlanes = this.activePlanes(); mat.clipShadows = true; }
    snapCameraToNormal()
    onSceneBoundsChanged(sphere)
}
```

`ThreeJSViewer` holds `this._anim` and `this._clip`. Everything that used to poke `this._animationX` becomes `this._anim.X`. No compat shims.

## Cross-subsystem seam — the one hard bit

`_CHANNEL_APPLY` (built at line 103 via `makeChannelApply(viewer)`) closes over:
- `viewer._objGeneration`, `viewer._mixerGeneration` (cache invalidation keys)
- `viewer._objects`, `viewer._mixers` (lookup maps)
- The `clip_times` handler calls `viewer._setClipTime(id, v)`.

Approach: `makeChannelApply(viewer)` keeps its current signature. `AnimationPlayer` assigns `this._channelApply = makeChannelApply(viewer)` in ctor. The `clip_times` handler switches from `viewer._setClipTime(...)` to `viewer._clip.setClipTime(...)`. Minimal mechanical change.

**Sequencing matters:** `ClippingController` must be constructed *before* `AnimationPlayer` in the viewer ctor, because the channel-apply factory captures a reference to `viewer._clip` transitively. Order in the viewer ctor:
```js
this._clip = new ClippingController(this);
this._anim = new AnimationPlayer(this);
```

## Work plan

Each step must compile, type-check, and pass all 154 tests before moving to the next. Commit each step separately so `git log -p viewer-refactor` stays readable.

### Step 1 — ClippingController extraction (simpler, do first)

- Add `class ClippingController` before `ThreeJSViewer`.
- Move all 11 state fields + 7 3D objects + 13 DOM refs + 2 scratch vectors onto `this._clip.X`.
- Move these methods in as methods: `_onClipGizmoChange`, `_syncClipUI`, `_updateClipHelpers`, `_buildClipPanel`, `_setClipDistance`, `_setClipAxis`, `_setClipSlabMode`, `_setClipSlabThickness`, `_snapCameraToClipNormal`, `_setClippingPlane`, `_setClippingSlab`, `_disableClippingPlane`, `_setClippingDefaults`, `_setClipTime`.
- Replace `_activeClippingPlanes()` call sites with `this._clip.activePlanes()`.
- Route `_onMessage` straight to `this._clip.enable(...)` etc. Delete the viewer wrappers — fewer layers.

### Step 2 — AnimationPlayer extraction

- Add `class AnimationPlayer` before `ThreeJSViewer`.
- Move all 30 animation fields (state, DOM refs, tracking) onto `this._anim.X`.
- Move the 18 animation methods into methods (listed in target-shape section).
- `_CHANNEL_APPLY` moves inside `AnimationPlayer`; `clip_times` handler switches to `viewer._clip.setClipTime(...)`.
- Update the render loop tick to call `this._anim.tick(now)` instead of the inline body.
- Migrate the two test files (`test_animation_interp.py`, `test_parametric_tube.py`) to the new names.

### Step 3 — Wire-up cleanup

- `ThreeJSViewer` ctor: instantiate controllers in the right order (clipping before animation).
- DOM setup step: call `this._clip.bind(this._rootEl); this._anim.bind(this._rootEl);`.
- Delete remaining compat getters left from pass one.
- Verify no `this._animationX` / `this._clipX` field references remain on the viewer.

## Critical files

- `src/threejs_viewer/viewer/viewer.js` — the entire refactor
- `src/threejs_viewer/viewer/build.py` — unchanged (concat-only, same file)
- `src/threejs_viewer/viewer.html` — rebuilt artifact
- `tests/test_animation_interp.py:89-93`, `tests/test_parametric_tube.py:575` — migrate to new names
- `jsconfig.json` — already in place; `npx tsc --noEmit -p jsconfig.json` is the type gate

## Docs to update in the same PR

Neither CLAUDE.md nor DESIGN.md need heavy changes — they describe user-facing behavior (WS API, examples, keybindings) which the refactor doesn't touch. But the one-line internal-structure descriptions should get a small expansion so a future agent reading these files knows the controller layout exists without having to open viewer.js.

- **`CLAUDE.md:55`** — current line: `  - \`viewer.js\` - ES module exporting \`ThreeJSViewer(container, options)\` class`. Update to note the in-file controller classes (`ParametricTube`, `CameraController`, `ShadingDebugController`, `ClippingController`, `AnimationPlayer`) and that they're intentionally kept in one file (concat-only build).
- **`DESIGN.md:235`** — current line: `│       ├── viewer.js    # ES module: ThreeJSViewer class`. Same update — one-line mention of the controller decomposition.
- **`DESIGN.md`** — add a short "Viewer internal structure" subsection under the existing architecture narrative (just a few lines: what each controller owns, where the seams are). Aim for ~10 lines, not a full architecture essay. This is the one doc where agents should be able to orient without reading viewer.js.
- **`CHANGELOG.md`** — add a line under the unreleased section noting internal viewer refactor (no behavior change). Keep it terse; end users don't care.
- **`README.md`** — no change needed (user-facing surface unchanged).
- **`plans/viewer-refactor-pass-two.md`** (this file) — decision-making artifact for undecided future work. Delete once pass two lands or is formally declined.

Grep gate before merging: `grep -rn "_animation\|_clip" src/threejs_viewer/viewer/viewer.js` should show zero `this._animationX` or `this._clipX` field references on `ThreeJSViewer` (only references via `this._anim.X` / `this._clip.X`). Same grep over `tests/` should only match the migrated new names.

## Verification

After each step:

```bash
npx tsc --noEmit -p jsconfig.json                       # clean
uv run python src/threejs_viewer/viewer/build.py        # builds
lsof -ti:5666,5667 2>/dev/null | xargs -I{} kill -9 {} 2>/dev/null; \
  uv run pytest -x -q                                   # 154/154
uv run ruff check . && uv run ruff format --check .     # clean
```

**Manual smoke is not optional** — pytest covers maybe 30% of the affected behavior. Play through each of these in a real browser:

- `examples/17_animation_interpolation.py` — HOLD vs LINEAR playback, timeline scrub, speed step, camera track cycle
- `examples/11_toolpath.py` — draw-range animation + color-mode swap (exercises `clip_times`-adjacent paths)
- `examples/16_clipping_plane.py` — GUI panel open, single/slab toggle, axis buttons, thickness slider, V-key snap, arrow-key nudge
- `examples/18_parametric_tube.py` — draw-range frontier morph during animation
- `examples/10_animation_stress_test.py` — 520 objects × 2499 frames, stresses `_CHANNEL_APPLY` and cache invalidation

## Risks & caveats

### High-risk areas (unit tests won't catch)

- **`_CHANNEL_APPLY` closure depth.** Four handlers close over `_objGeneration` / `_mixerGeneration` / `_objects` / `_mixers`. Relocating the factory means capturing those at a specific lifecycle point. Mess it up and cache invalidation breaks silently — stale object refs persist across scene changes, animation continues playing against deleted objects. Mitigation: grep `makeChannelApply` and `_CHANNEL_APPLY` before touching; after the move, verify with `examples/10_animation_stress_test.py` followed by `stop_animation()` + reload.
- **Scrub-while-playing state machine.** `_scrubbing`, `_wasPlayingBeforeScrub`, and `_animationPlaying` interact via mouse events that pytest doesn't drive. A wrong reorder during move = scrub leaves playback stuck or resumes at wrong time.
- **Interactive camera tracking.** `_trackInteractive`, `_trackHasLastPos`, `_trackLastPos` interact with the T-key handler, the auto-tracking mode from `animData.camera_follow` / `camera_lookat`, and the `_controls.target` read/write. Not covered by pytest.
- **Clipping gizmo feedback loop.** `_clipSyncFromGizmo` is a guard that prevents `onChange` → `_syncClipUI` → gizmo update → `onChange` recursion. Miss it during the move and you get either an infinite loop or a desynced gizmo. Manual test: drag the gizmo, then edit the nx/ny/nz inputs manually — both should reflect the other's state.
- **V-key camera snap.** Reads live `_camera.position` and `_clipPlane.normal`. If `ClippingController.snapCameraToNormal` can't reach the camera correctly, the feature silently no-ops.
- **Timeline marker positioning.** Marker DOM elements are created with `style.left` from the animation duration. If `bind()` is called before the root element is in the layout, `getBoundingClientRect()` returns zeros and markers stack at the origin.

### Medium-risk areas

- **DOM bind timing.** Both controllers need the root element to exist before `bind()`. Current viewer queries DOM inline during ctor — `bind()` has to land at the same point in the init sequence. Keep ctor pure (just allocate); call `bind()` from the existing DOM-setup site. If an event listener fires before `bind()` finishes, `this.ui.X` is undefined.
- **Render loop tick extraction.** Moving the animation-advance block out of the RAF closure into `AnimationPlayer.tick(now)` has to preserve: the `performance.now()` delta math, the loop wrap, the pause-at-end behavior, and the `_animGeneration` invalidation check (a stale tick must not apply frames after `stop()` is called).
- **`_setClipTime` call sites.** Moving it from viewer to `ClippingController` means two callers update: `AnimationPlayer.applyFrame` (channel handler + JSON frame path) and anywhere else that calls it. Grep before deleting the old method.

### Low-risk but annoying

- **Ordering in the WS dispatcher.** `_onMessage` is a big switch. Changing every animation/clipping case in lockstep is mechanical but easy to drop one. Run all examples, not just the two obvious ones.
- **JSDoc churn.** `@type {AnimationPlayer}` on `_anim` and `@type {ClippingController}` on `_clip` need to land so downstream member access type-checks. `tsc` will tell you.
- **Method-name collision.** `AnimationPlayer.step(delta)` is fine but there's also `CameraController` and the plan should not re-use method names across controllers in a way that's confusing to grep.

### What won't be a problem

- Build changes (none — concat-only, same file)
- External API stability (WS message types unchanged)
- `ParametricTube` / `CameraController` / `ShadingDebugController` (extracted in pass one, out of scope)

## Decision point

This document exists because the answer to "should we do this?" isn't automatic. The first pass had obvious wins; this pass is a judgment call. The information above is enough to make that call either way.

If the answer is "yes, do it":
```bash
git checkout viewer-refactor
# Work through steps 1-3; commit each.
```

If the answer is "no, not now":
- Close this file; the 7 existing commits are ready to PR as-is.
- Revisit when file size or a new subsystem forces a reopen.
