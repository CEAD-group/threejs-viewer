// @ts-check
// Bespoke camera controls for ThreeJSViewer.
//
// One implementation, two modes:
//   - 'turntable' (default): yaw around world-Z, pitch around camera-local right.
//                            Pitch is clamped so the view never flips through the pole;
//                            a pitch away from the pole is always applied, so a drag
//                            escapes a top/bottom view. Once forward is outside the
//                            pole cone, camera.up is re-levelled to world +Z.
//   - 'free':                yaw around camera-local up, pitch around camera-local right.
//                            No world-up lock; horizon can tilt. Never auto-levelled.
//
// Click-to-pivot: a left-button pointerdown (no modifier) raycasts the registered
// pickables. If a hit is found, `target` is moved to the hit point WITHOUT touching
// the camera. Because update() never calls camera.lookAt(target), the view does not
// shift; subsequent left-drag orbits exactly around the new pivot.
//
// Navigation mode (setNavigation): rotate, pan and zoom can each be switched off, and
// leftButton: 'pan' makes a plain left drag pan (a plan view). With rotation off a click
// never moves the pivot.
//
// Pan (right-drag or shift+left-drag): translates camera + target together in screen
// space. Wheel: dollies camera toward/away from the point under the cursor
// (perspective) or scales zoom about the cursor (orthographic), shifting `target`
// along with the camera so the world point under the cursor keeps its screen
// position. Rotation and pan have a brief exponential damping on release.
//
// Public API used by the viewer:
//   ctrl.target           — persistent THREE.Vector3 (read/write, copy/set/add OK)
//   ctrl.enabled          — boolean kill switch
//   ctrl.mode             — 'turntable' | 'free'
//   ctrl.camera           — settable; assign on camera swap then call update()
//   ctrl.update()         — call each frame; applies damping; fires 'change'
//   ctrl.dispose()        — removes listeners
//   ctrl.setMode(mode)
//   ctrl.setRaycastObjects(getterFn)  // () => Iterable<Object3D>
//   ctrl.setNavigation({rotate, pan, zoom, leftButton}) / ctrl.getNavigation()
//   ctrl.addEventListener('change', fn) / removeEventListener
//
// Implementation note on the no-view-shift guarantee:
//   Rotation builds a single combined quaternion `q = qYaw * qPitch`. We rotate the
//   camera position around `target` by `q` AND premultiply the camera quaternion by
//   the same `q`. No lookAt is performed anywhere. Therefore at dt=0 (no rotation)
//   the camera position and orientation are unchanged regardless of where target is.
//   Moving target alone (click-to-pivot) is a no-op for the rendered view.

import * as THREE from 'three';

// TOUCH is a two-finger gesture: pinch zooms about the fingers' midpoint and moving it pans.
const STATE = { NONE: 0, ROTATE: 1, PAN: 2, TOUCH: 3 };

const ROTATE_SPEED = 1.0;          // radians per (delta / element height)
const PAN_SPEED = 1.0;
const ZOOM_SCALE_PER_WHEEL = 0.95; // per 100 wheel deltaY
const DAMP_TIME_CONSTANT = 0.12;   // seconds; exp(-dt / tc)
const MIN_DISTANCE = 1e-3;
const MAX_DISTANCE = 1e6;
const MIN_ZOOM = 1e-4;
const MAX_ZOOM = 1e6;
const POLE_EPS = THREE.MathUtils.degToRad(5); // turntable pitch clamp
/**
 * True when `obj` and every ancestor are visible and none opts out of picking
 * (`userData.pickable === false`, recorded by the viewer's add handlers).
 * Shared by click-to-pivot, the viewer's object click / dblclick hit test and
 * its floor-grid pivot, because three's raycaster ignores `.visible`.
 * @param {import('three').Object3D | null} obj
 */
function isPickTarget(obj) {
    for (let n = obj; n; n = n.parent) {
        if (n.visible === false || (n.userData && n.userData.pickable === false)) return false;
    }
    return true;
}

const _changeEvent = { type: 'change' };

class ViewerControls extends THREE.EventDispatcher {
    constructor(camera, domElement) {
        super();
        this.camera = camera;
        this.domElement = domElement;
        if (domElement && domElement.style) {
            domElement.style.touchAction = 'none';
        }

        this.enabled = true;
        this.mode = 'turntable';
        // Navigation mode (setNavigation): which gestures move the camera.
        this.enableRotate = true;
        this.enablePan = true;
        this.enableZoom = true;
        /** @type {'rotate' | 'pan'} */
        this.leftButton = 'rotate';
        this.target = new THREE.Vector3();

        this._state = STATE.NONE;
        this._activePointerId = null;
        // Snapshot of effective rotation mode at pointerdown — locked for the drag.
        this._dragMode = null;
        // Alt held → next rotate-drag temporarily uses the opposite mode.
        // (We use Alt instead of Shift because Shift+left is already pan.)
        this._altHeld = false;

        // Pending (target) deltas that damping bleeds out toward zero.
        this._rotDeltaTheta = 0; // yaw radians
        this._rotDeltaPhi = 0;   // pitch radians
        this._panDeltaX = 0;     // world units (camera-right)
        this._panDeltaY = 0;     // world units (camera-up)

        this._lastUpdateTime = -1;
        this._raycastObjectsGetter = null;
        // On a click that hits no pickable component, the orbit pivot falls back
        // to this getter's point (scene bounding-box center) instead of the old
        // z=0 floor-plane intersection. () => THREE.Vector3 | null.
        this._fallbackPivotGetter = null;

        // Reusable scratch
        this._raycaster = new THREE.Raycaster();
        this._raycaster.params.Line = { threshold: 0.05 };
        this._raycaster.params.Points = { threshold: 0.05 };
        this._ndc = new THREE.Vector2();
        this._tmpV1 = new THREE.Vector3();
        this._tmpV2 = new THREE.Vector3();
        this._tmpV3 = new THREE.Vector3();
        this._tmpV4 = new THREE.Vector3();
        this._tmpQ1 = new THREE.Quaternion();
        this._tmpQ2 = new THREE.Quaternion();
        this._tmpQ3 = new THREE.Quaternion();
        this._worldZ = new THREE.Vector3(0, 0, 1);

        this._lastPointer = { x: 0, y: 0 };
        // Active touch points by pointerId, so a second finger can start a pinch.
        this._touches = new Map();
        this._pinch = { dist: 0, x: 0, y: 0 };

        // Bind handlers so we can remove later
        this._onPointerDown = this._onPointerDown.bind(this);
        this._onPointerMove = this._onPointerMove.bind(this);
        this._onPointerUp = this._onPointerUp.bind(this);
        this._onWheel = this._onWheel.bind(this);
        this._onContextMenu = this._onContextMenu.bind(this);
        this._onKeyDown = this._onKeyDown.bind(this);
        this._onKeyUp = this._onKeyUp.bind(this);

        domElement.addEventListener('pointerdown', this._onPointerDown);
        domElement.addEventListener('wheel', this._onWheel, { passive: false });
        domElement.addEventListener('contextmenu', this._onContextMenu);
        window.addEventListener('keydown', this._onKeyDown);
        window.addEventListener('keyup', this._onKeyUp);
        // pointermove / pointerup are added on pointerdown to window for capture-outside
    }

    isDragging() {
        return this._state !== STATE.NONE;
    }

    /**
     * True only during an orbit drag, never during a pan. Reads the
     * `_dragMode` snapshot taken at pointerdown (set for a rotate drag,
     * including the Alt-flipped mode, and cleared at pointerup) so the
     * answer is fixed for the whole stroke. The auto-ortho exit after a
     * gizmo snap gates on this: a pan keeps the axis-aligned view, so it
     * must keep the orthographic projection too (issue #193).
     */
    isOrbiting() {
        return this._state === STATE.ROTATE && this._dragMode !== null;
    }

    /**
     * Drop any pending damped rotation/pan inertia (the tail of a just-
     * finished drag). Programmatic camera moves (setView) call this so
     * residual deltas — which update() keeps bleeding into the camera every
     * frame — don't drag the camera off the pose being set (pan inertia
     * even moves `target`).
     */
    cancelInertia() {
        this._rotDeltaTheta = 0;
        this._rotDeltaPhi = 0;
        this._panDeltaX = 0;
        this._panDeltaY = 0;
    }

    setMode(mode) {
        if (mode !== 'turntable' && mode !== 'free') return;
        this.mode = mode;
    }

    /**
     * Choose which gestures move the camera. Fields left out are unchanged.
     * `leftButton: 'pan'` makes a plain left drag pan; Shift+left and the
     * right button pan either way. Rotation off also stops a click from
     * moving the orbit pivot. Switching rotation or pan off drops their inertia.
     * @param {{rotate?: boolean, pan?: boolean, zoom?: boolean, leftButton?: 'rotate' | 'pan'}} opts
     */
    setNavigation(opts) {
        if (typeof opts.rotate === 'boolean') this.enableRotate = opts.rotate;
        if (typeof opts.pan === 'boolean') this.enablePan = opts.pan;
        if (typeof opts.zoom === 'boolean') this.enableZoom = opts.zoom;
        if (opts.leftButton === 'rotate' || opts.leftButton === 'pan') this.leftButton = opts.leftButton;
        if (!this.enableRotate) { this._rotDeltaTheta = 0; this._rotDeltaPhi = 0; }
        if (!this.enablePan) { this._panDeltaX = 0; this._panDeltaY = 0; }
    }

    /** @returns {{rotate: boolean, pan: boolean, zoom: boolean, leftButton: 'rotate' | 'pan'}} */
    getNavigation() {
        return { rotate: this.enableRotate, pan: this.enablePan, zoom: this.enableZoom, leftButton: this.leftButton };
    }

    /**
     * The drag a button starts under the navigation mode, NONE when that gesture is off.
     * @param {number} button @param {boolean} shift
     */
    _dragState(button, shift) {
        let state;
        if (button === 0) state = (shift || this.leftButton === 'pan') ? STATE.PAN : STATE.ROTATE;
        else if (button === 2) state = STATE.PAN;
        else return STATE.NONE;
        if (state === STATE.ROTATE && !this.enableRotate) return STATE.NONE;
        if (state === STATE.PAN && !this.enablePan) return STATE.NONE;
        return state;
    }

    setRaycastObjects(getter) {
        this._raycastObjectsGetter = getter;
    }

    /**
     * Register the fallback orbit-pivot source used when a click hits no
     * pickable component. The getter receives the click ray (world space) so
     * it can intersect furniture the component raycast skips, such as floor
     * grids, and returns a world-space THREE.Vector3 (a grid hit or the scene
     * bounding-box center) or null to leave the pivot unchanged.
     * @param {(ray?: THREE.Ray) => (THREE.Vector3 | null)} getter
     */
    setFallbackPivot(getter) {
        this._fallbackPivotGetter = getter;
    }

    dispose() {
        this.domElement.removeEventListener('pointerdown', this._onPointerDown);
        this.domElement.removeEventListener('wheel', this._onWheel);
        this.domElement.removeEventListener('contextmenu', this._onContextMenu);
        window.removeEventListener('pointermove', this._onPointerMove);
        window.removeEventListener('pointerup', this._onPointerUp);
        window.removeEventListener('pointercancel', this._onPointerUp);
        window.removeEventListener('keydown', this._onKeyDown);
        window.removeEventListener('keyup', this._onKeyUp);
    }

    // Returns the rotation mode that should be used right now, accounting for
    // the Alt-held temporary flip. Used at pointerdown to snapshot _dragMode.
    _effectiveMode() {
        if (this._altHeld) {
            return this.mode === 'turntable' ? 'free' : 'turntable';
        }
        return this.mode;
    }

    _onKeyDown(e) {
        if (e.key === 'Alt' || e.altKey) this._altHeld = true;
    }

    _onKeyUp(e) {
        // Clear on actual Alt release, and also self-heal if any keyup event
        // reports altKey as false (covers cases where we missed the Alt keyup
        // due to focus loss / window switch while Alt was held).
        if (e.key === 'Alt' || !e.altKey) this._altHeld = false;
    }

    // ===== Event handlers =====

    _onContextMenu(e) {
        if (!this.enabled) return;
        e.preventDefault();
    }

    _onPointerDown(e) {
        if (!this.enabled) return;
        if (e.pointerType === 'touch') {
            this._touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
            if (this._touches.size === 2) { this._startPinch(); return; }
            if (this._touches.size > 2) return;
        }

        // Decide intent. Shift+left = pan (existing). Alt+left = rotate, but with
        // the OPPOSITE rotation mode (temporary flip — released when Alt comes up).
        // The navigation mode can map a plain left drag to pan or switch a gesture off.
        if (e.button !== 0 && e.button !== 2) return;
        const nextState = this._dragState(e.button, e.shiftKey);

        // Click-to-pivot: on left-click without pan modifiers. Alt is OK (it's
        // our temp-mode-flip modifier, not a pan modifier). The pivot only
        // serves orbiting, so a view that cannot rotate keeps its pivot.
        if (e.button === 0 && !e.shiftKey && !e.ctrlKey && !e.metaKey && this.enableRotate) {
            this._tryPickPivot(e);
        }
        if (nextState === STATE.NONE) return;

        // Lock the rotation mode for the entire drag — toggling Alt mid-stroke
        // must not switch axes mid-rotation.
        if (nextState === STATE.ROTATE) {
            // Sync alt state from this event (key listeners can miss focus changes).
            this._altHeld = !!e.altKey;
            this._dragMode = this._effectiveMode();
        }

        this._state = nextState;
        this._activePointerId = e.pointerId;
        this._lastPointer.x = e.clientX;
        this._lastPointer.y = e.clientY;

        window.addEventListener('pointermove', this._onPointerMove);
        window.addEventListener('pointerup', this._onPointerUp);
        // A touch or pen gesture the browser takes over ends with pointercancel, not pointerup.
        window.addEventListener('pointercancel', this._onPointerUp);

        // Stop further damping motion that might be lingering.
        this._rotDeltaTheta = 0;
        this._rotDeltaPhi = 0;
        this._panDeltaX = 0;
        this._panDeltaY = 0;

        // Don't preventDefault — let focus / other listeners proceed.
    }

    _startPinch() {
        // The first finger may have started no drag (its gesture is off), so listen here too.
        window.addEventListener('pointermove', this._onPointerMove);
        window.addEventListener('pointerup', this._onPointerUp);
        window.addEventListener('pointercancel', this._onPointerUp);
        this._state = STATE.TOUCH;
        this._activePointerId = null;
        this._dragMode = null;
        // A rotate begun by the first finger must not keep spinning under the pinch.
        this._rotDeltaTheta = 0;
        this._rotDeltaPhi = 0;
        this._pinchSample(this._pinch);
    }

    /** Distance and midpoint of the first two touch points, written into `out`. */
    _pinchSample(out) {
        const [a, b] = this._touches.values();
        out.dist = Math.hypot(b.x - a.x, b.y - a.y);
        out.x = (a.x + b.x) / 2;
        out.y = (a.y + b.y) / 2;
        return out;
    }

    _onPinchMove() {
        const prev = this._pinch;
        const next = this._pinchSample({ dist: 0, x: 0, y: 0 });
        const rect = this.domElement.getBoundingClientRect();
        if (prev.dist > 0 && next.dist > 0 && this.enableZoom) {
            const ndcX = ((next.x - rect.left) / Math.max(1, rect.width)) * 2 - 1;
            const ndcY = -((next.y - rect.top) / Math.max(1, rect.height)) * 2 + 1;
            this._applyZoom(next.dist / prev.dist, ndcX, ndcY);
        }
        if (this.enablePan) this._addPan(next.x - prev.x, next.y - prev.y, Math.max(1, rect.height));
        this._pinch = next;
        this.dispatchEvent(_changeEvent);
    }

    // Screen-pixel drag into pending pan, scaled so the scene follows the pointer.
    _addPan(dx, dy, h) {
        const cam = this.camera;
        let worldPerPxY;
        if (cam.isPerspectiveCamera) {
            const dist = this._tmpV1.copy(cam.position).sub(this.target).length();
            worldPerPxY = (2 * dist * Math.tan(THREE.MathUtils.degToRad(cam.fov / 2))) / h;
        } else {
            worldPerPxY = (cam.top - cam.bottom) / cam.zoom / h;
        }
        // Negative Y because clientY grows downward but world up is +Y screen-up.
        this._panDeltaX += -dx * worldPerPxY * PAN_SPEED;
        this._panDeltaY += dy * worldPerPxY * PAN_SPEED;
    }

    _onPointerMove(e) {
        if (!this.enabled) return;
        const touch = this._touches.get(e.pointerId);
        if (touch) {
            touch.x = e.clientX;
            touch.y = e.clientY;
            if (this._state === STATE.TOUCH) { this._onPinchMove(); return; }
        }
        if (e.pointerId !== this._activePointerId) return;

        const dx = e.clientX - this._lastPointer.x;
        const dy = e.clientY - this._lastPointer.y;
        this._lastPointer.x = e.clientX;
        this._lastPointer.y = e.clientY;

        const rect = this.domElement.getBoundingClientRect();
        const h = Math.max(1, rect.height);

        if (this._state === STATE.ROTATE) {
            // Add to the pending delta; damping in update() drains it toward 0.
            this._rotDeltaTheta += (dx / h) * Math.PI * ROTATE_SPEED;
            this._rotDeltaPhi += (dy / h) * Math.PI * ROTATE_SPEED;
        } else if (this._state === STATE.PAN) {
            this._addPan(dx, dy, h);
        }
    }

    _onPointerUp(e) {
        if (this._touches.delete(e.pointerId)) {
            if (this._state === STATE.TOUCH) {
                if (this._touches.size >= 2) { this._pinchSample(this._pinch); return; }
                if (this._touches.size === 1) {
                    // The finger left behind drags from where it is, so the view does not jump.
                    const [[id, t]] = this._touches;
                    this._state = this._dragState(0, false);
                    this._activePointerId = id;
                    this._altHeld = false;
                    this._dragMode = this._effectiveMode();
                    this._lastPointer.x = t.x;
                    this._lastPointer.y = t.y;
                    return;
                }
            } else if (this._touches.size > 0 && e.pointerId !== this._activePointerId) {
                return;
            }
        }
        if (e.pointerId !== this._activePointerId && this._state !== STATE.TOUCH) return;
        this._touches.clear();
        this._state = STATE.NONE;
        this._activePointerId = null;
        this._dragMode = null;
        window.removeEventListener('pointermove', this._onPointerMove);
        window.removeEventListener('pointerup', this._onPointerUp);
        window.removeEventListener('pointercancel', this._onPointerUp);
    }

    _onWheel(e) {
        if (!this.enabled || !this.enableZoom) return;
        e.preventDefault();
        // Positive deltaY = scroll down = zoom out; scale >1 zooms in (dist/scale)
        const scale = Math.pow(ZOOM_SCALE_PER_WHEEL, e.deltaY / 100);
        // Zoom about the cursor (issue #192): the world point under the pointer
        // keeps its screen position instead of everything converging on target.
        const rect = this.domElement.getBoundingClientRect();
        const ndcX = ((e.clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
        const ndcY = -((e.clientY - rect.top) / Math.max(1, rect.height)) * 2 + 1;
        this._applyZoom(scale, ndcX, ndcY);
        this.dispatchEvent(_changeEvent);
    }

    // ===== Click-to-pivot =====

    _tryPickPivot(e) {
        const rect = this.domElement.getBoundingClientRect();
        this._ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this._ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        this._raycaster.setFromCamera(this._ndc, this.camera);

        // First try registered pickables.
        const getter = this._raycastObjectsGetter;
        if (getter) {
            const candidates = getter();
            if (candidates) {
                const arr = Array.isArray(candidates) ? candidates : Array.from(candidates);
                if (arr.length > 0) {
                    const hits = this._raycaster.intersectObjects(arr, true);
                    // Nearest hit that is actually on screen (issue #215):
                    // three's raycaster ignores `.visible`, so a mesh under a
                    // hidden ancestor is still returned, and an object sent
                    // with pickable=false is a passthrough for the pivot.
                    const hit = hits.find((h) => isPickTarget(h.object));
                    if (hit) {
                        this.target.copy(hit.point);
                        this.dispatchEvent({ type: 'pivot', point: hit.point.clone(), hit: true });
                        this.dispatchEvent(_changeEvent);
                        return;
                    }
                }
            }
        }

        // No component hit: the viewer's fallback intersects visible floor
        // grids along the ray (finite planes, so a grazing click past the
        // grid edge misses instead of landing far off-model like the old
        // infinite z=0 plane did), then falls back to the scene bounding-box
        // center. If neither is available, leave the pivot where it is.
        const fallback = this._fallbackPivotGetter && this._fallbackPivotGetter(this._raycaster.ray);
        if (!fallback) return;
        this.target.copy(fallback);
        this.dispatchEvent({ type: 'pivot', point: fallback.clone(), hit: false });
        this.dispatchEvent(_changeEvent);
    }

    // ===== Update =====

    update() {
        if (this._lastUpdateTime < 0) {
            this._lastUpdateTime = performance.now();
        }
        const now = performance.now();
        const dt = Math.max(0, (now - this._lastUpdateTime) / 1000);
        this._lastUpdateTime = now;

        // Damping factor: at dt -> infinity, alpha -> 1 (consume all). At dt=0, alpha=0.
        const alpha = dt > 0 ? (1 - Math.exp(-dt / DAMP_TIME_CONSTANT)) : 0;

        let moved = false;

        // --- Rotation ---
        const dTheta = this._rotDeltaTheta * alpha;
        const dPhi = this._rotDeltaPhi * alpha;
        this._rotDeltaTheta -= dTheta;
        this._rotDeltaPhi -= dPhi;
        if (Math.abs(dTheta) > 1e-9 || Math.abs(dPhi) > 1e-9) {
            this._applyRotation(dTheta, dPhi);
            moved = true;
        } else {
            // Snap residuals to zero to avoid float drift
            if (Math.abs(this._rotDeltaTheta) < 1e-7) this._rotDeltaTheta = 0;
            if (Math.abs(this._rotDeltaPhi) < 1e-7) this._rotDeltaPhi = 0;
        }

        // --- Pan ---
        const dPanX = this._panDeltaX * alpha;
        const dPanY = this._panDeltaY * alpha;
        this._panDeltaX -= dPanX;
        this._panDeltaY -= dPanY;
        if (Math.abs(dPanX) > 1e-9 || Math.abs(dPanY) > 1e-9) {
            this._applyPan(dPanX, dPanY);
            moved = true;
        } else {
            if (Math.abs(this._panDeltaX) < 1e-7) this._panDeltaX = 0;
            if (Math.abs(this._panDeltaY) < 1e-7) this._panDeltaY = 0;
        }

        if (moved) this.dispatchEvent(_changeEvent);
    }

    // ===== Math helpers =====

    _applyRotation(dTheta, dPhi) {
        const cam = this.camera;
        const camRight = this._tmpV1.set(1, 0, 0).applyQuaternion(cam.quaternion);

        // Use the locked drag-mode if a drag is in flight; otherwise fall back to
        // the effective mode (covers the damping tail after pointerup).
        const activeMode = this._dragMode || this._effectiveMode();

        let yawAxis;
        if (activeMode === 'turntable') {
            yawAxis = this._worldZ;
        } else {
            yawAxis = this._tmpV2.set(0, 1, 0).applyQuaternion(cam.quaternion);
        }

        const qYaw = this._tmpQ1.setFromAxisAngle(yawAxis, -dTheta);
        const qPitch = this._tmpQ2.setFromAxisAngle(camRight, -dPhi);
        const q = this._tmpQ3.copy(qYaw).multiply(qPitch);

        // Provisional rotated offset (target -> camera) in scratch.
        const offset = this._tmpV3.copy(cam.position).sub(this.target).applyQuaternion(q);

        if (activeMode === 'turntable') {
            const cosPole = Math.cos(POLE_EPS);
            // Forward before and after: fwd = q * cam.quaternion * (0,0,-1).
            const fwd = this._tmpV4.set(0, 0, -1).applyQuaternion(cam.quaternion);
            const dotBefore = Math.abs(fwd.dot(this._worldZ));
            fwd.applyQuaternion(q);
            const dotZ = Math.abs(fwd.dot(this._worldZ));
            // Reject the pitch only when it lands inside the POLE_EPS cone AND
            // brings forward closer to the pole than it already is (issue #202):
            // a top/bottom view preset parks forward exactly on the pole, and a
            // direction-blind cone test refused every mouse-sized pitch away
            // from it, so the drag could only yaw. Yaw-only re-apply reclaims
            // _tmpQ3 since q is no longer needed.
            if (dotZ > cosPole && dotZ > dotBefore) {
                const qYawOnly = this._tmpQ3.copy(qYaw);
                offset.copy(cam.position).sub(this.target).applyQuaternion(qYawOnly);
                cam.position.copy(this.target).add(offset);
                cam.quaternion.premultiply(qYawOnly).normalize();
                return;
            }
            cam.position.copy(this.target).add(offset);
            cam.quaternion.premultiply(q).normalize();
            // Re-level once the orbit is outside the pole cone: the top/bottom
            // presets set camera.up to +Y (world Z is parallel to the view
            // there), and every later lookAt (framing, set_camera, setView)
            // reads camera.up, which would come out rolled. Turntable only.
            if (dotZ < cosPole) cam.up.copy(this._worldZ);
            return;
        }

        cam.position.copy(this.target).add(offset);
        cam.quaternion.premultiply(q).normalize();
    }

    _applyPan(dx, dy) {
        const cam = this.camera;
        const right = this._tmpV1.set(1, 0, 0).applyQuaternion(cam.quaternion);
        const up = this._tmpV2.set(0, 1, 0).applyQuaternion(cam.quaternion);
        const delta = this._tmpV3.copy(right).multiplyScalar(dx).addScaledVector(up, dy);
        cam.position.add(delta);
        this.target.add(delta);
    }

    /**
     * Zoom by `scale` (>1 zooms in). With an NDC cursor position the zoom is
     * anchored on the world point under the cursor: perspective dollies both
     * camera and target toward the cursor ray's point on the view plane at
     * target depth, ortho rescales then translates so that point keeps its
     * screen position. Without a cursor (programmatic callers) the zoom stays
     * about `target`, which is also what a cursor exactly on target yields.
     * @param {number} scale
     * @param {number} [ndcX]
     * @param {number} [ndcY]
     */
    _applyZoom(scale, ndcX, ndcY) {
        const cam = this.camera;
        const hasCursor = Number.isFinite(ndcX) && Number.isFinite(ndcY);
        if (cam.isPerspectiveCamera) {
            const offset = this._tmpV1.copy(cam.position).sub(this.target);
            const dist = offset.length();
            const newDist = THREE.MathUtils.clamp(dist / scale, MIN_DISTANCE, MAX_DISTANCE);
            if (!hasCursor) {
                offset.setLength(newDist);
                cam.position.copy(this.target).add(offset);
                return;
            }
            // The clamped distance decides the effective scale, so MIN/MAX
            // distance still hold when anchoring on the cursor.
            const k = dist > 0 ? 1 - newDist / dist : 0;
            if (k === 0) return;
            const fwd = this._tmpV2.set(0, 0, -1).applyQuaternion(cam.quaternion);
            // Built from the projection inverse + live quaternion rather than
            // unproject(): matrixWorld only refreshes at render time, so it is
            // stale for the second of two wheel events in one frame.
            const ray = this._tmpV3.set(ndcX, ndcY, 0.5)
                .applyMatrix4(cam.projectionMatrixInverse)
                .normalize()
                .applyQuaternion(cam.quaternion);
            const rayDotFwd = ray.dot(fwd);
            if (Math.abs(rayDotFwd) < 1e-6) return;
            // Cursor ray hit on the view-parallel plane through target.
            const depth = -offset.dot(fwd);
            const anchor = this._tmpV4.copy(cam.position).addScaledVector(ray, depth / rayDotFwd);
            // Camera and target lerp toward the anchor by the same fraction:
            // the offset shrinks to newDist and the anchor stays on the ray.
            cam.position.lerp(anchor, k);
            this.target.lerp(anchor, k);
        } else if (cam.isOrthographicCamera) {
            const oldZoom = cam.zoom;
            cam.zoom = THREE.MathUtils.clamp(cam.zoom * scale, MIN_ZOOM, MAX_ZOOM);
            cam.updateProjectionMatrix();
            if (!hasCursor || cam.zoom === oldZoom) return;
            // Cursor offset from the optical axis in unzoomed frustum units;
            // the world point there moves by the change in 1/zoom, so shift
            // the camera (and target) by the same amount to cancel it.
            const fx = cam.left + ((ndcX + 1) / 2) * (cam.right - cam.left);
            const fy = cam.bottom + ((ndcY + 1) / 2) * (cam.top - cam.bottom);
            const dInv = 1 / oldZoom - 1 / cam.zoom;
            const right = this._tmpV1.set(1, 0, 0).applyQuaternion(cam.quaternion);
            const up = this._tmpV2.set(0, 1, 0).applyQuaternion(cam.quaternion);
            const delta = this._tmpV3.copy(right).multiplyScalar(fx * dInv).addScaledVector(up, fy * dInv);
            cam.position.add(delta);
            this.target.add(delta);
        }
    }
}

export { ViewerControls, isPickTarget };
