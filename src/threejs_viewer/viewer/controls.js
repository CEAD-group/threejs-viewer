// Bespoke camera controls for ThreeJSViewer.
//
// One implementation, two modes:
//   - 'turntable' (default): yaw around world-Z, pitch around camera-local right.
//                            Pitch is clamped so the view never flips through the pole.
//   - 'free':                yaw around camera-local up, pitch around camera-local right.
//                            No world-up lock; horizon can tilt.
//
// Click-to-pivot: a left-button pointerdown (no modifier) raycasts the registered
// pickables. If a hit is found, `target` is moved to the hit point WITHOUT touching
// the camera. Because update() never calls camera.lookAt(target), the view does not
// shift; subsequent left-drag orbits exactly around the new pivot.
//
// Pan (right-drag or shift+left-drag): translates camera + target together in screen
// space. Wheel: dollies camera toward/away from target (perspective) or scales zoom
// (orthographic). Rotation and pan have a brief exponential damping on release.
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
//   ctrl.addEventListener('change', fn) / removeEventListener
//
// Implementation note on the no-view-shift guarantee:
//   Rotation builds a single combined quaternion `q = qYaw * qPitch`. We rotate the
//   camera position around `target` by `q` AND premultiply the camera quaternion by
//   the same `q`. No lookAt is performed anywhere. Therefore at dt=0 (no rotation)
//   the camera position and orientation are unchanged regardless of where target is.
//   Moving target alone (click-to-pivot) is a no-op for the rendered view.

import * as THREE from 'three';

const STATE = { NONE: 0, ROTATE: 1, PAN: 2 };

const ROTATE_SPEED = 1.0;          // radians per (delta / element height)
const PAN_SPEED = 1.0;
const ZOOM_SCALE_PER_WHEEL = 0.95; // per 100 wheel deltaY
const DAMP_TIME_CONSTANT = 0.12;   // seconds; exp(-dt / tc)
const MIN_DISTANCE = 1e-3;
const MAX_DISTANCE = 1e6;
const MIN_ZOOM = 1e-4;
const MAX_ZOOM = 1e6;
const POLE_EPS = THREE.MathUtils.degToRad(5); // turntable pitch clamp
// Reject floor-fallback pivots whose hit point is absurdly far from the camera
// (happens when the click ray is nearly parallel to the floor — intersection
// point shoots off to "infinity"). Keeps pivot relocation sensible.
const FLOOR_PIVOT_MAX_DISTANCE = 1e4;
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
        // Cached for floor-fallback pivot picking (XY plane, normal +Z, d=0).
        this._floorPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
        this._floorHit = new THREE.Vector3();

        this._lastPointer = { x: 0, y: 0 };

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

    setMode(mode) {
        if (mode !== 'turntable' && mode !== 'free') return;
        this.mode = mode;
    }

    setRaycastObjects(getter) {
        this._raycastObjectsGetter = getter;
    }

    dispose() {
        this.domElement.removeEventListener('pointerdown', this._onPointerDown);
        this.domElement.removeEventListener('wheel', this._onWheel);
        this.domElement.removeEventListener('contextmenu', this._onContextMenu);
        window.removeEventListener('pointermove', this._onPointerMove);
        window.removeEventListener('pointerup', this._onPointerUp);
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

        // Decide intent. Shift+left = pan (existing). Alt+left = rotate, but with
        // the OPPOSITE rotation mode (temporary flip — released when Alt comes up).
        let nextState = STATE.NONE;
        if (e.button === 0) {
            if (e.shiftKey) nextState = STATE.PAN;
            else nextState = STATE.ROTATE;
        } else if (e.button === 2) {
            nextState = STATE.PAN;
        } else {
            return;
        }

        // Click-to-pivot: on left-click without pan modifiers. Alt is OK (it's
        // our temp-mode-flip modifier, not a pan modifier).
        if (e.button === 0 && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
            this._tryPickPivot(e);
        }

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

        // Stop further damping motion that might be lingering.
        this._rotDeltaTheta = 0;
        this._rotDeltaPhi = 0;
        this._panDeltaX = 0;
        this._panDeltaY = 0;

        // Don't preventDefault — let focus / other listeners proceed.
    }

    _onPointerMove(e) {
        if (!this.enabled) return;
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
            const cam = this.camera;
            let worldPerPxY;
            if (cam.isPerspectiveCamera) {
                const dist = this._tmpV1.copy(cam.position).sub(this.target).length();
                worldPerPxY = (2 * dist * Math.tan(THREE.MathUtils.degToRad(cam.fov / 2))) / h;
            } else {
                // Ortho
                worldPerPxY = (cam.top - cam.bottom) / cam.zoom / h;
            }
            // Negative Y because clientY grows downward but world up is +Y screen-up.
            this._panDeltaX += -dx * worldPerPxY * PAN_SPEED;
            this._panDeltaY += dy * worldPerPxY * PAN_SPEED;
        }
    }

    _onPointerUp(e) {
        if (e.pointerId !== this._activePointerId) return;
        this._state = STATE.NONE;
        this._activePointerId = null;
        this._dragMode = null;
        window.removeEventListener('pointermove', this._onPointerMove);
        window.removeEventListener('pointerup', this._onPointerUp);
    }

    _onWheel(e) {
        if (!this.enabled) return;
        e.preventDefault();
        // Positive deltaY = scroll down = zoom out; scale >1 zooms in (dist/scale)
        const scale = Math.pow(ZOOM_SCALE_PER_WHEEL, e.deltaY / 100);
        this._applyZoom(scale);
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
                    if (hits.length > 0) {
                        this.target.copy(hits[0].point);
                        this.dispatchEvent({ type: 'pivot', point: hits[0].point.clone(), hit: true });
                        this.dispatchEvent(_changeEvent);
                        return;
                    }
                }
            }
        }

        // Floor fallback: intersect the click ray with the world XY plane (z=0).
        // intersectPlane returns null when the ray is parallel or aimed away.
        const floorPt = this._raycaster.ray.intersectPlane(this._floorPlane, this._floorHit);
        if (!floorPt) return;
        // Reject "at infinity" hits when the camera is nearly parallel to the floor.
        if (this._tmpV1.copy(floorPt).sub(this.camera.position).length() > FLOOR_PIVOT_MAX_DISTANCE) return;
        this.target.copy(floorPt);
        this.dispatchEvent({ type: 'pivot', point: floorPt.clone(), hit: false });
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
            // Prospective forward after rotation = q * cam.quaternion * (0,0,-1).
            const fwd = this._tmpV4.set(0, 0, -1)
                .applyQuaternion(cam.quaternion)
                .applyQuaternion(q);
            const dotZ = Math.abs(fwd.dot(this._worldZ));
            // If forward is within POLE_EPS of ±worldZ, reject the pitch component
            // and re-apply only the yaw (reclaiming _tmpQ3 since q is no longer needed).
            if (dotZ > Math.cos(POLE_EPS)) {
                const qYawOnly = this._tmpQ3.copy(qYaw);
                offset.copy(cam.position).sub(this.target).applyQuaternion(qYawOnly);
                cam.position.copy(this.target).add(offset);
                cam.quaternion.premultiply(qYawOnly).normalize();
                return;
            }
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

    _applyZoom(scale) {
        const cam = this.camera;
        if (cam.isPerspectiveCamera) {
            const offset = this._tmpV1.copy(cam.position).sub(this.target);
            const dist = offset.length();
            const newDist = THREE.MathUtils.clamp(dist / scale, MIN_DISTANCE, MAX_DISTANCE);
            offset.setLength(newDist);
            cam.position.copy(this.target).add(offset);
        } else if (cam.isOrthographicCamera) {
            cam.zoom = THREE.MathUtils.clamp(cam.zoom * scale, MIN_ZOOM, MAX_ZOOM);
            cam.updateProjectionMatrix();
        }
    }
}

export { ViewerControls };
