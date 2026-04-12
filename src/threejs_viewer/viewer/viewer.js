import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { ColladaLoader } from 'three/addons/loaders/ColladaLoader.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { TDSLoader } from 'three/addons/loaders/TDSLoader.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { ViewHelper } from 'three/addons/helpers/ViewHelper.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

const VIEWER_VERSION = '0.0.0-dev';

const ORTHO_FRUSTUM = 10;

// Logarithmic speed steps: 0.001x to 1000x
const SPEED_STEPS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000];

const CLIP_AXIS_NORMALS = {
    'x+': new THREE.Vector3(1, 0, 0),
    'x-': new THREE.Vector3(-1, 0, 0),
    'y+': new THREE.Vector3(0, 1, 0),
    'y-': new THREE.Vector3(0, -1, 0),
    'z+': new THREE.Vector3(0, 0, 1),
    'z-': new THREE.Vector3(0, 0, -1),
};


// Refresh cached refs array when the object/mixer map has changed.
// Called at most once per frame per channel (guarded by generation check).
function refreshRefs(refs, ids, map) {
    for (let i = 0; i < ids.length; i++) {
        refs[i] = map.get(ids[i]) || null;
    }
}

// Scratch objects for matrix decompose/lerp. Module-scope to avoid per-frame alloc.
const _lerpPosA = new THREE.Vector3();
const _lerpPosB = new THREE.Vector3();
const _lerpScaleA = new THREE.Vector3();
const _lerpScaleB = new THREE.Vector3();
const _lerpQuatA = new THREE.Quaternion();
const _lerpQuatB = new THREE.Quaternion();
const _lerpMatTmp = new THREE.Matrix4();
const _lerpMatOut = new THREE.Matrix4();

// Decompose two matrices (from a typed array or plain array) and write the
// linearly-interpolated pos/slerped-quat/linearly-interpolated-scale recomposition
// into `outMat`. Both sources are 16-element flat matrices (column-major).
function lerpMatrixInto(srcA, baseA, srcB, baseB, t, outMat) {
    _lerpMatTmp.fromArray(srcA, baseA);
    _lerpMatTmp.decompose(_lerpPosA, _lerpQuatA, _lerpScaleA);
    _lerpMatTmp.fromArray(srcB, baseB);
    _lerpMatTmp.decompose(_lerpPosB, _lerpQuatB, _lerpScaleB);
    _lerpPosA.lerp(_lerpPosB, t);
    _lerpScaleA.lerp(_lerpScaleB, t);
    _lerpQuatA.slerp(_lerpQuatB, t);
    outMat.compose(_lerpPosA, _lerpQuatA, _lerpScaleA);
}

// Returns true when a channel should interpolate linearly between `base` and
// `baseNext`. Every channel carries its own explicit interpolation mode, set
// Python-side per channel — this function just checks that mode plus the
// standard guards (there is a next keyframe and we're between keyframes).
// Visibility overrides this to always hold (see visibility applier).
function shouldInterpChannel(ch, baseNext, t) {
    if (baseNext === null || t <= 0) return false;
    return ch.interpolation === 'linear';
}

// Linear-interpolate two hex colors in 8-bit RGB space. Alpha byte (top 8
// bits) is preserved from `a`; three.js only consumes the low 24 bits.
function lerpHexColor(a, b, t) {
    const ar = (a >> 16) & 0xff, ag = (a >> 8) & 0xff, ab = a & 0xff;
    const br = (b >> 16) & 0xff, bg = (b >> 8) & 0xff, bb = b & 0xff;
    const r = Math.round(ar + (br - ar) * t);
    const g = Math.round(ag + (bg - ag) * t);
    const bl = Math.round(ab + (bb - ab) * t);
    return (r << 16) | (g << 8) | bl;
}

// Normalize interpolation metadata from the wire. Accepts only 'hold' and
// 'linear'; anything else (null, undefined, typo like 'cubic') falls back
// to the provided default so stray strings can't silently switch playback
// semantics.
function sanitizeInterpolation(value, fallback) {
    if (value === 'hold' || value === 'linear') return value;
    if (value != null) {
        console.warn(`Unknown interpolation '${value}', falling back to '${fallback}'`);
    }
    return fallback;
}

// Channel apply functions — keyed by channel name. Signature is
// (ch, refs, base, baseNext, t) where baseNext/t are optional; when baseNext
// is null or t <= 0, all channels take the step path (current keyframe only).
function makeChannelApply(viewer) {
    return {
        transforms(ch, refs, base, baseNext, t) {
            if (ch._objGen !== viewer._objGeneration) {
                refreshRefs(refs, ch.ids, viewer._objects);
                ch._objGen = viewer._objGeneration;
                for (let i = 0; i < refs.length; i++) { if (refs[i]) refs[i].matrixAutoUpdate = false; }
            }
            const nObj = ch.ids.length;
            if (shouldInterpChannel(ch, baseNext, t)) {
                for (let i = 0; i < nObj; i++) {
                    const obj = refs[i];
                    if (obj) {
                        lerpMatrixInto(ch.data, base + i * 16, ch.data, baseNext + i * 16, t, _lerpMatOut);
                        obj.matrix.copy(_lerpMatOut);
                        obj.matrixWorldNeedsUpdate = true;
                    }
                }
            } else {
                for (let i = 0; i < nObj; i++) {
                    const obj = refs[i];
                    if (obj) {
                        obj.matrix.fromArray(ch.data, base + i * 16);
                        obj.matrixWorldNeedsUpdate = true;
                    }
                }
            }
        },

        colors(ch, refs, base, baseNext, t) {
            // Linear mode crossfades hex values component-wise in RGB space.
            // Indexed colors (uint8 + colormap) look up both endpoints in the
            // palette and lerp the resulting hex — consistent with direct-hex.
            if (ch._objGen !== viewer._objGeneration) {
                refreshRefs(refs, ch.ids, viewer._objects);
                ch._objGen = viewer._objGeneration;
            }
            const nObj = ch.ids.length;
            const colormap = ch.colormap;
            const interp = shouldInterpChannel(ch, baseNext, t);
            for (let i = 0; i < nObj; i++) {
                const obj = refs[i];
                if (!obj) continue;
                const rawA = ch.data[base + i];
                let color = colormap ? colormap[rawA] : rawA;
                if (interp) {
                    const rawB = ch.data[baseNext + i];
                    const colorB = colormap ? colormap[rawB] : rawB;
                    color = lerpHexColor(color, colorB, t);
                }
                obj.traverse(child => {
                    if (!child.material) return;
                    const mats = Array.isArray(child.material) ? child.material : [child.material];
                    for (const mat of mats) { if (mat.color) mat.color.setHex(color); }
                });
            }
        },

        visibility(ch, refs, base) {
            // Booleans have no meaningful linear interpretation — "linear on
            // a bool" would be a step function at some arbitrary threshold.
            // We always left-hold on the floor keyframe regardless of
            // ch.interpolation; the setting is accepted but has no effect.
            if (ch._objGen !== viewer._objGeneration) {
                refreshRefs(refs, ch.ids, viewer._objects);
                ch._objGen = viewer._objGeneration;
            }
            const nObj = ch.ids.length;
            for (let i = 0; i < nObj; i++) {
                const obj = refs[i];
                if (obj) obj.visible = (ch.data[base + i] === 1);
            }
        },

        draw_ranges(ch, refs, base, baseNext, t) {
            if (ch._objGen !== viewer._objGeneration) {
                refreshRefs(refs, ch.ids, viewer._objects);
                ch._objGen = viewer._objGeneration;
            }
            const nObj = ch.ids.length;
            const interp = shouldInterpChannel(ch, baseNext, t);
            for (let i = 0; i < nObj; i++) {
                const obj = refs[i];
                if (!obj) continue;
                let value = ch.data[base + i];
                if (interp) {
                    value = value * (1 - t) + ch.data[baseNext + i] * t;
                }
                if (obj.userData.isPolyline) {
                    obj.geometry.instanceCount = Math.round(value * obj.userData.maxInstanceCount);
                } else if (obj.userData.isParametricTube) {
                    applyParametricTubeDrawRange(obj, value);
                } else if (obj.userData.isMesh) {
                    obj.geometry.setDrawRange(0, Math.round(value * obj.userData.totalIndexCount));
                }
            }
        },

        opacity(ch, refs, base, baseNext, t) {
            if (ch._objGen !== viewer._objGeneration) {
                refreshRefs(refs, ch.ids, viewer._objects);
                ch._objGen = viewer._objGeneration;
            }
            const nObj = ch.ids.length;
            const interp = shouldInterpChannel(ch, baseNext, t);
            for (let i = 0; i < nObj; i++) {
                const obj = refs[i];
                if (!obj) continue;
                let val = ch.data[base + i];
                if (interp) {
                    val = val * (1 - t) + ch.data[baseNext + i] * t;
                }
                applyOpacity(obj, val);
            }
        },

        clip_times(ch, refs, base, baseNext, t) {
            if (ch._mixerGen !== viewer._mixerGeneration) {
                refreshRefs(refs, ch.ids, viewer._mixers);
                ch._mixerGen = viewer._mixerGeneration;
            }
            const nObj = ch.ids.length;
            const interp = shouldInterpChannel(ch, baseNext, t);
            for (let i = 0; i < nObj; i++) {
                const mixer = refs[i];
                if (!mixer || !mixer.setTime) continue;
                let val = ch.data[base + i];
                if (interp) {
                    val = val * (1 - t) + ch.data[baseNext + i] * t;
                }
                mixer.setTime(val);
            }
        },

        // No-ops: data is read directly in _applyCameraTracking, not per-object
        camera_target: () => {},
        camera_position: () => {},
    };
}

function applyOpacity(obj, opacity) {
    obj.traverse(child => {
        if (!child.material) return;
        const mats = Array.isArray(child.material) ? child.material : [child.material];
        for (const mat of mats) {
            const wasTransparent = mat.transparent;
            mat.transparent = opacity < 1;
            mat.opacity = opacity;
            mat.depthWrite = opacity >= 1;
            if (mat.transparent !== wasTransparent) mat.needsUpdate = true;
        }
    });
}

// Primitive creators
const PRIMITIVES = {
    box: (params) => new THREE.BoxGeometry(
        params.width || 1, params.height || 1, params.depth || 1
    ),
    sphere: (params) => new THREE.SphereGeometry(
        params.radius || 0.5, params.widthSegments || 32, params.heightSegments || 16
    ),
    cylinder: (params) => new THREE.CylinderGeometry(
        params.radiusTop || 0.5, params.radiusBottom || 0.5,
        params.height || 1, params.radialSegments || 32
    ),
    plane: (params) => new THREE.PlaneGeometry(
        params.width || 1, params.height || 1
    ),
    cone: (params) => new THREE.ConeGeometry(
        params.radius || 0.5, params.height || 1, params.radialSegments || 32
    ),
    torus: (params) => new THREE.TorusGeometry(
        params.radius || 0.5, params.tube || 0.2,
        params.radialSegments || 16, params.tubularSegments || 48
    ),
    capsule: (params) => new THREE.CapsuleGeometry(
        params.radius || 0.25, params.length || 0.5,
        params.capSegments || 8, params.radialSegments || 16
    )
};

// Sample a rounded-rectangle cross-section in the local (u, v) plane by
// distributing nVerts points equally by arc length around the perimeter.
// The shape is: 4 straight edges + 4 quarter-circle corner arcs. Points
// are always distributed the same way relative to the shape, so changing
// width/height scales the shape without rotating the visible polygon.
// Radius is clamped to cornerRadiusFrac * min(width, height) with
// cornerRadiusFrac in [0, 0.5].
function sampleRoundedRectInto(out, nVerts, width, height, cornerRadiusFrac) {
    if (!Number.isFinite(width) || width <= 0) {
        throw new Error(`parametric_tube width must be finite and > 0, got ${width}`);
    }
    if (!Number.isFinite(height) || height <= 0) {
        throw new Error(`parametric_tube height must be finite and > 0, got ${height}`);
    }
    const hw = width * 0.5;
    const hh = height * 0.5;
    const frac = Math.max(0, Math.min(0.5, cornerRadiusFrac));
    const r = frac * Math.min(width, height);
    const cxA = hw - r;  // half-width of the flat section
    const cyA = hh - r;  // half-height of the flat section

    // Perimeter segments (starting at +hw, 0 going counter-clockwise):
    //   right edge:  (+hw, -cyA) → (+hw, +cyA)    length = 2*cyA
    //   TR corner:   quarter arc radius r           length = π*r/2
    //   top edge:    (+cxA, +hh) → (-cxA, +hh)    length = 2*cxA
    //   TL corner:   quarter arc                    length = π*r/2
    //   left edge:   (-hw, +cyA) → (-hw, -cyA)    length = 2*cyA
    //   BL corner:   quarter arc                    length = π*r/2
    //   bottom edge: (-cxA, -hh) → (+cxA, -hh)    length = 2*cxA
    //   BR corner:   quarter arc                    length = π*r/2
    const edgeH = 2 * cyA;  // right/left edge lengths
    const edgeW = 2 * cxA;  // top/bottom edge lengths
    const arcLen = (Math.PI * 0.5) * r; // quarter-arc length
    const perimeter = 2 * edgeH + 2 * edgeW + 4 * arcLen;

    // Segment lengths in order
    const segLens = [edgeH, arcLen, edgeW, arcLen, edgeH, arcLen, edgeW, arcLen];

    for (let i = 0; i < nVerts; i++) {
        const d = (i / nVerts) * perimeter;
        let u, v;
        let remaining = d;
        if (remaining < segLens[0]) {
            // Right edge: (+hw, -cyA) → (+hw, +cyA)
            const t = edgeH > 0 ? remaining / edgeH : 0;
            u = hw; v = -cyA + t * 2 * cyA;
        } else if ((remaining -= segLens[0]) < segLens[1]) {
            // TR corner arc: center (+cxA, +cyA), from 0 to π/2
            const a = arcLen > 0 ? (remaining / arcLen) * (Math.PI * 0.5) : 0;
            u = cxA + r * Math.cos(a); v = cyA + r * Math.sin(a);
        } else if ((remaining -= segLens[1]) < segLens[2]) {
            // Top edge: (+cxA, +hh) → (-cxA, +hh)
            const t = edgeW > 0 ? remaining / edgeW : 0;
            u = cxA - t * 2 * cxA; v = hh;
        } else if ((remaining -= segLens[2]) < segLens[3]) {
            // TL corner arc: center (-cxA, +cyA), from π/2 to π
            const a = (Math.PI * 0.5) + (arcLen > 0 ? (remaining / arcLen) * (Math.PI * 0.5) : 0);
            u = -cxA + r * Math.cos(a); v = cyA + r * Math.sin(a);
        } else if ((remaining -= segLens[3]) < segLens[4]) {
            // Left edge: (-hw, +cyA) → (-hw, -cyA)
            const t = edgeH > 0 ? remaining / edgeH : 0;
            u = -hw; v = cyA - t * 2 * cyA;
        } else if ((remaining -= segLens[4]) < segLens[5]) {
            // BL corner arc: center (-cxA, -cyA), from π to 3π/2
            const a = Math.PI + (arcLen > 0 ? (remaining / arcLen) * (Math.PI * 0.5) : 0);
            u = -cxA + r * Math.cos(a); v = -cyA + r * Math.sin(a);
        } else if ((remaining -= segLens[5]) < segLens[6]) {
            // Bottom edge: (-cxA, -hh) → (+cxA, -hh)
            const t = edgeW > 0 ? remaining / edgeW : 0;
            u = -cxA + t * 2 * cxA; v = -hh;
        } else {
            // BR corner arc: center (+cxA, -cyA), from 3π/2 to 2π
            remaining -= segLens[6];
            const a = Math.PI * 1.5 + (arcLen > 0 ? (remaining / arcLen) * (Math.PI * 0.5) : 0);
            u = cxA + r * Math.cos(a); v = -cyA + r * Math.sin(a);
        }
        out[i * 2] = u;
        out[i * 2 + 1] = v;
    }
}

// Build a variable-cross-section tube BufferGeometry from per-spine-point
// parameter arrays. Topology: nSpine rings × nCs vertices, connected by
// quad strips into (nSpine - 1) ring pairs. All rings share the same
// azimuthal ordering so indices are consistent.
//
// - spine:        Float32Array length nSpine*3
// - widths:       Float32Array length nSpine
// - heights:      Float32Array length nSpine
// - orientations: Float32Array length nSpine*4 quaternions, or null (constant-up derived)
// - upVector:     [x, y, z] up direction for constant-up frame (default [0,0,1])
// - ringColors:   Float32Array length nSpine*3 RGB (0..1), or null
// - crossSection: "rounded_rect"
// - cornerRadiusFrac: number in [0, 0.5]
// - nCs:          number of cross-section vertices
//
// Returns { geometry, ringPairs, indicesPerRingPair, nCs }.
function buildParametricTubeGeometry(
    spine, widths, heights,
    orientations, upVector, ringColors,
    crossSection, cornerRadiusFrac, nCs,
) {
    if (crossSection !== 'rounded_rect') {
        throw new Error(`Unsupported cross_section '${crossSection}'`);
    }
    if (!Number.isInteger(nCs) || nCs < 3) {
        throw new Error(`parametric_tube n_cross_section_verts must be an integer >=3, got ${nCs}`);
    }
    const nSpine = spine.length / 3;
    if (nSpine < 2) {
        throw new Error(`parametric_tube needs >=2 spine points, got ${nSpine}`);
    }
    if (widths.length !== nSpine) {
        throw new Error(`parametric_tube widths length ${widths.length} does not match spine ${nSpine}`);
    }
    if (heights.length !== nSpine) {
        throw new Error(`parametric_tube heights length ${heights.length} does not match spine ${nSpine}`);
    }

    // Dome cap vertex layout (appended after tube rings):
    //   startCapRings(nCapRings * nCs) | startPole(1) |
    //   endCapRings(nCapRings * nCs) | endPole(1)
    // Each cap has nCapRings latitude rings that shrink from the tube edge
    // ring toward a pole, forming a hemisphere-like dome.
    const nCapRings = 3;
    const capAngles = new Float32Array(nCapRings);
    for (let k = 0; k < nCapRings; k++) {
        capAngles[k] = ((k + 1) / (nCapRings + 1)) * (Math.PI * 0.5);
    }
    const startCapBase = nSpine * nCs;
    const startPoleIdx = startCapBase + nCapRings * nCs;
    const endCapBase = startPoleIdx + 1;
    const endPoleIdx = endCapBase + nCapRings * nCs;
    const totalVerts = endPoleIdx + 1;
    // Per cap: nCapRings quad strips + 1 triangle fan to pole
    const capIndicesPerCap = nCapRings * nCs * 6 + nCs * 3;

    const positions = new Float32Array(totalVerts * 3);
    const colors = ringColors ? new Float32Array(totalVerts * 3) : null;
    const section = new Float32Array(nCs * 2);
    // Store per-spine-point local frames (U, V) for frontier-ring morphing.
    const localFrames = new Float32Array(nSpine * 6);

    // Compute local frames: tangent T + width axis U + height axis V.
    // V is anchored to global +Z (up) so "height" means "up" for
    // horizontal beads. Parallel-transport (U, V) along the spine so
    // there is no azimuthal drift at curvature changes. When the tangent
    // is parallel to global up (vertical bead), seed V from global +X.
    const _T = new THREE.Vector3();
    const _U = new THREE.Vector3();
    const _V = new THREE.Vector3();
    const _quat = new THREE.Quaternion();
    // Constant-up direction: project onto plane perpendicular to tangent at each point.
    const up = new THREE.Vector3(
        upVector ? upVector[0] : 0,
        upVector ? upVector[1] : 0,
        upVector ? upVector[2] : 1,
    ).normalize();
    // Fallback when tangent is parallel to up.
    const upFallback = new THREE.Vector3();
    if (Math.abs(up.x) < 0.9) upFallback.set(1, 0, 0);
    else upFallback.set(0, 1, 0);

    // Precompute tangents via central difference.
    const tangents = new Float32Array(nSpine * 3);
    for (let i = 0; i < nSpine; i++) {
        const iPrev = Math.max(i - 1, 0);
        const iNext = Math.min(i + 1, nSpine - 1);
        let tx = spine[iNext * 3] - spine[iPrev * 3];
        let ty = spine[iNext * 3 + 1] - spine[iPrev * 3 + 1];
        let tz = spine[iNext * 3 + 2] - spine[iPrev * 3 + 2];
        const len = Math.hypot(tx, ty, tz);
        if (len > 1e-12) { tx /= len; ty /= len; tz /= len; }
        else { tx = 1; ty = 0; tz = 0; }
        tangents[i * 3] = tx;
        tangents[i * 3 + 1] = ty;
        tangents[i * 3 + 2] = tz;
    }

    for (let i = 0; i < nSpine; i++) {
        _T.set(tangents[i * 3], tangents[i * 3 + 1], tangents[i * 3 + 2]);
        if (orientations) {
            _quat.set(
                orientations[i * 4],
                orientations[i * 4 + 1],
                orientations[i * 4 + 2],
                orientations[i * 4 + 3],
            );
            _U.set(1, 0, 0).applyQuaternion(_quat);
            _V.set(0, 1, 0).applyQuaternion(_quat);
        } else {
            // Constant-up: project up vector onto plane perpendicular to tangent.
            const seed = Math.abs(_T.dot(up)) > 0.99 ? upFallback : up;
            _V.copy(seed).addScaledVector(_T, -seed.dot(_T)).normalize();
            _U.copy(_V).cross(_T).normalize();
            _V.copy(_T).cross(_U).normalize();
        }
        localFrames[i * 6]     = _U.x; localFrames[i * 6 + 1] = _U.y; localFrames[i * 6 + 2] = _U.z;
        localFrames[i * 6 + 3] = _V.x; localFrames[i * 6 + 4] = _V.y; localFrames[i * 6 + 5] = _V.z;

        const w = widths[i];
        const h = heights[i];
        sampleRoundedRectInto(section, nCs, w, h, cornerRadiusFrac);
        const sx = spine[i * 3];
        const sy = spine[i * 3 + 1];
        const sz = spine[i * 3 + 2];
        const ringBase = i * nCs * 3;
        for (let j = 0; j < nCs; j++) {
            const u = section[j * 2];
            const v = section[j * 2 + 1];
            positions[ringBase + j * 3] = sx + u * _U.x + v * _V.x;
            positions[ringBase + j * 3 + 1] = sy + u * _U.y + v * _V.y;
            positions[ringBase + j * 3 + 2] = sz + u * _U.z + v * _V.z;
        }
        if (colors) {
            const r = ringColors[i * 3];
            const g = ringColors[i * 3 + 1];
            const b = ringColors[i * 3 + 2];
            for (let j = 0; j < nCs; j++) {
                colors[ringBase + j * 3] = r;
                colors[ringBase + j * 3 + 1] = g;
                colors[ringBase + j * 3 + 2] = b;
            }
        }
    }

    // --- Revolution cap vertices ---
    // The cap is a surface of revolution: the flat cross-section face is
    // revolved around the V (height) axis. Each vertex at (u, v) sweeps
    // through angle θ: U_component = u*cos(θ), T_offset = |u|*sin(θ),
    // V_component = v (unchanged). In top view this gives a perfect
    // semicircle of radius hw extending along the tangent direction.
    // tangentSign = -1 for start cap (extends in -T), +1 for end cap.
    function buildDomeCap(spineIdx, capBaseVert, poleVertIdx, tangentSign) {
        const sx = spine[spineIdx * 3], sy = spine[spineIdx * 3 + 1], sz = spine[spineIdx * 3 + 2];
        const w = widths[spineIdx], h = heights[spineIdx];
        const ux = localFrames[spineIdx * 6], uy = localFrames[spineIdx * 6 + 1], uz = localFrames[spineIdx * 6 + 2];
        const vx = localFrames[spineIdx * 6 + 3], vy = localFrames[spineIdx * 6 + 4], vz = localFrames[spineIdx * 6 + 5];
        // Tangent = U × V
        const tx = uy * vz - uz * vy, ty = uz * vx - ux * vz, tz = ux * vy - uy * vx;
        // Sample cross-section once at full width/height
        sampleRoundedRectInto(section, nCs, w, h, cornerRadiusFrac);
        for (let k = 0; k < nCapRings; k++) {
            const theta = capAngles[k];
            const cosT = Math.cos(theta);
            const sinT = Math.sin(theta) * tangentSign;
            const ringBase = (capBaseVert + k * nCs) * 3;
            for (let j = 0; j < nCs; j++) {
                const cu = section[j * 2], cv = section[j * 2 + 1];
                const tOff = Math.abs(cu) * sinT;
                positions[ringBase + j * 3]     = sx + cu * cosT * ux + cv * vx + tOff * tx;
                positions[ringBase + j * 3 + 1] = sy + cu * cosT * uy + cv * vy + tOff * ty;
                positions[ringBase + j * 3 + 2] = sz + cu * cosT * uz + cv * vz + tOff * tz;
            }
        }
        // Pole at the tip of the semicircle: spine + hw * tangentSign * T
        const hw = w * 0.5;
        positions[poleVertIdx * 3]     = sx + hw * tangentSign * tx;
        positions[poleVertIdx * 3 + 1] = sy + hw * tangentSign * ty;
        positions[poleVertIdx * 3 + 2] = sz + hw * tangentSign * tz;
    }
    buildDomeCap(0, startCapBase, startPoleIdx, -1);
    buildDomeCap(nSpine - 1, endCapBase, endPoleIdx, +1);
    // Cap colors (replicate ring color to all cap vertices)
    if (colors) {
        const capVertsPerCap = nCapRings * nCs + 1;
        const r0 = ringColors[0], g0 = ringColors[1], b0 = ringColors[2];
        for (let v = 0; v < capVertsPerCap; v++) {
            const dst = (startCapBase + v) * 3;
            colors[dst] = r0; colors[dst + 1] = g0; colors[dst + 2] = b0;
        }
        const rN = ringColors[(nSpine - 1) * 3], gN = ringColors[(nSpine - 1) * 3 + 1], bN = ringColors[(nSpine - 1) * 3 + 2];
        for (let v = 0; v < capVertsPerCap; v++) {
            const dst = (endCapBase + v) * 3;
            colors[dst] = rN; colors[dst + 1] = gN; colors[dst + 2] = bN;
        }
        // pole
        colors[startPoleIdx * 3] = r0; colors[startPoleIdx * 3 + 1] = g0; colors[startPoleIdx * 3 + 2] = b0;
        colors[endPoleIdx * 3] = rN; colors[endPoleIdx * 3 + 1] = gN; colors[endPoleIdx * 3 + 2] = bN;
    }

    // --- Index buffer: [start_cap_dome | ring_pairs | end_cap_dome] ---
    const ringPairs = nSpine - 1;
    const indicesPerRingPair = nCs * 6;
    const totalIndexCount = capIndicesPerCap + ringPairs * indicesPerRingPair + capIndicesPerCap;
    const IndexCtor = totalVerts > 65535 ? Uint32Array : Uint16Array;
    const indices = new IndexCtor(totalIndexCount);
    let p = 0;

    // Helper: build dome cap indices.
    // tubeRingBase = vertex index of the tube ring that the cap connects to.
    // capBaseVert = first vertex index of the cap rings.
    // poleVertIdx = vertex index of the pole.
    // reverse = true for start cap (winding reversed since dome goes in -T).
    function buildDomeCapIndices(tubeRingBase, capBaseVert, poleVertIdx, reverse) {
        for (let k = 0; k < nCapRings; k++) {
            const innerBase = k === 0 ? tubeRingBase : capBaseVert + (k - 1) * nCs;
            const outerBase = capBaseVert + k * nCs;
            for (let j = 0; j < nCs; j++) {
                const jN = (j + 1) % nCs;
                if (reverse) {
                    indices[p++] = innerBase + j;
                    indices[p++] = outerBase + j;
                    indices[p++] = outerBase + jN;
                    indices[p++] = innerBase + j;
                    indices[p++] = outerBase + jN;
                    indices[p++] = innerBase + jN;
                } else {
                    indices[p++] = innerBase + j;
                    indices[p++] = innerBase + jN;
                    indices[p++] = outerBase + jN;
                    indices[p++] = innerBase + j;
                    indices[p++] = outerBase + jN;
                    indices[p++] = outerBase + j;
                }
            }
        }
        // Fan to pole
        const lastRing = capBaseVert + (nCapRings - 1) * nCs;
        for (let j = 0; j < nCs; j++) {
            const jN = (j + 1) % nCs;
            if (reverse) {
                indices[p++] = poleVertIdx;
                indices[p++] = lastRing + jN;
                indices[p++] = lastRing + j;
            } else {
                indices[p++] = poleVertIdx;
                indices[p++] = lastRing + j;
                indices[p++] = lastRing + jN;
            }
        }
    }
    buildDomeCapIndices(0, startCapBase, startPoleIdx, true);
    // Ring pairs
    for (let i = 0; i < ringPairs; i++) {
        const a0 = i * nCs;
        const b0 = (i + 1) * nCs;
        for (let j = 0; j < nCs; j++) {
            const jNext = (j + 1) % nCs;
            const a = a0 + j;
            const b = a0 + jNext;
            const c = b0 + jNext;
            const d = b0 + j;
            indices[p++] = a;
            indices[p++] = b;
            indices[p++] = c;
            indices[p++] = a;
            indices[p++] = c;
            indices[p++] = d;
        }
    }
    // End cap dome
    const endCapOffset = p;
    buildDomeCapIndices((nSpine - 1) * nCs, endCapBase, endPoleIdx, false);
    // Extract end cap index pattern for dynamic relocation during draw_range
    const endCapPattern = indices.slice(endCapOffset, endCapOffset + capIndicesPerCap);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    if (colors) geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeVertexNormals();

    return {
        geometry, ringPairs, indicesPerRingPair, nCs,
        localFrames, capAngles, nCapRings,
        capIndicesPerCap, endCapBase, endPoleIdx, endCapPattern,
    };
}

// Decode N packed uint32 RGB values (0x00RRGGBB) into a ring-major Float32 RGB
// attribute of length nSpine*nCs*3, replicating each ring color across its nCs
// vertices. Writes into `out` if provided (must be pre-sized) to avoid
// reallocating on every color update.
function expandRingColors(packedColors, nSpine, nCs, out) {
    const total = nSpine * nCs * 3;
    if (!out || out.length < total) {
        out = new Float32Array(total);
    }
    for (let i = 0; i < nSpine; i++) {
        const c = packedColors[i];
        const r = ((c >> 16) & 0xff) / 255;
        const g = ((c >> 8) & 0xff) / 255;
        const b = (c & 0xff) / 255;
        const base = i * nCs * 3;
        for (let j = 0; j < nCs; j++) {
            out[base + j * 3] = r;
            out[base + j * 3 + 1] = g;
            out[base + j * 3 + 2] = b;
        }
    }
    return out;
}

// Restore previously morphed frontier ring to its original positions (and
// colors if applicable).  Called when the frontier advances past a ring or
// when draw_range reaches an exact ring boundary.
function restoreFrontierRing(obj) {
    const md = obj.userData.tubeMorphData;
    if (!md || md.savedRingIndex == null) return;
    const nCs = obj.userData.tubeNCs;
    const ringBase = md.savedRingIndex * nCs * 3;
    const posAttr = obj.geometry.getAttribute('position');
    posAttr.array.set(md.savedRing, ringBase);
    posAttr.needsUpdate = true;
    if (md.savedRingColors) {
        const colAttr = obj.geometry.getAttribute('color');
        if (colAttr) {
            colAttr.array.set(md.savedRingColors, ringBase);
            colAttr.needsUpdate = true;
        }
    }
    md.savedRingIndex = null;
    md.morphedState = null;
}

// Morph the frontier ring of a parametric tube to the interpolated spine
// position.  Returns the number of ring pairs that should be visible
// (complete pairs + the morphed frontier).
function morphFrontierRing(obj, fracRingPairs) {
    const ud = obj.userData;
    const md = ud.tubeMorphData;
    if (!md) return Math.floor(fracRingPairs);

    const nCs = ud.tubeNCs;
    const ringPairs = ud.tubeRingPairs;
    const completePairs = Math.floor(fracRingPairs);
    const frac = fracRingPairs - completePairs;

    if (frac < 1e-6 || completePairs >= ringPairs) {
        restoreFrontierRing(obj);
        return completePairs;
    }

    const iA = completePairs;      // lerp FROM
    const iB = completePairs + 1;  // ring we overwrite (lerp TO)

    // If the frontier moved to a different ring, restore the old one first
    // and save the new ring's original data.
    if (md.savedRingIndex !== iB) {
        restoreFrontierRing(obj);
        const posArr = obj.geometry.getAttribute('position').array;
        const ringBase = iB * nCs * 3;
        md.savedRing.set(posArr.subarray(ringBase, ringBase + nCs * 3));
        // Save colors if applicable
        if (ud.tubeHasColors) {
            const colAttr = obj.geometry.getAttribute('color');
            if (colAttr) {
                if (!md.savedRingColors) md.savedRingColors = new Float32Array(nCs * 3);
                md.savedRingColors.set(colAttr.array.subarray(ringBase, ringBase + nCs * 3));
            }
        }
        md.savedRingIndex = iB;
    }

    // Lerp spine position
    const sx = md.spine[iA * 3]     * (1 - frac) + md.spine[iB * 3]     * frac;
    const sy = md.spine[iA * 3 + 1] * (1 - frac) + md.spine[iB * 3 + 1] * frac;
    const sz = md.spine[iA * 3 + 2] * (1 - frac) + md.spine[iB * 3 + 2] * frac;

    // Lerp width & height
    const w = md.widths[iA] * (1 - frac) + md.widths[iB] * frac;
    const h = md.heights[iA] * (1 - frac) + md.heights[iB] * frac;

    // Lerp local frame vectors (U, V) — component-wise + normalize
    let ux = md.localFrames[iA * 6]     * (1 - frac) + md.localFrames[iB * 6]     * frac;
    let uy = md.localFrames[iA * 6 + 1] * (1 - frac) + md.localFrames[iB * 6 + 1] * frac;
    let uz = md.localFrames[iA * 6 + 2] * (1 - frac) + md.localFrames[iB * 6 + 2] * frac;
    let uLen = Math.hypot(ux, uy, uz);
    if (uLen > 1e-12) { ux /= uLen; uy /= uLen; uz /= uLen; }

    let vx = md.localFrames[iA * 6 + 3] * (1 - frac) + md.localFrames[iB * 6 + 3] * frac;
    let vy = md.localFrames[iA * 6 + 4] * (1 - frac) + md.localFrames[iB * 6 + 4] * frac;
    let vz = md.localFrames[iA * 6 + 5] * (1 - frac) + md.localFrames[iB * 6 + 5] * frac;
    let vLen = Math.hypot(vx, vy, vz);
    if (vLen > 1e-12) { vx /= vLen; vy /= vLen; vz /= vLen; }

    // Store morphed state for updateEndCap to use
    md.morphedState = { sx, sy, sz, w, h, ux, uy, uz, vx, vy, vz };

    // Sample cross-section at interpolated width/height
    sampleRoundedRectInto(md.section, nCs, w, h, md.cornerRadiusFrac);

    // Write morphed vertices into the position buffer at ring iB
    const posAttr = obj.geometry.getAttribute('position');
    const pos = posAttr.array;
    const ringBase = iB * nCs * 3;
    for (let j = 0; j < nCs; j++) {
        const lu = md.section[j * 2];
        const lv = md.section[j * 2 + 1];
        pos[ringBase + j * 3]     = sx + lu * ux + lv * vx;
        pos[ringBase + j * 3 + 1] = sy + lu * uy + lv * vy;
        pos[ringBase + j * 3 + 2] = sz + lu * uz + lv * vz;
    }
    posAttr.needsUpdate = true;

    // Lerp colors if present
    if (ud.tubeHasColors && md.ringColors) {
        const colAttr = obj.geometry.getAttribute('color');
        if (colAttr) {
            const cols = colAttr.array;
            const rA = md.ringColors[iA * 3], gA = md.ringColors[iA * 3 + 1], bA = md.ringColors[iA * 3 + 2];
            const rB = md.ringColors[iB * 3], gB = md.ringColors[iB * 3 + 1], bB = md.ringColors[iB * 3 + 2];
            const cr = rA * (1 - frac) + rB * frac;
            const cg = gA * (1 - frac) + gB * frac;
            const cb = bA * (1 - frac) + bB * frac;
            for (let j = 0; j < nCs; j++) {
                cols[ringBase + j * 3]     = cr;
                cols[ringBase + j * 3 + 1] = cg;
                cols[ringBase + j * 3 + 2] = cb;
            }
            colAttr.needsUpdate = true;
        }
    }

    return completePairs + 1;
}

// Update the end cap revolution surface to match the last visible ring.
// Uses morphed spine/frame/width/height stored by morphFrontierRing, or reads
// original data for un-morphed rings.
function updateEndCap(obj, lastVisibleRing) {
    const ud = obj.userData;
    const md = ud.tubeMorphData;
    if (!md) return;
    const nCs = ud.tubeNCs;
    const nCapRings = md.capAngles.length;
    const posAttr = obj.geometry.getAttribute('position');
    const pos = posAttr.array;

    // Determine spine pos, width, height, frame at the frontier ring
    let sx, sy, sz, w, h, ux, uy, uz, vx, vy, vz;
    if (md.morphedState) {
        const ms = md.morphedState;
        sx = ms.sx; sy = ms.sy; sz = ms.sz;
        w = ms.w; h = ms.h;
        ux = ms.ux; uy = ms.uy; uz = ms.uz;
        vx = ms.vx; vy = ms.vy; vz = ms.vz;
    } else {
        const i = lastVisibleRing;
        sx = md.spine[i * 3]; sy = md.spine[i * 3 + 1]; sz = md.spine[i * 3 + 2];
        w = md.widths[i]; h = md.heights[i];
        ux = md.localFrames[i * 6]; uy = md.localFrames[i * 6 + 1]; uz = md.localFrames[i * 6 + 2];
        vx = md.localFrames[i * 6 + 3]; vy = md.localFrames[i * 6 + 4]; vz = md.localFrames[i * 6 + 5];
    }
    // Tangent = U × V
    const tx = uy * vz - uz * vy, ty = uz * vx - ux * vz, tz = ux * vy - uy * vx;

    // Sample cross-section once at full width/height
    sampleRoundedRectInto(md.section, nCs, w, h, md.cornerRadiusFrac);

    // Revolution: each vertex at (cu, cv) sweeps around V axis
    const ecBase = ud.tubeEndCapBase;
    for (let k = 0; k < nCapRings; k++) {
        const theta = md.capAngles[k];
        const cosT = Math.cos(theta);
        const sinT = Math.sin(theta);
        const ringBase = (ecBase + k * nCs) * 3;
        for (let j = 0; j < nCs; j++) {
            const cu = md.section[j * 2], cv = md.section[j * 2 + 1];
            const tOff = Math.abs(cu) * sinT;
            pos[ringBase + j * 3]     = sx + cu * cosT * ux + cv * vx + tOff * tx;
            pos[ringBase + j * 3 + 1] = sy + cu * cosT * uy + cv * vy + tOff * ty;
            pos[ringBase + j * 3 + 2] = sz + cu * cosT * uz + cv * vz + tOff * tz;
        }
    }
    // Pole at tip of semicircle: spine + hw * T
    const hw = w * 0.5;
    const poleIdx = ud.tubeEndPoleIdx;
    pos[poleIdx * 3]     = sx + hw * tx;
    pos[poleIdx * 3 + 1] = sy + hw * ty;
    pos[poleIdx * 3 + 2] = sz + hw * tz;
    posAttr.needsUpdate = true;

    // Update end cap colors if applicable
    if (ud.tubeHasColors && md.ringColors) {
        const colAttr = obj.geometry.getAttribute('color');
        if (colAttr) {
            const cols = colAttr.array;
            // Use frontier ring's color for all end cap vertices
            const colSrcBase = lastVisibleRing * nCs * 3;
            const cr = cols[colSrcBase], cg = cols[colSrcBase + 1], cb = cols[colSrcBase + 2];
            const capVerts = nCapRings * nCs + 1;
            for (let v = 0; v < capVerts; v++) {
                const dst = (ecBase + v) * 3;
                cols[dst] = cr; cols[dst + 1] = cg; cols[dst + 2] = cb;
            }
            colAttr.needsUpdate = true;
        }
    }
}

// Restore end cap indices that were relocated by a previous relocateEndCap call.
function restoreRelocatedEndCap(obj) {
    const md = obj.userData.tubeMorphData;
    if (!md || md.savedCapOffset < 0) return;
    const indexAttr = obj.geometry.getIndex();
    indexAttr.array.set(md.savedCapIndices, md.savedCapOffset);
    md.savedCapOffset = -1;
    indexAttr.needsUpdate = true;
}

// Move end cap fan indices to sit right after the visible ring pairs so
// setDrawRange(0, startCap + visiblePairs + endCap) draws them correctly.
function relocateEndCap(obj, visiblePairs) {
    const ud = obj.userData;
    const md = ud.tubeMorphData;
    const capPer = ud.tubeCapIndicesPerCap;
    const perPair = ud.tubeIndicesPerRingPair;
    const indexAttr = obj.geometry.getIndex();
    const idx = indexAttr.array;

    // Restore previously relocated end cap indices
    restoreRelocatedEndCap(obj);

    // Place end cap right after start_cap + visible ring pairs
    const offset = capPer + visiblePairs * perPair;
    // Save the indices we're about to overwrite
    md.savedCapIndices.set(idx.subarray(offset, offset + capPer));
    md.savedCapOffset = offset;
    // Write end cap pattern
    idx.set(md.endCapPattern, offset);
    indexAttr.needsUpdate = true;
}

function applyParametricTubeDrawRange(obj, value) {
    const ud = obj.userData;
    const ringPairs = ud.tubeRingPairs;
    const perPair = ud.tubeIndicesPerRingPair;
    const capPer = ud.tubeCapIndicesPerCap || 0;
    const clamped = Math.max(0, Math.min(1, value));
    const fracRingPairs = clamped * ringPairs;

    if (!ud.tubeMorphData) {
        const pairs = Math.floor(fracRingPairs);
        obj.geometry.setDrawRange(0, capPer + pairs * perPair + capPer);
        return;
    }

    if (clamped < 1e-6) {
        restoreFrontierRing(obj);
        restoreRelocatedEndCap(obj);
        obj.geometry.setDrawRange(0, 0);
        return;
    }

    const visiblePairs = morphFrontierRing(obj, fracRingPairs);
    updateEndCap(obj, visiblePairs);
    relocateEndCap(obj, visiblePairs);
    // start_cap + ring_pairs + end_cap (end cap now sits right after visible pairs)
    obj.geometry.setDrawRange(0, capPer + visiblePairs * perPair + capPer);
}

export class ThreeJSViewer {
    /**
     * @param {HTMLElement} container - The DOM element to mount into
     * @param {Object} [options]
     * @param {string}  [options.wsUrl]       - Full WebSocket URL (e.g., "ws://localhost:5666")
     * @param {number}  [options.wsPort]      - Alternative: just specify port
     * @param {boolean} [options.autoConnect=true] - Whether to connect WebSocket immediately
     * @param {string}  [options.htmlTemplate] - HTML template string for UI controls
     * @param {Object}  [options.cubemapData]  - {px,nx,py,ny,pz,nz} base64 JPEG strings
     */
    constructor(container, options = {}) {
        if (!container) throw new Error('ThreeJSViewer: container element is required');

        this.container = container;
        this._options = options;

        // Make container focusable for keyboard events
        if (!container.hasAttribute('tabindex')) {
            container.tabIndex = 0;
        }
        // Ensure positioned for absolute children
        const pos = getComputedStyle(container).position;
        if (pos === 'static') {
            container.style.position = 'relative';
        }

        // Create wrapper
        this.el = document.createElement('div');
        this.el.className = 'threejs-viewer';
        this.el.style.width = '100%';
        this.el.style.height = '100%';
        container.appendChild(this.el);

        // Inject HTML
        if (!options.htmlTemplate) throw new Error('ThreeJSViewer: options.htmlTemplate is required');
        this.el.innerHTML = options.htmlTemplate;

        // Resolve WebSocket URL
        const urlParams = new URLSearchParams(window.location.search);
        if (options.wsUrl) {
            this._wsUrl = options.wsUrl;
        } else {
            const port = options.wsPort || parseInt(urlParams.get('ws_port')) || 5666;
            this._wsUrl = `ws://localhost:${port}`;
        }

        // State
        this._objects = new Map();
        this._mixers = new Map();
        this._objGeneration = 0;
        this._mixerGeneration = 0;
        this._pendingFetches = 0;
        this._sceneGeneration = 0;
        this._animGeneration = 0;
        this._assetsComplete = false;
        this._ws = null;
        this._reconnectTimeout = null;
        this._animationFrameId = null;

        // Animation state
        this._animation = null;
        this._animationPlaying = false;
        this._animationTime = 0;
        this._animationSpeed = 1;
        this._animationLoop = true;
        this._lastAnimationUpdate = 0;
        this._baselineVisibility = new Map();
        this._speedIndex = 6; // starts at 1x

        // Clipping state
        this._clipEnabled = false;
        this._clipHelperVisible = true;
        this._clipAxis = 'x+';
        this._clipSlabMode = false;
        this._clipSlabThickness = 2.0;
        this._clipPosition = 0;
        this._clipDefaults = null;
        this._clipSyncFromGizmo = false;

        // Scrubbing state
        this._scrubbing = false;
        this._wasPlayingBeforeScrub = false;

        // Camera state
        this._isOrtho = false;

        // Camera tracking state
        this._trackTargetId = null;
        this._trackMode = 'off';       // 'off' | 'follow' | 'lookat' | 'scripted'
        this._trackLastPos = new THREE.Vector3();
        this._trackHasLastPos = false;
        this._trackInteractive = false; // true when user pressed T (interactive override)
        this._tmpTrackPos = new THREE.Vector3();   // reusable scratch vectors
        this._tmpTrackDelta = new THREE.Vector3();

        // Channel apply functions
        this._CHANNEL_APPLY = makeChannelApply(this);

        // Cache DOM refs
        this._cacheElements();

        // Init Three.js
        this._initThreeJS();

        // Init clipping
        this._initClipping();

        // Bind events
        this._bindEvents();

        // Connect
        if (options.autoConnect !== false) {
            this.connect();
        }

        // Start render loop
        this._lastFrameTime = performance.now();
        this._lastUIUpdate = 0;
        this._animate = this._animate.bind(this);
        this._animationFrameId = requestAnimationFrame(this._animate);

        console.log(`threejs-viewer v${VIEWER_VERSION}`);
    }

    _cacheElements() {
        const q = (sel) => this.el.querySelector(sel);
        this._statusDot = q('.tjsv-status-dot');
        this._statusText = q('.tjsv-status-text');
        this._btnOrtho = q('.tjsv-btn-ortho');
        this._btnClip = q('.tjsv-btn-clip');
        this._clipPanelEl = q('.tjsv-clipping-panel');
        this._clipDistanceSlider = q('.tjsv-clip-distance');
        this._clipDistanceValue = q('.tjsv-clip-distance-value');
        this._clipThicknessSlider = q('.tjsv-clip-thickness');
        this._clipThicknessValue = q('.tjsv-clip-thickness-value');
        this._clipNxInput = q('.tjsv-clip-nx');
        this._clipNyInput = q('.tjsv-clip-ny');
        this._clipNzInput = q('.tjsv-clip-nz');
        this._clipModeSingle = q('.tjsv-clip-mode-single');
        this._clipModeSlab = q('.tjsv-clip-mode-slab');
        this._clipThicknessSection = q('.tjsv-clip-thickness-section');
        this._clipClose = q('.tjsv-clip-close');
        this._animControlsEl = q('.tjsv-animation-controls');
        this._timelineProgressEl = q('.tjsv-timeline-progress');
        this._timelineMarkersEl = q('.tjsv-timeline-markers');
        this._currentTimeEl = q('.tjsv-current-time');
        this._totalTimeEl = q('.tjsv-total-time');
        this._currentFrameEl = q('.tjsv-current-frame');
        this._totalFramesEl = q('.tjsv-total-frames');
        this._btnPlay = q('.tjsv-btn-play');
        this._btnLoop = q('.tjsv-btn-loop');
        this._speedDisplayEl = q('.tjsv-speed-display');
        this._timelineContainer = q('.tjsv-timeline-container');
        this._btnTrack = q('.tjsv-btn-track');
    }

    _initThreeJS() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;

        // Scene
        this._scene = new THREE.Scene();
        this._scene.background = new THREE.Color(0x222222);

        // Cameras
        this._perspCamera = new THREE.PerspectiveCamera(75, w / h, 0.1, 1000);
        this._perspCamera.position.set(5, -5, 5);
        this._perspCamera.up.set(0, 0, 1);

        const aspect = w / h;
        this._orthoCamera = new THREE.OrthographicCamera(
            -ORTHO_FRUSTUM * aspect, ORTHO_FRUSTUM * aspect,
            ORTHO_FRUSTUM, -ORTHO_FRUSTUM, 0.1, 1000
        );
        this._orthoCamera.position.copy(this._perspCamera.position);
        this._orthoCamera.up.set(0, 0, 1);

        this._camera = this._perspCamera;

        // Renderer
        this._renderer = new THREE.WebGLRenderer({ antialias: true });
        this._renderer.setSize(w, h);
        this._renderer.setPixelRatio(window.devicePixelRatio);
        this._renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this._renderer.toneMappingExposure = 1.5;
        this._renderer.localClippingEnabled = true;
        this.el.appendChild(this._renderer.domElement);

        // Environment cubemap
        this._loadCubemap();

        // Controls
        this._controls = new OrbitControls(this._camera, this._renderer.domElement);
        this._controls.enableDamping = true;

        // ViewHelper
        this._viewHelper = new ViewHelper(this._camera, this._renderer.domElement);
        this._viewHelper.center = this._controls.target;
        this._renderer.domElement.addEventListener('click', (e) => {
            if (this._viewHelper.handleClick(e)) {
                this._controls.target.copy(this._viewHelper.center);
            }
        });

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 1.5);
        this._scene.add(ambientLight);

        // Grid helper on XY plane (Z-up) — hidden by default
        this._gridHelper = new THREE.GridHelper(10, 10);
        this._gridHelper.rotation.x = Math.PI / 2;
        this._gridHelper.visible = false;
        this._scene.add(this._gridHelper);

        // Loaders
        this._loaders = {
            obj: new OBJLoader(),
            gltf: new GLTFLoader(),
            glb: new GLTFLoader(),
            fbx: new FBXLoader(),
            dae: new ColladaLoader(),
            stl: new STLLoader(),
            ply: new PLYLoader(),
            '3ds': new TDSLoader()
        };

        // Scene bounds caching for dynamic near/far
        this._sceneSphere = new THREE.Sphere();
        this._sceneBoundsDirty = true;
        this._boundsFrameCounter = 0;

        // ResizeObserver
        this._resizeObserver = new ResizeObserver(entries => {
            const { width, height } = entries[0].contentRect;
            this.resize(width, height);
        });
        this._resizeObserver.observe(this.container);
    }

    _loadCubemap() {
        const cubemapData = this._options.cubemapData;
        if (!cubemapData) {
            console.warn('ThreeJSViewer: no cubemapData provided, skipping environment map');
            return;
        }
        const pmremGenerator = new THREE.PMREMGenerator(this._renderer);
        const faces = ['px', 'nx', 'py', 'ny', 'pz', 'nz'];
        const cubeTexture = new THREE.CubeTexture();
        cubeTexture.colorSpace = THREE.SRGBColorSpace;
        let loaded = 0;
        let failed = false;
        const scene = this._scene;
        faces.forEach((face, i) => {
            const img = new Image();
            img.onload = () => {
                if (failed) return;
                cubeTexture.images[i] = img;
                loaded++;
                if (loaded === 6) {
                    cubeTexture.needsUpdate = true;
                    const envMap = pmremGenerator.fromCubemap(cubeTexture).texture;
                    scene.environment = envMap;
                    scene.environmentIntensity = 2.0;
                    cubeTexture.dispose();
                    pmremGenerator.dispose();
                }
            };
            img.onerror = () => {
                if (failed) return;
                failed = true;
                console.error(`Failed to load cubemap face: ${face}`);
                cubeTexture.dispose();
                pmremGenerator.dispose();
            };
            img.src = 'data:image/jpeg;base64,' + cubemapData[face];
        });
    }

    _initClipping() {
        this._clipPlane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0);
        this._clipPlane2 = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0);
        this._clipPlanes = [this._clipPlane];

        // Anchor object
        this._clipAnchor = new THREE.Group();
        this._clipAnchor.visible = false;
        this._scene.add(this._clipAnchor);

        // Disc + ring helpers
        const discGeo = new THREE.CircleGeometry(5, 48);
        const discMat = new THREE.MeshBasicMaterial({
            color: 0x00aaff, transparent: true, opacity: 0.15,
            side: THREE.DoubleSide, depthWrite: false,
        });
        this._clipDisc = new THREE.Mesh(discGeo, discMat);
        this._clipAnchor.add(this._clipDisc);

        const ringGeo = new THREE.RingGeometry(4.95, 5, 48);
        const ringMat = new THREE.MeshBasicMaterial({
            color: 0x00aaff, transparent: true, opacity: 0.5,
            side: THREE.DoubleSide, depthWrite: false,
        });
        this._clipRing = new THREE.Mesh(ringGeo, ringMat);
        this._clipAnchor.add(this._clipRing);

        // Second disc + ring for slab mode
        this._clipDisc2 = new THREE.Mesh(discGeo, discMat.clone());
        this._clipDisc2.visible = false;
        this._clipAnchor.add(this._clipDisc2);
        this._clipRing2 = new THREE.Mesh(ringGeo, ringMat.clone());
        this._clipRing2.visible = false;
        this._clipAnchor.add(this._clipRing2);

        // TransformControls for rotating the plane
        this._clipGizmo = new TransformControls(this._camera, this._renderer.domElement);
        this._clipGizmo.attach(this._clipAnchor);
        this._clipGizmo.setMode('rotate');
        this._clipGizmo.visible = false;
        this._clipGizmo.enabled = false;
        this._clipGizmoHelper = this._clipGizmo.getHelper();
        this._clipGizmoHelper.visible = false;
        this._scene.add(this._clipGizmoHelper);

        // Disable orbit while dragging gizmo
        this._clipGizmo.addEventListener('dragging-changed', (e) => {
            this._controls.enabled = !e.value;
        });

        // Temp vectors for gizmo sync
        this._zAxis = new THREE.Vector3();
        this._localZ = new THREE.Vector3(0, 0, 1);

        this._clipGizmo.addEventListener('change', () => {
            if (!this._clipEnabled) return;
            this._clipSyncFromGizmo = true;
            this._clipAnchor.getWorldDirection(this._zAxis);
            this._clipPosition = this._clipAnchor.position.dot(this._zAxis);

            this._clipPlane.normal.copy(this._zAxis);
            this._updatePlaneConstants();
            this._updateDiscPositions();
            this._clipDistanceSlider.value = this._clipPosition;
            this._clipDistanceValue.textContent = this._clipPosition.toFixed(2);
            this._clipPanelEl.querySelectorAll('.clip-axis-buttons button').forEach(btn => {
                btn.classList.remove('active');
            });
            this._clipAxis = null;
            this._syncNormalInputs();
            this._clipSyncFromGizmo = false;
        });

        // Shared bbox objects for slider range calculation
        this._bbox = new THREE.Box3();
        this._bboxCenter = new THREE.Vector3();
    }

    _syncNormalInputs() {
        this._clipNxInput.value = this._clipPlane.normal.x.toFixed(2);
        this._clipNyInput.value = this._clipPlane.normal.y.toFixed(2);
        this._clipNzInput.value = this._clipPlane.normal.z.toFixed(2);
    }

    _syncAnchorFromPlane() {
        if (this._clipSyncFromGizmo) return;
        const normal = this._clipPlane.normal;
        this._clipAnchor.position.copy(normal).multiplyScalar(this._clipPosition);
        this._clipAnchor.quaternion.setFromUnitVectors(this._localZ, normal);
        this._updateDiscPositions();
    }

    _setClipAxis(axis) {
        this._clipAxis = axis;
        const normal = CLIP_AXIS_NORMALS[axis].clone();
        this._clipPlane.normal.copy(normal);
        this._updatePlaneConstants();
        this._syncAnchorFromPlane();
        this._syncNormalInputs();
        this._updateClipSliderRange();
        this._clipPanelEl.querySelectorAll('.clip-axis-buttons button').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.axis === axis);
        });
    }

    _setClipDistance(d) {
        this._clipPosition = d;
        this._updatePlaneConstants();
        this._clipDistanceSlider.value = d;
        this._clipDistanceValue.textContent = d.toFixed(2);
        this._syncAnchorFromPlane();
    }

    _updatePlaneConstants() {
        if (this._clipSlabMode) {
            const halfT = this._clipSlabThickness / 2;
            this._clipPlane.constant = -(this._clipPosition - halfT);
            this._clipPlane2.normal.copy(this._clipPlane.normal).negate();
            this._clipPlane2.constant = this._clipPosition + halfT;
        } else {
            this._clipPlane.constant = -this._clipPosition;
        }
    }

    _updateDiscPositions() {
        if (this._clipSlabMode) {
            const halfT = this._clipSlabThickness / 2;
            this._clipDisc.position.z = halfT;
            this._clipRing.position.z = halfT;
            this._clipDisc2.position.z = -halfT;
            this._clipRing2.position.z = -halfT;
            this._clipDisc2.visible = true;
            this._clipRing2.visible = true;
        } else {
            this._clipDisc.position.z = 0;
            this._clipRing.position.z = 0;
            this._clipDisc2.visible = false;
            this._clipRing2.visible = false;
        }
    }

    _setClipThickness(t) {
        this._clipSlabThickness = Math.max(0.01, Math.min(20, t));
        this._clipThicknessSlider.value = this._clipSlabThickness;
        this._clipThicknessValue.textContent = this._clipSlabThickness.toFixed(2);
        this._updatePlaneConstants();
        this._syncAnchorFromPlane();
    }

    _applyClipToObject(obj) {
        const planes = this._clipEnabled ? this._clipPlanes : [];
        obj.traverse(child => {
            if (!child.material) return;
            const mats = Array.isArray(child.material) ? child.material : [child.material];
            for (const mat of mats) {
                mat.clippingPlanes = planes;
                mat.clipShadows = true;
                if (this._clipEnabled) {
                    if (mat.userData.originalSide === undefined) mat.userData.originalSide = mat.side;
                    mat.side = THREE.DoubleSide;
                } else if (mat.userData.originalSide !== undefined) {
                    mat.side = mat.userData.originalSide;
                    delete mat.userData.originalSide;
                }
                mat.needsUpdate = true;
            }
        });
    }

    _isClipHelper(child) {
        let node = child;
        while (node) {
            if (node === this._clipAnchor || node === this._clipGizmoHelper) return true;
            node = node.parent;
        }
        return false;
    }

    _updateClipSliderRange() {
        this._bbox.makeEmpty();
        this._scene.traverse(child => {
            if (!child.geometry) return;
            if (this._isClipHelper(child)) return;
            if (child === this._gridHelper) return;
            child.updateWorldMatrix(true, false);
            const geo = child.geometry;
            if (!geo.boundingBox) geo.computeBoundingBox();
            this._bbox.expandByObject(child);
        });
        if (this._bbox.isEmpty()) return;
        const n = this._clipPlane.normal;
        const corners = [
            new THREE.Vector3(this._bbox.min.x, this._bbox.min.y, this._bbox.min.z),
            new THREE.Vector3(this._bbox.max.x, this._bbox.min.y, this._bbox.min.z),
            new THREE.Vector3(this._bbox.min.x, this._bbox.max.y, this._bbox.min.z),
            new THREE.Vector3(this._bbox.max.x, this._bbox.max.y, this._bbox.min.z),
            new THREE.Vector3(this._bbox.min.x, this._bbox.min.y, this._bbox.max.z),
            new THREE.Vector3(this._bbox.max.x, this._bbox.min.y, this._bbox.max.z),
            new THREE.Vector3(this._bbox.min.x, this._bbox.max.y, this._bbox.max.z),
            new THREE.Vector3(this._bbox.max.x, this._bbox.max.y, this._bbox.max.z),
        ];
        let lo = Infinity, hi = -Infinity;
        for (const c of corners) {
            const d = c.dot(n);
            if (d < lo) lo = d;
            if (d > hi) hi = d;
        }
        this._clipDistanceSlider.min = lo.toFixed(2);
        this._clipDistanceSlider.max = hi.toFixed(2);
        this._clipDistanceSlider.step = ((hi - lo) / 500).toFixed(4);
        const extent = hi - lo;
        if (extent > 0) {
            this._clipThicknessSlider.max = extent.toFixed(2);
            this._clipThicknessSlider.step = (extent / 500).toFixed(4);
        }
    }

    _updateClipMaterials() {
        this._scene.traverse(child => {
            if (!child.material) return;
            if (this._isClipHelper(child)) return;
            const mats = Array.isArray(child.material) ? child.material : [child.material];
            for (const mat of mats) {
                mat.clippingPlanes = this._clipEnabled ? this._clipPlanes : [];
                mat.clipShadows = true;
                if (this._clipEnabled) {
                    if (mat.userData.originalSide === undefined) mat.userData.originalSide = mat.side;
                    mat.side = THREE.DoubleSide;
                } else if (mat.userData.originalSide !== undefined) {
                    mat.side = mat.userData.originalSide;
                    delete mat.userData.originalSide;
                }
                mat.needsUpdate = true;
            }
        });
        this._clipAnchor.visible = this._clipEnabled && this._clipHelperVisible;
        const showGizmo = this._clipEnabled && this._clipHelperVisible;
        this._clipGizmo.visible = showGizmo;
        this._clipGizmo.enabled = showGizmo;
        this._clipGizmoHelper.visible = showGizmo;
    }

    _toggleClipPanel() {
        this._clipEnabled = !this._clipEnabled;
        if (this._clipEnabled && this._clipDefaults) {
            const d = this._clipDefaults;
            this._clipDefaults = null;
            if (d.normal) {
                this._clipPlane.normal.fromArray(d.normal).normalize();
                this._syncNormalInputs();
                let matched = null;
                for (const [axis, n] of Object.entries(CLIP_AXIS_NORMALS)) {
                    if (n.equals(this._clipPlane.normal)) { matched = axis; break; }
                }
                if (matched) this._setClipAxis(matched);
            }
            if (d.distance != null) this._setClipDistance(d.distance);
        }
        this._clipPanelEl.classList.toggle('visible', this._clipEnabled);
        this._btnClip.classList.toggle('active', this._clipEnabled);
        if (this._clipEnabled) this._updateClipSliderRange();
        this._syncAnchorFromPlane();
        this._updateClipMaterials();
    }

    _applyNormalInputs() {
        const x = parseFloat(this._clipNxInput.value) || 0;
        const y = parseFloat(this._clipNyInput.value) || 0;
        const z = parseFloat(this._clipNzInput.value) || 0;
        if (x === 0 && y === 0 && z === 0) return;
        this._clipPlane.normal.set(x, y, z).normalize();
        this._updatePlaneConstants();
        this._syncAnchorFromPlane();
        this._updateClipSliderRange();
        this._updateClipMaterials();
        this._clipAxis = null;
        this._clipPanelEl.querySelectorAll('.clip-axis-buttons button').forEach(btn => {
            btn.classList.remove('active');
        });
    }

    // ========== Camera ==========

    _switchCamera(toOrtho) {
        if (toOrtho === this._isOrtho) return;
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        const aspect = w / h;

        if (toOrtho) {
            const dist = this._perspCamera.position.distanceTo(this._controls.target);
            const halfHeight = dist * Math.tan(THREE.MathUtils.degToRad(this._perspCamera.fov / 2));
            this._orthoCamera.zoom = ORTHO_FRUSTUM / halfHeight;
            this._orthoCamera.left = -ORTHO_FRUSTUM * aspect;
            this._orthoCamera.right = ORTHO_FRUSTUM * aspect;
            this._orthoCamera.top = ORTHO_FRUSTUM;
            this._orthoCamera.bottom = -ORTHO_FRUSTUM;
            this._orthoCamera.position.copy(this._perspCamera.position);
            this._orthoCamera.quaternion.copy(this._perspCamera.quaternion);
            this._orthoCamera.updateProjectionMatrix();
            this._camera = this._orthoCamera;
        } else {
            const halfHeight = ORTHO_FRUSTUM / this._orthoCamera.zoom;
            const dist = halfHeight / Math.tan(THREE.MathUtils.degToRad(this._perspCamera.fov / 2));
            const dir = this._orthoCamera.position.clone().sub(this._controls.target).normalize();
            this._perspCamera.position.copy(this._controls.target).addScaledVector(dir, dist);
            this._perspCamera.quaternion.copy(this._orthoCamera.quaternion);
            this._perspCamera.aspect = aspect;
            this._perspCamera.updateProjectionMatrix();
            this._camera = this._perspCamera;
        }

        this._isOrtho = toOrtho;
        this._controls.object = this._camera;
        this._clipGizmo.camera = this._camera;
        this._viewHelper = new ViewHelper(this._camera, this._renderer.domElement);
        this._viewHelper.center = this._controls.target;
        this._btnOrtho.textContent = this._isOrtho ? 'O' : 'P';
        this._btnOrtho.classList.toggle('active', this._isOrtho);
    }

    // ========== Object Management ==========

    _createMaterial(params) {
        const color = params.color || 0x4a90d9;
        const materialType = params.materialType || 'standard';
        const opacity = params.opacity != null ? params.opacity : 1;
        const transparent = opacity < 1;
        const clip = this._clipEnabled ? this._clipPlanes : [];

        switch (materialType) {
            case 'basic':
                return new THREE.MeshBasicMaterial({ color, opacity, transparent, clippingPlanes: clip });
            case 'phong':
                return new THREE.MeshPhongMaterial({ color, opacity, transparent, clippingPlanes: clip });
            case 'lambert':
                return new THREE.MeshLambertMaterial({ color, opacity, transparent, clippingPlanes: clip });
            default: {
                const roughness = params.roughness != null ? params.roughness : 0.7;
                const metalness = params.metalness != null ? params.metalness : 0.3;
                return new THREE.MeshStandardMaterial({ color, roughness, metalness, opacity, transparent, clippingPlanes: clip });
            }
        }
    }

    _setOpacity(id, opacity) {
        const obj = this._objects.get(id);
        if (!obj) return;
        applyOpacity(obj, opacity);
    }

    _addToParentOrScene(obj, parentId) {
        if (parentId) {
            const parent = this._objects.get(parentId);
            if (parent) {
                parent.add(obj);
            } else {
                console.warn(`Parent '${parentId}' not found, adding to scene`);
                this._scene.add(obj);
            }
        } else {
            this._scene.add(obj);
        }
        this._sceneBoundsDirty = true;
    }

    _applyTransform(obj, transform) {
        if (!transform) return;
        if (transform.matrix) {
            const m = new THREE.Matrix4();
            m.fromArray(transform.matrix);
            obj.matrix.copy(m);
            obj.matrix.decompose(obj.position, obj.quaternion, obj.scale);
        } else {
            if (transform.position) obj.position.fromArray(transform.position);
            if (transform.rotation) obj.rotation.fromArray(transform.rotation);
            if (transform.quaternion) obj.quaternion.fromArray(transform.quaternion);
            if (transform.scale) obj.scale.fromArray(transform.scale);
        }
    }

    async _addObject(id, objData, parentId) {
        let obj;

        if (objData.primitive) {
            const geometry = PRIMITIVES[objData.primitive](objData.params || {});
            const material = this._createMaterial(objData.params || {});
            obj = new THREE.Mesh(geometry, material);
        } else if (objData.model) {
            const format = objData.format || 'gltf';
            const loader = this._loaders[format];
            if (!loader) {
                console.error(`Unknown format: ${format}`);
                return;
            }
            try {
                const result = await this._loadModel(loader, objData.model, format, objData.yUp === true);
                obj = result.obj;
                if (result.animations.length > 0) {
                    const mixerRoot = obj.userData.gltfScene || obj;
                    const mixer = new THREE.AnimationMixer(mixerRoot);
                    for (const clip of result.animations) {
                        const action = mixer.clipAction(clip);
                        action.play();
                    }
                    mixer.setTime(0);
                    this._mixers.set(id, mixer);
                    this._mixerGeneration++;
                    console.log(`Model ${id}: ${result.animations.length} animation clip(s) available`);
                }
            } catch (e) {
                console.error(`Failed to load model: ${e}`);
                return;
            }
        }

        if (obj) {
            obj.name = id;
            obj.userData.id = id;
            this._applyTransform(obj, objData.transform);
            if (objData.visible === false) obj.visible = false;
            this._addToParentOrScene(obj, parentId);
            this._objects.set(id, obj);
            this._objGeneration++;
            if (this._clipEnabled) this._applyClipToObject(obj);
        }
    }

    _loadModel(loader, url, format, yUp) {
        return new Promise((resolve, reject) => {
            loader.load(
                url,
                (result) => {
                    let obj;
                    let animations = [];
                    if (format === 'gltf' || format === 'glb') {
                        if (yUp) {
                            const correction = new THREE.Group();
                            correction.rotation.x = Math.PI / 2;
                            correction.add(result.scene);
                            obj = new THREE.Group();
                            obj.add(correction);
                            obj.userData.gltfScene = result.scene;
                        } else {
                            obj = new THREE.Group();
                            obj.add(result.scene);
                            obj.userData.gltfScene = result.scene;
                        }
                        animations = result.animations || [];
                    } else if (format === 'dae') {
                        obj = new THREE.Group();
                        result.scene.traverse((child) => {
                            if (child.isMesh) obj.add(child.clone());
                        });
                    } else if (format === 'stl' || format === 'ply') {
                        const material = new THREE.MeshStandardMaterial({ color: 0x4a90d9 });
                        obj = new THREE.Mesh(result, material);
                    } else {
                        obj = result;
                    }
                    resolve({ obj, animations });
                },
                undefined,
                reject
            );
        });
    }

    _setClipTime(id, time) {
        const mixer = this._mixers.get(id);
        if (!mixer) return;
        mixer.setTime(time);
    }

    _setDrawRange(id, value) {
        const obj = this._objects.get(id);
        if (!obj) return;
        if (obj.userData.isPolyline) {
            obj.geometry.instanceCount = Math.round(value * obj.userData.maxInstanceCount);
        } else if (obj.userData.isParametricTube) {
            applyParametricTubeDrawRange(obj, value);
        } else if (obj.userData.isMesh) {
            obj.geometry.setDrawRange(0, Math.round(value * obj.userData.totalIndexCount));
        }
    }

    _updateTransform(id, transform) {
        const obj = this._objects.get(id);
        if (obj) this._applyTransform(obj, transform);
    }

    _deleteObject(id) {
        const obj = this._objects.get(id);
        if (obj) {
            const childIds = [];
            obj.traverse((child) => {
                if (child !== obj && child.userData.id) childIds.push(child.userData.id);
            });
            for (const childId of childIds) {
                this._objects.delete(childId);
                const childMixer = this._mixers.get(childId);
                if (childMixer) { childMixer.stopAllAction(); this._mixers.delete(childId); this._mixerGeneration++; }
            }
            if (obj.parent) obj.parent.remove(obj);
            this._objects.delete(id);
            this._objGeneration++;
            this._sceneBoundsDirty = true;
            const mixer = this._mixers.get(id);
            if (mixer) { mixer.stopAllAction(); this._mixers.delete(id); this._mixerGeneration++; }
            obj.traverse((child) => {
                if (child.userData.blobUrl) URL.revokeObjectURL(child.userData.blobUrl);
                if (child.geometry) child.geometry.dispose();
                if (child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach(m => m.dispose());
                    } else {
                        child.material.dispose();
                    }
                }
            });
        }
    }

    _setVisibility(id, visible) {
        const obj = this._objects.get(id);
        if (obj) obj.visible = visible;
    }

    _setSceneVisibility(visibility) {
        for (const [id, visible] of Object.entries(visibility)) {
            const obj = this._objects.get(id);
            if (obj) {
                obj.visible = visible;
                this._baselineVisibility.set(id, visible);
            }
        }
    }

    _clearScene() {
        this._sceneGeneration++;
        this._resetAnimationState();
        for (const id of this._objects.keys()) {
            this._deleteObject(id);
        }
    }

    _batchUpdate(transforms) {
        for (const [id, transform] of Object.entries(transforms)) {
            this._updateTransform(id, transform);
            if (transform.opacity != null) this._setOpacity(id, transform.opacity);
        }
    }

    // ========== Animation ==========

    _loadAnimation(animData) {
        this._animGeneration++;
        this._animation = animData;
        this._animationTime = 0;
        this._animationPlaying = false;
        this._lastAnimationUpdate = performance.now();

        if (this._animation.channels) {
            for (const [name, ch] of Object.entries(this._animation.channels)) {
                ch.refs = ch.ids.map(id => {
                    const obj = this._objects.get(id);
                    if (obj && name === 'transforms') obj.matrixAutoUpdate = false;
                    return obj || null;
                });
            }
        }

        if (this._animation.frames.length >= 2) {
            const dt = this._animation.frames[1].time - this._animation.frames[0].time;
            this._animation.uniformDt = dt > 0 ? dt : 0;
        }

        this._baselineVisibility.clear();
        for (const [id, obj] of this._objects) {
            this._baselineVisibility.set(id, obj.visible);
        }

        this._animControlsEl.classList.add('visible');
        this._totalTimeEl.textContent = this._animation.duration.toFixed(2);
        this._totalFramesEl.textContent = this._animation.frames.length;
        this._animationLoop = this._animation.loop;
        this._btnLoop.classList.toggle('active', this._animationLoop);
        this._updateAnimationUI();

        this._timelineMarkersEl.innerHTML = '';
        if (this._animation.markers) {
            for (const marker of this._animation.markers) {
                const el = document.createElement('div');
                el.className = 'timeline-marker';
                el.style.left = `${(marker.time / this._animation.duration) * 100}%`;
                el.setAttribute('data-label', marker.label);
                el.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this._seekToTime(marker.time);
                });
                this._timelineMarkersEl.appendChild(el);
            }
        }

        this._applyFrame(0);

        // Camera tracking from animation metadata
        this._trackHasLastPos = false;
        this._trackInteractive = false;
        if (animData.camera_follow) {
            this._trackTargetId = animData.camera_follow;
            this._trackMode = 'follow';
        } else if (animData.camera_lookat) {
            this._trackTargetId = animData.camera_lookat;
            this._trackMode = 'lookat';
        } else if (animData.channels?.camera_target || animData.channels?.camera_position) {
            this._trackMode = 'scripted';
            this._trackTargetId = null;
        } else {
            this._trackMode = 'off';
            this._trackTargetId = null;
        }
        this._updateTrackingUI();

        this._animationPlaying = true;
        this._lastAnimationUpdate = performance.now();
        console.log(`Animation loaded: ${this._animation.frames.length} frames, ${this._animation.duration.toFixed(2)}s`);
    }

    _resetAnimationState() {
        this._animation = null;
        this._animationPlaying = false;
        this._baselineVisibility.clear();
        this._animControlsEl.classList.remove('visible');
        this._trackMode = 'off';
        this._trackTargetId = null;
        this._trackHasLastPos = false;
        this._trackInteractive = false;
        this._updateTrackingUI();
    }

    _stopAnimation(restoreVisibility = true) {
        this._animGeneration++;
        this._objects.forEach((obj) => { obj.matrixAutoUpdate = true; });
        if (restoreVisibility) {
            for (const [id, baselineVisible] of this._baselineVisibility) {
                const obj = this._objects.get(id);
                if (obj) obj.visible = baselineVisible;
            }
        }
        // Reset draw ranges to full
        for (const id of this._objects.keys()) {
            this._setDrawRange(id, 1.0);
        }
        this._resetAnimationState();
    }

    _applyFrame(frameIndex, t = 0) {
        if (!this._animation || frameIndex < 0 || frameIndex >= this._animation.frames.length) return;

        const frames = this._animation.frames;
        const frame = frames[frameIndex];
        // Between-keyframe lerp needs a next keyframe and a non-zero t. We
        // skip the lerp at t≈0 as a no-op optimization; there is no upper
        // epsilon because t is always in [0, 1) from _getFrameAtTime and
        // clamping near 1 would produce a visible "hold" stutter on the last
        // sliver of each interval. Channels still make their own linear/hold
        // decision on top of this.
        const hasNext = t > 1e-6 && frameIndex < frames.length - 1;
        const nextFrame = hasNext ? frames[frameIndex + 1] : null;

        if (this._animation.channels) {
            if (this._animation.channels.visibility) {
                for (const [id, baselineVisible] of this._baselineVisibility) {
                    const obj = this._objects.get(id);
                    if (obj) obj.visible = baselineVisible;
                }
            }
            for (const [name, ch] of Object.entries(this._animation.channels)) {
                const applyFn = this._CHANNEL_APPLY[name];
                if (applyFn) {
                    const nObj = ch.ids.length;
                    const base = frameIndex * nObj * ch.stride;
                    // shouldInterpChannel checks ch.interpolation; pass baseNext
                    // regardless of mode so the applier owns the decision.
                    const baseNext = hasNext ? (frameIndex + 1) * nObj * ch.stride : null;
                    applyFn(ch, ch.refs, base, baseNext, hasNext ? t : 0);
                }
            }
        }

        // JSON frame path: every lerp-able field interpolates by default,
        // matching the binary-channel defaults. Visibility left-holds (no
        // meaningful linear for booleans).
        if (frame.transforms) {
            const nextT = (hasNext && nextFrame.transforms) ? nextFrame.transforms : null;
            for (const [id, matrix] of Object.entries(frame.transforms)) {
                const obj = this._objects.get(id);
                if (!obj) continue;
                obj.matrixAutoUpdate = false;
                const nextMat = nextT ? nextT[id] : null;
                if (nextMat) {
                    lerpMatrixInto(matrix, 0, nextMat, 0, t, _lerpMatOut);
                    obj.matrix.copy(_lerpMatOut);
                } else {
                    obj.matrix.fromArray(matrix);
                }
                obj.matrixWorldNeedsUpdate = true;
            }
        }
        if (frame.colors) {
            const nextC = (hasNext && nextFrame.colors) ? nextFrame.colors : null;
            for (const [id, color] of Object.entries(frame.colors)) {
                const obj = this._objects.get(id);
                if (!obj) continue;
                let hex = color;
                if (nextC && nextC[id] != null) {
                    hex = lerpHexColor(color, nextC[id], t);
                }
                obj.traverse((child) => {
                    if (!child.material) return;
                    const mats = Array.isArray(child.material) ? child.material : [child.material];
                    for (const mat of mats) { if (mat.color) mat.color.setHex(hex); }
                });
            }
        }
        if (!this._animation.channels || !this._animation.channels.visibility) {
            for (const [id, baselineVisible] of this._baselineVisibility) {
                const obj = this._objects.get(id);
                if (obj) obj.visible = baselineVisible;
            }
        }
        if (frame.visibility) {
            // Left-hold regardless (booleans have no meaningful lerp).
            for (const [id, visible] of Object.entries(frame.visibility)) {
                const obj = this._objects.get(id);
                if (obj) obj.visible = visible;
            }
        }
        if (frame.opacity) {
            const nextO = (hasNext && nextFrame.opacity) ? nextFrame.opacity : null;
            for (const [id, opacity] of Object.entries(frame.opacity)) {
                const obj = this._objects.get(id);
                if (!obj) continue;
                let val = opacity;
                if (nextO && nextO[id] != null) {
                    val = opacity * (1 - t) + nextO[id] * t;
                }
                applyOpacity(obj, val);
            }
        }
        if (frame.clip_times) {
            const nextCt = (hasNext && nextFrame.clip_times) ? nextFrame.clip_times : null;
            for (const [id, time] of Object.entries(frame.clip_times)) {
                let v = time;
                if (nextCt && nextCt[id] != null) v = time * (1 - t) + nextCt[id] * t;
                this._setClipTime(id, v);
            }
        }
        if (frame.draw_ranges) {
            const nextD = (hasNext && nextFrame.draw_ranges) ? nextFrame.draw_ranges : null;
            for (const [id, value] of Object.entries(frame.draw_ranges)) {
                let v = value;
                if (nextD && nextD[id] != null) {
                    v = value * (1 - t) + nextD[id] * t;
                }
                this._setDrawRange(id, v);
            }
        }

        this._applyCameraTracking(frameIndex, hasNext ? (frameIndex + 1) : frameIndex, hasNext ? t : 0);
    }

    _cycleTrackMode() {
        this._trackInteractive = true;
        const hasChannels = this._animation?.channels?.camera_target || this._animation?.channels?.camera_position;

        if (this._trackMode === 'scripted') {
            // Scripted → off
            this._trackMode = 'off';
        } else if (hasChannels && this._trackMode === 'off' && !this._trackTargetId) {
            // Off → re-enable scripted (if no object target, only channels)
            this._trackInteractive = false;
            this._trackMode = 'scripted';
        } else {
            // Object-based: off → follow → lookat → off
            if (!this._trackTargetId && this._trackMode === 'off') {
                this._trackTargetId = this._guessTrackTarget();
            }
            if (this._trackTargetId) {
                const modes = ['off', 'follow', 'lookat'];
                const idx = modes.indexOf(this._trackMode);
                this._trackMode = modes[(idx + 1) % modes.length];
            }
        }
        this._trackHasLastPos = false;
        this._updateTrackingUI();
    }

    _guessTrackTarget() {
        if (!this._animation?.channels?.transforms) return null;
        const ids = this._animation.channels.transforms.ids;
        const hints = ['nozzle', 'tip', 'tool', 'effector'];
        for (const hint of hints) {
            const match = ids.find(id => id.toLowerCase().includes(hint));
            if (match) return match;
        }
        return null;
    }

    _updateTrackingUI() {
        if (!this._btnTrack) return;
        const hasTracking = this._trackTargetId || this._animation?.channels?.camera_target || this._animation?.channels?.camera_position;
        this._btnTrack.style.display = (this._animation && hasTracking) ? '' : 'none';
        this._btnTrack.classList.toggle('active', this._trackMode !== 'off');

        let title = 'Camera tracking (T)';
        if (this._trackMode === 'scripted') {
            title = 'Camera: scripted (T)';
        } else if (this._trackMode !== 'off') {
            const modeLabel = this._trackMode === 'follow' ? 'Follow' : 'Look-at';
            title = `${modeLabel}: ${this._trackTargetId || ''} (T)`;
        }
        this._btnTrack.title = title;
    }

    _applyCameraTracking(frameIndex, frameIndexNext = frameIndex, t = 0) {
        if (this._trackMode === 'off') return;

        // Scripted camera channels
        if (this._trackMode === 'scripted' && this._animation?.channels) {
            const ctCh = this._animation.channels.camera_target;
            const cpCh = this._animation.channels.camera_position;
            if (ctCh || cpCh) {
                const interp = t > 0 && frameIndexNext !== frameIndex;
                if (ctCh) {
                    const base = frameIndex * 3;
                    if (interp && ctCh.interpolation === 'linear') {
                        const bN = frameIndexNext * 3;
                        this._controls.target.set(
                            ctCh.data[base]     * (1 - t) + ctCh.data[bN]     * t,
                            ctCh.data[base + 1] * (1 - t) + ctCh.data[bN + 1] * t,
                            ctCh.data[base + 2] * (1 - t) + ctCh.data[bN + 2] * t
                        );
                    } else {
                        this._controls.target.set(
                            ctCh.data[base], ctCh.data[base + 1], ctCh.data[base + 2]
                        );
                    }
                }
                if (cpCh) {
                    const base = frameIndex * 3;
                    if (interp && cpCh.interpolation === 'linear') {
                        const bN = frameIndexNext * 3;
                        this._camera.position.set(
                            cpCh.data[base]     * (1 - t) + cpCh.data[bN]     * t,
                            cpCh.data[base + 1] * (1 - t) + cpCh.data[bN + 1] * t,
                            cpCh.data[base + 2] * (1 - t) + cpCh.data[bN + 2] * t
                        );
                    } else {
                        this._camera.position.set(
                            cpCh.data[base], cpCh.data[base + 1], cpCh.data[base + 2]
                        );
                    }
                }
                return;
            }
        }

        // Object-based tracking (follow / lookat)
        if (!this._trackTargetId) return;
        const obj = this._objects.get(this._trackTargetId);
        if (!obj) return;

        obj.updateWorldMatrix(true, false);
        const targetPos = this._tmpTrackPos;
        targetPos.setFromMatrixPosition(obj.matrixWorld);

        if (this._trackMode === 'follow') {
            if (this._trackHasLastPos) {
                const delta = this._tmpTrackDelta.copy(targetPos).sub(this._trackLastPos);
                this._controls.target.add(delta);
                this._camera.position.add(delta);
            } else {
                // First frame: snap orbit target, keep camera offset
                const offset = this._tmpTrackDelta.copy(this._camera.position).sub(this._controls.target);
                this._controls.target.copy(targetPos);
                this._camera.position.copy(targetPos).add(offset);
            }
        } else if (this._trackMode === 'lookat') {
            this._controls.target.copy(targetPos);
        }

        this._trackLastPos.copy(targetPos);
        this._trackHasLastPos = true;
    }

    // Returns {index, t} where index is the floor keyframe and t ∈ [0, 1)
    // is the fractional position between frames[index] and frames[index+1].
    // At (or past) the last frame, t is clamped to 0 so callers short-circuit
    // to step behavior.
    _getFrameAtTime(time) {
        if (!this._animation || this._animation.frames.length === 0) {
            return { index: 0, t: 0 };
        }
        const frames = this._animation.frames;
        const lastIdx = frames.length - 1;
        if (this._animation.uniformDt > 0) {
            const raw = time / this._animation.uniformDt;
            const idx = Math.floor(raw);
            if (idx < 0) return { index: 0, t: 0 };
            if (idx >= lastIdx) return { index: lastIdx, t: 0 };
            return { index: idx, t: raw - idx };
        }
        for (let i = lastIdx; i >= 0; i--) {
            if (frames[i].time <= time) {
                if (i >= lastIdx) return { index: lastIdx, t: 0 };
                const dt = frames[i + 1].time - frames[i].time;
                return {
                    index: i,
                    t: dt > 0 ? (time - frames[i].time) / dt : 0,
                };
            }
        }
        return { index: 0, t: 0 };
    }

    _updateAnimationUI() {
        if (!this._animation) return;
        const { index: frameIndex } = this._getFrameAtTime(this._animationTime);
        const progress = this._animation.duration > 0 ? (this._animationTime / this._animation.duration) * 100 : 0;
        this._timelineProgressEl.style.width = `${progress}%`;
        this._currentTimeEl.textContent = this._animationTime.toFixed(2);
        this._currentFrameEl.textContent = frameIndex + 1;
        this._btnPlay.textContent = this._animationPlaying ? '\u23F8' : '\u25B6';
    }

    _stepFrames(delta) {
        if (!this._animation) return;
        const { index: currentFrame } = this._getFrameAtTime(this._animationTime);
        const newFrame = Math.max(0, Math.min(this._animation.frames.length - 1, currentFrame + delta));
        this._animationTime = this._animation.frames[newFrame].time;
        this._applyFrame(newFrame);
        this._updateAnimationUI();
    }

    _seekToTime(time) {
        if (!this._animation) return;
        this._animationTime = Math.max(0, Math.min(this._animation.duration, time));
        const { index, t } = this._getFrameAtTime(this._animationTime);
        this._applyFrame(index, t);
        this._updateAnimationUI();
    }

    _togglePlay() {
        this._animationPlaying = !this._animationPlaying;
        if (this._animationPlaying) {
            this._lastAnimationUpdate = performance.now();
        }
        this._updateAnimationUI();
    }

    _setSpeed(speed) {
        this._animationSpeed = speed;
        this._speedDisplayEl.textContent = `${speed}x`;
    }

    _stepSpeed(delta) {
        this._speedIndex = Math.max(0, Math.min(SPEED_STEPS.length - 1, this._speedIndex + delta));
        this._setSpeed(SPEED_STEPS[this._speedIndex]);
    }

    _scrubFromEvent(e) {
        if (!this._animation) return;
        const rect = this._timelineContainer.getBoundingClientRect();
        const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
        const ratio = x / rect.width;
        this._seekToTime(ratio * this._animation.duration);
    }

    // ========== Asset Tracking ==========

    _maybeNotifyAssetsLoaded() {
        if (this._pendingFetches === 0 && this._assetsComplete && this._ws && this._ws.readyState === WebSocket.OPEN) {
            this._ws.send(JSON.stringify({ type: 'assets_loaded' }));
        }
    }

    _onFetchStart() { this._pendingFetches++; }
    _onFetchEnd() {
        this._pendingFetches = Math.max(0, this._pendingFetches - 1);
        this._maybeNotifyAssetsLoaded();
    }

    // ========== Events ==========

    _bindEvents() {
        // Ortho button
        this._btnOrtho.addEventListener('click', () => this._switchCamera(!this._isOrtho));

        // Clip button
        this._btnClip.addEventListener('click', () => this._toggleClipPanel());

        // Clip axis buttons
        this._clipPanelEl.querySelectorAll('.clip-axis-buttons button').forEach(btn => {
            btn.addEventListener('click', () => this._setClipAxis(btn.dataset.axis));
        });

        // Normal inputs
        [this._clipNxInput, this._clipNyInput, this._clipNzInput].forEach(input => {
            input.addEventListener('change', () => this._applyNormalInputs());
            input.addEventListener('keydown', (e) => e.stopPropagation());
        });

        // Distance slider
        this._clipDistanceSlider.addEventListener('input', () => {
            this._setClipDistance(parseFloat(this._clipDistanceSlider.value));
        });
        this._clipDistanceSlider.addEventListener('wheel', (e) => {
            e.preventDefault();
            const step = e.shiftKey ? 0.1 : 0.01;
            const delta = e.deltaY > 0 ? -step : step;
            const newVal = Math.max(-20, Math.min(20, parseFloat(this._clipDistanceSlider.value) + delta));
            this._setClipDistance(newVal);
        });

        // Slab mode toggles
        this._clipModeSingle.addEventListener('click', (e) => {
            this._clipSlabMode = false;
            e.currentTarget.classList.add('active');
            this._clipModeSlab.classList.remove('active');
            this._clipThicknessSection.style.display = 'none';
            this._clipPlanes = [this._clipPlane];
            this._updatePlaneConstants();
            this._syncAnchorFromPlane();
            this._updateClipMaterials();
        });
        this._clipModeSlab.addEventListener('click', (e) => {
            this._clipSlabMode = true;
            e.currentTarget.classList.add('active');
            this._clipModeSingle.classList.remove('active');
            this._clipThicknessSection.style.display = '';
            this._clipPlanes = [this._clipPlane, this._clipPlane2];
            this._updatePlaneConstants();
            this._syncAnchorFromPlane();
            this._updateClipMaterials();
        });

        // Thickness slider
        this._clipThicknessSlider.addEventListener('input', () => {
            this._setClipThickness(parseFloat(this._clipThicknessSlider.value));
        });
        this._clipThicknessSlider.addEventListener('wheel', (e) => {
            e.preventDefault();
            const step = e.shiftKey ? 0.1 : 0.01;
            const delta = e.deltaY > 0 ? -step : step;
            this._setClipThickness(this._clipSlabThickness + delta);
        });

        // Close button
        this._clipClose.addEventListener('click', () => this._toggleClipPanel());

        // Animation controls
        this.el.querySelector('.tjsv-btn-start').addEventListener('click', () => this._seekToTime(0));
        this.el.querySelector('.tjsv-btn-end').addEventListener('click', () => this._seekToTime(this._animation?.duration || 0));
        this.el.querySelector('.tjsv-btn-prev-frame').addEventListener('click', () => this._stepFrames(-1));
        this.el.querySelector('.tjsv-btn-next-frame').addEventListener('click', () => this._stepFrames(1));
        this._btnPlay.addEventListener('click', () => this._togglePlay());
        this._btnLoop.addEventListener('click', () => {
            this._animationLoop = !this._animationLoop;
            this._btnLoop.classList.toggle('active', this._animationLoop);
        });
        this._btnTrack.addEventListener('click', () => this._cycleTrackMode());
        this.el.querySelector('.tjsv-btn-slower').addEventListener('click', () => this._stepSpeed(-1));
        this.el.querySelector('.tjsv-btn-faster').addEventListener('click', () => this._stepSpeed(1));

        // Timeline scrubbing
        this._timelineContainer.addEventListener('mousedown', (e) => {
            if (!this._animation) return;
            this._scrubbing = true;
            this._wasPlayingBeforeScrub = this._animationPlaying;
            this._animationPlaying = false;
            this._scrubFromEvent(e);
        });

        // Document-level listeners for scrubbing (stored for cleanup)
        this._onDocMouseMove = (e) => {
            if (this._scrubbing) this._scrubFromEvent(e);
        };
        this._onDocMouseUp = () => {
            if (this._scrubbing) {
                this._scrubbing = false;
                if (this._wasPlayingBeforeScrub) {
                    this._animationPlaying = true;
                    this._lastAnimationUpdate = performance.now();
                }
                this._updateAnimationUI();
            }
        };
        document.addEventListener('mousemove', this._onDocMouseMove);
        document.addEventListener('mouseup', this._onDocMouseUp);

        // Keyboard shortcuts — scoped to container
        this._onKeyDown = (e) => {
            if (e.target.tagName === 'INPUT') return;

            // Global shortcuts
            if (e.code === 'KeyO' && !e.ctrlKey && !e.metaKey) {
                this._switchCamera(!this._isOrtho);
                return;
            }
            if (e.code === 'KeyC' && !e.ctrlKey && !e.metaKey) {
                this._toggleClipPanel();
                return;
            }

            // Clipping shortcuts
            if (this._clipEnabled) {
                if (e.code === 'KeyV') {
                    const center = this._clipPlane.normal.clone().multiplyScalar(this._clipPosition);
                    this._controls.target.copy(center);
                    const camDist = 50;
                    const camPos = center.clone().addScaledVector(this._clipPlane.normal, camDist);
                    if (!this._isOrtho) this._switchCamera(true);
                    this._camera.position.copy(camPos);
                    this._camera.lookAt(center);
                    this._camera.zoom = 1;
                    this._camera.updateProjectionMatrix();
                    this._controls.update();
                    return;
                }
                if (e.code === 'KeyS') {
                    if (this._clipSlabMode) {
                        this._clipModeSingle.click();
                    } else {
                        this._clipModeSlab.click();
                    }
                    return;
                }
                if (e.code === 'KeyH') {
                    this._clipHelperVisible = !this._clipHelperVisible;
                    this._clipPanelEl.classList.toggle('visible', this._clipHelperVisible);
                    this._updateClipMaterials();
                    return;
                }
                {
                    const step = e.shiftKey ? 0.5 : 0.05;
                    switch (e.code) {
                        case 'ArrowLeft':
                            e.preventDefault();
                            this._setClipDistance(this._clipPosition - step);
                            return;
                        case 'ArrowRight':
                            e.preventDefault();
                            this._setClipDistance(this._clipPosition + step);
                            return;
                        case 'ArrowUp':
                            if (this._clipSlabMode) {
                                e.preventDefault();
                                this._setClipThickness(this._clipSlabThickness + step);
                                return;
                            }
                            break;
                        case 'ArrowDown':
                            if (this._clipSlabMode) {
                                e.preventDefault();
                                this._setClipThickness(this._clipSlabThickness - step);
                                return;
                            }
                            break;
                    }
                }
            }

            if (!this._animation) return;

            switch (e.code) {
                case 'Space':
                    e.preventDefault();
                    this._togglePlay();
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    this._stepFrames(e.shiftKey ? -10 : -1);
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this._stepFrames(e.shiftKey ? 10 : 1);
                    break;
                case 'Home':
                    e.preventDefault();
                    this._seekToTime(0);
                    break;
                case 'End':
                    e.preventDefault();
                    this._seekToTime(this._animation.duration);
                    break;
                case 'KeyL':
                    this._animationLoop = !this._animationLoop;
                    this._btnLoop.classList.toggle('active', this._animationLoop);
                    break;
                case 'KeyT':
                    this._cycleTrackMode();
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    this._stepSpeed(1);
                    break;
                case 'ArrowDown':
                    e.preventDefault();
                    this._stepSpeed(-1);
                    break;
            }
        };
        this.container.addEventListener('keydown', this._onKeyDown);
    }

    // ========== WebSocket ==========

    connect() {
        const doConnect = () => {
            if (this._destroyed) return;

            this._ws = new WebSocket(this._wsUrl);

            this._ws.onopen = () => {
                this._statusDot.className = 'tjsv-status-dot connected';
                this._statusDot.title = 'Connected';
                this._statusText.textContent = 'Connected';
                // Don't reset _pendingFetches to 0: in-flight HTTP fetches from the
                // previous connection may still complete and run _onFetchEnd(), which
                // would drive the counter negative. Generation counters ensure stale
                // fetches are discarded; let their finally blocks balance the count.
                this._sceneGeneration++;
                this._animGeneration++;
                this._assetsComplete = false;
                this._ws.send(JSON.stringify({ type: 'hello', viewer_version: VIEWER_VERSION }));
            };

            this._ws.onclose = () => {
                this._statusDot.className = 'tjsv-status-dot disconnected';
                this._statusDot.title = 'Waiting for Python...';
                this._statusText.textContent = 'Waiting...';
                this._reconnectTimeout = setTimeout(doConnect, 500);
            };

            this._ws.onerror = () => {};

            this._ws.onmessage = async (event) => {
                const tParse = performance.now();
                const data = JSON.parse(event.data);
                const parseMs = performance.now() - tParse;
                if (parseMs > 200) {
                    console.warn(
                        `JSON.parse took ${parseMs.toFixed(0)}ms (${(event.data.length / 1024 / 1024).toFixed(1)}MB). ` +
                        `Consider using binary channels (animation.add_channel()) for large animation data.`
                    );
                }

                switch (data.type) {
                    case 'hello':
                        console.log(`Python client v${data.client_version}`);
                        if (data.client_version !== VIEWER_VERSION) {
                            console.warn(
                                `Version mismatch: viewer v${VIEWER_VERSION}, client v${data.client_version}. ` +
                                `Close this tab and re-open viewer.html to pick up the latest version.`
                            );
                        }
                        break;
                    case 'add_group': {
                        const group = new THREE.Group();
                        group.name = data.id;
                        group.userData.id = data.id;
                        if (data.transform) this._applyTransform(group, data.transform);
                        if (data.visible === false) group.visible = false;
                        this._addToParentOrScene(group, data.parent);
                        this._objects.set(data.id, group);
                        this._objGeneration++;
                        break;
                    }
                    case 'add_object':
                        await this._addObject(data.id, data.object, data.parent);
                        break;
                    case 'update_transform':
                        this._updateTransform(data.id, data.transform);
                        break;
                    case 'delete_object':
                        this._deleteObject(data.id);
                        break;
                    case 'set_visibility':
                        this._setVisibility(data.id, data.visible);
                        break;
                    case 'set_scene_visibility':
                        this._setSceneVisibility(data.visibility);
                        break;
                    case 'clear_scene':
                        this._clearScene();
                        break;
                    case 'batch_update':
                        this._batchUpdate(data.transforms);
                        break;
                    case 'set_color': {
                        const colorObj = this._objects.get(data.id);
                        if (colorObj) {
                            colorObj.traverse((child) => {
                                if (!child.material) return;
                                const mats = Array.isArray(child.material) ? child.material : [child.material];
                                for (const mat of mats) { if (mat.color) mat.color.setHex(data.color); }
                            });
                            if (data.opacity != null) applyOpacity(colorObj, data.opacity);
                        }
                        break;
                    }
                    case 'set_opacity':
                        this._setOpacity(data.id, data.opacity);
                        break;
                    case 'list_objects':
                        this._ws.send(JSON.stringify({
                            type: 'list_objects_response',
                            requestId: data.requestId,
                            objects: Array.from(this._objects.keys())
                        }));
                        break;
                    case 'query_scene': {
                        const tree = {};
                        for (const [id, obj] of this._objects) {
                            let drawRange = 1.0;
                            const geom = obj.geometry;
                            if (geom) {
                                if (obj.userData.isPolyline) {
                                    const max = obj.userData.maxInstanceCount;
                                    drawRange = max > 0 ? Math.min(geom.instanceCount / max, 1.0) : 1.0;
                                } else if (obj.userData.isMesh || obj.userData.isParametricTube) {
                                    const total = obj.userData.totalIndexCount;
                                    if (total > 0) {
                                        const cnt = geom.drawRange.count;
                                        drawRange = Number.isFinite(cnt) ? Math.min(cnt / total, 1.0) : 1.0;
                                    }
                                }
                            }
                            tree[id] = {
                                type: obj.type,
                                parent: obj.parent?.userData?.id || null,
                                children: obj.children
                                    .filter(c => c.userData?.id)
                                    .map(c => c.userData.id),
                                visible: obj.visible,
                                drawRange: drawRange,
                            };
                        }
                        this._ws.send(JSON.stringify({
                            type: 'query_scene_response',
                            requestId: data.requestId,
                            tree: tree,
                            meta: {
                                animation: { playing: this._animationPlaying },
                                grid: { visible: this._gridHelper.visible },
                                pending_fetches: this._pendingFetches,
                            },
                        }));
                        break;
                    }
                    case 'load_animation':
                        this._loadAnimation(data.animation);
                        break;
                    case 'load_animation_http': {
                        this._onFetchStart();
                        const capturedScene = this._sceneGeneration;
                        const capturedAnim = ++this._animGeneration;
                        (async () => {
                            try {
                                const t0 = performance.now();
                                const resp = await fetch(data.blob_url);
                                const buffer = await resp.arrayBuffer();
                                if (this._sceneGeneration !== capturedScene || this._animGeneration !== capturedAnim) {
                                    console.log('Discarding stale animation fetch');
                                    return;
                                }
                                const t1 = performance.now();
                                console.log(`HTTP animation fetch: ${(buffer.byteLength / 1024 / 1024).toFixed(1)}MB in ${(t1 - t0).toFixed(0)}ms`);

                                const nFrames = data.frame_count;
                                const DTYPE_INFO = {
                                    float32: { ArrayType: Float32Array, bytes: 4 },
                                    uint32:  { ArrayType: Uint32Array,  bytes: 4 },
                                    uint8:   { ArrayType: Uint8Array,   bytes: 1 },
                                };

                                // Each channel carries its own interpolation mode,
                                // set explicitly Python-side (defaults to 'linear').
                                // Visibility is special-cased by its applier to always
                                // hold regardless — see makeChannelApply.visibility.
                                let byteOffset = 0;
                                const channels = {};
                                if (data.channels) {
                                    for (const ch of data.channels) {
                                        const info = DTYPE_INFO[ch.dtype];
                                        if (!info) { console.error(`Unknown channel dtype '${ch.dtype}' for '${ch.name}', skipping`); continue; }
                                        const count = nFrames * ch.ids.length * ch.stride;
                                        channels[ch.name] = {
                                            data: new info.ArrayType(buffer, byteOffset, count),
                                            ids: ch.ids,
                                            stride: ch.stride,
                                            refs: null,
                                            colormap: ch.colormap || null,
                                            interpolation: sanitizeInterpolation(ch.interpolation, 'linear'),
                                        };
                                        byteOffset += count * info.bytes;
                                    }
                                }

                                const tMeta = performance.now();
                                const metaByFrame = {};
                                if (data.frames_meta) {
                                    for (const meta of data.frames_meta) {
                                        metaByFrame[meta.index] = meta;
                                    }
                                }

                                const frames = [];
                                for (let fi = 0; fi < nFrames; fi++) {
                                    const frame = { time: data.frame_times[fi] };
                                    const meta = metaByFrame[fi];
                                    if (meta) {
                                        if (meta.colors) frame.colors = meta.colors;
                                        if (meta.visibility) frame.visibility = meta.visibility;
                                        if (meta.opacity) frame.opacity = meta.opacity;
                                        if (meta.clip_times) frame.clip_times = meta.clip_times;
                                        if (meta.draw_ranges) frame.draw_ranges = meta.draw_ranges;
                                    }
                                    frames.push(frame);
                                }

                                const metaMs = performance.now() - tMeta;
                                if (metaMs > 500) {
                                    console.warn(
                                        `frames_meta processing took ${metaMs.toFixed(0)}ms. ` +
                                        `Consider using animation.add_channel() on the Python side ` +
                                        `for colors/visibility/opacity/draw_ranges for much faster transfer.`
                                    );
                                }

                                this._loadAnimation({
                                    duration: data.duration,
                                    fps: data.fps,
                                    loop: data.loop,
                                    frames: frames,
                                    markers: data.markers || [],
                                    channels: channels,
                                    camera_follow: data.camera_follow || null,
                                    camera_lookat: data.camera_lookat || null,
                                });
                                console.log(`  total: ${(performance.now() - t0).toFixed(0)}ms`);
                            } catch (e) {
                                console.error('Error loading animation via HTTP:', e);
                            } finally {
                                this._onFetchEnd();
                            }
                        })();
                        break;
                    }
                    case 'add_model_binary': {
                        this._onFetchStart();
                        const capturedScene = this._sceneGeneration;
                        (async () => {
                            try {
                                const resp = await fetch(data.blob_url);
                                const meshBytes = await resp.arrayBuffer();
                                if (this._sceneGeneration !== capturedScene) {
                                    console.log('Discarding stale model fetch');
                                    return;
                                }
                                const blob = new Blob([meshBytes]);
                                const blobUrl = URL.createObjectURL(blob);
                                console.log(`Loading model ${data.id} (${data.format}) via HTTP`);
                                await this._addObject(data.id, {
                                    model: blobUrl,
                                    format: data.format || 'stl',
                                    yUp: data.yUp === true,
                                }, data.parent);
                                const obj = this._objects.get(data.id);
                                if (obj) {
                                    obj.userData.blobUrl = blobUrl;
                                    if (data.transform) this._updateTransform(data.id, data.transform);
                                }
                            } catch (e) {
                                console.error(`Error loading model via HTTP:`, e);
                            } finally {
                                this._onFetchEnd();
                            }
                        })();
                        break;
                    }
                    case 'add_polyline_binary': {
                        this._onFetchStart();
                        const capturedScene = this._sceneGeneration;
                        (async () => {
                            try {
                                const resp = await fetch(data.blob_url);
                                const buffer = await resp.arrayBuffer();
                                if (this._sceneGeneration !== capturedScene) {
                                    console.log('Discarding stale polyline fetch');
                                    return;
                                }
                                const rawData = new Uint8Array(buffer);
                                const numPoints = data.numPoints || (rawData.length / 12);
                                const positionBytes = numPoints * 12;

                                const posBuffer = new ArrayBuffer(positionBytes);
                                new Uint8Array(posBuffer).set(rawData.slice(0, positionBytes));
                                const pointData = new Float32Array(posBuffer);

                                console.log(`Creating polyline ${data.id} with ${numPoints} points via HTTP`);

                                const geometry = new LineGeometry();
                                geometry.setPositions(pointData);

                                const w = this.container.clientWidth;
                                const h = this.container.clientHeight;
                                let material;
                                if (data.hasVertexColors) {
                                    const colorBytes = rawData.slice(positionBytes);
                                    const colorData = new Float32Array(numPoints * 3);
                                    for (let i = 0; i < numPoints; i++) {
                                        colorData[i * 3] = colorBytes[i * 3] / 255;
                                        colorData[i * 3 + 1] = colorBytes[i * 3 + 1] / 255;
                                        colorData[i * 3 + 2] = colorBytes[i * 3 + 2] / 255;
                                    }
                                    geometry.setColors(colorData);
                                    material = new LineMaterial({
                                        color: 0xffffff,
                                        vertexColors: true,
                                        linewidth: data.lineWidth || 2,
                                        resolution: new THREE.Vector2(w, h),
                                    });
                                } else {
                                    material = new LineMaterial({
                                        color: data.color || 0xffffff,
                                        linewidth: data.lineWidth || 2,
                                        resolution: new THREE.Vector2(w, h),
                                    });
                                }

                                const line = new Line2(geometry, material);
                                line.computeLineDistances();
                                line.name = data.id;
                                line.userData.id = data.id;
                                line.userData.isPolyline = true;
                                line.userData.maxInstanceCount = numPoints - 1;
                                this._addToParentOrScene(line, data.parent);
                                this._objects.set(data.id, line);
                                this._objGeneration++;
                            } catch (e) {
                                console.error(`Error creating polyline via HTTP:`, e);
                            } finally {
                                this._onFetchEnd();
                            }
                        })();
                        break;
                    }
                    case 'add_mesh_binary': {
                        this._onFetchStart();
                        const capturedScene = this._sceneGeneration;
                        (async () => {
                            try {
                                const resp = await fetch(data.blob_url);
                                const buffer = await resp.arrayBuffer();
                                if (this._sceneGeneration !== capturedScene) {
                                    console.log('Discarding stale mesh fetch');
                                    return;
                                }
                                const nv = data.numVertices;
                                const ni = data.numIndices;

                                let offset = 0;
                                const positions = new Float32Array(buffer, offset, nv * 3);
                                offset += nv * 3 * 4;

                                let normals = null;
                                if (data.hasNormals) {
                                    normals = new Float32Array(buffer, offset, nv * 3);
                                    offset += nv * 3 * 4;
                                }

                                let colors = null;
                                if (data.hasVertexColors) {
                                    colors = new Float32Array(buffer, offset, nv * 3);
                                    offset += nv * 3 * 4;
                                }

                                const indices = new Uint32Array(buffer, offset, ni);

                                const geometry = new THREE.BufferGeometry();
                                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                                geometry.setIndex(new THREE.BufferAttribute(indices, 1));
                                if (normals) {
                                    geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
                                } else {
                                    geometry.computeVertexNormals();
                                }
                                if (colors) {
                                    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                                }

                                const meshOpacity = data.opacity !== undefined ? data.opacity : 1;
                                const meshMaterial = new THREE.MeshStandardMaterial({
                                    color: colors ? 0xffffff : (data.color || 0x7ab8cc),
                                    metalness: data.metalness !== undefined ? data.metalness : 0.1,
                                    roughness: data.roughness !== undefined ? data.roughness : 0.8,
                                    opacity: meshOpacity,
                                    transparent: meshOpacity < 1,
                                    depthWrite: meshOpacity >= 1,
                                    side: THREE.DoubleSide,
                                    vertexColors: !!colors,
                                    clippingPlanes: this._clipEnabled ? this._clipPlanes : [],
                                });

                                const mesh = new THREE.Mesh(geometry, meshMaterial);
                                mesh.name = data.id;
                                mesh.userData.id = data.id;
                                mesh.userData.isMesh = true;
                                mesh.userData.totalIndexCount = ni;
                                this._addToParentOrScene(mesh, data.parent);
                                this._objects.set(data.id, mesh);
                                this._objGeneration++;
                                if (data.transform) this._applyTransform(mesh, data.transform);
                                console.log(`Created mesh ${data.id}: ${nv} verts, ${(ni / 3)|0} tris`);
                            } catch (e) {
                                console.error(`Error creating mesh:`, e);
                            } finally {
                                this._onFetchEnd();
                            }
                        })();
                        break;
                    }
                    case 'add_parametric_tube_binary': {
                        this._onFetchStart();
                        const capturedScene = this._sceneGeneration;
                        (async () => {
                            try {
                                const resp = await fetch(data.blob_url);
                                const buffer = await resp.arrayBuffer();
                                if (this._sceneGeneration !== capturedScene) {
                                    console.log('Discarding stale parametric tube fetch');
                                    return;
                                }
                                const n = data.numSpinePoints;
                                let offset = 0;
                                const spine = new Float32Array(buffer, offset, n * 3);
                                offset += n * 3 * 4;
                                const widths = new Float32Array(buffer, offset, n);
                                offset += n * 4;
                                const heights = new Float32Array(buffer, offset, n);
                                offset += n * 4;
                                let orientations = null;
                                if (data.hasOrientations) {
                                    orientations = new Float32Array(buffer, offset, n * 4);
                                    offset += n * 4 * 4;
                                }
                                let ringColors = null;
                                if (data.hasColors) {
                                    const packed = new Uint32Array(buffer, offset, n);
                                    offset += n * 4;
                                    // Decode packed uint32 0x00RRGGBB → Float32Array(n*3).
                                    ringColors = new Float32Array(n * 3);
                                    for (let i = 0; i < n; i++) {
                                        const c = packed[i];
                                        ringColors[i * 3] = ((c >> 16) & 0xff) / 255;
                                        ringColors[i * 3 + 1] = ((c >> 8) & 0xff) / 255;
                                        ringColors[i * 3 + 2] = (c & 0xff) / 255;
                                    }
                                }
                                const nCs = data.nCrossSectionVerts || 8;
                                const cornerRadiusFrac = data.cornerRadiusFrac != null ? data.cornerRadiusFrac : 0.25;
                                const { geometry, ringPairs, indicesPerRingPair, localFrames, capAngles, capIndicesPerCap, endCapBase, endPoleIdx, endCapPattern } = buildParametricTubeGeometry(
                                    spine, widths, heights,
                                    orientations, data.upVector || null, ringColors,
                                    data.crossSection || 'rounded_rect',
                                    cornerRadiusFrac,
                                    nCs,
                                );
                                const hasColors = !!ringColors;
                                const opacity = data.opacity !== undefined ? data.opacity : 1;
                                const material = new THREE.MeshStandardMaterial({
                                    color: hasColors ? 0xffffff : (data.color || 0x7ab8cc),
                                    metalness: data.metalness !== undefined ? data.metalness : 0.1,
                                    roughness: data.roughness !== undefined ? data.roughness : 0.8,
                                    opacity,
                                    transparent: opacity < 1,
                                    depthWrite: opacity >= 1,
                                    side: THREE.DoubleSide,
                                    vertexColors: hasColors,
                                    wireframe: !!data.wireframe,
                                    clippingPlanes: this._clipEnabled ? this._clipPlanes : [],
                                });
                                const mesh = new THREE.Mesh(geometry, material);
                                mesh.name = data.id;
                                mesh.userData.id = data.id;
                                mesh.userData.isParametricTube = true;
                                mesh.userData.tubeNumSpinePoints = n;
                                mesh.userData.tubeNCs = nCs;
                                mesh.userData.tubeRingPairs = ringPairs;
                                mesh.userData.tubeIndicesPerRingPair = indicesPerRingPair;
                                mesh.userData.totalIndexCount = capIndicesPerCap + ringPairs * indicesPerRingPair + capIndicesPerCap;
                                mesh.userData.tubeHasColors = hasColors;
                                mesh.userData.tubeCapIndicesPerCap = capIndicesPerCap;
                                mesh.userData.tubeEndCapBase = endCapBase;
                                mesh.userData.tubeEndPoleIdx = endPoleIdx;
                                mesh.userData.tubeMorphData = {
                                    spine: new Float32Array(spine),
                                    widths: new Float32Array(widths),
                                    heights: new Float32Array(heights),
                                    localFrames, cornerRadiusFrac, capAngles,
                                    ringColors: ringColors ? new Float32Array(ringColors) : null,
                                    section: new Float32Array(nCs * 2),
                                    savedRing: new Float32Array(nCs * 3),
                                    savedRingColors: null,
                                    savedRingIndex: null,
                                    morphedState: null,
                                    endCapPattern,
                                    savedCapIndices: new endCapPattern.constructor(endCapPattern.length),
                                    savedCapOffset: -1,
                                };
                                this._addToParentOrScene(mesh, data.parent);
                                this._objects.set(data.id, mesh);
                                this._objGeneration++;
                                if (data.transform) this._applyTransform(mesh, data.transform);
                                console.log(`Created parametric_tube ${data.id}: ${n} spine pts × ${nCs} cs verts, ${ringPairs} ring pairs`);
                            } catch (e) {
                                console.error(`Error creating parametric_tube:`, e);
                            } finally {
                                this._onFetchEnd();
                            }
                        })();
                        break;
                    }
                    case 'update_parametric_tube_colors': {
                        this._onFetchStart();
                        const capturedScene = this._sceneGeneration;
                        (async () => {
                            try {
                                const resp = await fetch(data.blob_url);
                                const buffer = await resp.arrayBuffer();
                                if (this._sceneGeneration !== capturedScene) {
                                    console.log('Discarding stale parametric tube color fetch');
                                    return;
                                }
                                const obj = this._objects.get(data.id);
                                if (!obj || !obj.userData.isParametricTube) {
                                    console.warn(`update_parametric_tube_colors: '${data.id}' is not a parametric_tube`);
                                    return;
                                }
                                const n = obj.userData.tubeNumSpinePoints;
                                const nCs = obj.userData.tubeNCs;
                                if (buffer.byteLength < n * 4) {
                                    console.warn(`update_parametric_tube_colors: blob too small (${buffer.byteLength} < ${n * 4})`);
                                    return;
                                }
                                const packed = new Uint32Array(buffer, 0, n);
                                // Decode per-ring colors
                                const rc = new Float32Array(n * 3);
                                for (let i = 0; i < n; i++) {
                                    const c = packed[i];
                                    rc[i * 3]     = ((c >> 16) & 0xff) / 255;
                                    rc[i * 3 + 1] = ((c >> 8) & 0xff) / 255;
                                    rc[i * 3 + 2] = (c & 0xff) / 255;
                                }
                                // Fill cap dome vertices with a single color
                                function fillCapColors(arr, baseVert, capVerts, r, g, b) {
                                    for (let j = 0; j < capVerts; j++) {
                                        arr[(baseVert + j) * 3]     = r;
                                        arr[(baseVert + j) * 3 + 1] = g;
                                        arr[(baseVert + j) * 3 + 2] = b;
                                    }
                                }
                                const posCount = obj.geometry.getAttribute('position').count;
                                const capVertsPerCap = (posCount - n * nCs) / 2;
                                const startCapBaseVert = n * nCs;
                                const endCapBaseVert = startCapBaseVert + capVertsPerCap;
                                const lr = (n - 1) * 3;
                                const existing = obj.geometry.getAttribute('color');
                                if (existing) {
                                    expandRingColors(packed, n, nCs, existing.array);
                                    fillCapColors(existing.array, startCapBaseVert, capVertsPerCap, rc[0], rc[1], rc[2]);
                                    fillCapColors(existing.array, endCapBaseVert, capVertsPerCap, rc[lr], rc[lr + 1], rc[lr + 2]);
                                    existing.needsUpdate = true;
                                } else {
                                    const allColors = new Float32Array(posCount * 3);
                                    expandRingColors(packed, n, nCs, allColors);
                                    fillCapColors(allColors, startCapBaseVert, capVertsPerCap, rc[0], rc[1], rc[2]);
                                    fillCapColors(allColors, endCapBaseVert, capVertsPerCap, rc[lr], rc[lr + 1], rc[lr + 2]);
                                    obj.geometry.setAttribute('color', new THREE.BufferAttribute(allColors, 3));
                                }
                                obj.material.vertexColors = true;
                                obj.material.color.setHex(0xffffff);
                                obj.material.needsUpdate = true;
                                obj.userData.tubeHasColors = true;
                                // Sync morph data so frontier ring lerps use new colors.
                                const md = obj.userData.tubeMorphData;
                                if (md) {
                                    md.ringColors = rc;
                                    md.savedRingIndex = null;
                                }
                            } catch (e) {
                                console.error(`Error updating parametric_tube colors:`, e);
                            } finally {
                                this._onFetchEnd();
                            }
                        })();
                        break;
                    }
                    case 'stop_animation':
                        this._stopAnimation();
                        break;
                    case 'clear_animation':
                        this._stopAnimation(false);
                        break;
                    case 'set_clip_time':
                        this._setClipTime(data.id, data.time);
                        break;
                    case 'set_draw_range':
                        this._setDrawRange(data.id, data.value);
                        break;
                    case 'set_clipping_plane': {
                        this._clipEnabled = true;
                        this._clipSlabMode = false;
                        this._clipPlanes = [this._clipPlane];
                        this._clipModeSingle.classList.add('active');
                        this._clipModeSlab.classList.remove('active');
                        this._clipThicknessSection.style.display = 'none';
                        if (data.normal) {
                            this._clipPlane.normal.fromArray(data.normal).normalize();
                            this._syncNormalInputs();
                        }
                        if (data.distance != null) this._setClipDistance(data.distance);
                        if (data.show_helper != null) this._clipHelperVisible = data.show_helper;
                        let matchedAxis = null;
                        for (const [axis, n] of Object.entries(CLIP_AXIS_NORMALS)) {
                            if (n.equals(this._clipPlane.normal)) { matchedAxis = axis; break; }
                        }
                        if (matchedAxis) this._setClipAxis(matchedAxis);
                        this._updateClipSliderRange();
                        this._syncAnchorFromPlane();
                        this._clipPanelEl.classList.add('visible');
                        this._btnClip.classList.add('active');
                        this._updateClipMaterials();
                        break;
                    }
                    case 'set_clipping_slab': {
                        this._clipEnabled = true;
                        this._clipSlabMode = true;
                        this._clipPlanes = [this._clipPlane, this._clipPlane2];
                        this._clipModeSlab.classList.add('active');
                        this._clipModeSingle.classList.remove('active');
                        this._clipThicknessSection.style.display = '';
                        if (data.normal) {
                            this._clipPlane.normal.fromArray(data.normal).normalize();
                            this._syncNormalInputs();
                        }
                        if (data.thickness != null) this._setClipThickness(data.thickness);
                        if (data.center != null) this._setClipDistance(data.center);
                        if (data.show_helper != null) this._clipHelperVisible = data.show_helper;
                        let matchedAxis2 = null;
                        for (const [axis, n] of Object.entries(CLIP_AXIS_NORMALS)) {
                            if (n.equals(this._clipPlane.normal)) { matchedAxis2 = axis; break; }
                        }
                        if (matchedAxis2) this._setClipAxis(matchedAxis2);
                        this._updateClipSliderRange();
                        this._syncAnchorFromPlane();
                        this._clipPanelEl.classList.add('visible');
                        this._btnClip.classList.add('active');
                        this._updateClipMaterials();
                        break;
                    }
                    case 'disable_clipping_plane':
                        this._clipEnabled = false;
                        this._clipPanelEl.classList.remove('visible');
                        this._btnClip.classList.remove('active');
                        this._updateClipMaterials();
                        break;
                    case 'set_clipping_defaults':
                        this._clipDefaults = { normal: data.normal, distance: data.distance };
                        break;
                    case 'show_grid':
                        this._gridHelper.visible = !!data.visible;
                        if (data.size != null && data.divisions != null) {
                            const parent = this._gridHelper.parent;
                            parent.remove(this._gridHelper);
                            this._gridHelper.geometry.dispose();
                            this._gridHelper.material.dispose();
                            this._gridHelper = new THREE.GridHelper(data.size, data.divisions);
                            this._gridHelper.rotation.x = Math.PI / 2;
                            this._gridHelper.visible = !!data.visible;
                            parent.add(this._gridHelper);
                        }
                        break;
                    case 'mark_assets_complete':
                        this._assetsComplete = true;
                        this._maybeNotifyAssetsLoaded();
                        break;
                    default:
                        console.warn(`Unknown message type: '${data.type}'`);
                }
            };
        };

        doConnect();
    }

    // ========== Dynamic Near/Far ==========

    _updateSceneBounds() {
        const box = new THREE.Box3();
        for (const obj of this._objects.values()) {
            obj.updateWorldMatrix(true, true);
            box.expandByObject(obj);
        }
        if (box.isEmpty()) {
            this._sceneSphere.set(new THREE.Vector3(), 0);
        } else {
            box.getBoundingSphere(this._sceneSphere);
        }
        this._sceneBoundsDirty = false;
    }

    _updateNearFar() {
        if (this._isOrtho) return;
        // Recompute bounds on dirty flag or every 30 frames (~0.5s) to catch transform changes
        this._boundsFrameCounter++;
        if (this._sceneBoundsDirty || this._boundsFrameCounter >= 30) {
            this._updateSceneBounds();
            this._boundsFrameCounter = 0;
        }
        const radius = this._sceneSphere.radius;
        if (radius === 0) return;
        const dist = this._perspCamera.position.distanceTo(this._sceneSphere.center);
        const nextNear = Math.max(0.001, (dist - radius * 1.5) * 0.5);
        const nextFar = Math.max(dist + radius * 1.5, 100);
        if (
            Math.abs(this._perspCamera.near - nextNear) < 1e-6 &&
            Math.abs(this._perspCamera.far - nextFar) < 1e-6
        ) return;
        this._perspCamera.near = nextNear;
        this._perspCamera.far = nextFar;
        this._perspCamera.updateProjectionMatrix();
    }

    // ========== Render Loop ==========

    _animate() {
        this._animationFrameId = requestAnimationFrame(this._animate);
        const frameNow = performance.now();
        const frameDelta = (frameNow - this._lastFrameTime) / 1000;
        this._lastFrameTime = frameNow;

        // Update animation playback
        if (this._animation && this._animationPlaying) {
            const now = performance.now();
            const deltaTime = (now - this._lastAnimationUpdate) / 1000;
            this._lastAnimationUpdate = now;

            this._animationTime += deltaTime * this._animationSpeed;

            if (this._animationTime >= this._animation.duration) {
                if (this._animationLoop) {
                    this._animationTime = this._animationTime % this._animation.duration;
                } else {
                    this._animationTime = this._animation.duration;
                    this._animationPlaying = false;
                }
            }

            const { index, t } = this._getFrameAtTime(this._animationTime);
            this._applyFrame(index, t);
            if (now - this._lastUIUpdate > 100) {
                this._updateAnimationUI();
                this._lastUIUpdate = now;
            }
        }

        this._controls.update();
        this._updateNearFar();

        if (this._viewHelper.animating) this._viewHelper.update(frameDelta);
        this._renderer.autoClear = true;
        this._renderer.render(this._scene, this._camera);
        this._renderer.autoClear = false;
        this._viewHelper.render(this._renderer);
    }

    // ========== Public API ==========

    resize(width, height) {
        width = width ?? this.container.clientWidth;
        height = height ?? this.container.clientHeight;
        if (width === 0 || height === 0) return;
        const aspect = width / height;
        this._perspCamera.aspect = aspect;
        this._perspCamera.updateProjectionMatrix();
        this._orthoCamera.left = -ORTHO_FRUSTUM * aspect;
        this._orthoCamera.right = ORTHO_FRUSTUM * aspect;
        this._orthoCamera.updateProjectionMatrix();
        this._renderer.setSize(width, height);
        this._objects.forEach((obj) => {
            if (obj.material && obj.material.resolution) {
                obj.material.resolution.set(width, height);
            }
        });
    }

    frameAll() {
        const bbox = new THREE.Box3();
        this._scene.traverse(child => {
            if (!child.geometry) return;
            if (child === this._gridHelper) return;
            if (this._isClipHelper(child)) return;
            child.updateWorldMatrix(true, false);
            bbox.expandByObject(child);
        });
        if (bbox.isEmpty()) return;

        const center = bbox.getCenter(new THREE.Vector3());
        const size = bbox.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);

        this._controls.target.copy(center);

        if (this._isOrtho) {
            const w = this.container.clientWidth;
            const h = this.container.clientHeight;
            if (w === 0 || h === 0) return;
            const aspect = w / h;
            const halfHeight = Math.max(size.z, size.y) / 2 * 1.2;
            const halfWidth = Math.max(size.x, size.y) / 2 * 1.2;
            const fitHalf = Math.max(halfHeight, halfWidth / aspect, 1e-6);
            this._orthoCamera.zoom = ORTHO_FRUSTUM / fitHalf;
            this._orthoCamera.left = -ORTHO_FRUSTUM * aspect;
            this._orthoCamera.right = ORTHO_FRUSTUM * aspect;
            this._orthoCamera.top = ORTHO_FRUSTUM;
            this._orthoCamera.bottom = -ORTHO_FRUSTUM;
            this._orthoCamera.updateProjectionMatrix();
            const dir = this._camera.position.clone().sub(this._controls.target);
            if (dir.lengthSq() < 1e-10) dir.set(1, -1, 1).normalize();
            else dir.normalize();
            this._camera.position.copy(center).addScaledVector(dir, maxDim * 2);
        } else {
            const vFov = THREE.MathUtils.degToRad(this._perspCamera.fov / 2);
            const aspect = this._perspCamera.aspect || 1;
            const hFov = Math.atan(Math.tan(vFov) * aspect);
            const distV = Math.max(size.y, size.z) / 2 / Math.tan(vFov);
            const distH = Math.max(size.x, size.y) / 2 / Math.tan(hFov);
            const dist = Math.max(distV, distH) * 1.2;
            const dir = this._camera.position.clone().sub(this._controls.target);
            if (dir.lengthSq() < 1e-10) dir.set(1, -1, 1).normalize();
            else dir.normalize();
            this._camera.position.copy(center).addScaledVector(dir, dist);
        }

        this._controls.update();
    }

    // ========== Destroy ==========

    destroy() {
        this._destroyed = true;
        cancelAnimationFrame(this._animationFrameId);
        if (this._ws) {
            this._ws.onclose = null;
            this._ws.close();
            this._ws = null;
        }
        clearTimeout(this._reconnectTimeout);
        this._resizeObserver.disconnect();
        this.container.removeEventListener('keydown', this._onKeyDown);
        document.removeEventListener('mousemove', this._onDocMouseMove);
        document.removeEventListener('mouseup', this._onDocMouseUp);
        this._renderer.dispose();
        this._controls.dispose();
        this._clipGizmo.dispose();
        this.el.remove();
    }
}
