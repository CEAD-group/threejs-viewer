// @ts-check
import * as THREE from 'three';
import { ViewerControls } from './controls.js';
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
import { VertexNormalsHelper } from 'three/addons/helpers/VertexNormalsHelper.js';
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


// ========== Typedefs ==========
//
// These describe recurring shapes used across viewer.js. They are intentionally
// loose — WebSocket payloads come in as `unknown` and are validated at the call
// site — but they document the expected fields well enough for editors to catch
// typos and wrong property accesses.

/**
 * @typedef {Object} ThreeJSViewerOptions
 * @property {string} htmlTemplate                        HTML template string for UI controls (required — constructor throws without it)
 * @property {string} [wsUrl]                             Full WebSocket URL override. When omitted, falls back to `ws://localhost:${port}` where port comes from the `ws_port` query param, `wsPort`, or 5666
 * @property {number} [wsPort]                            WebSocket port used when `wsUrl` is not provided (default 5666)
 * @property {boolean} [autoConnect]                      Auto-connect on construction (default true)
 * @property {Object<string, string>} [cubemapData]       Map of face name -> base64 JPEG
 * @property {number} [toneMappingExposure]               Tone-mapping exposure (default 1.0)
 * @property {number} [environmentIntensity]              Scene environment intensity (default 2.0)
 * @property {number} [ambientIntensity]                  Ambient-light intensity (default 1.5)
 * @property {string} [toneMapping]                       Tone-mapping mode: one of none/linear/reinhard/cineon/aces/agx/neutral (default "aces")
 */

/**
 * A binary animation channel once it has been materialised into a TypedArray
 * view on the main thread. `refs` is a lazily-populated parallel array of the
 * Three.js objects looked up by `ids`. `_objGen` / `_mixerGen` are cached
 * generation counters used by the appliers to detect when the scene graph
 * changed and refs need re-resolution.
 *
 * @typedef {Object} BinaryChannel
 * @property {ArrayLike<number>} data                Flat typed array of length nFrames * ids.length * stride
 * @property {string[]} ids                          Object ids addressed by this channel
 * @property {number} stride                         Elements per (frame, object) tuple
 * @property {Array<any>|null} refs                  Cached lookups for `ids` (null until first refresh)
 * @property {number[]|null} [colormap]              Optional palette for indexed uint8 channels
 * @property {'linear'|'hold'} interpolation         Per-channel interpolation mode
 * @property {number} [_objGen]                      Cached viewer._objGeneration when refs were last resolved
 * @property {number} [_mixerGen]                    Cached viewer._mixerGeneration (clip_times only)
 */

/**
 * Minimal shape of a JSON animation frame. Fields beyond `time` are optional
 * and only present when the producer emitted that channel; binary channels on
 * the enclosing Animation supersede same-named frame fields.
 *
 * @typedef {Object} AnimationFrame
 * @property {number} time
 * @property {Record<string, number[]>} [transforms]
 * @property {Record<string, number>}   [colors]
 * @property {Record<string, boolean>}  [visibility]
 * @property {Record<string, number>}   [opacity]
 * @property {Record<string, number>}   [clip_times]
 * @property {Record<string, number>}   [draw_ranges]
 */

/**
 * Animation payload (either JSON-frame or binary-channel backed) received from
 * Python. Frames and channels may coexist.
 *
 * @typedef {Object} AnimationData
 * @property {number} duration
 * @property {AnimationFrame[]} [frames]
 * @property {Record<string, BinaryChannel>} [channels]
 * @property {number} [generation]
 */


// Refresh cached refs array when the object/mixer map has changed.
// Called at most once per frame per channel (guarded by generation check).
/**
 * @param {Array<any>} refs
 * @param {string[]} ids
 * @param {Map<string, any>} map
 */
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
/**
 * @param {ArrayLike<number>} srcA
 * @param {number} baseA
 * @param {ArrayLike<number>} srcB
 * @param {number} baseB
 * @param {number} t
 * @param {THREE.Matrix4} outMat
 */
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
/**
 * @param {{ interpolation?: string }} ch
 * @param {number | null} baseNext
 * @param {number} t
 */
function shouldInterpChannel(ch, baseNext, t) {
    if (baseNext === null || t <= 0) return false;
    return ch.interpolation === 'linear';
}

// Linear-interpolate two hex colors in 8-bit RGB space. Alpha byte (top 8
// bits) is preserved from `a`; three.js only consumes the low 24 bits.
/**
 * @param {number} a
 * @param {number} b
 * @param {number} t
 */
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
/**
 * @param {unknown} value
 * @param {string} fallback
 */
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
// TODO(types): `viewer` is ThreeJSViewer but its instance fields aren't all
// declared up-front (many are attached inside methods), so a strict
// ThreeJSViewer annotation surfaces noisy "property X does not exist" errors.
// Leaving `any` until we either declare all fields in the constructor or
// introduce a dedicated interface.
/** @param {any} viewer */
function makeChannelApply(viewer) {
    return {
        /**
         * @param {BinaryChannel} ch
         * @param {Array<any>} refs
         * @param {number} base
         * @param {number | null} baseNext
         * @param {number} t
         */
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

        /**
         * @param {BinaryChannel} ch
         * @param {Array<any>} refs
         * @param {number} base
         * @param {number | null} baseNext
         * @param {number} t
         */
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
                obj.traverse(/** @param {any} child */ (child) => {
                    if (!child.material) return;
                    const mats = Array.isArray(child.material) ? child.material : [child.material];
                    for (const mat of mats) { if (mat.color) mat.color.setHex(color); }
                });
            }
        },

        /**
         * @param {BinaryChannel} ch
         * @param {Array<any>} refs
         * @param {number} base
         */
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

        /**
         * @param {BinaryChannel} ch
         * @param {Array<any>} refs
         * @param {number} base
         * @param {number | null} baseNext
         * @param {number} t
         */
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
                if (obj.userData.isToolpathGroup) {
                    applyToolpathGroupDrawRange(obj, value, viewer._objects);
                } else if (obj.userData.isPolyline) {
                    obj.geometry.instanceCount = Math.round(value * obj.userData.maxInstanceCount);
                } else if (obj.userData.isParametricTube) {
                    applyParametricTubeDrawRange(obj, value);
                } else if (obj.userData.isMesh) {
                    obj.geometry.setDrawRange(0, Math.round(value * obj.userData.totalIndexCount));
                }
            }
        },

        /**
         * @param {BinaryChannel} ch
         * @param {Array<any>} refs
         * @param {number} base
         * @param {number | null} baseNext
         * @param {number} t
         */
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

        /**
         * @param {BinaryChannel} ch
         * @param {Array<any>} refs
         * @param {number} base
         * @param {number | null} baseNext
         * @param {number} t
         */
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

// Default lighting values. Panel ranges: exposure 0.0–3.0, env intensity 0.0–4.0, ambient 0.0–3.0.
const DEFAULT_TONE_MAPPING_EXPOSURE = 1.0;
const DEFAULT_ENVIRONMENT_INTENSITY = 2.0;
const DEFAULT_AMBIENT_INTENSITY = 1.5;
const DEFAULT_TONE_MAPPING = 'aces';
const LS_KEY_TONE_MAPPING_EXPOSURE = 'tjsv.toneMappingExposure';
const LS_KEY_ENVIRONMENT_INTENSITY = 'tjsv.environmentIntensity';
const LS_KEY_AMBIENT_INTENSITY = 'tjsv.ambientIntensity';
const LS_KEY_TONE_MAPPING = 'tjsv.toneMapping';

// Tone-mapping mode name -> THREE.* constant. Resolved lazily so THREE only
// needs to be loaded when the viewer actually instantiates.
/** @returns {Record<string, number>} */
function toneMappingModes() {
    return {
        none: THREE.NoToneMapping,
        linear: THREE.LinearToneMapping,
        reinhard: THREE.ReinhardToneMapping,
        cineon: THREE.CineonToneMapping,
        aces: THREE.ACESFilmicToneMapping,
        agx: THREE.AgXToneMapping,
        neutral: THREE.NeutralToneMapping,
    };
}
const TONE_MAPPING_MODE_NAMES = ['none', 'linear', 'reinhard', 'cineon', 'aces', 'agx', 'neutral'];

/**
 * Resolve lighting values with two layered views:
 *
 *   `reset`  — the baseline the panel's Reset button restores to. Uses
 *              URL param > options > hard defaults (localStorage is ignored
 *              here, because "Reset" means "go back to the developer default
 *              for this page load", not "re-apply the last session's tweak").
 *   top-level — the effective initial values applied at startup and used
 *              to seed the panel UI. Uses URL param > options > localStorage
 *              > hard defaults.
 *
 * URL-pinned / option-pinned values end up identical in both views, so
 * reloading a URL with `tone_mapping_exposure=2.3` sticks across reloads
 * even if the user previously tweaked the panel.
 *
 * Invalid tone-mapping strings (not one of the seven known modes) and
 * non-finite numeric values silently fall through to the next level — we
 * never throw in the browser here; Python already validates its kwargs.
 *
 * @param {ThreeJSViewerOptions} options
 * @param {URLSearchParams} urlParams
 * @returns {{
 *     exposure: number,
 *     envIntensity: number,
 *     ambientIntensity: number,
 *     toneMapping: string,
 *     reset: {
 *         exposure: number,
 *         envIntensity: number,
 *         ambientIntensity: number,
 *         toneMapping: string,
 *     },
 * }}
 */
function resolveLightingDefaults(options, urlParams) {
    /** @type {(raw: (string|null|number|undefined)) => (number|null)} */
    const parseFinite = (raw) => {
        if (raw === null || raw === undefined || raw === '') return null;
        const n = typeof raw === 'number' ? raw : parseFloat(raw);
        return Number.isFinite(n) ? n : null;
    };
    /** @type {(raw: (string|null|undefined)) => (string|null)} */
    const parseToneMapping = (raw) => {
        if (raw === null || raw === undefined || raw === '') return null;
        const s = String(raw).toLowerCase();
        return TONE_MAPPING_MODE_NAMES.includes(s) ? s : null;
    };
    const urlExp = parseFinite(urlParams.get('tone_mapping_exposure'));
    const urlEnv = parseFinite(urlParams.get('environment_intensity'));
    const urlAmb = parseFinite(urlParams.get('ambient_intensity'));
    const urlTm = parseToneMapping(urlParams.get('tone_mapping'));
    const optExp = parseFinite(options.toneMappingExposure);
    const optEnv = parseFinite(options.environmentIntensity);
    const optAmb = parseFinite(options.ambientIntensity);
    const optTm = parseToneMapping(options.toneMapping);
    let lsExp = null;
    let lsEnv = null;
    let lsAmb = null;
    let lsTm = null;
    try {
        lsExp = parseFinite(localStorage.getItem(LS_KEY_TONE_MAPPING_EXPOSURE));
        lsEnv = parseFinite(localStorage.getItem(LS_KEY_ENVIRONMENT_INTENSITY));
        lsAmb = parseFinite(localStorage.getItem(LS_KEY_AMBIENT_INTENSITY));
        lsTm = parseToneMapping(localStorage.getItem(LS_KEY_TONE_MAPPING));
    } catch (e) { /* ignore storage errors */ }

    // Reset baseline: URL > options > hard defaults (no localStorage).
    const resetExposure = urlExp != null ? urlExp
        : optExp != null ? optExp
        : DEFAULT_TONE_MAPPING_EXPOSURE;
    const resetEnvIntensity = urlEnv != null ? urlEnv
        : optEnv != null ? optEnv
        : DEFAULT_ENVIRONMENT_INTENSITY;
    const resetAmbientIntensity = urlAmb != null ? urlAmb
        : optAmb != null ? optAmb
        : DEFAULT_AMBIENT_INTENSITY;
    const resetToneMapping = urlTm != null ? urlTm
        : optTm != null ? optTm
        : DEFAULT_TONE_MAPPING;

    // Effective initial: localStorage overlays the reset baseline, but only
    // when URL/options haven't already pinned the value.
    const exposure = urlExp != null ? urlExp
        : optExp != null ? optExp
        : lsExp != null ? lsExp
        : DEFAULT_TONE_MAPPING_EXPOSURE;
    const envIntensity = urlEnv != null ? urlEnv
        : optEnv != null ? optEnv
        : lsEnv != null ? lsEnv
        : DEFAULT_ENVIRONMENT_INTENSITY;
    const ambientIntensity = urlAmb != null ? urlAmb
        : optAmb != null ? optAmb
        : lsAmb != null ? lsAmb
        : DEFAULT_AMBIENT_INTENSITY;
    const toneMapping = urlTm != null ? urlTm
        : optTm != null ? optTm
        : lsTm != null ? lsTm
        : DEFAULT_TONE_MAPPING;
    return {
        exposure,
        envIntensity,
        ambientIntensity,
        toneMapping,
        reset: {
            exposure: resetExposure,
            envIntensity: resetEnvIntensity,
            ambientIntensity: resetAmbientIntensity,
            toneMapping: resetToneMapping,
        },
    };
}

/**
 * @param {THREE.Object3D} obj
 * @param {number} opacity
 */
function applyOpacity(obj, opacity) {
    obj.traverse(/** @param {any} child */ child => {
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
/** @type {Record<string, (params: any) => THREE.BufferGeometry>} */
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

// Chamfered rectangle cross-section: 45° chamfers on all corners, depth =
// min(width, height) / 2.  Always emits N_CROSS_SECTION (6) vertices CCW.
// When w > h: flat top & bottom, pointed left & right (hexagon).
// When h > w: flat left & right, pointed top & bottom (hexagon).
// When w == h: diamond — two vertex pairs coincide, giving 4 unique corners.
const N_CROSS_SECTION = 6;
// LOD: epsilon = camera_distance / LOD_EPSILON_DIVISOR.
// Higher → more points kept (finer detail). Lower → more aggressive simplification.
// Per-tube overrides: `lod` option on `add_parametric_tube` (see parseLodConfig).
const LOD_EPSILON_DIVISOR = 2500;
// Spine-length gate: tubes with fewer points skip LOD entirely (overridable
// per-tube via `lod.threshold`).
const LOD_DEFAULT_THRESHOLD = 25000;
// Max original points that can be skipped between two kept points.
// Safety valve for pathological inputs; augmented-space RDP below already
// catches attribute variation on geometrically straight segments.
const LOD_MAX_SKIP = 100;
// Augmented-RDP weights: convert each attribute delta into "world-equivalent
// units" so the single camera-scaled epsilon bounds geometric and attribute
// error together. Widths/heights are already in world units; their weights
// scale the boundary shift they produce (delta-width of 1 moves each wall by
// ~0.5). Color channels are unitless (0..1) and get multiplied by the tube's
// bounding radius × LOD_COLOR_WEIGHT_FRAC at RDP time.
const LOD_WIDTH_WEIGHT = 0.5;
const LOD_HEIGHT_WEIGHT = 0.5;
const LOD_COLOR_WEIGHT_FRAC = 0.05;

/**
 * Parse the optional per-tube `lod` payload sent from Python.
 *
 * Accepts `false` (disable), `undefined`/`null` (defaults), or a config
 * object `{ epsilonDivisor?, threshold? }`. Uses `!== undefined` rather
 * than `||` so `threshold: 0` survives as "force LOD on for short spines."
 *
 * @param {unknown} lod
 * @returns {{enabled: boolean, epsilonDivisor: number, threshold: number}}
 */
function parseLodConfig(lod) {
    if (lod === false) {
        return { enabled: false, epsilonDivisor: LOD_EPSILON_DIVISOR, threshold: LOD_DEFAULT_THRESHOLD };
    }
    const cfg = (lod && typeof lod === 'object') ? /** @type {any} */ (lod) : {};
    return {
        enabled: true,
        epsilonDivisor: cfg.epsilonDivisor !== undefined ? cfg.epsilonDivisor : LOD_EPSILON_DIVISOR,
        threshold: cfg.threshold !== undefined ? cfg.threshold : LOD_DEFAULT_THRESHOLD,
    };
}
/**
 * @param {Float32Array} out
 * @param {number} width
 * @param {number} height
 */
function sampleChamferedRect(out, width, height) {
    if (!Number.isFinite(width) || width < 0) {
        throw new Error(`parametric_tube width must be finite and >= 0, got ${width}`);
    }
    if (!Number.isFinite(height) || height < 0) {
        throw new Error(`parametric_tube height must be finite and >= 0, got ${height}`);
    }
    if (width === 0 || height === 0) {
        for (let i = 0; i < N_CROSS_SECTION * 2; i++) out[i] = 0;
        return;
    }
    const hw = width * 0.5;
    const hh = height * 0.5;
    const c = Math.min(hw, hh); // chamfer depth (45°)

    // 6 corners CCW.  Which edges are flat depends on aspect ratio:
    //   w >= h: flat top/bottom, pointed right/left
    //   h > w:  flat left/right, pointed top/bottom
    if (width >= height) {
        out[0]  = +hw;        out[1]  = 0;            // right tip
        out[2]  = +(hw - c);  out[3]  = +hh;          // top-right
        out[4]  = -(hw - c);  out[5]  = +hh;          // top-left
        out[6]  = -hw;        out[7]  = 0;             // left tip
        out[8]  = -(hw - c);  out[9]  = -hh;          // bottom-left
        out[10] = +(hw - c);  out[11] = -hh;           // bottom-right
    } else {
        out[0]  = +hw;        out[1]  = -(hh - c);    // right-bottom
        out[2]  = +hw;        out[3]  = +(hh - c);    // right-top
        out[4]  = 0;          out[5]  = +hh;           // top tip
        out[6]  = -hw;        out[7]  = +(hh - c);    // left-top
        out[8]  = -hw;        out[9]  = -(hh - c);    // left-bottom
        out[10] = 0;          out[11] = -hh;            // bottom tip
    }
}

// Per-vertex 2D outward normals for a CCW cross-section (chamfered hex).
// Computed as the bisector of the two adjacent edge normals, giving smooth
// shading that depends ONLY on the cross-section shape — independent of
// spine spacing. This is the key to killing LOD-induced normal artifacts:
// face-area-weighted vertex normals skew heavily when adjacent ring-to-ring
// quads have wildly different sizes (common after distance-weighted RDP).
/**
 * @param {Float32Array} section
 * @param {number} nCs
 * @param {Float32Array} out
 */
function computeSectionNormals(section, nCs, out) {
    for (let j = 0; j < nCs; j++) {
        const jPrev = (j - 1 + nCs) % nCs;
        const jNext = (j + 1) % nCs;
        const ptx = section[j * 2]     - section[jPrev * 2];
        const pty = section[j * 2 + 1] - section[jPrev * 2 + 1];
        const ntx = section[jNext * 2] - section[j * 2];
        const nty = section[jNext * 2 + 1] - section[j * 2 + 1];
        // Outward normal of a CCW edge with tangent (tx,ty) is (ty, -tx).
        let nu = pty + nty;
        let nv = -(ptx + ntx);
        const len = Math.hypot(nu, nv);
        if (len > 1e-12) { nu /= len; nv /= len; }
        out[j * 2] = nu;
        out[j * 2 + 1] = nv;
    }
}

// Analytic normals for a revolution cap's (nCapRings * nCs) vertices.
// At angle θ: n(θ) = normalize( cos(θ)·(nu·U + nv·V) + sin(θ)·tSign·T ).
// This matches the tube-side section normal at θ=0 (no shading seam) and
// points along ±T at θ=90° (correct axial normal at the dome tip).
/**
 * @param {Float32Array} normalArr
 * @param {number} capBaseVert
 * @param {number} nCs
 * @param {number} nCapRings
 * @param {Float32Array} capAngles
 * @param {number} width
 * @param {number} height
 * @param {Float32Array} localFrames
 * @param {number} spineIdx
 * @param {number} tSign
 * @param {number} Tx @param {number} Ty @param {number} Tz
 */
function writeAnalyticCapNormals(normalArr, capBaseVert, nCs, nCapRings, capAngles,
                                 width, height, localFrames, spineIdx, tSign,
                                 Tx, Ty, Tz) {
    const Ux = localFrames[spineIdx * 6],     Uy = localFrames[spineIdx * 6 + 1], Uz = localFrames[spineIdx * 6 + 2];
    const Vx = localFrames[spineIdx * 6 + 3], Vy = localFrames[spineIdx * 6 + 4], Vz = localFrames[spineIdx * 6 + 5];
    const section = _capScratchSection;
    sampleChamferedRect(section, width, height);
    const sectionNormals = _capScratchSectionNormals;
    computeSectionNormals(section, nCs, sectionNormals);
    for (let k = 0; k < nCapRings; k++) {
        const theta = capAngles[k];
        const c = Math.cos(theta);
        const s = Math.sin(theta) * tSign;
        for (let j = 0; j < nCs; j++) {
            const nu = sectionNormals[j * 2], nv = sectionNormals[j * 2 + 1];
            let nx = c * (nu * Ux + nv * Vx) + s * Tx;
            let ny = c * (nu * Uy + nv * Vy) + s * Ty;
            let nz = c * (nu * Uz + nv * Vz) + s * Tz;
            const len = Math.hypot(nx, ny, nz);
            if (len > 1e-12) { nx /= len; ny /= len; nz /= len; }
            const dst = (capBaseVert + k * nCs + j) * 3;
            normalArr[dst] = nx;
            normalArr[dst + 1] = ny;
            normalArr[dst + 2] = nz;
        }
    }
}
const _capScratchSection = new Float32Array(N_CROSS_SECTION * 2);
const _capScratchSectionNormals = new Float32Array(N_CROSS_SECTION * 2);

// SVG-style miter limit. Past this ratio the miter is clamped so a near-hairpin
// corner doesn't spike off to infinity.
const TUBE_MITER_LIMIT = 4;

// Per-spine-point miter frames + u-axis scale factor. At interior points the
// tangent is the unit bisector of the incoming/outgoing segment directions
// (not the central-difference average — the two agree for equal segment
// lengths but diverge for unequal). Miter scale = 1/cos(half_turn_angle),
// clamped by TUBE_MITER_LIMIT. Endpoints get scale 1.
//
// We only scale the U-axis component of cross-section offsets (not V). For
// predominantly-horizontal paths with a constant-up frame this is exact: the
// turn plane is horizontal, U is the in-turn-plane width direction, V stays
// vertical. For turns in the vertical plane the scaling direction would be
// V — not handled here; that case degrades to "no miter correction on the
// vertical component", which is still not worse than the pre-miter build.
//
// `outFrames`, `outScales`, and `outTangents` must be pre-allocated:
// Float32Array(nSpine*6), Float32Array(nSpine), and Float32Array(nSpine*3).
// `outTangents` receives the unit-bisector tangent at each spine point — this
// is what downstream cap construction reads (not U × V), so that the hairpin
// U-flip sweep that runs after this function doesn't leave caps extruding
// backwards into the tube body.
/**
 * @param {Float32Array} spine
 * @param {number} nSpine
 * @param {Float32Array} outFrames
 * @param {Float32Array} outScales
 * @param {Float32Array} outTangents
 * @param {number} upX @param {number} upY @param {number} upZ
 * @param {number} fbX @param {number} fbY @param {number} fbZ
 */
function computeMiterFrames(spine, nSpine, outFrames, outScales, outTangents,
                            upX, upY, upZ, fbX, fbY, fbZ) {
    const nSeg = nSpine - 1;
    // Rolling segment directions — avoids a nSeg*3 scratch buffer on large
    // spines (~12 MB throwaway at 1M points). At spine point i, (pX,pY,pZ)
    // holds segDir[i-1] (incoming) and (cX,cY,cZ) holds segDir[i] (outgoing).
    // Endpoints use both = the single adjacent segment direction. A single-
    // point spine (nSeg===0) falls through with the default (1,0,0).
    let pX = 1, pY = 0, pZ = 0;
    let cX = 1, cY = 0, cZ = 0;
    if (nSeg > 0) {
        let dx = spine[3]     - spine[0];
        let dy = spine[4]     - spine[1];
        let dz = spine[5]     - spine[2];
        const len = Math.hypot(dx, dy, dz);
        if (len > 1e-12) { dx /= len; dy /= len; dz /= len; }
        else { dx = 1; dy = 0; dz = 0; }
        pX = cX = dx; pY = cY = dy; pZ = cZ = dz;
    }
    for (let i = 0; i < nSpine; i++) {
        // Advance for interior points: shift c→p, compute new c = segDir[i].
        // Endpoints (i=0 or i=nSpine-1) skip the advance, so p and c both hold
        // their single adjacent segment direction.
        if (i > 0 && i < nSeg) {
            pX = cX; pY = cY; pZ = cZ;
            let dx = spine[(i + 1) * 3]     - spine[i * 3];
            let dy = spine[(i + 1) * 3 + 1] - spine[i * 3 + 1];
            let dz = spine[(i + 1) * 3 + 2] - spine[i * 3 + 2];
            const len = Math.hypot(dx, dy, dz);
            if (len > 1e-12) { dx /= len; dy /= len; dz /= len; }
            else { dx = 1; dy = 0; dz = 0; }
            cX = dx; cY = dy; cZ = dz;
        }
        const isEnd = i === 0 || i === nSpine - 1;
        const inX  = isEnd ? cX : pX, inY  = isEnd ? cY : pY, inZ  = isEnd ? cZ : pZ;
        const outX = cX, outY = cY, outZ = cZ;
        // Unit-bisector tangent.
        let tx = inX + outX, ty = inY + outY, tz = inZ + outZ;
        const tlen = Math.hypot(tx, ty, tz);
        if (tlen > 1e-12) {
            tx /= tlen; ty /= tlen; tz /= tlen;
        } else {
            // Near-hairpin (incoming ≈ -outgoing): bisector undefined. Fall
            // back to the incoming direction; miter scale will be clamped to
            // the limit below. The hairpin itself is a separate topology
            // problem the miter fix can't solve.
            tx = inX; ty = inY; tz = inZ;
        }
        // V axis: constant-up projection.
        const dotTu = tx * upX + ty * upY + tz * upZ;
        const seedX = Math.abs(dotTu) > 0.99 ? fbX : upX;
        const seedY = Math.abs(dotTu) > 0.99 ? fbY : upY;
        const seedZ = Math.abs(dotTu) > 0.99 ? fbZ : upZ;
        const sdot = seedX * tx + seedY * ty + seedZ * tz;
        let vx = seedX - sdot * tx, vy = seedY - sdot * ty, vz = seedZ - sdot * tz;
        const vlen = Math.hypot(vx, vy, vz);
        if (vlen > 1e-12) { vx /= vlen; vy /= vlen; vz /= vlen; }
        // U = V × T.
        let Ux = vy * tz - vz * ty, Uy = vz * tx - vx * tz, Uz = vx * ty - vy * tx;
        const ulen = Math.hypot(Ux, Uy, Uz);
        if (ulen > 1e-12) { Ux /= ulen; Uy /= ulen; Uz /= ulen; }
        // Re-orthogonalize V = T × U.
        vx = ty * Uz - tz * Uy; vy = tz * Ux - tx * Uz; vz = tx * Uy - ty * Ux;

        outFrames[i * 6]     = Ux; outFrames[i * 6 + 1] = Uy; outFrames[i * 6 + 2] = Uz;
        outFrames[i * 6 + 3] = vx; outFrames[i * 6 + 4] = vy; outFrames[i * 6 + 5] = vz;
        outTangents[i * 3] = tx; outTangents[i * 3 + 1] = ty; outTangents[i * 3 + 2] = tz;

        // Miter scale: 1 / cos(half_turn). cos(turn) = in · out.
        // cos²(half) = (1 + cos(turn)) / 2.
        // Past the miter limit (e.g. near-hairpins) we fall back to scale = 1
        // — SVG-style bevel-on-overflow. Strictly no worse than a non-mitered
        // build at those corners, and avoids a spike where the bisector frame
        // is anyway ill-defined.
        const dDot = inX * outX + inY * outY + inZ * outZ;
        const cosHalfSq = Math.max(0, (1 + dDot) * 0.5);
        const cosHalf = Math.sqrt(cosHalfSq);
        outScales[i] = cosHalf < 1 / TUBE_MITER_LIMIT ? 1 : 1 / cosHalf;
    }
}

// Write one tube ring: `nCs` cross-section samples laid out around spine
// point (sx,sy,sz) using local frame (U,V). `vOff` shifts along V for the
// anchor offset (see the `heightOffset` / anchor parameter). `uScale`
// multiplies the u-component so miter-joined corners can extend the
// cross-section along the width direction without changing normals.
/**
 * @param {any} positions - Float32Array (raw) or typed-array view from BufferAttribute
 * @param {number} ringBase
 * @param {Float32Array} section
 * @param {number} nCs
 * @param {number} Ux @param {number} Uy @param {number} Uz
 * @param {number} Vx @param {number} Vy @param {number} Vz
 * @param {number} sx @param {number} sy @param {number} sz
 * @param {number} vOff
 * @param {number} uScale
 */
function writeRingVerts(positions, ringBase, section, nCs,
                        Ux, Uy, Uz, Vx, Vy, Vz, sx, sy, sz, vOff, uScale) {
    for (let j = 0; j < nCs; j++) {
        const u = section[j * 2] * uScale;
        const v = section[j * 2 + 1] + vOff;
        positions[ringBase + j * 3]     = sx + u * Ux + v * Vx;
        positions[ringBase + j * 3 + 1] = sy + u * Uy + v * Vy;
        positions[ringBase + j * 3 + 2] = sz + u * Uz + v * Vz;
    }
}

// Write one revolution-cap ring at angle (cosT, sinT). Each vertex j sweeps
// along the tangent T by |cu|·sinT, so the ring collapses to a line at θ=90°.
/**
 * @param {any} positions - Float32Array (raw) or typed-array view from BufferAttribute
 * @param {number} ringBase
 * @param {Float32Array} section
 * @param {number} nCs
 * @param {number} Ux @param {number} Uy @param {number} Uz
 * @param {number} Vx @param {number} Vy @param {number} Vz
 * @param {number} Tx @param {number} Ty @param {number} Tz
 * @param {number} sx @param {number} sy @param {number} sz
 * @param {number} cosT @param {number} sinT
 * @param {number} vOff
 * @param {number} uScale
 */
function writeCapRingVerts(positions, ringBase, section, nCs,
                           Ux, Uy, Uz, Vx, Vy, Vz, Tx, Ty, Tz,
                           sx, sy, sz, cosT, sinT, vOff, uScale) {
    for (let j = 0; j < nCs; j++) {
        const cu = section[j * 2] * uScale;
        const cv = section[j * 2 + 1] + vOff;
        const tOff = Math.abs(cu) * sinT;
        positions[ringBase + j * 3]     = sx + cu * cosT * Ux + cv * Vx + tOff * Tx;
        positions[ringBase + j * 3 + 1] = sy + cu * cosT * Uy + cv * Vy + tOff * Ty;
        positions[ringBase + j * 3 + 2] = sz + cu * cosT * Uz + cv * Vz + tOff * Tz;
    }
}

// Sliding-window self-intersection collapse for the nCs "strand" polylines
// running along the tube. Each strand connects vertex j across all rings; at
// inside corners with κ·W/2 > 1 the offset strand folds back on itself, and
// non-adjacent segments (i, i+1) and (k, k+1) end up arbitrarily close. The
// detector scans each strand with a sliding window of TUBE_STRAND_COLLAPSE_WIN
// rings: for every node p_i, it tests the perpendicular foot of p_i onto each
// segment (p_k, p_{k+1}) within the window, and flags the run [min(i,k) ..
// max(i,k+1)] when the foot lands inside the segment and the perpendicular
// distance falls below `tol`. Overlapping detection ranges merge; each merged
// range collapses to its centroid.
//
// Tolerance is computed internally as 5% of the tube's largest cross-section
// extent — invariant of placement / overall scale.
//
// Discrete topology fix: a borderline κ·W/2 ≈ 1 fold either fully collapses
// or doesn't, no graceful ramp. The visible result at a collapsed fold is a
// hard crease — coincident vertices produce zero-area quads that three.js
// skips in raster.
//
// Endpoints are protected (collapse range is clamped to [1, nSpine-2]) so the
// caps' anchor rings stay on the spine frame.
//
// Note: collapsed vertices are coincident, so neighbouring quads degenerate
// to zero area — three.js skips them in raster, so the visible result is a
// hard crease at the fold. Other strands at the same ring may not collapse,
// so the ring becomes non-planar through the fold; this is intentional and
// only happens where the bead was already geometrically degenerate.
const TUBE_STRAND_COLLAPSE_WIN = 30;
/**
 * @param {Float32Array} positions
 * @param {Float32Array} widths
 * @param {Float32Array} heights
 * @param {number} nSpine
 * @param {number} nCs
 */
function collapseTubeStrandFolds(positions, widths, heights, nSpine, nCs) {
    if (nSpine < 4) return;
    let maxDim = 0;
    for (let i = 0; i < nSpine; i++) {
        if (widths[i] > maxDim) maxDim = widths[i];
        if (heights[i] > maxDim) maxDim = heights[i];
    }
    if (maxDim <= 0) return;
    const tol = 0.05 * maxDim;
    const tolSq = tol * tol;
    const ringStride = nCs * 3;
    const window = TUBE_STRAND_COLLAPSE_WIN;
    /** @type {number[]} */
    const rangeStart = [];
    /** @type {number[]} */
    const rangeEnd = [];
    for (let j = 0; j < nCs; j++) {
        rangeStart.length = 0;
        rangeEnd.length = 0;
        for (let i = 0; i < nSpine; i++) {
            const ip = i * ringStride + j * 3;
            const px = positions[ip], py = positions[ip + 1], pz = positions[ip + 2];
            const kHi = Math.min(nSpine - 2, i + window);
            // Only scan k > i; (k, i) was covered when i was at k's position.
            // Skip the two adjacent segments touching p_i (k = i and k = i-1).
            // Scan k backwards from kHi and break on the first hit: any later
            // (smaller-k) hit from this same i would produce a sub-range that
            // the merge step subsumes. Cuts work substantially in fold regions
            // where many k's fire from the same i; equivalent to V0 elsewhere.
            for (let k = kHi; k >= i + 2; k--) {
                const ka = k * ringStride + j * 3;
                const kb = (k + 1) * ringStride + j * 3;
                const ax = positions[ka],     ay = positions[ka + 1], az = positions[ka + 2];
                const ex = positions[kb] - ax, ey = positions[kb + 1] - ay, ez = positions[kb + 2] - az;
                const eL2 = ex * ex + ey * ey + ez * ez;
                if (eL2 < 1e-24) continue;
                const t = ((px - ax) * ex + (py - ay) * ey + (pz - az) * ez) / eL2;
                if (t < 0 || t > 1) continue;
                const dx = px - (ax + t * ex);
                const dy = py - (ay + t * ey);
                const dz = pz - (az + t * ez);
                if (dx * dx + dy * dy + dz * dz < tolSq) {
                    rangeStart.push(i);
                    rangeEnd.push(k + 1);
                    break;
                }
            }
        }
        if (rangeStart.length === 0) continue;
        // Detections are produced in non-decreasing order of `i`, so a single
        // forward sweep merges overlapping ranges in place.
        const mStart = [rangeStart[0]];
        const mEnd = [rangeEnd[0]];
        for (let m = 1; m < rangeStart.length; m++) {
            const s = rangeStart[m], e = rangeEnd[m];
            const last = mEnd.length - 1;
            if (s <= mEnd[last]) {
                if (e > mEnd[last]) mEnd[last] = e;
            } else {
                mStart.push(s);
                mEnd.push(e);
            }
        }
        for (let m = 0; m < mStart.length; m++) {
            const s = Math.max(1, mStart[m]);
            const e = Math.min(nSpine - 2, mEnd[m]);
            if (e < s) continue;
            let cx = 0, cy = 0, cz = 0;
            const cnt = e - s + 1;
            for (let i = s; i <= e; i++) {
                const ip = i * ringStride + j * 3;
                cx += positions[ip]; cy += positions[ip + 1]; cz += positions[ip + 2];
            }
            cx /= cnt; cy /= cnt; cz /= cnt;
            for (let i = s; i <= e; i++) {
                const ip = i * ringStride + j * 3;
                positions[ip] = cx; positions[ip + 1] = cy; positions[ip + 2] = cz;
            }
        }
    }
}

// Fill `count` RGB triplets starting at float offset `floatBase`.
/**
 * @param {any} colors - Float32Array (raw) or typed-array view from BufferAttribute
 * @param {number} floatBase
 * @param {number} count
 * @param {number} r @param {number} g @param {number} b
 */
function fillRGBBlock(colors, floatBase, count, r, g, b) {
    for (let v = 0; v < count; v++) {
        const dst = floatBase + v * 3;
        colors[dst] = r; colors[dst + 1] = g; colors[dst + 2] = b;
    }
}

// ========== LOD: Augmented-Space RDP ==========
//
// RDP runs in a Web Worker to avoid blocking the render loop. Perpendicular
// distance is computed in an augmented space (xyz + weighted width + weighted
// height + weighted color channels) so attribute variation on a geometrically
// straight segment is preserved. All weights are camera-independent scalars;
// the camera-distance-scaled epsilon applies uniformly because attribute
// wiggles are sub-pixel at the same rate as geometric wiggles when the tube
// is far from the camera.
//
// The worker code is defined as a string and loaded via Blob URL. A
// synchronous fallback (distanceWeightedRDP) is kept for initial LOD at tube
// creation time, before the worker is ready.

const _LOD_WORKER_CODE = `
const LOD_EPSILON_DIVISOR = ${LOD_EPSILON_DIVISOR};
const LOD_MAX_SKIP = ${LOD_MAX_SKIP};
const LOD_WIDTH_WEIGHT = ${LOD_WIDTH_WEIGHT};
const LOD_HEIGHT_WEIGHT = ${LOD_HEIGHT_WEIGHT};
const LOD_COLOR_WEIGHT_FRAC = ${LOD_COLOR_WEIGHT_FRAC};
const LOD_CHUNK_SIZE = 5000;
const LOD_CHUNK_REUSE_RATIO = 1.5;
const N_CS = ${N_CROSS_SECTION};
const N_CAP_RINGS = 8;
const TUBE_MITER_LIMIT = ${TUBE_MITER_LIMIT};

// ---- Augmented-space RDP helpers ----
//
// Projection-based perpendicular distance (works in any dimension; equivalent
// to the 3D cross-product form when only xyz terms are present):
//   |perp|^2 = |AP|^2 - (AP . AB)^2 / |AB|^2
// Each extra dimension adds one term to |AP|^2, (AP.AB), and |AB|^2.

function _augPerpDistSq(spine, widths, heights, ringColors, iP, iA, iB, wColor) {
    const ax = spine[iA*3], ay = spine[iA*3+1], az = spine[iA*3+2];
    const abx = spine[iB*3] - ax, aby = spine[iB*3+1] - ay, abz = spine[iB*3+2] - az;
    const apx = spine[iP*3] - ax, apy = spine[iP*3+1] - ay, apz = spine[iP*3+2] - az;
    let apSq = apx*apx + apy*apy + apz*apz;
    let abSq = abx*abx + aby*aby + abz*abz;
    let apDot = apx*abx + apy*aby + apz*abz;
    if (widths) {
        const a = widths[iA];
        const apW = (widths[iP] - a) * LOD_WIDTH_WEIGHT;
        const abW = (widths[iB] - a) * LOD_WIDTH_WEIGHT;
        apSq += apW*apW; abSq += abW*abW; apDot += apW*abW;
    }
    if (heights) {
        const a = heights[iA];
        const apH = (heights[iP] - a) * LOD_HEIGHT_WEIGHT;
        const abH = (heights[iB] - a) * LOD_HEIGHT_WEIGHT;
        apSq += apH*apH; abSq += abH*abH; apDot += apH*abH;
    }
    if (ringColors && wColor > 0) {
        for (let c = 0; c < 3; c++) {
            const aC = ringColors[iA*3 + c];
            const apC = (ringColors[iP*3 + c] - aC) * wColor;
            const abC = (ringColors[iB*3 + c] - aC) * wColor;
            apSq += apC*apC; abSq += abC*abC; apDot += apC*abC;
        }
    }
    if (abSq < 1e-24) return apSq;
    // Clamp: fp cancellation can drive this slightly negative on near-colinear points.
    return Math.max(0, apSq - (apDot * apDot) / abSq);
}

function rdpChunk(spine, widths, heights, ringColors, wColor, epsSq, chunkStart, chunkEnd) {
    const keep = [chunkStart];
    if (chunkEnd > chunkStart) keep.push(chunkEnd);
    const stack = [[chunkStart, chunkEnd]];
    while (stack.length > 0) {
        const [start, end] = stack.pop();
        if (end - start < 2) continue;
        let maxPerpSq = 0, maxIdx = start;
        for (let i = start + 1; i < end; i++) {
            const dSq = _augPerpDistSq(spine, widths, heights, ringColors, i, start, end, wColor);
            if (dSq > maxPerpSq) { maxPerpSq = dSq; maxIdx = i; }
        }
        if (maxPerpSq > epsSq[maxIdx]) {
            keep.push(maxIdx);
            stack.push([start, maxIdx]);
            stack.push([maxIdx, end]);
        } else if (end - start > LOD_MAX_SKIP) {
            const mid = (start + end) >> 1;
            keep.push(mid);
            stack.push([start, mid]);
            stack.push([mid, end]);
        }
    }
    keep.sort((a, b) => a - b);
    return keep;
}
// ---- Cross-section (chamfered hex) ----

function sampleChamferedRect(out, width, height) {
    if (width === 0 || height === 0) {
        for (let i = 0; i < N_CS * 2; i++) out[i] = 0;
        return;
    }
    const hw = width * 0.5, hh = height * 0.5;
    const c = Math.min(hw, hh);
    if (width >= height) {
        out[0]=+hw;       out[1]=0;       out[2]=+(hw-c); out[3]=+hh;
        out[4]=-(hw-c);   out[5]=+hh;     out[6]=-hw;     out[7]=0;
        out[8]=-(hw-c);   out[9]=-hh;     out[10]=+(hw-c);out[11]=-hh;
    } else {
        out[0]=+hw;       out[1]=-(hh-c); out[2]=+hw;     out[3]=+(hh-c);
        out[4]=0;         out[5]=+hh;     out[6]=-hw;     out[7]=+(hh-c);
        out[8]=-hw;       out[9]=-(hh-c); out[10]=0;      out[11]=-hh;
    }
}

function computeSectionNormals(section, nCs, out) {
    for (let j = 0; j < nCs; j++) {
        const jPrev = (j - 1 + nCs) % nCs;
        const jNext = (j + 1) % nCs;
        const ptx = section[j*2]     - section[jPrev*2];
        const pty = section[j*2+1]   - section[jPrev*2+1];
        const ntx = section[jNext*2] - section[j*2];
        const nty = section[jNext*2+1] - section[j*2+1];
        let nu = pty + nty;
        let nv = -(ptx + ntx);
        const len = Math.hypot(nu, nv);
        if (len > 1e-12) { nu /= len; nv /= len; }
        out[j*2] = nu; out[j*2+1] = nv;
    }
}

// Per-spine-point miter frames + u-axis scale. Injected from the main-thread
// definition via Function.prototype.toString() to keep a single source of
// truth — the only dependency is TUBE_MITER_LIMIT, which is defined at this
// scope's header.
${computeMiterFrames.toString()}

function writeAnalyticCapNormalsW(normalArr, capBaseVert, nCapRings, capAngles,
                                  width, height, localFrames, spineIdx, tSign,
                                  sectionScratch, sectionNormalsScratch,
                                  Tx, Ty, Tz) {
    const Ux = localFrames[spineIdx*6],   Uy = localFrames[spineIdx*6+1], Uz = localFrames[spineIdx*6+2];
    const Vx = localFrames[spineIdx*6+3], Vy = localFrames[spineIdx*6+4], Vz = localFrames[spineIdx*6+5];
    sampleChamferedRect(sectionScratch, width, height);
    computeSectionNormals(sectionScratch, N_CS, sectionNormalsScratch);
    for (let k = 0; k < nCapRings; k++) {
        const theta = capAngles[k];
        const c = Math.cos(theta);
        const s = Math.sin(theta) * tSign;
        for (let j = 0; j < N_CS; j++) {
            const nu = sectionNormalsScratch[j*2], nv = sectionNormalsScratch[j*2+1];
            let nx = c*(nu*Ux + nv*Vx) + s*Tx;
            let ny = c*(nu*Uy + nv*Vy) + s*Ty;
            let nz = c*(nu*Uz + nv*Vz) + s*Tz;
            const len = Math.hypot(nx, ny, nz);
            if (len > 1e-12) { nx /= len; ny /= len; nz /= len; }
            const dst = (capBaseVert + k*N_CS + j) * 3;
            normalArr[dst] = nx; normalArr[dst+1] = ny; normalArr[dst+2] = nz;
        }
    }
}

// ---- Geometry build (plain math, no Three.js) ----

function buildGeometry(spine, widths, heights, upVec, ringColors, heightOffset) {
    const nSpine = spine.length / 3;
    const capAngles = new Float32Array(N_CAP_RINGS);
    for (let k = 0; k < N_CAP_RINGS; k++) capAngles[k] = ((k + 1) / N_CAP_RINGS) * (Math.PI * 0.5);

    const startCapBase = nSpine * N_CS;
    const endCapBase = startCapBase + N_CAP_RINGS * N_CS;
    const totalVerts = endCapBase + N_CAP_RINGS * N_CS;
    const capIndicesPerCap = N_CAP_RINGS * N_CS * 6;

    const positions = new Float32Array(totalVerts * 3);
    const colors = ringColors ? new Float32Array(totalVerts * 3) : null;
    const section = new Float32Array(N_CS * 2);
    const localFrames = new Float32Array(nSpine * 6);
    const miterScales = new Float32Array(nSpine);
    const tangents = new Float32Array(nSpine * 3);

    // Up vector
    let ux0 = upVec ? upVec[0] : 0, uy0 = upVec ? upVec[1] : 0, uz0 = upVec ? upVec[2] : 1;
    const uLen = Math.hypot(ux0, uy0, uz0);
    if (uLen > 1e-12) { ux0 /= uLen; uy0 /= uLen; uz0 /= uLen; }
    // Fallback when tangent parallel to up
    let fbx, fby, fbz;
    if (Math.abs(ux0) < 0.9) { fbx = 1; fby = 0; fbz = 0; }
    else { fbx = 0; fby = 1; fbz = 0; }

    computeMiterFrames(spine, nSpine, localFrames, miterScales, tangents,
                       ux0, uy0, uz0, fbx, fby, fbz);

    // Positions
    for (let i = 0; i < nSpine; i++) {
        const Ux = localFrames[i*6],   Uy = localFrames[i*6+1], Uz = localFrames[i*6+2];
        const vx = localFrames[i*6+3], vy = localFrames[i*6+4], vz = localFrames[i*6+5];
        sampleChamferedRect(section, widths[i], heights[i]);
        const vOff = heightOffset ? heightOffset * heights[i] : 0;
        const px = spine[i*3], py = spine[i*3+1], pz = spine[i*3+2];
        const rb = i * N_CS * 3;
        const us = miterScales[i];
        for (let j = 0; j < N_CS; j++) {
            const cu = section[j*2] * us, cv = section[j*2+1] + vOff;
            positions[rb+j*3]   = px + cu*Ux + cv*vx;
            positions[rb+j*3+1] = py + cu*Uy + cv*vy;
            positions[rb+j*3+2] = pz + cu*Uz + cv*vz;
        }
        if (colors) {
            const r = ringColors[i*3], g = ringColors[i*3+1], b = ringColors[i*3+2];
            for (let j = 0; j < N_CS; j++) {
                colors[rb+j*3]=r; colors[rb+j*3+1]=g; colors[rb+j*3+2]=b;
            }
        }
    }

    // Hairpin fixup — mirror of the main-thread sweep. See
    // buildParametricTubeGeometry for the rationale. Runs on the reduced spine
    // so the LOD mesh matches the full-resolution one at reversals. Re-applies
    // the per-ring miter scale so mitered corners survive a U flip.
    for (let i = 1; i < nSpine; i++) {
        const pUx = localFrames[(i-1)*6],     pUy = localFrames[(i-1)*6+1], pUz = localFrames[(i-1)*6+2];
        const Ux  = localFrames[i*6],         Uy  = localFrames[i*6+1],     Uz  = localFrames[i*6+2];
        if (Ux*pUx + Uy*pUy + Uz*pUz >= -0.95) continue;
        const nUx = -Ux, nUy = -Uy, nUz = -Uz;
        localFrames[i*6] = nUx; localFrames[i*6+1] = nUy; localFrames[i*6+2] = nUz;
        const vx = localFrames[i*6+3], vy = localFrames[i*6+4], vz = localFrames[i*6+5];
        sampleChamferedRect(section, widths[i], heights[i]);
        const vOff = heightOffset ? heightOffset * heights[i] : 0;
        const px = spine[i*3], py = spine[i*3+1], pz = spine[i*3+2];
        const rb = i * N_CS * 3;
        const us = miterScales[i];
        for (let j = 0; j < N_CS; j++) {
            const cu = section[j*2] * us, cv = section[j*2+1] + vOff;
            positions[rb+j*3]   = px + cu*nUx + cv*vx;
            positions[rb+j*3+1] = py + cu*nUy + cv*vy;
            positions[rb+j*3+2] = pz + cu*nUz + cv*vz;
        }
    }

    // Revolution caps. T is read from the precomputed tangents array rather
    // than U x V because the hairpin sweep above may have flipped U on the
    // last ring; U x V would then point opposite the true spine tangent and
    // the end cap would extrude backwards into the tube body. Endpoints have
    // miter_scale = 1 by construction, so the cap writer doesn't need to
    // scale here.
    function buildCap(spineIdx, capBase, tSign) {
        const px = spine[spineIdx*3], py = spine[spineIdx*3+1], pz = spine[spineIdx*3+2];
        const Ux = localFrames[spineIdx*6], Uy = localFrames[spineIdx*6+1], Uz = localFrames[spineIdx*6+2];
        const vx = localFrames[spineIdx*6+3], vy = localFrames[spineIdx*6+4], vz = localFrames[spineIdx*6+5];
        const Tx = tangents[spineIdx*3], Ty = tangents[spineIdx*3+1], Tz = tangents[spineIdx*3+2];
        sampleChamferedRect(section, widths[spineIdx], heights[spineIdx]);
        const capVOff = heightOffset ? heightOffset * heights[spineIdx] : 0;
        for (let k = 0; k < N_CAP_RINGS; k++) {
            const cosT = Math.cos(capAngles[k]);
            const sinT = Math.sin(capAngles[k]) * tSign;
            const rb = (capBase + k * N_CS) * 3;
            for (let j = 0; j < N_CS; j++) {
                const cu = section[j*2], cv = section[j*2+1] + capVOff;
                const tOff = Math.abs(cu) * sinT;
                positions[rb+j*3]   = px + cu*cosT*Ux + cv*vx + tOff*Tx;
                positions[rb+j*3+1] = py + cu*cosT*Uy + cv*vy + tOff*Ty;
                positions[rb+j*3+2] = pz + cu*cosT*Uz + cv*vz + tOff*Tz;
            }
        }
    }
    buildCap(0, startCapBase, -1);
    buildCap(nSpine - 1, endCapBase, +1);

    if (colors) {
        const cpv = N_CAP_RINGS * N_CS;
        const r0=ringColors[0], g0=ringColors[1], b0=ringColors[2];
        for (let v = 0; v < cpv; v++) {
            const d=(startCapBase+v)*3; colors[d]=r0; colors[d+1]=g0; colors[d+2]=b0;
        }
        const li=(nSpine-1)*3;
        const rN=ringColors[li], gN=ringColors[li+1], bN=ringColors[li+2];
        for (let v = 0; v < cpv; v++) {
            const d=(endCapBase+v)*3; colors[d]=rN; colors[d+1]=gN; colors[d+2]=bN;
        }
    }

    // Index buffer: [start_cap | ring_pairs | end_cap]
    const ringPairs = nSpine - 1;
    const indicesPerRingPair = N_CS * 6;
    const totalIdx = capIndicesPerCap + ringPairs * indicesPerRingPair + capIndicesPerCap;
    const indices = totalVerts > 65535 ? new Uint32Array(totalIdx) : new Uint16Array(totalIdx);
    let p = 0;

    function buildCapIdx(tubeRingBase, capBase, reverse) {
        for (let k = 0; k < N_CAP_RINGS; k++) {
            const inner = k === 0 ? tubeRingBase : capBase + (k-1)*N_CS;
            const outer = capBase + k*N_CS;
            for (let j = 0; j < N_CS; j++) {
                const jN = (j+1)%N_CS;
                if (reverse) {
                    indices[p++]=inner+j; indices[p++]=outer+j; indices[p++]=inner+jN;
                    indices[p++]=outer+j; indices[p++]=outer+jN; indices[p++]=inner+jN;
                } else {
                    indices[p++]=inner+j; indices[p++]=inner+jN; indices[p++]=outer+j;
                    indices[p++]=inner+jN; indices[p++]=outer+jN; indices[p++]=outer+j;
                }
            }
        }
    }
    buildCapIdx(0, startCapBase, true);
    for (let i = 0; i < ringPairs; i++) {
        const a0 = i*N_CS, b0 = (i+1)*N_CS;
        for (let j = 0; j < N_CS; j++) {
            const jN = (j+1)%N_CS;
            indices[p++]=a0+j; indices[p++]=a0+jN; indices[p++]=b0+jN;
            indices[p++]=a0+j; indices[p++]=b0+jN; indices[p++]=b0+j;
        }
    }
    const endCapOff = p;
    buildCapIdx((nSpine-1)*N_CS, endCapBase, false);
    const endCapPattern = indices.slice(endCapOff, endCapOff + capIndicesPerCap);

    // Analytic normals throughout — LOD-invariant, no face-area weighting.
    const normals = new Float32Array(totalVerts * 3);
    const sectionNormals = new Float32Array(N_CS * 2);
    for (let i = 0; i < nSpine; i++) {
        sampleChamferedRect(section, widths[i], heights[i]);
        computeSectionNormals(section, N_CS, sectionNormals);
        const Ux = localFrames[i*6],   Uy = localFrames[i*6+1], Uz = localFrames[i*6+2];
        const Vx = localFrames[i*6+3], Vy = localFrames[i*6+4], Vz = localFrames[i*6+5];
        const rb = i * N_CS * 3;
        for (let j = 0; j < N_CS; j++) {
            const nu = sectionNormals[j*2], nv = sectionNormals[j*2+1];
            normals[rb + j*3]     = nu*Ux + nv*Vx;
            normals[rb + j*3 + 1] = nu*Uy + nv*Vy;
            normals[rb + j*3 + 2] = nu*Uz + nv*Vz;
        }
    }
    writeAnalyticCapNormalsW(normals, startCapBase, N_CAP_RINGS, capAngles,
        widths[0], heights[0], localFrames, 0, -1, section, sectionNormals,
        tangents[0], tangents[1], tangents[2]);
    const lastI = nSpine - 1;
    writeAnalyticCapNormalsW(normals, endCapBase, N_CAP_RINGS, capAngles,
        widths[lastI], heights[lastI], localFrames, lastI, +1, section, sectionNormals,
        tangents[lastI*3], tangents[lastI*3+1], tangents[lastI*3+2]);

    return { positions, normals, colors, indices, localFrames, miterScales, tangents, capAngles, endCapPattern,
             ringPairs, indicesPerRingPair, capIndicesPerCap, endCapBase,
             nSpine, totalVerts, is32bit: totalVerts > 65535 };
}

// ---- Per-tube state cache ----
const _tubes = new Map();   // tubeId -> { spine, widths, heights, ringColors, upVec, nPoints, epsilonDivisor }
const _rdpCache = new Map(); // tubeId -> { chunkDists, chunkIndices }

self.onmessage = function(e) {
    const msg = e.data;

    // 'init': register original arrays for a tube
    if (msg.type === 'init') {
        _tubes.set(msg.tubeId, {
            spine: msg.spine,
            widths: msg.widths,
            heights: msg.heights,
            ringColors: msg.ringColors,
            upVec: msg.upVec,
            nPoints: msg.nPoints,
            heightOffset: msg.heightOffset || 0,
            boundingRadius: msg.boundingRadius || 0,
            epsilonDivisor: msg.epsilonDivisor || LOD_EPSILON_DIVISOR,
        });
        _rdpCache.delete(msg.tubeId);
        return;
    }

    // 'updateColors': update cached ring colors. Invalidate the RDP chunk
    // cache — chunk splits depend on color deltas under augmented-space RDP.
    if (msg.type === 'updateColors') {
        const tube = _tubes.get(msg.tubeId);
        if (tube) tube.ringColors = msg.ringColors;
        _rdpCache.delete(msg.tubeId);
        return;
    }

    // 'dispose': clean up cached state for a deleted tube
    if (msg.type === 'dispose') {
        _tubes.delete(msg.tubeId);
        _rdpCache.delete(msg.tubeId);
        return;
    }

    // 'update': run LOD (RDP + geometry build)
    const { tubeId, camX, camY, camZ } = msg;
    const tube = _tubes.get(tubeId);
    if (!tube) return;
    const { spine, widths, heights, ringColors, upVec, nPoints, heightOffset, boundingRadius, epsilonDivisor } = tube;

    if (nPoints <= 2) {
        self.postMessage({ tubeId, allReused: true, nReduced: nPoints });
        return;
    }

    // Per-point epsilon². Worker path — keep in sync with distanceWeightedRDP
    // (the main-thread copy used for initial reduction).
    const epsSq = new Float32Array(nPoints);
    let minDist = Infinity, maxDist = 0;
    for (let i = 0; i < nPoints; i++) {
        const dx = spine[i*3]-camX, dy = spine[i*3+1]-camY, dz = spine[i*3+2]-camZ;
        const d = Math.sqrt(dx*dx+dy*dy+dz*dz);
        if (d < minDist) minDist = d;
        if (d > maxDist) maxDist = d;
        const ep = d / epsilonDivisor;
        epsSq[i] = ep * ep;
    }
    // Color weight: full per-channel swing (delta=1) costs this many world units.
    const wColor = LOD_COLOR_WEIGHT_FRAC * boundingRadius;

    // Chunked RDP with caching
    const nChunks = Math.ceil(Math.max(1, (nPoints - 1) / LOD_CHUNK_SIZE));
    let cached = _rdpCache.get(tubeId);
    if (!cached || cached.chunkDists.length !== nChunks) {
        cached = { chunkDists: new Float64Array(nChunks), chunkIndices: new Array(nChunks) };
        _rdpCache.set(tubeId, cached);
    }

    let totalKept = 0, chunksReused = 0;
    for (let ci = 0; ci < nChunks; ci++) {
        const cs = ci * LOD_CHUNK_SIZE;
        const ce = Math.min(cs + LOD_CHUNK_SIZE, nPoints - 1);
        const mid = Math.floor((cs + ce) / 2);
        const dx = spine[mid*3]-camX, dy = spine[mid*3+1]-camY, dz = spine[mid*3+2]-camZ;
        const dist = Math.sqrt(dx*dx+dy*dy+dz*dz);
        const prev = cached.chunkDists[ci];
        if (prev > 0 && cached.chunkIndices[ci]) {
            const ratio = dist / prev;
            if (ratio > 1/LOD_CHUNK_REUSE_RATIO && ratio < LOD_CHUNK_REUSE_RATIO) {
                totalKept += cached.chunkIndices[ci].length;
                chunksReused++;
                continue;
            }
        }
        const kept = rdpChunk(spine, widths, heights, ringColors, wColor, epsSq, cs, ce);
        cached.chunkDists[ci] = dist;
        cached.chunkIndices[ci] = kept;
        totalKept += kept.length;
    }

    if (chunksReused === nChunks) {
        self.postMessage({ tubeId, allReused: true, nReduced: totalKept, chunksReused, chunksTotal: nChunks });
        return;
    }

    // Merge kept indices (deduplicate boundaries)
    const keptRaw = new Uint32Array(totalKept);
    let j = 0;
    for (let ci = 0; ci < nChunks; ci++) {
        const kept = cached.chunkIndices[ci];
        for (let k = 0; k < kept.length; k++) {
            if (j > 0 && kept[k] === keptRaw[j-1]) continue;
            keptRaw[j++] = kept[k];
        }
    }
    const nRed = j;
    const keptIndices = keptRaw.subarray(0, nRed);

    // Extract reduced arrays
    const redSpine = new Float32Array(nRed * 3);
    const redWidths = new Float32Array(nRed);
    const redHeights = new Float32Array(nRed);
    let redColors = ringColors ? new Float32Array(nRed * 3) : null;
    for (let i = 0; i < nRed; i++) {
        const oi = keptIndices[i];
        redSpine[i*3]=spine[oi*3]; redSpine[i*3+1]=spine[oi*3+1]; redSpine[i*3+2]=spine[oi*3+2];
        redWidths[i]=widths[oi]; redHeights[i]=heights[oi];
        if (redColors) { redColors[i*3]=ringColors[oi*3]; redColors[i*3+1]=ringColors[oi*3+1]; redColors[i*3+2]=ringColors[oi*3+2]; }
    }

    // Build geometry in worker
    const geo = buildGeometry(redSpine, redWidths, redHeights, upVec, redColors, heightOffset);

    // Transfer ownership of large buffers
    const transfer = [geo.positions.buffer, geo.normals.buffer, geo.indices.buffer, geo.localFrames.buffer,
                      geo.miterScales.buffer, geo.tangents.buffer,
                      geo.endCapPattern.buffer, keptRaw.buffer, redSpine.buffer, redWidths.buffer, redHeights.buffer];
    if (geo.colors) transfer.push(geo.colors.buffer);
    if (redColors) transfer.push(redColors.buffer);

    self.postMessage({
        tubeId, allReused: false,
        positions: geo.positions, normals: geo.normals, colors: geo.colors, indices: geo.indices,
        localFrames: geo.localFrames, miterScales: geo.miterScales, tangents: geo.tangents,
        capAngles: geo.capAngles, endCapPattern: geo.endCapPattern,
        ringPairs: geo.ringPairs, indicesPerRingPair: geo.indicesPerRingPair,
        capIndicesPerCap: geo.capIndicesPerCap, endCapBase: geo.endCapBase,
        nSpine: geo.nSpine, is32bit: geo.is32bit,
        keptIndices: keptRaw.subarray(0, nRed),
        reducedSpine: redSpine, reducedWidths: redWidths, reducedHeights: redHeights,
        reducedColors: redColors,
        minDist, maxDist, chunksReused, chunksTotal: nChunks,
    }, transfer);
};
`;

/** @type {Worker | null} */
let _lodWorker = null;
function _getLodWorker() {
    if (!_lodWorker) {
        const blob = new Blob([_LOD_WORKER_CODE], { type: 'application/javascript' });
        const url = URL.createObjectURL(blob);
        _lodWorker = new Worker(url);
        URL.revokeObjectURL(url);  // worker keeps running after URL is revoked
    }
    return _lodWorker;
}

// Synchronous fallback for initial LOD at tube creation (before worker is ready).
// Mirrors the worker's augmented-space RDP so first-render matches subsequent
// worker-produced LOD levels.
const LOD_CHUNK_SIZE = 5000;

/**
 * @param {Float32Array} spine
 * @param {Float32Array | null} widths
 * @param {Float32Array | null} heights
 * @param {Float32Array | null} ringColors
 * @param {number} iP @param {number} iA @param {number} iB
 * @param {number} wColor
 */
function _augPerpDistSqSync(spine, widths, heights, ringColors, iP, iA, iB, wColor) {
    const ax = spine[iA * 3], ay = spine[iA * 3 + 1], az = spine[iA * 3 + 2];
    const abx = spine[iB * 3] - ax, aby = spine[iB * 3 + 1] - ay, abz = spine[iB * 3 + 2] - az;
    const apx = spine[iP * 3] - ax, apy = spine[iP * 3 + 1] - ay, apz = spine[iP * 3 + 2] - az;
    let apSq = apx * apx + apy * apy + apz * apz;
    let abSq = abx * abx + aby * aby + abz * abz;
    let apDot = apx * abx + apy * aby + apz * abz;
    if (widths) {
        const a = widths[iA];
        const apW = (widths[iP] - a) * LOD_WIDTH_WEIGHT;
        const abW = (widths[iB] - a) * LOD_WIDTH_WEIGHT;
        apSq += apW * apW; abSq += abW * abW; apDot += apW * abW;
    }
    if (heights) {
        const a = heights[iA];
        const apH = (heights[iP] - a) * LOD_HEIGHT_WEIGHT;
        const abH = (heights[iB] - a) * LOD_HEIGHT_WEIGHT;
        apSq += apH * apH; abSq += abH * abH; apDot += apH * abH;
    }
    if (ringColors && wColor > 0) {
        for (let c = 0; c < 3; c++) {
            const aC = ringColors[iA * 3 + c];
            const apC = (ringColors[iP * 3 + c] - aC) * wColor;
            const abC = (ringColors[iB * 3 + c] - aC) * wColor;
            apSq += apC * apC; abSq += abC * abC; apDot += apC * abC;
        }
    }
    if (abSq < 1e-24) return apSq;
    // Clamp: fp cancellation can drive this slightly negative on near-colinear points.
    return Math.max(0, apSq - (apDot * apDot) / abSq);
}

/**
 * @param {Float32Array} spine
 * @param {Float32Array | null} widths
 * @param {Float32Array | null} heights
 * @param {Float32Array | null} ringColors
 * @param {number} boundingRadius
 * @param {number} nPoints
 * @param {number} camX @param {number} camY @param {number} camZ
 * @param {number} [epsilonDivisor] Optional override; defaults to LOD_EPSILON_DIVISOR.
 */
function distanceWeightedRDP(spine, widths, heights, ringColors, boundingRadius, nPoints, camX, camY, camZ, epsilonDivisor) {
    if (nPoints <= 2) {
        const indices = nPoints < 1 ? new Uint32Array(0)
            : nPoints === 1 ? Uint32Array.of(0)
            : Uint32Array.of(0, 1);
        return { indices, minDist: 0, maxDist: 0 };
    }
    const divisor = epsilonDivisor !== undefined ? epsilonDivisor : LOD_EPSILON_DIVISOR;
    // Main-thread path — keep in sync with the worker's chunked RDP above.
    const epsSq = new Float32Array(nPoints);
    let minDist = Infinity, maxDist = 0;
    for (let i = 0; i < nPoints; i++) {
        const dx = spine[i * 3] - camX;
        const dy = spine[i * 3 + 1] - camY;
        const dz = spine[i * 3 + 2] - camZ;
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (d < minDist) minDist = d;
        if (d > maxDist) maxDist = d;
        const e = d / divisor;
        epsSq[i] = e * e;
    }
    const wColor = LOD_COLOR_WEIGHT_FRAC * boundingRadius;
    const keep = new Uint8Array(nPoints);
    keep[0] = 1;
    keep[nPoints - 1] = 1;
    for (let chunkStart = 0; chunkStart < nPoints - 1; chunkStart += LOD_CHUNK_SIZE) {
        const chunkEnd = Math.min(chunkStart + LOD_CHUNK_SIZE, nPoints - 1);
        keep[chunkStart] = 1;
        keep[chunkEnd] = 1;
        const stack = [[chunkStart, chunkEnd]];
        while (stack.length > 0) {
            const [start, end] = stack.pop();
            if (end - start < 2) continue;
            let maxPerpSq = 0;
            let maxIdx = start;
            for (let i = start + 1; i < end; i++) {
                const dSq = _augPerpDistSqSync(spine, widths, heights, ringColors, i, start, end, wColor);
                if (dSq > maxPerpSq) {
                    maxPerpSq = dSq;
                    maxIdx = i;
                }
            }
            if (maxPerpSq > epsSq[maxIdx]) {
                keep[maxIdx] = 1;
                stack.push([start, maxIdx]);
                stack.push([maxIdx, end]);
            } else if (end - start > LOD_MAX_SKIP) {
                const mid = (start + end) >> 1;
                keep[mid] = 1;
                stack.push([start, mid]);
                stack.push([mid, end]);
            }
        }
    }
    let count = 0;
    for (let i = 0; i < nPoints; i++) if (keep[i]) count++;
    const indices = new Uint32Array(count);
    let j = 0;
    for (let i = 0; i < nPoints; i++) if (keep[i]) indices[j++] = i;
    return { indices, minDist, maxDist };
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
// - strandCollapse: When true, each strand polyline is scanned for
//                 self-intersections within a sliding window of 30 rings;
//                 detected fold runs collapse to their centroid (see
//                 collapseTubeStrandFolds).
//
// Returns { geometry, ringPairs, indicesPerRingPair, nCs }.
/**
 * @param {Float32Array} spine
 * @param {Float32Array} widths
 * @param {Float32Array} heights
 * @param {Float32Array | null} orientations
 * @param {number[] | null} upVector
 * @param {Float32Array | null} ringColors
 * @param {number} heightOffset
 * @param {boolean} [strandCollapse]
 */
function buildParametricTubeGeometry(
    spine, widths, heights,
    orientations, upVector, ringColors, heightOffset,
    strandCollapse,
) {
    const nCs = N_CROSS_SECTION;
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

    // Cap vertex layout (appended after tube rings):
    //   startCapRings(nCapRings * nCs) | endCapRings(nCapRings * nCs)
    // Each cap has nCapRings revolution rings (including the 90° endpoint).
    // Revolution is around the V (up) axis through the spine center, so each
    // vertex sweeps by its own cu: pos = spine + cu*cos(θ)*U + cv*V + cu*sin(θ)*T.
    // At θ=90° the ring collapses to a vertical line segment (cu→0), closing
    // the cap without a pole vertex.
    const nCapRings = 8;
    const capAngles = new Float32Array(nCapRings);
    for (let k = 0; k < nCapRings; k++) {
        capAngles[k] = ((k + 1) / nCapRings) * (Math.PI * 0.5);
    }
    const startCapBase = nSpine * nCs;
    const endCapBase = startCapBase + nCapRings * nCs;
    const totalVerts = endCapBase + nCapRings * nCs;
    // Per cap: nCapRings quad strips with spoke triangulation
    const capIndicesPerCap = nCapRings * nCs * 6;

    const positions = new Float32Array(totalVerts * 3);
    const colors = ringColors ? new Float32Array(totalVerts * 3) : null;
    const section = new Float32Array(nCs * 2);
    // Store per-spine-point local frames (U, V) for frontier-ring morphing.
    const localFrames = new Float32Array(nSpine * 6);
    // Per-spine-point miter u-scale. 1.0 on straight segments; 1/cos(half_turn)
    // at interior corners, clamped by TUBE_MITER_LIMIT. Applied only to
    // positions; normals use the orthonormal U,V so the shading stays clean.
    const miterScales = new Float32Array(nSpine);
    // Per-spine-point unit tangent (bisector of incoming/outgoing segments for
    // the constant-up path; quaternion Z = U × V for explicit orientations).
    // Consumed by revolution-cap construction in place of U × V because the
    // hairpin sweep below may flip U on the last ring.
    const tangents = new Float32Array(nSpine * 3);

    // Up vector (normalized) + fallback seed for when the tangent is parallel
    // to up at a specific point.
    let upX = upVector ? upVector[0] : 0;
    let upY = upVector ? upVector[1] : 0;
    let upZ = upVector ? upVector[2] : 1;
    const upLen = Math.hypot(upX, upY, upZ);
    if (upLen > 1e-12) { upX /= upLen; upY /= upLen; upZ /= upLen; }
    const fbX = Math.abs(upX) < 0.9 ? 1 : 0;
    const fbY = Math.abs(upX) < 0.9 ? 0 : 1;
    const fbZ = 0;

    if (orientations) {
        // Explicit per-point quaternion override: use as-is, no miter scaling.
        // T for downstream caps is the quaternion's Z axis (U × V), written
        // into `tangents[i]` so caps consume the same array regardless of path.
        const _quat = new THREE.Quaternion();
        const _U = new THREE.Vector3();
        const _V = new THREE.Vector3();
        for (let i = 0; i < nSpine; i++) {
            _quat.set(
                orientations[i * 4],
                orientations[i * 4 + 1],
                orientations[i * 4 + 2],
                orientations[i * 4 + 3],
            );
            _U.set(1, 0, 0).applyQuaternion(_quat);
            _V.set(0, 1, 0).applyQuaternion(_quat);
            localFrames[i * 6]     = _U.x; localFrames[i * 6 + 1] = _U.y; localFrames[i * 6 + 2] = _U.z;
            localFrames[i * 6 + 3] = _V.x; localFrames[i * 6 + 4] = _V.y; localFrames[i * 6 + 5] = _V.z;
            tangents[i * 3]     = _U.y * _V.z - _U.z * _V.y;
            tangents[i * 3 + 1] = _U.z * _V.x - _U.x * _V.z;
            tangents[i * 3 + 2] = _U.x * _V.y - _U.y * _V.x;
            miterScales[i] = 1;
        }
    } else {
        computeMiterFrames(spine, nSpine, localFrames, miterScales, tangents,
                           upX, upY, upZ, fbX, fbY, fbZ);
    }

    for (let i = 0; i < nSpine; i++) {
        const Ux = localFrames[i * 6],     Uy = localFrames[i * 6 + 1], Uz = localFrames[i * 6 + 2];
        const Vx = localFrames[i * 6 + 3], Vy = localFrames[i * 6 + 4], Vz = localFrames[i * 6 + 5];
        const w = widths[i];
        const h = heights[i];
        sampleChamferedRect(section, w, h);
        // Apply height anchor offset: shift cross-section along V axis
        // so that "top" anchor places the spine at the top surface.
        const vOff = heightOffset ? heightOffset * h : 0;
        const sx = spine[i * 3];
        const sy = spine[i * 3 + 1];
        const sz = spine[i * 3 + 2];
        const ringBase = i * nCs * 3;
        writeRingVerts(positions, ringBase, section, nCs,
                       Ux, Uy, Uz, Vx, Vy, Vz, sx, sy, sz, vOff, miterScales[i]);
        if (colors) {
            fillRGBBlock(colors, ringBase, nCs,
                         ringColors[i * 3], ringColors[i * 3 + 1], ringColors[i * 3 + 2]);
        }
    }

    // --- Hairpin fixup: preserve ring-vertex-index continuity at 180° reversals ---
    // At a true tangent reversal (retrace, zigzag infill hairpin), U = V × T
    // flips sign even though V is unchanged. Adjacent rings then point their
    // cross-section vertices at opposite physical points and the connecting
    // quad twists into a figure-8. Sweep sequentially after the main loop:
    // if U_i is anti-parallel to U_{i-1}, flip U_i (V stays — it's even in T)
    // and rewrite ring i. The sweep is sticky: once a reversal kicks in, every
    // ring on the return leg compares against the flipped previous U and stays
    // aligned. Threshold stays near -1 so gradual curves and non-hairpin sharp
    // corners are untouched. Skipped when the caller provides orientations —
    // explicit frames own their own continuity.
    if (!orientations) {
        for (let i = 1; i < nSpine; i++) {
            const pUx = localFrames[(i - 1) * 6];
            const pUy = localFrames[(i - 1) * 6 + 1];
            const pUz = localFrames[(i - 1) * 6 + 2];
            const Ux = localFrames[i * 6];
            const Uy = localFrames[i * 6 + 1];
            const Uz = localFrames[i * 6 + 2];
            if (Ux * pUx + Uy * pUy + Uz * pUz >= -0.95) continue;
            localFrames[i * 6]     = -Ux;
            localFrames[i * 6 + 1] = -Uy;
            localFrames[i * 6 + 2] = -Uz;
            const Vx = localFrames[i * 6 + 3];
            const Vy = localFrames[i * 6 + 4];
            const Vz = localFrames[i * 6 + 5];
            sampleChamferedRect(section, widths[i], heights[i]);
            const vOff = heightOffset ? heightOffset * heights[i] : 0;
            const sx = spine[i * 3], sy = spine[i * 3 + 1], sz = spine[i * 3 + 2];
            const ringBase = i * nCs * 3;
            writeRingVerts(positions, ringBase, section, nCs,
                           -Ux, -Uy, -Uz, Vx, Vy, Vz, sx, sy, sz, vOff, miterScales[i]);
        }
    }

    // Post-process strand collapse: runs after all ring vertices (and hairpin
    // fixups) so it sees the final offset polylines, but before caps bolt on
    // — caps read positions from the spine/frame arrays directly, not from
    // the modified rings, so the cap seam stays analytic.
    if (strandCollapse) {
        collapseTubeStrandFolds(positions, widths, heights, nSpine, nCs);
    }

    // --- Revolution cap vertices ---
    // Revolution around V (up axis) through the spine center.  Each vertex
    // at (cu, cv) sweeps: U shrinks as cu*cos(θ), T extends by |cu|*sin(θ).
    // At θ=90° the ring collapses to a vertical line (cu→0).
    // tangentSign = -1 for start cap (extends in -T), +1 for end cap.
    /**
     * @param {number} spineIdx
     * @param {number} capBaseVert
     * @param {number} tangentSign
     */
    function buildRevolutionCap(spineIdx, capBaseVert, tangentSign) {
        const sx = spine[spineIdx * 3], sy = spine[spineIdx * 3 + 1], sz = spine[spineIdx * 3 + 2];
        const w = widths[spineIdx], h = heights[spineIdx];
        const ux = localFrames[spineIdx * 6], uy = localFrames[spineIdx * 6 + 1], uz = localFrames[spineIdx * 6 + 2];
        const vx = localFrames[spineIdx * 6 + 3], vy = localFrames[spineIdx * 6 + 4], vz = localFrames[spineIdx * 6 + 5];
        // Read T from the precomputed `tangents` array (not U × V): the
        // hairpin sweep above may have flipped U on the last ring, which would
        // make U × V point opposite the real spine tangent and the end cap
        // would extrude backwards into the tube body.
        const tx = tangents[spineIdx * 3], ty = tangents[spineIdx * 3 + 1], tz = tangents[spineIdx * 3 + 2];
        sampleChamferedRect(section, w, h);
        const capVOff = heightOffset ? heightOffset * h : 0;
        for (let k = 0; k < nCapRings; k++) {
            const theta = capAngles[k];
            const cosT = Math.cos(theta);
            const sinT = Math.sin(theta) * tangentSign;
            const ringBase = (capBaseVert + k * nCs) * 3;
            writeCapRingVerts(positions, ringBase, section, nCs,
                              ux, uy, uz, vx, vy, vz, tx, ty, tz,
                              sx, sy, sz, cosT, sinT, capVOff, miterScales[spineIdx]);
        }
    }
    buildRevolutionCap(0, startCapBase, -1);
    buildRevolutionCap(nSpine - 1, endCapBase, +1);
    // Cap colors (replicate ring color to all cap vertices)
    if (colors) {
        const capVertsPerCap = nCapRings * nCs;
        fillRGBBlock(colors, startCapBase * 3, capVertsPerCap,
                     ringColors[0], ringColors[1], ringColors[2]);
        const iN = nSpine - 1;
        fillRGBBlock(colors, endCapBase * 3, capVertsPerCap,
                     ringColors[iN * 3], ringColors[iN * 3 + 1], ringColors[iN * 3 + 2]);
    }

    // --- Index buffer: [start_cap_dome | ring_pairs | end_cap_dome] ---
    const ringPairs = nSpine - 1;
    const indicesPerRingPair = nCs * 6;
    const totalIndexCount = capIndicesPerCap + ringPairs * indicesPerRingPair + capIndicesPerCap;
    const IndexCtor = totalVerts > 65535 ? Uint32Array : Uint16Array;
    const indices = new IndexCtor(totalIndexCount);
    let p = 0;

    // Helper: build revolution cap indices with radial spoke pattern.
    // Quads between consecutive ring layers, split along the spoke direction.
    /**
     * @param {number} tubeRingBase
     * @param {number} capBaseVert
     * @param {boolean} reverse
     */
    function buildCapSpokeIndices(tubeRingBase, capBaseVert, reverse) {
        for (let k = 0; k < nCapRings; k++) {
            const innerBase = k === 0 ? tubeRingBase : capBaseVert + (k - 1) * nCs;
            const outerBase = capBaseVert + k * nCs;
            for (let j = 0; j < nCs; j++) {
                const jN = (j + 1) % nCs;
                if (reverse) {
                    indices[p++] = innerBase + j;
                    indices[p++] = outerBase + j;
                    indices[p++] = innerBase + jN;
                    indices[p++] = outerBase + j;
                    indices[p++] = outerBase + jN;
                    indices[p++] = innerBase + jN;
                } else {
                    indices[p++] = innerBase + j;
                    indices[p++] = innerBase + jN;
                    indices[p++] = outerBase + j;
                    indices[p++] = innerBase + jN;
                    indices[p++] = outerBase + jN;
                    indices[p++] = outerBase + j;
                }
            }
        }
    }
    buildCapSpokeIndices(0, startCapBase, true);
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
    // End cap spoke pattern
    const endCapOffset = p;
    buildCapSpokeIndices((nSpine - 1) * nCs, endCapBase, false);
    // Extract end cap index pattern for dynamic relocation during draw_range
    const endCapPattern = indices.slice(endCapOffset, endCapOffset + capIndicesPerCap);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    if (colors) geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    // Analytic normals everywhere — no reliance on face-area accumulation.
    // Tube-side rings use the cross-section bisector normal (pure UV radial).
    // Revolution caps use cos(θ)·radial + sin(θ)·T, which equals the tube-side
    // normal at θ=0 (smooth seam) and pure axial T at θ=90°.
    const normalArr = new Float32Array(totalVerts * 3);
    const sectionNormals = new Float32Array(nCs * 2);
    for (let i = 0; i < nSpine; i++) {
        sampleChamferedRect(section, widths[i], heights[i]);
        computeSectionNormals(section, nCs, sectionNormals);
        const Ux = localFrames[i * 6],     Uy = localFrames[i * 6 + 1], Uz = localFrames[i * 6 + 2];
        const Vx = localFrames[i * 6 + 3], Vy = localFrames[i * 6 + 4], Vz = localFrames[i * 6 + 5];
        const rb = i * nCs * 3;
        for (let j = 0; j < nCs; j++) {
            const nu = sectionNormals[j * 2], nv = sectionNormals[j * 2 + 1];
            normalArr[rb + j * 3]     = nu * Ux + nv * Vx;
            normalArr[rb + j * 3 + 1] = nu * Uy + nv * Vy;
            normalArr[rb + j * 3 + 2] = nu * Uz + nv * Vz;
        }
    }
    writeAnalyticCapNormals(normalArr, startCapBase, nCs, nCapRings, capAngles,
        widths[0], heights[0], localFrames, 0, -1,
        tangents[0], tangents[1], tangents[2]);
    const lastIdx = nSpine - 1;
    writeAnalyticCapNormals(normalArr, endCapBase, nCs, nCapRings, capAngles,
        widths[lastIdx], heights[lastIdx], localFrames, lastIdx, +1,
        tangents[lastIdx * 3], tangents[lastIdx * 3 + 1], tangents[lastIdx * 3 + 2]);
    geometry.setAttribute('normal', new THREE.BufferAttribute(normalArr, 3));

    return {
        geometry, ringPairs, indicesPerRingPair, nCs,
        localFrames, miterScales, tangents, capAngles,
        capIndicesPerCap, endCapBase, endCapPattern,
    };
}

// ========== LOD: Geometry Rebuild ==========

// Binary-search cumulative arc lengths to convert a draw_range value (0-1,
// fraction of total original arc length) into fractional ring pairs in the
// reduced spine.
// Remap a draw_range value (0-1, fraction of original spine points) to
// fractional ring pairs in the reduced spine.  Preserves the original
// "fraction-of-points" semantics so animation pacing stays in sync.
/**
 * @param {any} lod
 * @param {number} value
 */
function remapDrawRangeToReducedPairs(lod, value) {
    const kept = lod.keptIndices;
    const nRed = kept.length;
    if (nRed < 2) return 0;
    // Target original spine index (fractional)
    const targetIdx = value * (lod.originalCount - 1);
    // Binary search keptIndices for targetIdx
    let lo = 0, hi = nRed - 1;
    while (lo < hi - 1) {
        const mid = (lo + hi) >> 1;
        if (kept[mid] <= targetIdx) lo = mid; else hi = mid;
    }
    // Interpolate between kept[lo] and kept[hi]
    const span = kept[hi] - kept[lo];
    const frac = span > 0 ? (targetIdx - kept[lo]) / span : 0;
    return Math.min(lo + frac, nRed - 1);
}

// Decode N packed uint32 RGB values (0x00RRGGBB) into a ring-major Float32 RGB
// attribute of length nSpine*nCs*3, replicating each ring color across its nCs
// vertices. Writes into `out` if provided (must be pre-sized) to avoid
// reallocating on every color update.
/**
 * @param {Uint32Array | Uint8Array | number[]} packedColors
 * @param {number} nSpine
 * @param {number} nCs
 * @param {Float32Array | null} [out]
 */
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

// ParametricTube — namespaces the operations that read / mutate a tube mesh
// built by `buildParametricTubeGeometry`. State continues to live on
// `mesh.userData.tube*` so stream-mode code, tests, and the LOD worker can
// keep using it; this class is the single place where tube-specific methods
// live. One instance per tube mesh, stored at `mesh.userData.parametricTube`.
class ParametricTube {
    /** @param {any} mesh */
    constructor(mesh) {
        this._mesh = mesh;
    }

    // Apply pre-built geometry from LOD worker.
    // msg: worker message containing typed arrays + metadata.
    /** @param {any} msg */
    applyWorkerGeometry(msg) {
        const mesh = this._mesh;
        const lod = mesh.userData.tubeLOD;
        const nCs = N_CROSS_SECTION;

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(msg.positions, 3));
        geometry.setAttribute('normal', new THREE.BufferAttribute(msg.normals, 3));
        if (msg.colors) geometry.setAttribute('color', new THREE.BufferAttribute(msg.colors, 3));
        geometry.setIndex(new THREE.BufferAttribute(msg.indices, 1));

        mesh.geometry.dispose();
        mesh.geometry = geometry;
        if (mesh.userData.wireframeOverlay) {
            mesh.userData.wireframeOverlay.geometry = geometry;
        }

        const ud = mesh.userData;
        ud.tubeNumSpinePoints = msg.nSpine;
        ud.tubeRingPairs = msg.ringPairs;
        ud.tubeIndicesPerRingPair = msg.indicesPerRingPair;
        ud.totalIndexCount = msg.capIndicesPerCap + msg.ringPairs * msg.indicesPerRingPair + msg.capIndicesPerCap;
        ud.tubeCapIndicesPerCap = msg.capIndicesPerCap;
        ud.tubeEndCapBase = msg.endCapBase;

        ud.tubeMorphData = {
            spine: msg.reducedSpine,
            widths: msg.reducedWidths,
            heights: msg.reducedHeights,
            localFrames: msg.localFrames,
            miterScales: msg.miterScales,
            tangents: msg.tangents,
            capAngles: msg.capAngles,
            ringColors: msg.reducedColors,
            section: new Float32Array(nCs * 2),
            savedRing: new Float32Array(nCs * 3),
            savedRingNormals: null,
            savedRingColors: null,
            savedRingIndex: null,
            morphedState: null,
            endCapPattern: msg.endCapPattern,
            savedCapIndices: new (/** @type {any} */ (msg.endCapPattern.constructor))(msg.endCapPattern.length),
            savedCapOffset: -1,
            heightOffset: ud.tubeHeightOffset || 0,
        };

        lod.keptIndices = msg.keptIndices;
        this.setDrawRange(lod.currentDrawRange);
    }

    // Restore a previously-morphed frontier ring to its original positions
    // (and normals / colors if saved). Called when the frontier advances
    // past a ring or when draw_range reaches an exact ring boundary.
    restoreFrontierRing() {
        const obj = this._mesh;
        const md = obj.userData.tubeMorphData;
        if (!md || md.savedRingIndex == null) return;
        const nCs = obj.userData.tubeNCs;
        const ringBase = md.savedRingIndex * nCs * 3;
        const posAttr = obj.geometry.getAttribute('position');
        const rangeCount = nCs * 3;
        posAttr.array.set(md.savedRing, ringBase);
        posAttr.addUpdateRange(ringBase, rangeCount);
        posAttr.needsUpdate = true;
        if (md.savedRingNormals) {
            const norAttr = obj.geometry.getAttribute('normal');
            if (norAttr) {
                norAttr.array.set(md.savedRingNormals, ringBase);
                norAttr.addUpdateRange(ringBase, rangeCount);
                norAttr.needsUpdate = true;
            }
        }
        if (md.savedRingColors) {
            const colAttr = obj.geometry.getAttribute('color');
            if (colAttr) {
                colAttr.array.set(md.savedRingColors, ringBase);
                colAttr.addUpdateRange(ringBase, rangeCount);
                colAttr.needsUpdate = true;
            }
        }
        md.savedRingIndex = null;
        md.morphedState = null;
    }

    // Morph the frontier ring to the interpolated spine position. Returns
    // the number of ring pairs that should be visible (complete pairs + the
    // morphed frontier).
    /** @param {number} fracRingPairs */
    morphFrontierRing(fracRingPairs) {
        const obj = this._mesh;
        const ud = obj.userData;
        const md = ud.tubeMorphData;
        if (!md) return Math.floor(fracRingPairs);

        const nCs = ud.tubeNCs;
        const ringPairs = ud.tubeRingPairs;
        const completePairs = Math.floor(fracRingPairs);
        const frac = fracRingPairs - completePairs;

        if (frac < 1e-6 || completePairs >= ringPairs) {
            this.restoreFrontierRing();
            return completePairs;
        }

        const iA = completePairs;      // lerp FROM
        const iB = completePairs + 1;  // ring we overwrite (lerp TO)

        // Frontier moved to a different ring — restore the old one first, then
        // snapshot the new ring's original data.
        if (md.savedRingIndex !== iB) {
            this.restoreFrontierRing();
            const posArr = obj.geometry.getAttribute('position').array;
            const ringBase = iB * nCs * 3;
            md.savedRing.set(posArr.subarray(ringBase, ringBase + nCs * 3));
            const norAttr = obj.geometry.getAttribute('normal');
            if (norAttr) {
                if (!md.savedRingNormals) md.savedRingNormals = new Float32Array(nCs * 3);
                md.savedRingNormals.set(norAttr.array.subarray(ringBase, ringBase + nCs * 3));
            }
            if (ud.tubeHasColors) {
                const colAttr = obj.geometry.getAttribute('color');
                if (colAttr) {
                    if (!md.savedRingColors) md.savedRingColors = new Float32Array(nCs * 3);
                    md.savedRingColors.set(colAttr.array.subarray(ringBase, ringBase + nCs * 3));
                }
            }
            md.savedRingIndex = iB;
        }

        const sx = md.spine[iA * 3]     * (1 - frac) + md.spine[iB * 3]     * frac;
        const sy = md.spine[iA * 3 + 1] * (1 - frac) + md.spine[iB * 3 + 1] * frac;
        const sz = md.spine[iA * 3 + 2] * (1 - frac) + md.spine[iB * 3 + 2] * frac;

        const w = md.widths[iA] * (1 - frac) + md.widths[iB] * frac;
        const h = md.heights[iA] * (1 - frac) + md.heights[iB] * frac;

        // Lerp local frame vectors (U, V) — component-wise + normalize.
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

        const miterA = md.miterScales ? md.miterScales[iA] : 1;
        const miterB = md.miterScales ? md.miterScales[iB] : 1;
        const miter = miterA * (1 - frac) + miterB * frac;

        // Lerp the spine tangent too — caps derive their axial direction from
        // this, not from U × V (see updateEndCap / buildRevolutionCap).
        let tx = md.tangents[iA * 3]     * (1 - frac) + md.tangents[iB * 3]     * frac;
        let ty = md.tangents[iA * 3 + 1] * (1 - frac) + md.tangents[iB * 3 + 1] * frac;
        let tz = md.tangents[iA * 3 + 2] * (1 - frac) + md.tangents[iB * 3 + 2] * frac;
        let tLen = Math.hypot(tx, ty, tz);
        if (tLen > 1e-12) { tx /= tLen; ty /= tLen; tz /= tLen; }

        md.morphedState = { sx, sy, sz, w, h, ux, uy, uz, vx, vy, vz, tx, ty, tz, miter };

        sampleChamferedRect(md.section, w, h);
        const vOff = md.heightOffset ? md.heightOffset * h : 0;

        const posAttr = obj.geometry.getAttribute('position');
        const pos = posAttr.array;
        const ringBase = iB * nCs * 3;
        writeRingVerts(pos, ringBase, md.section, nCs,
                       ux, uy, uz, vx, vy, vz, sx, sy, sz, vOff, miter);
        const rangeCount = nCs * 3;
        posAttr.addUpdateRange(ringBase, rangeCount);
        posAttr.needsUpdate = true;

        if (ud.tubeHasColors && md.ringColors) {
            const colAttr = obj.geometry.getAttribute('color');
            if (colAttr) {
                const cols = colAttr.array;
                const rA = md.ringColors[iA * 3], gA = md.ringColors[iA * 3 + 1], bA = md.ringColors[iA * 3 + 2];
                const rB = md.ringColors[iB * 3], gB = md.ringColors[iB * 3 + 1], bB = md.ringColors[iB * 3 + 2];
                const cr = rA * (1 - frac) + rB * frac;
                const cg = gA * (1 - frac) + gB * frac;
                const cb = bA * (1 - frac) + bB * frac;
                fillRGBBlock(cols, ringBase, nCs, cr, cg, cb);
                colAttr.addUpdateRange(ringBase, rangeCount);
                colAttr.needsUpdate = true;
            }
        }

        return completePairs + 1;
    }

    // Update the end cap revolution surface to match the last visible ring.
    // Uses morphed spine/frame/width/height stored by morphFrontierRing, or
    // reads original data for un-morphed rings.
    /** @param {number} lastVisibleRing */
    updateEndCap(lastVisibleRing) {
        const obj = this._mesh;
        const ud = obj.userData;
        const md = ud.tubeMorphData;
        if (!md) return;
        const nCs = ud.tubeNCs;
        const nCapRings = md.capAngles.length;
        const posAttr = obj.geometry.getAttribute('position');
        const pos = posAttr.array;

        let sx, sy, sz, w, h, ux, uy, uz, vx, vy, vz, tx, ty, tz, miter;
        if (md.morphedState) {
            const ms = md.morphedState;
            sx = ms.sx; sy = ms.sy; sz = ms.sz;
            w = ms.w; h = ms.h;
            ux = ms.ux; uy = ms.uy; uz = ms.uz;
            vx = ms.vx; vy = ms.vy; vz = ms.vz;
            tx = ms.tx; ty = ms.ty; tz = ms.tz;
            miter = ms.miter !== undefined ? ms.miter : 1;
        } else {
            const i = lastVisibleRing;
            sx = md.spine[i * 3]; sy = md.spine[i * 3 + 1]; sz = md.spine[i * 3 + 2];
            w = md.widths[i]; h = md.heights[i];
            ux = md.localFrames[i * 6]; uy = md.localFrames[i * 6 + 1]; uz = md.localFrames[i * 6 + 2];
            vx = md.localFrames[i * 6 + 3]; vy = md.localFrames[i * 6 + 4]; vz = md.localFrames[i * 6 + 5];
            // T from the stored spine-tangent array, not U × V — the hairpin
            // fixup may have flipped U on return-leg rings.
            tx = md.tangents[i * 3]; ty = md.tangents[i * 3 + 1]; tz = md.tangents[i * 3 + 2];
            miter = md.miterScales ? md.miterScales[i] : 1;
        }

        sampleChamferedRect(md.section, w, h);
        const vOff = md.heightOffset ? md.heightOffset * h : 0;
        const ecBase = ud.tubeEndCapBase;
        for (let k = 0; k < nCapRings; k++) {
            const theta = md.capAngles[k];
            const cosT = Math.cos(theta);
            const sinT = Math.sin(theta);
            const ringBase = (ecBase + k * nCs) * 3;
            writeCapRingVerts(pos, ringBase, md.section, nCs,
                              ux, uy, uz, vx, vy, vz, tx, ty, tz,
                              sx, sy, sz, cosT, sinT, vOff, miter);
        }
        const capRangeStart = ecBase * 3;
        const capRangeCount = nCapRings * nCs * 3;
        posAttr.addUpdateRange(capRangeStart, capRangeCount);
        posAttr.needsUpdate = true;

        if (ud.tubeHasColors && md.ringColors) {
            const colAttr = obj.geometry.getAttribute('color');
            if (colAttr) {
                const cols = colAttr.array;
                const colSrcBase = lastVisibleRing * nCs * 3;
                const cr = cols[colSrcBase], cg = cols[colSrcBase + 1], cb = cols[colSrcBase + 2];
                const capVerts = nCapRings * nCs;
                fillRGBBlock(cols, ecBase * 3, capVerts, cr, cg, cb);
                colAttr.addUpdateRange(capRangeStart, capRangeCount);
                colAttr.needsUpdate = true;
            }
        }
    }

    // Write analytic normals for the frontier ring + end cap during animation.
    // Frontier ring gets the pure cross-section bisector (pure UV radial); the
    // end cap uses cos(θ)·radial + sin(θ)·T so the seam at the frontier
    // matches and the dome tip points along +T.
    /** @param {number} visiblePairs */
    updateMorphedNormals(visiblePairs) {
        const obj = this._mesh;
        const ud = obj.userData;
        const md = ud.tubeMorphData;
        if (!md) return;

        const nCs = ud.tubeNCs;
        const nCapRings = md.capAngles.length;
        const ecBase = ud.tubeEndCapBase;
        const norAttr = obj.geometry.getAttribute('normal');
        if (!norAttr) return;
        const nor = norAttr.array;

        let w, h, ux, uy, uz, vx, vy, vz, tx, ty, tz;
        if (md.morphedState) {
            const ms = md.morphedState;
            w = ms.w; h = ms.h;
            ux = ms.ux; uy = ms.uy; uz = ms.uz;
            vx = ms.vx; vy = ms.vy; vz = ms.vz;
            tx = ms.tx; ty = ms.ty; tz = ms.tz;
        } else {
            const i = visiblePairs;
            w = md.widths[i]; h = md.heights[i];
            ux = md.localFrames[i * 6];     uy = md.localFrames[i * 6 + 1]; uz = md.localFrames[i * 6 + 2];
            vx = md.localFrames[i * 6 + 3]; vy = md.localFrames[i * 6 + 4]; vz = md.localFrames[i * 6 + 5];
            // T from the stored tangents, not U × V (hairpin fixup may have flipped U).
            tx = md.tangents[i * 3]; ty = md.tangents[i * 3 + 1]; tz = md.tangents[i * 3 + 2];
        }

        sampleChamferedRect(md.section, w, h);
        if (!md._sectionNormalsScratch) md._sectionNormalsScratch = new Float32Array(nCs * 2);
        const sn = md._sectionNormalsScratch;
        computeSectionNormals(md.section, nCs, sn);

        const fBase = visiblePairs * nCs;
        for (let j = 0; j < nCs; j++) {
            const nu = sn[j * 2], nv = sn[j * 2 + 1];
            const dst = (fBase + j) * 3;
            nor[dst]     = nu * ux + nv * vx;
            nor[dst + 1] = nu * uy + nv * vy;
            nor[dst + 2] = nu * uz + nv * vz;
        }

        for (let k = 0; k < nCapRings; k++) {
            const theta = md.capAngles[k];
            const c = Math.cos(theta);
            const s = Math.sin(theta);
            for (let j = 0; j < nCs; j++) {
                const nu = sn[j * 2], nv = sn[j * 2 + 1];
                let nx = c * (nu * ux + nv * vx) + s * tx;
                let ny = c * (nu * uy + nv * vy) + s * ty;
                let nz = c * (nu * uz + nv * vz) + s * tz;
                const len = Math.hypot(nx, ny, nz);
                if (len > 1e-12) { nx /= len; ny /= len; nz /= len; }
                const dst = (ecBase + k * nCs + j) * 3;
                nor[dst] = nx;
                nor[dst + 1] = ny;
                nor[dst + 2] = nz;
            }
        }

        norAttr.addUpdateRange(fBase * 3, nCs * 3);
        norAttr.addUpdateRange(ecBase * 3, nCapRings * nCs * 3);
        norAttr.needsUpdate = true;
    }

    // Restore end cap indices that were relocated by a previous relocateEndCap.
    restoreRelocatedEndCap() {
        const obj = this._mesh;
        const md = obj.userData.tubeMorphData;
        if (!md || md.savedCapOffset < 0) return;
        const capPer = obj.userData.tubeCapIndicesPerCap;
        const indexAttr = obj.geometry.getIndex();
        indexAttr.array.set(md.savedCapIndices, md.savedCapOffset);
        indexAttr.addUpdateRange(md.savedCapOffset, capPer);
        md.savedCapOffset = -1;
        indexAttr.needsUpdate = true;
    }

    // Move end cap fan indices to sit right after the visible ring pairs so
    // setDrawRange(0, startCap + visiblePairs + endCap) draws them correctly.
    /** @param {number} visiblePairs */
    relocateEndCap(visiblePairs) {
        const obj = this._mesh;
        const ud = obj.userData;
        const md = ud.tubeMorphData;
        const capPer = ud.tubeCapIndicesPerCap;
        const perPair = ud.tubeIndicesPerRingPair;
        const indexAttr = obj.geometry.getIndex();
        const idx = indexAttr.array;

        this.restoreRelocatedEndCap();

        const offset = capPer + visiblePairs * perPair;
        md.savedCapIndices.set(idx.subarray(offset, offset + capPer));
        md.savedCapOffset = offset;
        // Write end cap pattern, adjusting tube ring references from the
        // original last ring to the current frontier ring.
        const nCs = ud.tubeNCs;
        const origRingBase = (ud.tubeNumSpinePoints - 1) * nCs;
        const frontierRingBase = visiblePairs * nCs;
        for (let i = 0; i < capPer; i++) {
            const v = md.endCapPattern[i];
            if (v >= origRingBase && v < origRingBase + nCs) {
                idx[offset + i] = v - origRingBase + frontierRingBase;
            } else {
                idx[offset + i] = v;
            }
        }
        indexAttr.addUpdateRange(offset, capPer);
        indexAttr.needsUpdate = true;
    }

    /** @param {number} value */
    setDrawRange(value) {
        const obj = this._mesh;
        const ud = obj.userData;
        const ringPairs = ud.tubeRingPairs;
        const perPair = ud.tubeIndicesPerRingPair;
        const capPer = ud.tubeCapIndicesPerCap || 0;
        const clamped = Math.max(0, Math.min(1, value));

        if (ud.tubeLOD) ud.tubeLOD.currentDrawRange = value;

        // LOD active: preserve original point-index pacing.
        let fracRingPairs;
        if (ud.tubeLOD && ud.tubeLOD.keptIndices) {
            fracRingPairs = remapDrawRangeToReducedPairs(ud.tubeLOD, clamped);
        } else {
            fracRingPairs = clamped * ringPairs;
        }

        if (!ud.tubeMorphData) {
            const pairs = Math.floor(fracRingPairs);
            obj.geometry.setDrawRange(0, capPer + pairs * perPair + capPer);
            return;
        }

        if (clamped < 1e-6) {
            this.restoreFrontierRing();
            this.restoreRelocatedEndCap();
            obj.geometry.setDrawRange(0, 0);
            return;
        }

        const visiblePairs = this.morphFrontierRing(fracRingPairs);
        this.updateEndCap(visiblePairs);
        this.updateMorphedNormals(visiblePairs);
        this.relocateEndCap(visiblePairs);
        obj.geometry.setDrawRange(0, capPer + visiblePairs * perPair + capPer);

        // After a full color rewrite (_colorFullUploadNeeded), the morph path
        // may have added partial addUpdateRange calls on the color attribute.
        // With pending ranges, Three.js only uploads those ranges — not the
        // full buffer. Clear them so needsUpdate triggers a complete upload
        // on the next render.
        if (ud._colorFullUploadNeeded) {
            const colAttr = obj.geometry.getAttribute('color');
            if (colAttr) {
                colAttr.clearUpdateRanges();
                colAttr.needsUpdate = true;
            }
            ud._colorFullUploadNeeded = false;
        }
    }
}

// Back-compat free-function shims — these keep existing call sites simple
// while the ParametricTube class is the source of truth.
/**
 * @param {THREE.Object3D} obj
 * @param {number} value
 */
function applyParametricTubeDrawRange(obj, value) {
    obj.userData.parametricTube.setDrawRange(value);
}
/**
 * @param {THREE.Mesh} mesh
 * @param {any} msg
 */
function applyWorkerGeometry(mesh, msg) {
    mesh.userData.parametricTube.applyWorkerGeometry(msg);
}
/** @param {THREE.Object3D} obj */
function restoreFrontierRing(obj) {
    const t = obj.userData.parametricTube;
    if (t) t.restoreFrontierRing();
}

/**
 * Groups camera-related methods (persp/ortho switch, dynamic near/far,
 * frame-to-bbox). Camera objects themselves remain on the viewer.
 */
class CameraController {
    // TODO(types): see comment on makeChannelApply — keeping `viewer` loose.
    /** @param {any} viewer */
    constructor(viewer) { this.v = viewer; }

    /** @param {boolean} toOrtho */
    switch(toOrtho) {
        const v = this.v;
        if (toOrtho === v._isOrtho) return;
        const w = v.container.clientWidth;
        const h = v.container.clientHeight;
        const aspect = w / h;

        const tgt = v._tmpSwitchTarget || (v._tmpSwitchTarget = new THREE.Vector3());
        tgt.copy(v._controls.target);

        if (toOrtho) {
            const dist = v._perspCamera.position.distanceTo(tgt);
            const halfHeight = dist * Math.tan(THREE.MathUtils.degToRad(v._perspCamera.fov / 2));
            v._orthoCamera.zoom = ORTHO_FRUSTUM / halfHeight;
            v._orthoCamera.left = -ORTHO_FRUSTUM * aspect;
            v._orthoCamera.right = ORTHO_FRUSTUM * aspect;
            v._orthoCamera.top = ORTHO_FRUSTUM;
            v._orthoCamera.bottom = -ORTHO_FRUSTUM;
            v._orthoCamera.position.copy(v._perspCamera.position);
            v._orthoCamera.quaternion.copy(v._perspCamera.quaternion);
            v._orthoCamera.updateProjectionMatrix();
            v._camera = v._orthoCamera;
        } else {
            const halfHeight = ORTHO_FRUSTUM / v._orthoCamera.zoom;
            const dist = halfHeight / Math.tan(THREE.MathUtils.degToRad(v._perspCamera.fov / 2));
            const dir = v._orthoCamera.position.clone().sub(tgt).normalize();
            v._perspCamera.position.copy(tgt).addScaledVector(dir, dist);
            v._perspCamera.quaternion.copy(v._orthoCamera.quaternion);
            v._perspCamera.aspect = aspect;
            v._perspCamera.updateProjectionMatrix();
            v._camera = v._perspCamera;
        }

        v._isOrtho = toOrtho;
        // ViewerControls is camera-agnostic — just reassign and re-update.
        v._controls.camera = v._camera;
        v._controls.target.copy(tgt);
        v._controls.update();
        v._clipGizmo.camera = v._camera;
        v._setGizmoHoverSprite(null);
        v._viewHelper = new ViewHelper(v._camera, v._renderer.domElement);
        v._viewHelper.center = v._controls.target;
        v._configureViewHelper(v._viewHelper);
        v._btnOrtho.textContent = '\u2B1A O';
        v._btnOrtho.classList.toggle('active', v._isOrtho);
    }

    updateSceneBounds() {
        const v = this.v;
        const box = new THREE.Box3();
        for (const obj of v._objects.values()) {
            obj.updateWorldMatrix(true, true);
            box.expandByObject(obj);
        }
        if (box.isEmpty()) {
            v._sceneSphere.set(new THREE.Vector3(), 0);
        } else {
            box.getBoundingSphere(v._sceneSphere);
        }
        v._sceneBoundsDirty = false;
    }

    updateNearFar() {
        const v = this.v;
        if (v._isOrtho) return;
        // Recompute bounds on dirty flag or every 30 frames (~0.5s) to catch transform changes
        v._boundsFrameCounter++;
        if (v._sceneBoundsDirty || v._boundsFrameCounter >= 30) {
            this.updateSceneBounds();
            v._boundsFrameCounter = 0;
        }
        const radius = v._sceneSphere.radius;
        if (radius === 0) return;
        const dist = v._perspCamera.position.distanceTo(v._sceneSphere.center);
        const nextNear = Math.max(0.001, (dist - radius * 1.5) * 0.5);
        const nextFar = Math.max(dist + radius * 1.5, 100);
        if (
            Math.abs(v._perspCamera.near - nextNear) < 1e-6 &&
            Math.abs(v._perspCamera.far - nextFar) < 1e-6
        ) return;
        v._perspCamera.near = nextNear;
        v._perspCamera.far = nextFar;
        v._perspCamera.updateProjectionMatrix();
    }

    /**
     * Aim the active camera at `bbox` with `perspMargin` extra distance multiplier
     * (ortho always uses a 1.2 half-extent padding; perspMargin applies to perspective).
     * Shared by frameObject / frameAll.
     * @param {any} bbox
     * @param {number} perspMargin
     */
    fitToBox(bbox, perspMargin) {
        const v = this.v;
        if (bbox.isEmpty()) return;
        const center = bbox.getCenter(new THREE.Vector3());
        const size = bbox.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z, 1e-6);
        v._controls.target.copy(center);
        const dir = v._camera.position.clone().sub(center);
        if (dir.lengthSq() < 1e-10) dir.set(1, -1, 1).normalize();
        else dir.normalize();
        if (v._isOrtho) {
            const w = v.container.clientWidth;
            const h = v.container.clientHeight;
            // Preserve pre-refactor behavior: if the container isn't laid out
            // yet (e.g. frameObject called while hidden), fall back to aspect=1
            // instead of silently no-opping.
            const aspect = (w > 0 && h > 0) ? (w / h) : 1;
            const halfHeight = Math.max(size.z, size.y) / 2 * 1.2;
            const halfWidth = Math.max(size.x, size.y) / 2 * 1.2;
            const fitHalf = Math.max(halfHeight, halfWidth / aspect, 1e-6);
            v._orthoCamera.zoom = ORTHO_FRUSTUM / fitHalf;
            v._orthoCamera.updateProjectionMatrix();
            v._camera.position.copy(center).addScaledVector(dir, maxDim * 2);
        } else {
            const vFov = THREE.MathUtils.degToRad(v._perspCamera.fov / 2);
            const aspect = v._perspCamera.aspect || 1;
            const hFov = Math.atan(Math.tan(vFov) * aspect);
            const distV = Math.max(size.y, size.z) / 2 / Math.tan(vFov);
            const distH = Math.max(size.x, size.y) / 2 / Math.tan(hFov);
            const dist = Math.max(distV, distH) * perspMargin;
            v._camera.position.copy(center).addScaledVector(dir, dist);
        }
        // ViewerControls never calls camera.lookAt, so explicitly re-orient
        // to actually frame the object (not just translate along old view ray).
        v._camera.lookAt(center);
        v._controls.update();
    }
}

/**
 * Owns the M-key wireframe cycle and N-key shading-debug cycle.
 * Holds cached debug materials + per-mesh helpers.
 */
class ShadingDebugController {
    // TODO(types): see comment on makeChannelApply — keeping `viewer` loose.
    /** @param {any} viewer */
    constructor(viewer) {
        this.v = viewer;
        this.wireframeMode = 0;
        this.shadingMode = 0;
        this._normalMat = null;
        this._uvMat = null;
        this._uvTex = null;
    }

    cycleWireframe() {
        this.wireframeMode = (this.wireframeMode + 1) % 3;
        this.applyWireframe();
    }

    applyWireframe() {
        const mode = this.wireframeMode;
        const wantOverlay = mode === 2;
        const wantWire = mode === 1;
        this.v._scene.traverse(/** @param {any} obj */ (obj) => {
            if (!obj.isMesh) return;
            if (obj.userData.isWireOverlay) return;
            if (!obj.material) return;
            const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
            for (const m of mats) {
                if ('wireframe' in m) m.wireframe = wantWire;
            }
            let overlay = obj.userData.wireframeOverlay;
            if (wantOverlay) {
                if (!overlay) {
                    overlay = this._createWireOverlay(obj);
                    obj.userData.wireframeOverlay = overlay;
                    obj.add(overlay);
                } else {
                    overlay.material.clippingPlanes = this.v._activeClippingPlanes();
                    overlay.material.needsUpdate = true;
                }
                overlay.visible = true;
            } else if (overlay) {
                overlay.visible = false;
            }
        });
    }

    /** @param {any} parentMesh */
    _createWireOverlay(parentMesh) {
        const mat = new THREE.MeshBasicMaterial({
            color: 0x000000,
            wireframe: true,
            polygonOffset: true,
            polygonOffsetFactor: -1,
            polygonOffsetUnits: -1,
            side: THREE.DoubleSide,
            clippingPlanes: this.v._activeClippingPlanes(),
        });
        const overlay = new THREE.Mesh(parentMesh.geometry, mat);
        overlay.userData.isWireOverlay = true;
        overlay.raycast = () => {};
        overlay.name = (parentMesh.name || 'mesh') + '_wireOverlay';
        return overlay;
    }

    cycleShading() {
        this.shadingMode = (this.shadingMode + 1) % 4;
        this.applyShading();
    }

    _getUvCheckerTexture() {
        if (this._uvTex) return this._uvTex;
        const size = 256;
        const cells = 8;
        const cell = size / cells;
        const canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;
        const ctx = /** @type {CanvasRenderingContext2D} */ (canvas.getContext('2d'));
        for (let y = 0; y < cells; y++) {
            for (let x = 0; x < cells; x++) {
                const dark = (x + y) % 2 === 0;
                ctx.fillStyle = dark ? '#202020' : '#e0e0e0';
                ctx.fillRect(x * cell, y * cell, cell, cell);
            }
        }
        // Colored axis bands so U/V orientation is legible.
        ctx.fillStyle = '#d03030';
        ctx.fillRect(0, 0, size, 2);
        ctx.fillStyle = '#30a030';
        ctx.fillRect(0, 0, 2, size);
        const tex = new THREE.CanvasTexture(canvas);
        tex.wrapS = THREE.RepeatWrapping;
        tex.wrapT = THREE.RepeatWrapping;
        tex.magFilter = THREE.NearestFilter;
        tex.minFilter = THREE.NearestMipMapLinearFilter;
        tex.colorSpace = THREE.SRGBColorSpace;
        this._uvTex = tex;
        return tex;
    }

    /** @param {number} mode */
    _getDebugMaterial(mode) {
        const clip = this.v._activeClippingPlanes();
        if (mode === 1) {
            if (!this._normalMat) {
                this._normalMat = new THREE.MeshNormalMaterial({
                    side: THREE.DoubleSide,
                    clippingPlanes: clip,
                });
            } else {
                this._normalMat.clippingPlanes = clip;
                this._normalMat.needsUpdate = true;
            }
            return this._normalMat;
        }
        if (mode === 2) {
            if (!this._uvMat) {
                this._uvMat = new THREE.MeshBasicMaterial({
                    map: this._getUvCheckerTexture(),
                    side: THREE.DoubleSide,
                    clippingPlanes: clip,
                });
            } else {
                this._uvMat.clippingPlanes = clip;
                this._uvMat.needsUpdate = true;
            }
            return this._uvMat;
        }
        return null;
    }

    /** @param {any} obj @param {number} mode */
    _applyDebugMaterial(obj, mode) {
        const debugMat = this._getDebugMaterial(mode);
        if (!debugMat) return;
        if (obj.userData.originalMaterial === undefined) {
            obj.userData.originalMaterial = obj.material;
        }
        obj.material = debugMat;
    }

    /** @param {any} obj */
    _restoreOriginalMaterial(obj) {
        if (obj.userData.originalMaterial === undefined) return;
        obj.material = obj.userData.originalMaterial;
        delete obj.userData.originalMaterial;
        // Clipping state may have changed while the debug material was active.
        // Re-sync clippingPlanes + side + clipShadows to match what
        // _updateClipMaterials would set right now, so toggling N-key off
        // while clipping is enabled doesn't leave the restored material with
        // stale sidedness/clip flags.
        const planes = this.v._activeClippingPlanes();
        const clipEnabled = this.v._clipEnabled;
        const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
        for (const m of mats) {
            if (!m) continue;
            m.clippingPlanes = planes;
            m.clipShadows = true;
            if (clipEnabled) {
                if (m.userData.originalSide === undefined) m.userData.originalSide = m.side;
                m.side = THREE.DoubleSide;
            } else if (m.userData.originalSide !== undefined) {
                m.side = m.userData.originalSide;
                delete m.userData.originalSide;
            }
            m.needsUpdate = true;
        }
    }

    /** @param {(obj: any) => void} cb */
    _forEachUserMesh(cb) {
        // Only visit user-registered objects and their descendants — skips
        // viewer furniture (TransformControls gizmo, ViewHelper, grid, pivot
        // marker, clip anchor) whose materials often lack the fields our
        // debug swaps assume (e.g. MeshNormalMaterial has no `.color`).
        for (const root of this.v._objects.values()) {
            root.traverse((/** @type {any} */ obj) => {
                if (!obj.isMesh) return;
                if (obj.userData.isWireOverlay) return;
                if (obj.userData.isDebugHelper) return;
                cb(obj);
            });
        }
    }

    applyShading() {
        const mode = this.shadingMode;
        this._forEachUserMesh((obj) => {
            if (mode === 1 || mode === 2) {
                this._applyDebugMaterial(obj, mode);
            } else {
                this._restoreOriginalMaterial(obj);
            }

            let helper = obj.userData.vertexNormalsHelper;
            if (mode === 3) {
                if (!helper && obj.geometry && obj.geometry.attributes && obj.geometry.attributes.normal) {
                    helper = new VertexNormalsHelper(obj, this.cameraRelativeNormalSize(obj), 0x00ffff);
                    helper.userData.isDebugHelper = true;
                    helper.raycast = () => {};
                    obj.userData.vertexNormalsHelper = helper;
                    this.v._scene.add(helper);
                }
                if (helper) {
                    helper.visible = true;
                    helper.size = this.cameraRelativeNormalSize(obj);
                    helper.update();
                }
            } else if (helper) {
                helper.visible = false;
            }
        });
    }

    /** @param {any} obj */
    cameraRelativeNormalSize(obj) {
        // Target ~30 pixels on screen regardless of zoom.
        const cam = /** @type {THREE.PerspectiveCamera & THREE.OrthographicCamera} */ (this.v._camera);
        const canvasH = Math.max(1, this.v._renderer.domElement.clientHeight);
        const targetPx = 30;
        if (!obj.geometry) return 0.1;
        if (!obj.geometry.boundingSphere) obj.geometry.computeBoundingSphere();
        obj.updateWorldMatrix(true, false);
        const center = obj.geometry.boundingSphere
            ? obj.geometry.boundingSphere.center.clone().applyMatrix4(obj.matrixWorld)
            : obj.getWorldPosition(new THREE.Vector3());
        let worldPerPixel;
        if (cam.isPerspectiveCamera) {
            const dist = cam.position.distanceTo(center);
            worldPerPixel = (2 * dist * Math.tan(THREE.MathUtils.degToRad(cam.fov / 2))) / canvasH;
        } else {
            worldPerPixel = (cam.top - cam.bottom) / cam.zoom / canvasH;
        }
        return Math.max(worldPerPixel * targetPx, 1e-6);
    }
}

/**
 * Distribute a [0,1] draw_range fraction to children of a toolpath group.
 * Each child segment knows its (start, end) position in the overall toolpath.
 * Segments fully before the current position get 1.0, the active segment gets
 * a proportional fraction, and segments after get 0.0.
 * @param {THREE.Object3D} grp
 * @param {number} value
 * @param {Map<string, any>} objects
 */
function applyToolpathGroupDrawRange(grp, value, objects) {
    const segIds = grp.userData.toolpathSegmentIds;
    const segRanges = grp.userData.toolpathSegmentRanges;
    const clamped = Math.max(0, Math.min(1, value));
    for (let i = 0; i < segIds.length; i++) {
        const [sStart, sEnd] = segRanges[i];
        const span = sEnd - sStart;
        let segFrac;
        if (clamped >= sEnd) {
            segFrac = 1.0;
        } else if (clamped <= sStart || span < 1e-10) {
            segFrac = 0.0;
        } else {
            segFrac = (clamped - sStart) / span;
        }
        const child = objects.get(segIds[i]);
        if (child && child.userData.isParametricTube) {
            applyParametricTubeDrawRange(child, segFrac);
        }
    }
}

export class ThreeJSViewer {
    /**
     * @param {HTMLElement} container - The DOM element to mount into
     * @param {ThreeJSViewerOptions} [options]
     *
     * Embedding contract: before opening each WebSocket, `connect()` issues a
     * `mode: 'no-cors'` HTTP GET to the URL derived from `wsUrl` by swapping
     * the scheme (`ws:` → `http:`, `wss:` → `https:`). Path and query are
     * preserved, so the probe targets the *full* `wsUrl` with only the scheme
     * changed — an embedder using `ws://host:port/my-path` must answer plain
     * HTTP on `http://host:port/my-path`, not just on `/`. In `no-cors` mode,
     * any HTTP response counts as "server is up" (different servers/proxies
     * may return 200, 400, 404, or 426 for a plain GET to a WS URL — all fine);
     * only a TCP-level failure aborts the attempt. The standard `websockets`
     * Python library satisfies this for free as part of the WS upgrade
     * handshake. If your WS host sits behind a proxy that drops non-upgrade
     * HTTP on that path, make it answer *something*, or the browser will
     * never attempt the WebSocket.
     */
    constructor(container, options = /** @type {ThreeJSViewerOptions} */ ({})) {
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

        // Resolve lighting defaults. Precedence: URL param > options > localStorage > hard default.
        // URL param is always authoritative (developer's explicit choice) — panel edits go to
        // localStorage but don't override a URL-provided value on reload.
        this._lightingDefaults = resolveLightingDefaults(options, urlParams);

        // State
        this._objects = new Map();
        this._mixers = new Map();
        this._objGeneration = 0;
        this._mixerGeneration = 0;
        this._pendingFetches = 0;
        this._sceneGeneration = 0;
        // Per-id load token: each add/fetch for a given id bumps the counter.
        // Async completions check their captured token against this map and
        // bail if a newer add has started — prevents out-of-order fetches
        // from letting stale data replace newer geometry when the same id
        // is re-added rapidly.
        this._loadTokens = new Map();
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

        // M-key wireframe cycle + N-key shading-debug cycle. Owns cached
        // debug materials and per-mesh vertex-normals helpers.
        this._shading = new ShadingDebugController(this);

        // Camera methods (persp/ortho switch, near/far, frame-to-bbox).
        // Camera objects themselves live on the viewer.
        this._camController = new CameraController(this);

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
        /** @type {string | null} */
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
        /** @type {(sel: string) => any} */
        const q = (sel) => this.el.querySelector(sel);
        this._statusDot = q('.tjsv-status-dot');
        this._statusText = q('.tjsv-status-text');
        this._btnOrtho = q('.tjsv-btn-ortho');
        this._btnOrbitMode = q('.tjsv-btn-orbit-mode');
        this._btnClip = q('.tjsv-btn-clip');
        this._btnLighting = q('.tjsv-btn-lighting');
        this._clipPanelEl = q('.tjsv-clipping-panel');
        this._lightingPanelEl = q('.tjsv-lighting-panel');
        this._lightingExposureSlider = q('.tjsv-lighting-exposure');
        this._lightingExposureValue = q('.tjsv-lighting-exposure-value');
        this._lightingEnvSlider = q('.tjsv-lighting-env');
        this._lightingEnvValue = q('.tjsv-lighting-env-value');
        this._lightingAmbientSlider = q('.tjsv-lighting-ambient');
        this._lightingAmbientValue = q('.tjsv-lighting-ambient-value');
        this._lightingToneMappingSelect = q('.tjsv-lighting-tone-mapping');
        this._lightingResetBtn = q('.tjsv-lighting-reset');
        this._lightingCloseBtn = q('.tjsv-lighting-close');
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
        this._animLiftCss = 0;
        // Toolbar height tracks viewport width: the timeline-row wraps when
        // controls don't fit, so the lift is a function of viewport size, not
        // just show/hide state. Observe the toolbar and push its height into
        // the cache + CSS var whenever layout reports a new size. display:none
        // reports 0 — matches the "not visible" state naturally. We pass the
        // observer's own measurement (borderBoxSize, which matches offsetHeight
        // semantics — includes padding+border) into _refreshAnimLift so the
        // handler doesn't trigger a second synchronous layout read.
        this._animLiftObserver = new ResizeObserver((entries) => {
            const entry = entries[0];
            const box = entry.borderBoxSize && entry.borderBoxSize[0];
            const h = box ? box.blockSize : entry.contentRect.height;
            this._refreshAnimLift(h);
        });
        this._animLiftObserver.observe(this._animControlsEl);
        this._viewHomeBtn = q('.tjsv-view-home');
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
        this._perspCamera.lookAt(0, 0, 0);

        const aspect = w / h;
        this._orthoCamera = new THREE.OrthographicCamera(
            -ORTHO_FRUSTUM * aspect, ORTHO_FRUSTUM * aspect,
            ORTHO_FRUSTUM, -ORTHO_FRUSTUM, 0.1, 1000
        );
        this._orthoCamera.position.copy(this._perspCamera.position);
        this._orthoCamera.up.set(0, 0, 1);
        this._orthoCamera.lookAt(0, 0, 0);

        this._camera = this._perspCamera;

        // Renderer
        this._renderer = new THREE.WebGLRenderer({ antialias: true });
        this._renderer.setSize(w, h);
        this._renderer.setPixelRatio(window.devicePixelRatio);
        this._renderer.toneMapping = /** @type {THREE.ToneMapping} */ (
            toneMappingModes()[this._lightingDefaults.toneMapping]
        );
        this._renderer.toneMappingExposure = this._lightingDefaults.exposure;
        this._renderer.localClippingEnabled = true;
        this.el.appendChild(this._renderer.domElement);

        // Environment cubemap
        this._loadCubemap();

        // Controls: bespoke ViewerControls (one implementation, two modes).
        // - turntable: yaw around world-Z, pitch around camera-right (clamped near pole)
        // - free: yaw around camera-up, pitch around camera-right (no world-up lock)
        // Toggle via toolbar orbit-mode button or R; persisted to localStorage.
        // Click-to-pivot moves the orbit pivot to the raycast hit without view shift.
        this._orbitMode = localStorage.getItem('tjsv.orbitMode') || 'turntable';
        this._controls = new ViewerControls(this._camera, this._renderer.domElement);
        this._controls.setMode(this._orbitMode);
        this._controls.setRaycastObjects(() => {
            const arr = [];
            for (const o of this._objects.values()) if (o && o.visible) arr.push(o);
            return arr;
        });
        this._controls.addEventListener('change', () => { this._lodDirty = true; });

        // Pivot marker: small screen-space-sized yellow sphere + ring shown
        // briefly when a click sets a new orbit pivot. Lives in the scene only
        // (NOT in this._objects) so it can't be raycast-picked or treated as
        // a public object — the raycast getter above iterates this._objects.
        this._pivotMarker = new THREE.Group();
        const pivotSphere = new THREE.Mesh(
            new THREE.SphereGeometry(0.1, 16, 12),
            new THREE.MeshBasicMaterial({ color: 0xffd76b, depthTest: false, transparent: true, opacity: 0.95 })
        );
        pivotSphere.renderOrder = 999;
        const pivotRing = new THREE.Mesh(
            new THREE.RingGeometry(0.14, 0.18, 32),
            new THREE.MeshBasicMaterial({ color: 0xffd76b, depthTest: false, transparent: true, opacity: 0.7, side: THREE.DoubleSide })
        );
        pivotRing.renderOrder = 999;
        this._pivotMarker.add(pivotSphere);
        this._pivotMarker.add(pivotRing);
        this._pivotMarker.visible = false;
        this._pivotMarkerRing = pivotRing;
        this._pivotShownAt = 0;
        this._scene.add(this._pivotMarker);

        this._controls.addEventListener('pivot', (e) => {
            this._pivotMarker.position.copy(e.point);
            this._pivotMarker.visible = true;
            this._pivotShownAt = performance.now();
        });

        // LOD: Web Worker for async RDP computation
        this._lodThrottleMs = 500;
        this._lodLastRunTime = 0;
        this._lodDirty = false;
        this._lodWorkerBusy = false;  // true while worker is computing
        this._lodWorker = _getLodWorker();
        this._lodWorker.onmessage = /** @param {MessageEvent} e */ (e) => {
            const msg = e.data;
            this._lodWorkerBusy = false;
            const obj = this._objects.get(msg.tubeId);
            if (!obj || !obj.userData.tubeLOD) return;

            if (msg.allReused) {
                // All chunks cached — no geometry change needed
                return;
            }

            // Worker built geometry — upload to GPU
            applyWorkerGeometry(obj, msg);

            // Re-sync colors: the worker may have used stale colors if a
            // color update arrived while it was busy rebuilding geometry.
            const lod = obj.userData.tubeLOD;
            if (lod.colorVersion > 0 && lod.originalRingColors) {
                const nRed = lod.keptIndices.length;
                const nCs = obj.userData.tubeNCs;
                const rc = lod.originalRingColors;
                // Restore frontier ring before writing new colors
                const md = obj.userData.tubeMorphData;
                if (md) restoreFrontierRing(obj);
                // Extract reduced Float32 RGB from original colors at kept indices
                const redRc = new Float32Array(nRed * 3);
                for (let i = 0; i < nRed; i++) {
                    const oi = lod.keptIndices[i];
                    redRc[i * 3] = rc[oi * 3]; redRc[i * 3 + 1] = rc[oi * 3 + 1]; redRc[i * 3 + 2] = rc[oi * 3 + 2];
                }
                const colAttr = obj.geometry.getAttribute('color');
                if (colAttr) {
                    const out = colAttr.array;
                    // Expand per-ring colors to per-vertex
                    for (let i = 0; i < nRed; i++) {
                        const r = redRc[i * 3], g = redRc[i * 3 + 1], b = redRc[i * 3 + 2];
                        const base = i * nCs * 3;
                        for (let j = 0; j < nCs; j++) {
                            out[base + j * 3] = r; out[base + j * 3 + 1] = g; out[base + j * 3 + 2] = b;
                        }
                    }
                    // Fill cap colors (start cap = first ring color, end cap = last ring color)
                    const posCount = colAttr.count;
                    const capVertsPerCap = (posCount - nRed * nCs) / 2;
                    const startCapBase = nRed * nCs;
                    const endCapBase = startCapBase + capVertsPerCap;
                    const lr = (nRed - 1) * 3;
                    for (let j = 0; j < capVertsPerCap; j++) {
                        out[(startCapBase + j) * 3]     = redRc[0]; out[(startCapBase + j) * 3 + 1] = redRc[1]; out[(startCapBase + j) * 3 + 2] = redRc[2];
                        out[(endCapBase + j) * 3]       = redRc[lr]; out[(endCapBase + j) * 3 + 1] = redRc[lr + 1]; out[(endCapBase + j) * 3 + 2] = redRc[lr + 2];
                    }
                    colAttr.clearUpdateRanges();
                    colAttr.needsUpdate = true;
                }
                obj.userData._colorFullUploadNeeded = true;
                if (md) md.ringColors = redRc;
            }
        };
        // ViewHelper — axis sprites are enlarged for easier hit-testing, and a
        // custom hover raycast lights up the sprite under the cursor so users
        // can see when a click will actually land. Pointerdown inside the gizmo
        // rect is suppressed at capture to prevent click-to-pivot from firing
        // on near-misses.
        this._gizmoDim = 128;
        this._gizmoBaseScale = 1.4;
        this._gizmoHoverScale = 1.75;
        this._gizmoHoverRaycaster = new THREE.Raycaster();
        this._gizmoHoverOrthoCam = new THREE.OrthographicCamera(-2, 2, 2, -2, 0, 4);
        this._gizmoHoverOrthoCam.position.set(0, 0, 2);
        this._gizmoHoverOrthoCam.updateMatrixWorld();
        this._gizmoHovered = null;
        this._viewHelper = new ViewHelper(this._camera, this._renderer.domElement);
        this._viewHelper.center = this._controls.target;
        this._configureViewHelper(this._viewHelper);

        // Suppress click-to-pivot when the pointer is inside the gizmo rect.
        // Runs at capture so it fires before ViewerControls' pointerdown listener.
        this._renderer.domElement.addEventListener('pointerdown', (e) => {
            if (this._gizmoHitTest(e).insideRect) {
                e.stopImmediatePropagation();
            }
        }, true);

        // Hover feedback: highlight the sprite under the cursor + pointer cursor.
        this._renderer.domElement.addEventListener('pointermove', (e) => {
            if (this._viewHelper.animating) return;
            const { hit } = this._gizmoHitTest(e);
            this._setGizmoHoverSprite(hit);
            this._renderer.domElement.style.cursor = hit ? 'pointer' : '';
        });
        this._renderer.domElement.addEventListener('pointerleave', () => {
            this._setGizmoHoverSprite(null);
            this._renderer.domElement.style.cursor = '';
        });

        this._renderer.domElement.addEventListener('click', (e) => {
            if (this._viewHelper.animating) return;
            // ViewHelper.handleClick assumes the gizmo sits at bottom:0, but we
            // lift it above the animation toolbar when visible — shim clientY
            // so handleClick's raycast maps to the rendered location.
            const liftCss = this._gizmoLiftCss();
            const shim = { clientX: e.clientX, clientY: e.clientY + liftCss };
            if (this._viewHelper.handleClick(/** @type {any} */ (shim))) {
                this._controls.target.copy(this._viewHelper.center);
                this._setGizmoHoverSprite(null);
            }
        });
        // Double-click an object to frame it; double-click empty space to reset.
        this._dblclickRaycaster = new THREE.Raycaster();
        this._dblclickRaycaster.params.Line.threshold = 0.05;
        this._dblclickRaycaster.params.Points.threshold = 0.05;
        this._renderer.domElement.addEventListener('dblclick', (e) => {
            const rect = this._renderer.domElement.getBoundingClientRect();
            const ndc = new THREE.Vector2(
                ((e.clientX - rect.left) / rect.width) * 2 - 1,
                -((e.clientY - rect.top) / rect.height) * 2 + 1,
            );
            this._dblclickRaycaster.setFromCamera(ndc, this._camera);
            const candidates = [];
            for (const obj of this._objects.values()) {
                if (obj && obj.visible) candidates.push(obj);
            }
            const hits = this._dblclickRaycaster.intersectObjects(candidates, true);
            if (!hits.length) { this.resetView(); return; }
            // Walk up to the top-level object the user added (a value of _objects).
            const objSet = new Set(this._objects.values());
            let target = hits[0].object;
            while (target && !objSet.has(target)) target = target.parent;
            if (!target) { this.resetView(); return; }
            this.frameObject(target);
        });

        // Lighting — kept as an instance ref so the Lighting panel can tune intensity at runtime.
        this._ambientLight = new THREE.AmbientLight(0xffffff, this._lightingDefaults.ambientIntensity);
        this._scene.add(this._ambientLight);

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
                    scene.environmentIntensity = this._lightingDefaults.envIntensity;
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
            this._clipPanelEl.querySelectorAll('.clip-axis-buttons button').forEach(/** @param {Element} btn */ (btn) => {
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

    /** @param {string} axis */
    _setClipAxis(axis) {
        this._clipAxis = axis;
        const normal = CLIP_AXIS_NORMALS[/** @type {keyof typeof CLIP_AXIS_NORMALS} */ (axis)].clone();
        this._clipPlane.normal.copy(normal);
        this._updatePlaneConstants();
        this._syncAnchorFromPlane();
        this._syncNormalInputs();
        this._updateClipSliderRange();
        this._clipPanelEl.querySelectorAll('.clip-axis-buttons button').forEach(/** @param {any} btn */ btn => {
            btn.classList.toggle('active', btn.dataset.axis === axis);
        });
    }

    /** @param {number} d */
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

    /** @param {number} t */
    _setClipThickness(t) {
        this._clipSlabThickness = Math.max(0.01, Math.min(20, t));
        this._clipThicknessSlider.value = this._clipSlabThickness;
        this._clipThicknessValue.textContent = this._clipSlabThickness.toFixed(2);
        this._updatePlaneConstants();
        this._syncAnchorFromPlane();
    }

    /** Planes array to stamp on material.clippingPlanes — empty when clipping is off. */
    _activeClippingPlanes() {
        return this._clipEnabled ? this._clipPlanes : [];
    }

    /** @param {THREE.Object3D} obj */
    _applyClipToObject(obj) {
        const planes = this._activeClippingPlanes();
        obj.traverse(/** @param {any} child */ child => {
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

    /** @param {any} child */
    _isClipHelper(child) {
        let node = child;
        while (node) {
            if (node === this._clipAnchor || node === this._clipGizmoHelper) return true;
            node = node.parent;
        }
        return false;
    }

    /** @param {any} child */
    _isPivotMarkerDescendant(child) {
        if (!this._pivotMarker) return false;
        let node = child;
        while (node) {
            if (node === this._pivotMarker) return true;
            node = node.parent;
        }
        return false;
    }

    _updateClipSliderRange() {
        this._bbox.makeEmpty();
        this._scene.traverse(/** @param {any} child */ child => {
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
        this._scene.traverse(/** @param {any} child */ child => {
            if (!child.material) return;
            if (this._isClipHelper(child)) return;
            const mats = Array.isArray(child.material) ? child.material : [child.material];
            for (const mat of mats) {
                mat.clippingPlanes = this._activeClippingPlanes();
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
        this._clipPanelEl.querySelectorAll('.clip-axis-buttons button').forEach(/** @param {any} btn */ btn => {
            btn.classList.remove('active');
        });
    }

    // ========== Lighting Panel ==========

    _toggleLightingPanel() {
        const visible = this._lightingPanelEl.classList.toggle('visible');
        this._btnLighting.classList.toggle('active', visible);
    }

    _applyToneMappingExposure(value) {
        this._renderer.toneMappingExposure = value;
    }

    _applyEnvironmentIntensity(value) {
        this._scene.environmentIntensity = value;
    }

    _applyAmbientIntensity(value) {
        if (this._ambientLight) this._ambientLight.intensity = value;
    }

    /**
     * Set the renderer tone-mapping mode and flush every material's shader.
     *
     * Three.js bakes `renderer.toneMapping` into each material's shader at
     * compile time, so changing the renderer value alone has no visible
     * effect on already-compiled materials. Setting `material.needsUpdate =
     * true` marks the program for recompilation on the next draw, which is
     * the documented invalidation path in three.js 0.183.
     *
     * @param {string} mode - one of the keys in toneMappingModes().
     */
    _applyToneMapping(mode) {
        const modes = toneMappingModes();
        if (!(mode in modes)) return;
        this._renderer.toneMapping = /** @type {THREE.ToneMapping} */ (modes[mode]);
        // Flush shaders on every material currently in the scene so they
        // recompile against the new tone-mapping constant.
        this._scene.traverse((obj) => {
            const mat = /** @type {any} */ (obj).material;
            if (!mat) return;
            if (Array.isArray(mat)) {
                for (const m of mat) { if (m) m.needsUpdate = true; }
            } else {
                mat.needsUpdate = true;
            }
        });
    }

    _writeLightingLocalStorage(key, value) {
        try { localStorage.setItem(key, String(value)); } catch (e) { /* ignore quota */ }
    }

    _resetLightingPanel() {
        // Reset target: the pre-localStorage baseline (URL > options > hard
        // defaults). Using _lightingDefaults directly would be a no-op when
        // the page was seeded from localStorage, because localStorage already
        // participated in that resolution — Reset is meant to *escape* the
        // user's persisted tweaks, not re-apply them.
        const d = this._lightingDefaults.reset;
        try { localStorage.removeItem(LS_KEY_TONE_MAPPING_EXPOSURE); } catch (e) { /* ignore */ }
        try { localStorage.removeItem(LS_KEY_ENVIRONMENT_INTENSITY); } catch (e) { /* ignore */ }
        try { localStorage.removeItem(LS_KEY_AMBIENT_INTENSITY); } catch (e) { /* ignore */ }
        try { localStorage.removeItem(LS_KEY_TONE_MAPPING); } catch (e) { /* ignore */ }
        this._applyToneMapping(d.toneMapping);
        this._applyToneMappingExposure(d.exposure);
        this._applyEnvironmentIntensity(d.envIntensity);
        this._applyAmbientIntensity(d.ambientIntensity);
        this._lightingToneMappingSelect.value = d.toneMapping;
        this._lightingExposureSlider.value = String(d.exposure);
        this._lightingExposureValue.textContent = d.exposure.toFixed(2);
        this._lightingEnvSlider.value = String(d.envIntensity);
        this._lightingEnvValue.textContent = d.envIntensity.toFixed(2);
        this._lightingAmbientSlider.value = String(d.ambientIntensity);
        this._lightingAmbientValue.textContent = d.ambientIntensity.toFixed(2);
    }

    _initLightingPanelUI() {
        const d = this._lightingDefaults;
        // Seed sliders/readouts with the effective initial values.
        this._lightingToneMappingSelect.value = d.toneMapping;
        this._lightingExposureSlider.value = String(d.exposure);
        this._lightingExposureValue.textContent = d.exposure.toFixed(2);
        this._lightingEnvSlider.value = String(d.envIntensity);
        this._lightingEnvValue.textContent = d.envIntensity.toFixed(2);
        this._lightingAmbientSlider.value = String(d.ambientIntensity);
        this._lightingAmbientValue.textContent = d.ambientIntensity.toFixed(2);

        this._btnLighting.addEventListener('click', () => this._toggleLightingPanel());
        this._lightingCloseBtn.addEventListener('click', () => this._toggleLightingPanel());

        this._lightingToneMappingSelect.addEventListener('change', () => {
            const mode = this._lightingToneMappingSelect.value;
            if (!TONE_MAPPING_MODE_NAMES.includes(mode)) return;
            this._applyToneMapping(mode);
            this._writeLightingLocalStorage(LS_KEY_TONE_MAPPING, mode);
        });
        this._lightingExposureSlider.addEventListener('input', () => {
            const v = parseFloat(this._lightingExposureSlider.value);
            if (!Number.isFinite(v)) return;
            this._applyToneMappingExposure(v);
            this._lightingExposureValue.textContent = v.toFixed(2);
            this._writeLightingLocalStorage(LS_KEY_TONE_MAPPING_EXPOSURE, v);
        });
        this._lightingEnvSlider.addEventListener('input', () => {
            const v = parseFloat(this._lightingEnvSlider.value);
            if (!Number.isFinite(v)) return;
            this._applyEnvironmentIntensity(v);
            this._lightingEnvValue.textContent = v.toFixed(2);
            this._writeLightingLocalStorage(LS_KEY_ENVIRONMENT_INTENSITY, v);
        });
        this._lightingAmbientSlider.addEventListener('input', () => {
            const v = parseFloat(this._lightingAmbientSlider.value);
            if (!Number.isFinite(v)) return;
            this._applyAmbientIntensity(v);
            this._lightingAmbientValue.textContent = v.toFixed(2);
            this._writeLightingLocalStorage(LS_KEY_AMBIENT_INTENSITY, v);
        });
        this._lightingResetBtn.addEventListener('click', () => this._resetLightingPanel());
    }

    // ========== Camera ==========

    /** @param {string} mode */
    _setOrbitMode(mode) {
        if (mode !== 'turntable' && mode !== 'free') return;
        this._orbitMode = mode;
        try { localStorage.setItem('tjsv.orbitMode', mode); } catch (e) { /* ignore */ }
        this._controls.setMode(mode);
        this._updateOrbitModeButton();
    }

    _updateOrbitModeButton() {
        if (!this._btnOrbitMode) return;
        const isFree = this._orbitMode === 'free';
        this._btnOrbitMode.classList.toggle('active', isFree);
        this._btnOrbitMode.textContent = '\u27F3 R';
        this._btnOrbitMode.title = isFree
            ? 'Orbit: Free (trackball-style, no world-up lock). Press R or click to switch to Turntable. Hold Alt while dragging to temporarily use the other mode.'
            : 'Orbit: Turntable (Z-up locked \u2014 level horizon). Press R or click to switch to Free. Hold Alt while dragging to temporarily use the other mode.';
    }

    /** @param {boolean} toOrtho */
    _switchCamera(toOrtho) { this._camController.switch(toOrtho); }

    // ========== Object Management ==========

    /** @param {any} params */
    _createMaterial(params) {
        const color = params.color || 0x4a90d9;
        const materialType = params.materialType || 'standard';
        const opacity = params.opacity != null ? params.opacity : 1;
        const transparent = opacity < 1;
        const clip = this._activeClippingPlanes();

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

    /** @param {string} id @param {number} opacity */
    _setOpacity(id, opacity) {
        const obj = this._objects.get(id);
        if (!obj) return;
        applyOpacity(obj, opacity);
    }

    /** @param {THREE.Object3D} obj @param {string | null | undefined} parentId */
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
        if (this._shading.wireframeMode !== 0) this._shading.applyWireframe();
    }

    /** @param {any} obj @param {any} transform */
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

    // TODO(types): objData is a highly polymorphic add_object payload
    // (primitive | model | polyline | mesh | tube | group); tightening it
    // requires splitting the dispatch into per-kind helpers or a tagged-union
    // typedef. Out of scope for the drive-by type tighten.
    /** @param {string} id @param {any} objData @param {string} [parentId] */
    async _addObject(id, objData, parentId) {
        let obj;
        const token = this._claimLoadToken(id);

        if (objData.primitive) {
            const geometry = PRIMITIVES[objData.primitive](objData.params || {});
            const material = this._createMaterial(objData.params || {});
            obj = new THREE.Mesh(geometry, material);
        } else if (objData.model) {
            const format = objData.format || 'gltf';
            const loader = this._loaders[/** @type {keyof typeof this._loaders} */ (format)];
            if (!loader) {
                console.error(`Unknown format: ${format}`);
                return;
            }
            try {
                const result = await this._loadModel(loader, objData.model, format, objData.yUp === true);
                if (!this._isLoadTokenCurrent(id, token)) {
                    console.log(`Discarding stale model load for '${id}'`);
                    return;
                }
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
            this._deleteObject(id);
            this._addToParentOrScene(obj, parentId);
            this._objects.set(id, obj);
            this._objGeneration++;
            if (this._clipEnabled) this._applyClipToObject(obj);
        }
    }

    /**
     * @param {any} loader
     * @param {string} url
     * @param {string} format
     * @param {boolean} yUp
     */
    _loadModel(loader, url, format, yUp) {
        return new Promise((resolve, reject) => {
            loader.load(
                url,
                /** @param {any} result */
                (result) => {
                    /** @type {any} */
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
                        result.scene.traverse(/** @param {any} child */ (child) => {
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

    /** @param {string} id @param {number} time */
    _setClipTime(id, time) {
        const mixer = this._mixers.get(id);
        if (!mixer) return;
        mixer.setTime(time);
    }

    /** @param {string} id @param {number} value */
    _setDrawRange(id, value) {
        const obj = this._objects.get(id);
        if (!obj) return;
        if (obj.userData.isToolpathGroup) {
            applyToolpathGroupDrawRange(obj, value, this._objects);
        } else if (obj.userData.isPolyline) {
            obj.geometry.instanceCount = Math.round(value * obj.userData.maxInstanceCount);
        } else if (obj.userData.isParametricTube) {
            applyParametricTubeDrawRange(obj, value);
        } else if (obj.userData.isMesh) {
            obj.geometry.setDrawRange(0, Math.round(value * obj.userData.totalIndexCount));
        }
    }

    /** @param {string} id @param {any} transform */
    _updateTransform(id, transform) {
        const obj = this._objects.get(id);
        if (obj) this._applyTransform(obj, transform);
    }

    /**
     * Bump the load token for `id` and return the new value. Capture this
     * synchronously at the start of an async add/fetch; after each `await`,
     * check with `_isLoadTokenCurrent(id, token)` and bail on mismatch.
     * @param {string} id
     */
    _claimLoadToken(id) {
        const next = (this._loadTokens.get(id) || 0) + 1;
        this._loadTokens.set(id, next);
        return next;
    }

    /** @param {string} id @param {number} token */
    _isLoadTokenCurrent(id, token) {
        return this._loadTokens.get(id) === token;
    }

    /** @param {string} id */
    _deleteObject(id) {
        // Invalidate any in-flight async add/fetch for this id so a late
        // completion can't re-add an object that was explicitly deleted or
        // cleared. Safe to call unconditionally — the load handlers'
        // post-delete insert path has already passed its own token check.
        this._claimLoadToken(id);
        const obj = this._objects.get(id);
        if (obj) {
            /** @type {string[]} */
            const childIds = [];
            obj.traverse(/** @param {any} child */ (child) => {
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
            obj.traverse(/** @param {any} child */ (child) => {
                if (child.userData.blobUrl) URL.revokeObjectURL(child.userData.blobUrl);
                if (child.userData.tubeLOD) {
                    this._lodWorker.postMessage({ type: 'dispose', tubeId: child.userData.id });
                }
                if (child.userData.vertexNormalsHelper) {
                    const h = child.userData.vertexNormalsHelper;
                    if (h.parent) h.parent.remove(h);
                    if (h.geometry) h.geometry.dispose();
                    if (h.material) h.material.dispose();
                    delete child.userData.vertexNormalsHelper;
                }
                if (child.userData.originalMaterial !== undefined) {
                    delete child.userData.originalMaterial;
                }
                if (child.geometry) child.geometry.dispose();
                if (child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach(/** @param {any} m */ m => m.dispose());
                    } else {
                        child.material.dispose();
                    }
                }
            });
        }
    }

    /** @param {string} id @param {boolean} visible */
    _setVisibility(id, visible) {
        const obj = this._objects.get(id);
        if (obj) obj.visible = visible;
    }

    /** @param {Record<string, boolean>} visibility */
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

    /** @param {Record<string, any>} transforms */
    _batchUpdate(transforms) {
        for (const [id, transform] of Object.entries(transforms)) {
            this._updateTransform(id, transform);
            if (transform.opacity != null) this._setOpacity(id, transform.opacity);
        }
    }

    // ========== Animation ==========

    /**
     * @param {any} animData
     * @param {{ restart?: boolean, autoplay?: boolean, initial_time?: number | 'end' }} [opts]
     */
    _loadAnimation(animData, opts = {}) {
        // First load (no animation yet) or explicit restart resets the playhead
        // to t=0 (or opts.initial_time, if provided) and installs camera-tracking
        // from the new metadata; whether playback starts immediately is
        // controlled by the caller's autoplay setting (applied by client.py
        // before sending the load message). A subsequent load (animation already
        // loaded, restart not set) preserves playhead time, play state, and
        // camera-tracking — only the underlying frame data is swapped (and
        // opts.initial_time is ignored on a swap).
        const isSwap = this._animation != null && !opts.restart;
        const prevTime = this._animationTime;
        const prevPlaying = this._animationPlaying;
        const prevTrackMode = this._trackMode;
        const prevTrackTargetId = this._trackTargetId;
        const prevTrackInteractive = this._trackInteractive;

        this._animGeneration++;
        this._animation = animData;
        this._animationTime = 0;
        this._animationPlaying = false;
        this._lastAnimationUpdate = performance.now();

        if (this._animation.channels) {
            for (const [name, chRaw] of Object.entries(this._animation.channels)) {
                const ch = /** @type {BinaryChannel} */ (chRaw);
                ch.refs = ch.ids.map(/** @param {string} id */ id => {
                    const obj = this._objects.get(id);
                    if (obj && name === 'transforms') obj.matrixAutoUpdate = false;
                    return obj || null;
                });
            }
        }

        if (this._animation.frames.length >= 2) {
            const dt0 = this._animation.frames[1].time - this._animation.frames[0].time;
            let uniform = dt0 > 0;
            if (uniform) {
                for (let i = 2; i < this._animation.frames.length; i++) {
                    const dt = this._animation.frames[i].time - this._animation.frames[i - 1].time;
                    if (Math.abs(dt - dt0) > dt0 * 1e-3) { uniform = false; break; }
                }
            }
            this._animation.uniformDt = uniform ? dt0 : 0;
        }

        this._baselineVisibility.clear();
        for (const [id, obj] of this._objects) {
            this._baselineVisibility.set(id, obj.visible);
        }

        this._animControlsEl.classList.add('visible');
        // Prime the cache synchronously so the very first rendered frame
        // after the show already has the lift applied — ResizeObserver would
        // otherwise fire a tick later and produce a one-frame flash. Further
        // updates (reflow on resize, content changes) flow through the
        // observer.
        this._refreshAnimLift();
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

        if (isSwap) {
            // Preserve playback continuity across the swap.
            this._animationTime = animData.duration > 0 ? Math.min(prevTime, animData.duration) : 0;
            this._animationPlaying = prevPlaying;
            this._trackMode = prevTrackMode;
            this._trackTargetId = prevTrackTargetId;
            this._trackInteractive = prevTrackInteractive;
            // Trajectory is replaced — treat the next tracking tick as a first
            // frame (snap orbit target, keep camera offset) rather than
            // computing a delta against the old bead's last position.
            this._trackHasLastPos = false;
            this._updateTrackingUI();
            this._seekToTime(this._animationTime);
        } else {
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

            // Set play state before the optional seek — _seekToTime calls
            // _updateAnimationUI, which paints the play/pause icon from
            // _animationPlaying. Setting it after would leave the icon stale.
            this._animationPlaying = opts.autoplay !== false;

            // Caller-chosen playhead on first load / restart. Seeking through
            // _seekToTime paints the correct frame before the first tick, so
            // "paused at end" or "start at t=5" doesn't flash t=0 first.
            // Silently ignore bad values (NaN, wrong type) — they fall through
            // to the default t=0 we already applied.
            if (opts.initial_time !== undefined) {
                const duration = animData.duration || 0;
                let target = null;
                if (opts.initial_time === 'end') {
                    target = duration;
                } else if (typeof opts.initial_time === 'number' && Number.isFinite(opts.initial_time)) {
                    target = opts.initial_time;
                }
                if (target !== null) this._seekToTime(target);
            }

            // Refresh the animation UI so the play/pause icon reflects the
            // just-set _animationPlaying (the earlier _updateAnimationUI call
            // ran before we knew the autoplay choice).
            this._updateAnimationUI();
        }
        this._lastAnimationUpdate = performance.now();
        console.log(`Animation loaded: ${this._animation.frames.length} frames, ${this._animation.duration.toFixed(2)}s${isSwap ? ' (swap)' : ''}`);
    }

    _resetAnimationState() {
        this._animation = null;
        this._animationPlaying = false;
        this._baselineVisibility.clear();
        this._animControlsEl.classList.remove('visible');
        // display:none makes offsetHeight 0; the observer will sync shortly
        // anyway, but zero it synchronously so the next render/hit-test sees
        // the unlifted position immediately.
        this._refreshAnimLift();
        this._trackMode = 'off';
        this._trackTargetId = null;
        this._trackHasLastPos = false;
        this._trackInteractive = false;
        this._updateTrackingUI();
    }

    _unloadAnimation(restoreVisibility = true) {
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

    /** @param {number} frameIndex @param {number} [t] */
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
                const applyFn = this._CHANNEL_APPLY[/** @type {keyof typeof this._CHANNEL_APPLY} */ (name)];
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
                obj.traverse(/** @param {any} child */ (child) => {
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
            const match = ids.find(/** @param {string} id */ id => id.toLowerCase().includes(hint));
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

    /** @param {number} frameIndex @param {number} [frameIndexNext] @param {number} [t] */
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
                    let tx, ty, tz;
                    if (interp && ctCh.interpolation === 'linear') {
                        const bN = frameIndexNext * 3;
                        tx = ctCh.data[base]     * (1 - t) + ctCh.data[bN]     * t;
                        ty = ctCh.data[base + 1] * (1 - t) + ctCh.data[bN + 1] * t;
                        tz = ctCh.data[base + 2] * (1 - t) + ctCh.data[bN + 2] * t;
                    } else {
                        tx = ctCh.data[base]; ty = ctCh.data[base + 1]; tz = ctCh.data[base + 2];
                    }
                    this._controls.target.set(tx, ty, tz);
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

        const curTgt = this._tmpTrackTarget || (this._tmpTrackTarget = new THREE.Vector3());
        curTgt.copy(this._controls.target);

        if (this._trackMode === 'follow') {
            if (this._trackHasLastPos) {
                const delta = this._tmpTrackDelta.copy(targetPos).sub(this._trackLastPos);
                curTgt.add(delta);
                this._controls.target.copy(curTgt);
                this._camera.position.add(delta);
            } else {
                // First frame: snap orbit target, keep camera offset
                const offset = this._tmpTrackDelta.copy(this._camera.position).sub(curTgt);
                this._controls.target.copy(targetPos);
                this._camera.position.copy(targetPos).add(offset);
            }
        } else if (this._trackMode === 'lookat') {
            this._controls.target.copy(targetPos);
            // ViewerControls never calls camera.lookAt (would break click-to-pivot),
            // so re-orient the camera explicitly here to actually track the target.
            this._camera.lookAt(targetPos);
        }

        this._trackLastPos.copy(targetPos);
        this._trackHasLastPos = true;
    }

    // Returns {index, t} where index is the floor keyframe and t ∈ [0, 1)
    // is the fractional position between frames[index] and frames[index+1].
    // At (or past) the last frame, t is clamped to 0 so callers short-circuit
    // to step behavior.
    /** @param {number} time */
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
        // Binary search for the floor keyframe.
        let lo = 0, hi = lastIdx;
        while (lo < hi) {
            const mid = (lo + hi + 1) >>> 1;
            if (frames[mid].time <= time) lo = mid;
            else hi = mid - 1;
        }
        if (lo >= lastIdx) return { index: lastIdx, t: 0 };
        const dt = frames[lo + 1].time - frames[lo].time;
        return { index: lo, t: dt > 0 ? (time - frames[lo].time) / dt : 0 };
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

    /** @param {number} delta */
    _stepFrames(delta) {
        if (!this._animation) return;
        const { index: currentFrame } = this._getFrameAtTime(this._animationTime);
        const newFrame = Math.max(0, Math.min(this._animation.frames.length - 1, currentFrame + delta));
        this._animationTime = this._animation.frames[newFrame].time;
        this._applyFrame(newFrame);
        this._updateAnimationUI();
    }

    /** @param {number} time */
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

    /** @param {number} speed */
    _setSpeed(speed) {
        this._animationSpeed = speed;
        this._speedDisplayEl.textContent = `${speed}x`;
    }

    /** @param {number} delta */
    _stepSpeed(delta) {
        this._speedIndex = Math.max(0, Math.min(SPEED_STEPS.length - 1, this._speedIndex + delta));
        this._setSpeed(SPEED_STEPS[this._speedIndex]);
    }

    /** @param {MouseEvent} e */
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

        // Orbit-mode toggle (Turntable <-> Free)
        this._updateOrbitModeButton();
        this._btnOrbitMode.addEventListener('click', () => {
            this._setOrbitMode(this._orbitMode === 'turntable' ? 'free' : 'turntable');
        });

        // Clip button
        this._btnClip.addEventListener('click', () => this._toggleClipPanel());

        // Lighting panel
        this._initLightingPanelUI();

        // Clip axis buttons
        this._clipPanelEl.querySelectorAll('.clip-axis-buttons button').forEach(/** @param {Element} btn */ (btn) => {
            btn.addEventListener('click', () => this._setClipAxis(/** @type {HTMLElement} */ (btn).dataset.axis));
        });

        // Normal inputs
        [this._clipNxInput, this._clipNyInput, this._clipNzInput].forEach((input) => {
            input.addEventListener('change', () => this._applyNormalInputs());
            input.addEventListener('keydown', /** @param {Event} e */ (e) => e.stopPropagation());
        });

        // Distance slider
        this._clipDistanceSlider.addEventListener('input', () => {
            this._setClipDistance(parseFloat(this._clipDistanceSlider.value));
        });
        this._clipDistanceSlider.addEventListener('wheel', /** @param {WheelEvent} e */ (e) => {
            e.preventDefault();
            const step = e.shiftKey ? 0.1 : 0.01;
            const delta = e.deltaY > 0 ? -step : step;
            const newVal = Math.max(-20, Math.min(20, parseFloat(this._clipDistanceSlider.value) + delta));
            this._setClipDistance(newVal);
        });

        // Slab mode toggles
        this._clipModeSingle.addEventListener('click', /** @param {MouseEvent} e */ (e) => {
            this._clipSlabMode = false;
            /** @type {HTMLElement} */ (e.currentTarget).classList.add('active');
            this._clipModeSlab.classList.remove('active');
            this._clipThicknessSection.style.display = 'none';
            this._clipPlanes = [this._clipPlane];
            this._updatePlaneConstants();
            this._syncAnchorFromPlane();
            this._updateClipMaterials();
        });
        this._clipModeSlab.addEventListener('click', /** @param {MouseEvent} e */ (e) => {
            this._clipSlabMode = true;
            /** @type {HTMLElement} */ (e.currentTarget).classList.add('active');
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
        this._clipThicknessSlider.addEventListener('wheel', /** @param {WheelEvent} e */ (e) => {
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

        // Home button: sits centered in the ViewHelper area and resets the view.
        if (this._viewHomeBtn) {
            this._viewHomeBtn.addEventListener('click', () => {
                this.resetView();
                this._viewHomeBtn.blur();
            });
        }

        // Timeline scrubbing
        this._timelineContainer.addEventListener('mousedown', /** @param {MouseEvent} e */ (e) => {
            if (!this._animation) return;
            this._scrubbing = true;
            this._wasPlayingBeforeScrub = this._animationPlaying;
            this._animationPlaying = false;
            this._scrubFromEvent(e);
        });

        // Document-level listeners for scrubbing (stored for cleanup)
        this._onDocMouseMove = /** @param {MouseEvent} e */ (e) => {
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
        this._onKeyDown = /** @param {KeyboardEvent} e */ (e) => {
            if (/** @type {HTMLElement} */ (e.target).tagName === 'INPUT') return;

            // Global shortcuts
            if (e.code === 'KeyO' && !e.ctrlKey && !e.metaKey) {
                this._switchCamera(!this._isOrtho);
                return;
            }
            if (e.code === 'KeyC' && !e.ctrlKey && !e.metaKey) {
                this._toggleClipPanel();
                return;
            }
            if (e.code === 'KeyE' && !e.ctrlKey && !e.metaKey) {
                this._toggleLightingPanel();
                return;
            }
            if (e.code === 'KeyM' && !e.ctrlKey && !e.metaKey) {
                this._shading.cycleWireframe();
                return;
            }
            if (e.code === 'KeyN' && !e.ctrlKey && !e.metaKey) {
                this._shading.cycleShading();
                return;
            }
            if (e.code === 'KeyR' && !e.ctrlKey && !e.metaKey) {
                this._setOrbitMode(this._orbitMode === 'turntable' ? 'free' : 'turntable');
                return;
            }
            if (e.code === 'KeyF' && !e.ctrlKey && !e.metaKey) {
                this.resetView();
                return;
            }
            if (e.code === 'Home' && !e.ctrlKey && !e.metaKey && !e.shiftKey && !this._animation) {
                e.preventDefault();
                this.resetView();
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
        // Derive the probe URL from the WS URL by swapping only the scheme
        // (ws→http / wss→https). Path and query are preserved, so the probe
        // targets the same endpoint as the eventual WebSocket upgrade.
        const probeUrl = this._wsUrl.replace(/^ws/, 'http');

        const doConnect = async () => {
            if (this._destroyed) return;

            // Probe HTTP on the same URL (minus scheme) as the pending
            // WebSocket: a failed `new WebSocket()` always logs `WebSocket
            // connection to '...' failed` to devtools and there's no way to
            // silence it, so we only attempt the upgrade once we know
            // something is listening. `mode: 'no-cors'` makes *any* HTTP
            // response count as success (200/400/404/426/... — the exact
            // status varies by server/proxy, all of them satisfy the probe);
            // only a TCP-level failure throws and triggers retry. The Python
            // `websockets` server answers plain HTTP for free as part of the
            // upgrade handshake; embedders pointing at a different WS host
            // must ensure that host (or its proxy) returns *something* on GET
            // for the same path/query as `wsUrl`, not just for `/`.
            try {
                await fetch(probeUrl, { mode: 'no-cors', signal: AbortSignal.timeout(400) });
            } catch {
                // Server not reachable — retry later without creating a WebSocket
                this._reconnectTimeout = setTimeout(doConnect, 500);
                return;
            }

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

            this._ws.onmessage = /** @param {MessageEvent<string>} event */ async (event) => {
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
                            colorObj.traverse(/** @param {any} child */ (child) => {
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
                        /** @type {Record<string, any>} */
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
                                    .filter(/** @param {any} c */ (c) => c.userData?.id)
                                    .map(/** @param {any} c */ (c) => c.userData.id),
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
                        this._loadAnimation(data.animation, {
                            restart: !!data.restart,
                            autoplay: data.autoplay !== false,
                            initial_time: data.initial_time,
                        });
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
                                /** @type {Record<string, {ArrayType: any, bytes: number}>} */
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
                                /** @type {Record<string, any>} */
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
                                /** @type {Record<number, any>} */
                                const metaByFrame = {};
                                if (data.frames_meta) {
                                    for (const meta of data.frames_meta) {
                                        metaByFrame[meta.index] = meta;
                                    }
                                }

                                /** @type {Array<Record<string, any>>} */
                                const frames = [];
                                for (let fi = 0; fi < nFrames; fi++) {
                                    /** @type {Record<string, any>} */
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
                                }, {
                                    restart: !!data.restart,
                                    autoplay: data.autoplay !== false,
                                    initial_time: data.initial_time,
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
                        const loadToken = this._claimLoadToken(data.id);
                        (async () => {
                            try {
                                const resp = await fetch(data.blob_url);
                                const buffer = await resp.arrayBuffer();
                                if (this._sceneGeneration !== capturedScene) {
                                    console.log('Discarding stale polyline fetch');
                                    return;
                                }
                                if (!this._isLoadTokenCurrent(data.id, loadToken)) {
                                    console.log(`Discarding stale polyline fetch for '${data.id}'`);
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
                                this._deleteObject(data.id);
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
                        const loadToken = this._claimLoadToken(data.id);
                        (async () => {
                            try {
                                const resp = await fetch(data.blob_url);
                                const buffer = await resp.arrayBuffer();
                                if (this._sceneGeneration !== capturedScene) {
                                    console.log('Discarding stale mesh fetch');
                                    return;
                                }
                                if (!this._isLoadTokenCurrent(data.id, loadToken)) {
                                    console.log(`Discarding stale mesh fetch for '${data.id}'`);
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
                                    clippingPlanes: this._activeClippingPlanes(),
                                });

                                const mesh = new THREE.Mesh(geometry, meshMaterial);
                                mesh.name = data.id;
                                mesh.userData.id = data.id;
                                mesh.userData.isMesh = true;
                                mesh.userData.totalIndexCount = ni;
                                this._deleteObject(data.id);
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
                        const loadToken = this._claimLoadToken(data.id);
                        (async () => {
                            try {
                                const resp = await fetch(data.blob_url);
                                const buffer = await resp.arrayBuffer();
                                if (this._sceneGeneration !== capturedScene) {
                                    console.log('Discarding stale parametric tube fetch');
                                    return;
                                }
                                if (!this._isLoadTokenCurrent(data.id, loadToken)) {
                                    console.log(`Discarding stale parametric tube fetch for '${data.id}'`);
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
                                const nCs = N_CROSS_SECTION;
                                const hasColors = !!ringColors;
                                const upVector = data.upVector || null;
                                const heightOffset = data.heightOffset || 0;

                                // LOD: for large tubes, reduce spine before building geometry.
                                // Per-tube config via `data.lod` (see parseLodConfig).
                                const lodCfg = parseLodConfig(data.lod);
                                let tubeLOD = null;
                                let buildSpine = spine, buildWidths = widths, buildHeights = heights;
                                let buildOrientations = orientations, buildRingColors = ringColors;
                                let buildN = n;
                                if (lodCfg.enabled && n >= lodCfg.threshold) {
                                    // Bounding sphere from original spine (stable across LOD rebuilds).
                                    // Includes cross-section extent so `boundingRadius` reflects the
                                    // tube's actual on-screen size — color weight scales with it.
                                    const _center = new THREE.Vector3();
                                    for (let i = 0; i < n; i++) {
                                        _center.x += spine[i * 3]; _center.y += spine[i * 3 + 1]; _center.z += spine[i * 3 + 2];
                                    }
                                    _center.divideScalar(n);
                                    let _maxR2 = 0;
                                    for (let i = 0; i < n; i++) {
                                        const dx = spine[i * 3] - _center.x, dy = spine[i * 3 + 1] - _center.y, dz = spine[i * 3 + 2] - _center.z;
                                        _maxR2 = Math.max(_maxR2, dx * dx + dy * dy + dz * dz);
                                    }
                                    let _maxHalfExtent = 0;
                                    for (let i = 0; i < n; i++) {
                                        const h = Math.max(widths[i], heights[i]) * 0.5;
                                        if (h > _maxHalfExtent) _maxHalfExtent = h;
                                    }
                                    const boundingRadius = Math.sqrt(_maxR2) + _maxHalfExtent;
                                    const cam = this._camera;
                                    const { indices: keptIndices, minDist: _lodMinDist, maxDist: _lodMaxDist } = distanceWeightedRDP(
                                        spine, widths, heights, ringColors, boundingRadius, n,
                                        cam.position.x, cam.position.y, cam.position.z,
                                        lodCfg.epsilonDivisor,
                                    );
                                    const nRed = keptIndices.length;
                                    // Extract reduced arrays
                                    buildSpine = new Float32Array(nRed * 3);
                                    buildWidths = new Float32Array(nRed);
                                    buildHeights = new Float32Array(nRed);
                                    buildRingColors = ringColors ? new Float32Array(nRed * 3) : null;
                                    buildOrientations = null; // orientations not preserved through LOD
                                    buildN = nRed;
                                    for (let i = 0; i < nRed; i++) {
                                        const oi = keptIndices[i];
                                        buildSpine[i * 3] = spine[oi * 3]; buildSpine[i * 3 + 1] = spine[oi * 3 + 1]; buildSpine[i * 3 + 2] = spine[oi * 3 + 2];
                                        buildWidths[i] = widths[oi];
                                        buildHeights[i] = heights[oi];
                                        if (buildRingColors) {
                                            buildRingColors[i * 3] = ringColors[oi * 3]; buildRingColors[i * 3 + 1] = ringColors[oi * 3 + 1]; buildRingColors[i * 3 + 2] = ringColors[oi * 3 + 2];
                                        }
                                    }
                                    tubeLOD = {
                                        originalSpine: new Float32Array(spine),
                                        originalWidths: new Float32Array(widths),
                                        originalHeights: new Float32Array(heights),
                                        originalRingColors: ringColors ? new Float32Array(ringColors) : null,
                                        originalCount: n,
                                        upVector,
                                        keptIndices,
                                        currentDrawRange: 1.0,
                                        lastCameraPos: cam.position.clone(),
                                        boundingCenter: _center,
                                        boundingRadius,
                                        colorVersion: 0,
                                        epsilonDivisor: lodCfg.epsilonDivisor,
                                    };
                                    // LOD initial reduction logged at debug level only
                                }

                                const strandCollapse = !!data.strandCollapse;
                                const { geometry, ringPairs, indicesPerRingPair, localFrames, miterScales, tangents: builtTangents, capAngles, capIndicesPerCap, endCapBase, endCapPattern } = buildParametricTubeGeometry(
                                    buildSpine, buildWidths, buildHeights,
                                    buildOrientations, upVector, buildRingColors, heightOffset,
                                    strandCollapse,
                                );
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
                                    clippingPlanes: this._activeClippingPlanes(),
                                });
                                const mesh = new THREE.Mesh(geometry, material);
                                mesh.name = data.id;
                                mesh.userData.id = data.id;
                                mesh.userData.isParametricTube = true;
                                mesh.userData.parametricTube = new ParametricTube(mesh);
                                mesh.userData.tubeNumSpinePoints = buildN;
                                mesh.userData.tubeNCs = nCs;
                                mesh.userData.tubeRingPairs = ringPairs;
                                mesh.userData.tubeIndicesPerRingPair = indicesPerRingPair;
                                mesh.userData.totalIndexCount = capIndicesPerCap + ringPairs * indicesPerRingPair + capIndicesPerCap;
                                mesh.userData.tubeHasColors = hasColors;
                                mesh.userData.tubeCapIndicesPerCap = capIndicesPerCap;
                                mesh.userData.tubeEndCapBase = endCapBase;
                                mesh.userData.tubeHeightOffset = heightOffset;
                                mesh.userData.tubeMorphData = {
                                    spine: new Float32Array(buildSpine),
                                    widths: new Float32Array(buildWidths),
                                    heights: new Float32Array(buildHeights),
                                    localFrames, miterScales, tangents: builtTangents, capAngles,
                                    ringColors: buildRingColors ? new Float32Array(buildRingColors) : null,
                                    section: new Float32Array(nCs * 2),
                                    savedRing: new Float32Array(nCs * 3),
                                    savedRingNormals: null,
                                    savedRingColors: null,
                                    savedRingIndex: null,
                                    morphedState: null,
                                    endCapPattern,
                                    savedCapIndices: new (/** @type {any} */ (endCapPattern.constructor))(endCapPattern.length),
                                    savedCapOffset: -1,
                                    heightOffset,
                                };
                                if (tubeLOD) {
                                    mesh.userData.tubeLOD = tubeLOD;
                                    // Register original arrays with LOD worker
                                    this._lodWorker.postMessage({
                                        type: 'init',
                                        tubeId: data.id,
                                        spine: tubeLOD.originalSpine,
                                        widths: tubeLOD.originalWidths,
                                        heights: tubeLOD.originalHeights,
                                        ringColors: tubeLOD.originalRingColors,
                                        upVec: upVector,
                                        nPoints: tubeLOD.originalCount,
                                        heightOffset: heightOffset,
                                        boundingRadius: tubeLOD.boundingRadius,
                                        epsilonDivisor: tubeLOD.epsilonDivisor,
                                    });
                                }
                                this._deleteObject(data.id);
                                this._addToParentOrScene(mesh, data.parent);
                                this._objects.set(data.id, mesh);
                                this._objGeneration++;
                                if (data.transform) this._applyTransform(mesh, data.transform);
                                console.log(`Created parametric_tube ${data.id}: ${buildN} spine pts × ${nCs} cs verts, ${ringPairs} ring pairs`);
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
                                const lod = obj.userData.tubeLOD;
                                // Color data from Python is always for the original spine count
                                const nOrig = lod ? lod.originalCount : obj.userData.tubeNumSpinePoints;
                                const nCs = obj.userData.tubeNCs;
                                if (buffer.byteLength < nOrig * 4) {
                                    console.warn(`update_parametric_tube_colors: blob too small (${buffer.byteLength} < ${nOrig * 4})`);
                                    return;
                                }
                                const packed = new Uint32Array(buffer, 0, nOrig);
                                // Decode per-ring colors (full original count)
                                const rc = new Float32Array(nOrig * 3);
                                for (let i = 0; i < nOrig; i++) {
                                    const c = packed[i];
                                    rc[i * 3]     = ((c >> 16) & 0xff) / 255;
                                    rc[i * 3 + 1] = ((c >> 8) & 0xff) / 255;
                                    rc[i * 3 + 2] = (c & 0xff) / 255;
                                }

                                // When LOD is active, store full colors and rebuild with reduced subset
                                if (lod && lod.keptIndices) {
                                    lod.originalRingColors = new Float32Array(rc);
                                    lod.colorVersion = (lod.colorVersion || 0) + 1;
                                    this._lodWorker.postMessage({ type: 'updateColors', tubeId: data.id, ringColors: lod.originalRingColors });
                                    // Extract reduced colors for current LOD level
                                    const nRed = lod.keptIndices.length;
                                    const redRc = new Float32Array(nRed * 3);
                                    const redPacked = new Uint32Array(nRed);
                                    for (let i = 0; i < nRed; i++) {
                                        const oi = lod.keptIndices[i];
                                        redRc[i * 3] = rc[oi * 3]; redRc[i * 3 + 1] = rc[oi * 3 + 1]; redRc[i * 3 + 2] = rc[oi * 3 + 2];
                                        redPacked[i] = packed[oi];
                                    }
                                    // Update geometry colors for reduced mesh
                                    const n = nRed;
                                    // Restore frontier ring BEFORE writing new colors
                                    const md = obj.userData.tubeMorphData;
                                    if (md) restoreFrontierRing(obj);
                                    // Fill cap dome vertices with a single color
                                    /**
                                     * @param {ArrayLike<number> & {[i: number]: number}} arr
                                     * @param {number} baseVert
                                     * @param {number} capVerts
                                     * @param {number} r
                                     * @param {number} g
                                     * @param {number} b
                                     */
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
                                        expandRingColors(redPacked, n, nCs, existing.array);
                                        fillCapColors(existing.array, startCapBaseVert, capVertsPerCap, redRc[0], redRc[1], redRc[2]);
                                        fillCapColors(existing.array, endCapBaseVert, capVertsPerCap, redRc[lr], redRc[lr + 1], redRc[lr + 2]);
                                        existing.clearUpdateRanges();
                                        existing.needsUpdate = true;
                                    } else {
                                        const allColors = new Float32Array(posCount * 3);
                                        expandRingColors(redPacked, n, nCs, allColors);
                                        fillCapColors(allColors, startCapBaseVert, capVertsPerCap, redRc[0], redRc[1], redRc[2]);
                                        fillCapColors(allColors, endCapBaseVert, capVertsPerCap, redRc[lr], redRc[lr + 1], redRc[lr + 2]);
                                        obj.geometry.setAttribute('color', new THREE.BufferAttribute(allColors, 3));
                                    }
                                    obj.material.vertexColors = true;
                                    obj.material.color.setHex(0xffffff);
                                    obj.material.needsUpdate = true;
                                    obj.userData.tubeHasColors = true;
                                    obj.userData._colorFullUploadNeeded = true;
                                    if (md) md.ringColors = redRc;
                                } else {
                                    // No LOD active — original path
                                    if (lod) {
                                        lod.originalRingColors = new Float32Array(rc);
                                        lod.colorVersion = (lod.colorVersion || 0) + 1;
                                        this._lodWorker.postMessage({ type: 'updateColors', tubeId: data.id, ringColors: lod.originalRingColors });
                                    }
                                    const n = nOrig;
                                    // Fill cap dome vertices with a single color
                                    /**
                                     * @param {ArrayLike<number> & {[i: number]: number}} arr
                                     * @param {number} baseVert
                                     * @param {number} capVerts
                                     * @param {number} r
                                     * @param {number} g
                                     * @param {number} b
                                     */
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
                                    // Restore frontier ring BEFORE writing new colors so
                                    // the stale savedRingColors don't overwrite the update.
                                    const md = obj.userData.tubeMorphData;
                                    if (md) restoreFrontierRing(obj);
                                    const existing = obj.geometry.getAttribute('color');
                                    if (existing) {
                                        expandRingColors(packed, n, nCs, existing.array);
                                        fillCapColors(existing.array, startCapBaseVert, capVertsPerCap, rc[0], rc[1], rc[2]);
                                        fillCapColors(existing.array, endCapBaseVert, capVertsPerCap, rc[lr], rc[lr + 1], rc[lr + 2]);
                                        existing.clearUpdateRanges();
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
                                    obj.userData._colorFullUploadNeeded = true;
                                    if (md) md.ringColors = rc;
                                }
                            } catch (e) {
                                console.error(`Error updating parametric_tube colors:`, e);
                            } finally {
                                this._onFetchEnd();
                            }
                        })();
                        break;
                    }
                    case 'register_toolpath_group': {
                        const grp = this._objects.get(data.id);
                        if (grp) {
                            grp.userData.isToolpathGroup = true;
                            grp.userData.toolpathSegmentIds = data.segmentIds;
                            grp.userData.toolpathSegmentRanges = data.segmentRanges;
                        }
                        break;
                    }
                    case 'unload_animation':
                        this._unloadAnimation(data.restore_visibility !== false);
                        break;
                    case 'pause_animation':
                        if (this._animation) {
                            this._animationPlaying = false;
                            this._updateAnimationUI();
                        }
                        break;
                    case 'resume_animation':
                        if (this._animation) {
                            this._animationPlaying = true;
                            this._lastAnimationUpdate = performance.now();
                            this._updateAnimationUI();
                        }
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

    _updateSceneBounds() { this._camController.updateSceneBounds(); }
    _updateNearFar() { this._camController.updateNearFar(); }

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

        // Pivot marker: scale to a constant ~6px screen radius, ring faces camera,
        // hide 900ms after the pivot was set (but only once the user stops dragging).
        if (this._pivotMarker && this._pivotMarker.visible) {
            const cam = /** @type {THREE.PerspectiveCamera & THREE.OrthographicCamera} */ (this._camera);
            const canvasH = Math.max(1, this._renderer.domElement.clientHeight);
            let worldPerPixel;
            if (cam.isPerspectiveCamera) {
                const dist = cam.position.distanceTo(this._pivotMarker.position);
                worldPerPixel = (2 * dist * Math.tan(THREE.MathUtils.degToRad(cam.fov / 2))) / canvasH;
            } else {
                worldPerPixel = (cam.top - cam.bottom) / cam.zoom / canvasH;
            }
            this._pivotMarker.scale.setScalar(worldPerPixel * 60);
            this._pivotMarkerRing.quaternion.copy(cam.quaternion);
            const elapsed = performance.now() - this._pivotShownAt;
            if (elapsed > 900 && !this._controls.isDragging()) {
                this._pivotMarker.visible = false;
            }
        }

        if (this._viewHelper.animating) this._viewHelper.update(frameDelta);
        if (this._shading.shadingMode === 3) {
            this._scene.traverse((obj) => {
                const h = obj.userData && obj.userData.vertexNormalsHelper;
                if (!h || !h.visible) return;
                h.size = this._shading.cameraRelativeNormalSize(obj);
                h.update();
            });
        }
        this._renderer.autoClear = true;
        this._renderer.render(this._scene, this._camera);
        this._renderer.autoClear = false;
        // Lift the ViewHelper above the animation toolbar when it's visible.
        // ViewHelper hardcodes setViewport(x, 0, dim, dim); we shim that one
        // call to add a Y offset matching the toolbar height.
        const lift = (this._animLiftCss || 0) * window.devicePixelRatio;
        if (lift > 0) {
            // Cache the true original once so we don't re-wrap the wrapped
            // setViewport each frame (which would deepen the call chain by
            // one level per frame and eventually blow the stack).
            if (!this._rendererSetViewportOriginal) {
                this._rendererSetViewportOriginal = this._renderer.setViewport.bind(this._renderer);
            }
            const orig = this._rendererSetViewportOriginal;
            const r = this._renderer;
            r.setViewport = (x, y, w, h) => orig(x, (y === 0 && w === h) ? lift : y, w, h);
            try {
                this._viewHelper.render(this._renderer);
            } finally {
                r.setViewport = orig;
            }
        } else {
            this._viewHelper.render(this._renderer);
        }

        // LOD: dispatch to Web Worker after render (non-blocking)
        if (this._lodDirty && !this._lodWorkerBusy && performance.now() - this._lodLastRunTime >= this._lodThrottleMs) {
            this._lodLastRunTime = performance.now();
            this._lodDirty = false;
            this._dispatchLodWorker();
        }
    }

    _dispatchLodWorker() {
        const cam = this._camera;
        const camX = cam.position.x, camY = cam.position.y, camZ = cam.position.z;
        for (const obj of this._objects.values()) {
            if (!obj.userData.isParametricTube || !obj.userData.tubeLOD) continue;
            const lod = obj.userData.tubeLOD;

            // Skip if camera hasn't moved enough (5% of distance to tube center)
            if (lod.keptIndices) {
                const dx = camX - lod.lastCameraPos.x;
                const dy = camY - lod.lastCameraPos.y;
                const dz = camZ - lod.lastCameraPos.z;
                const deltaSq = dx * dx + dy * dy + dz * dz;
                const centerDist = Math.max(1e-6, lod.boundingRadius * 2);
                const threshold = centerDist * 0.05;
                if (deltaSq < threshold * threshold) continue;
            }

            lod.lastCameraPos.set(camX, camY, camZ);
            this._lodWorkerBusy = true;
            this._lodWorker.postMessage({
                type: 'update',
                tubeId: obj.name,
                camX, camY, camZ,
            });
            // Only one tube per dispatch cycle (worker handles one at a time)
            return;
        }
    }

    // ========== Public API ==========

    /**
     * @param {number} [width]
     * @param {number} [height]
     */
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

    /** @param {THREE.Object3D} object */
    frameObject(object) {
        const bbox = new THREE.Box3();
        object.updateWorldMatrix(true, true);
        bbox.expandByObject(object);
        this._fitCameraToBox(bbox, 1.5);
    }

    // ========== ViewHelper (corner gizmo) ==========

    /**
     * Enlarge the ViewHelper's axis sprites so they have a bigger hit target
     * and a more visible cue. Baseline opacity is captured for the hover
     * restore. Called once per ViewHelper instance — the helper is re-created
     * on every perspective/ortho swap.
     * @param {any} helper
     */
    _configureViewHelper(helper) {
        const sprites = [];
        for (const child of helper.children) {
            if (!child.userData || !child.userData.type) continue;
            child.scale.setScalar(this._gizmoBaseScale);
            child.userData.baseOpacity = child.material.opacity;
            sprites.push(child);
        }
        helper.userData.interactiveSprites = sprites;
    }

    /**
     * CSS-pixel lift applied to the gizmo when the animation toolbar is
     * visible — matches the render-time shim in the main loop. Cached to
     * avoid layout reads on every pointermove (hit-test) and every frame
     * (render shim). Kept in sync by a ResizeObserver on the toolbar (+
     * synchronous priming in the show/hide paths) so viewport-driven
     * wrap/unwrap of the timeline-row updates the lift too.
     */
    _gizmoLiftCss() {
        return this._animLiftCss || 0;
    }

    /**
     * Push the toolbar's current height into both the cache (hit-test +
     * render shim) and the --tjsv-anim-lift CSS var (Home button).
     * display:none yields 0, which matches the "toolbar hidden" state.
     * Called on show/hide (no arg → reads offsetHeight to flush layout and
     * get the post-transition height synchronously) and from the
     * ResizeObserver (arg → borderBoxSize, avoiding another layout read).
     * @param {number} [measuredHeight]
     */
    _refreshAnimLift(measuredHeight) {
        const h = Number.isFinite(measuredHeight)
            ? /** @type {number} */ (measuredHeight)
            : this._animControlsEl.offsetHeight;
        if (h === this._animLiftCss) return;
        this._animLiftCss = h;
        this.el.style.setProperty('--tjsv-anim-lift', `${h}px`);
    }

    /**
     * Hit-test a pointer event against the ViewHelper's axis sprites. Returns
     * whether the pointer is inside the 128×128 gizmo rect at all, plus the
     * hovered sprite (null if no sprite under cursor).
     * @param {PointerEvent | MouseEvent} e
     * @returns {{ insideRect: boolean, hit: any }}
     */
    _gizmoHitTest(e) {
        const dom = this._renderer.domElement;
        const rect = dom.getBoundingClientRect();
        const dim = this._gizmoDim;
        const liftCss = this._gizmoLiftCss();
        const offsetX = rect.left + rect.width - dim;
        const offsetY = rect.top + rect.height - dim - liftCss;
        const insideRect =
            e.clientX >= offsetX && e.clientX <= offsetX + dim &&
            e.clientY >= offsetY && e.clientY <= offsetY + dim;
        if (!insideRect) return { insideRect: false, hit: null };
        const sprites = this._viewHelper.userData.interactiveSprites;
        if (!sprites || sprites.length === 0) return { insideRect: true, hit: null };
        const ndcX = ((e.clientX - offsetX) / dim) * 2 - 1;
        const ndcY = -((e.clientY - offsetY) / dim) * 2 + 1;
        this._gizmoHoverRaycaster.setFromCamera(
            new THREE.Vector2(ndcX, ndcY),
            this._gizmoHoverOrthoCam,
        );
        const hits = this._gizmoHoverRaycaster.intersectObjects(sprites, false);
        return { insideRect: true, hit: hits.length ? hits[0].object : null };
    }

    /**
     * Apply the hover visual (enlarged + fully opaque) to a sprite, restoring
     * the previously hovered one. Pass `null` to clear.
     * @param {any} sprite
     */
    _setGizmoHoverSprite(sprite) {
        if (sprite === this._gizmoHovered) return;
        if (this._gizmoHovered) {
            this._gizmoHovered.scale.setScalar(this._gizmoBaseScale);
            const base = this._gizmoHovered.userData.baseOpacity;
            if (typeof base === 'number') this._gizmoHovered.material.opacity = base;
        }
        if (sprite) {
            sprite.scale.setScalar(this._gizmoHoverScale);
            sprite.material.opacity = 1.0;
        }
        this._gizmoHovered = sprite;
    }

    resetView() {
        // Canonical home view: world-Z up, isometric-ish direction
        // (+X, -Y, +Z) looking at the scene bbox center, then framed to fit.
        // Replaces the old "hail-mary" reset — orientation is now always the
        // same regardless of prior camera state, so Home is predictable.
        this._camera.up.set(0, 0, 1);
        const bbox = new THREE.Box3();
        this._scene.traverse(/** @param {any} child */ child => {
            if (!child.geometry) return;
            if (child === this._gridHelper) return;
            if (this._isClipHelper(child)) return;
            if (this._isPivotMarkerDescendant(child)) return;
            child.updateWorldMatrix(true, false);
            bbox.expandByObject(child);
        });
        const target = bbox.isEmpty()
            ? new THREE.Vector3()
            : bbox.getCenter(new THREE.Vector3());
        this._controls.target.copy(target);
        // Pre-orient along the canonical direction so _fitCameraToBox (which
        // preserves whatever direction the camera is already pointing) puts
        // the camera in the right spot after fitting.
        const isoDir = new THREE.Vector3(1, -1, 1).normalize();
        this._camera.position.copy(target).addScaledVector(isoDir, 1);
        this._camera.lookAt(target);
        this._controls.update();
        if (bbox.isEmpty()) {
            // No scene yet — fall back to a sensible distance so the user
            // isn't staring down at near-clipping at the origin.
            this._camera.position.copy(target).addScaledVector(isoDir, 10);
            this._camera.lookAt(target);
            this._controls.update();
        } else {
            this._fitCameraToBox(bbox, 1.2);
        }
    }

    frameAll() {
        const bbox = new THREE.Box3();
        this._scene.traverse(/** @param {any} child */ child => {
            if (!child.geometry) return;
            if (child === this._gridHelper) return;
            if (this._isClipHelper(child)) return;
            if (this._isPivotMarkerDescendant(child)) return;
            child.updateWorldMatrix(true, false);
            bbox.expandByObject(child);
        });
        this._fitCameraToBox(bbox, 1.2);
    }

    /**
     * @param {THREE.Box3} bbox
     * @param {number} perspMargin
     */
    _fitCameraToBox(bbox, perspMargin) { this._camController.fitToBox(bbox, perspMargin); }

    // ========== Destroy ==========

    destroy() {
        this._destroyed = true;
        cancelAnimationFrame(this._animationFrameId);
        if (this._ws) {
            this._ws.onclose = null;
            this._ws.close();
            this._ws = null;
        }
        if (this._lodWorker) {
            this._lodWorker.terminate();
            this._lodWorker = null;
        }
        clearTimeout(this._reconnectTimeout);
        this._resizeObserver.disconnect();
        this._animLiftObserver.disconnect();
        this.container.removeEventListener('keydown', this._onKeyDown);
        document.removeEventListener('mousemove', this._onDocMouseMove);
        document.removeEventListener('mouseup', this._onDocMouseUp);
        this._renderer.dispose();
        this._controls.dispose();
        this._clipGizmo.dispose();
        this.el.remove();
    }
}
