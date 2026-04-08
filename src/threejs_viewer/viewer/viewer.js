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


// Channel apply functions — keyed by channel name
function makeChannelApply(viewer) {
    return {
        transforms(ch, refs, base) {
            const nObj = ch.ids.length;
            for (let i = 0; i < nObj; i++) {
                let obj = refs[i];
                if (!obj) {
                    obj = viewer._objects.get(ch.ids[i]);
                    if (obj) { refs[i] = obj; obj.matrixAutoUpdate = false; }
                }
                if (obj) {
                    obj.matrix.fromArray(ch.data, base + i * 16);
                    obj.matrixWorldNeedsUpdate = true;
                }
            }
        },

        colors(ch, refs, base) {
            const nObj = ch.ids.length;
            const colormap = ch.colormap;
            for (let i = 0; i < nObj; i++) {
                const raw = ch.data[base + i];
                const color = colormap ? colormap[raw] : raw;
                let obj = refs[i];
                if (!obj) { obj = viewer._objects.get(ch.ids[i]); if (obj) refs[i] = obj; }
                if (obj) {
                    obj.traverse(child => {
                        if (!child.material) return;
                        const mats = Array.isArray(child.material) ? child.material : [child.material];
                        for (const mat of mats) { if (mat.color) mat.color.setHex(color); }
                    });
                }
            }
        },

        visibility(ch, refs, base) {
            const nObj = ch.ids.length;
            for (let i = 0; i < nObj; i++) {
                let obj = refs[i];
                if (!obj) { obj = viewer._objects.get(ch.ids[i]); if (obj) refs[i] = obj; }
                if (obj) obj.visible = (ch.data[base + i] === 1);
            }
        },

        draw_ranges(ch, refs, base) {
            const nObj = ch.ids.length;
            for (let i = 0; i < nObj; i++) {
                viewer._setDrawRange(ch.ids[i], ch.data[base + i]);
            }
        },

        opacity(ch, refs, base) {
            const nObj = ch.ids.length;
            for (let i = 0; i < nObj; i++) {
                const val = ch.data[base + i];
                let obj = refs[i];
                if (!obj) { obj = viewer._objects.get(ch.ids[i]); if (obj) refs[i] = obj; }
                if (obj) applyOpacity(obj, val);
            }
        },

        clip_times(ch, refs, base) {
            const nObj = ch.ids.length;
            for (let i = 0; i < nObj; i++) {
                viewer._setClipTime(ch.ids[i], ch.data[base + i]);
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
        this._pendingFetches = 0;
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

        // Grid helper on XY plane (Z-up)
        this._gridHelper = new THREE.GridHelper(10, 10);
        this._gridHelper.rotation.x = Math.PI / 2;
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
                    scene.environmentRotation = new THREE.Euler(Math.PI / 2, 0, 0);
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
                if (childMixer) { childMixer.stopAllAction(); this._mixers.delete(childId); }
            }
            if (obj.parent) obj.parent.remove(obj);
            this._objects.delete(id);
            this._sceneBoundsDirty = true;
            const mixer = this._mixers.get(id);
            if (mixer) { mixer.stopAllAction(); this._mixers.delete(id); }
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

    _clearScene() {
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

    _stopAnimation() {
        this._objects.forEach((obj) => { obj.matrixAutoUpdate = true; });
        for (const [id, baselineVisible] of this._baselineVisibility) {
            const obj = this._objects.get(id);
            if (obj) obj.visible = baselineVisible;
        }
        this._baselineVisibility.clear();
        this._animation = null;
        this._animationPlaying = false;
        this._animControlsEl.classList.remove('visible');

        this._trackMode = 'off';
        this._trackTargetId = null;
        this._trackHasLastPos = false;
        this._trackInteractive = false;
        this._updateTrackingUI();
    }

    _applyFrame(frameIndex) {
        if (!this._animation || frameIndex < 0 || frameIndex >= this._animation.frames.length) return;

        const frame = this._animation.frames[frameIndex];

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
                    applyFn(ch, ch.refs, base);
                }
            }
        }

        if (frame.transforms) {
            for (const [id, matrix] of Object.entries(frame.transforms)) {
                const obj = this._objects.get(id);
                if (obj) {
                    obj.matrixAutoUpdate = false;
                    obj.matrix.fromArray(matrix);
                    obj.matrixWorldNeedsUpdate = true;
                }
            }
        }
        if (frame.colors) {
            for (const [id, color] of Object.entries(frame.colors)) {
                const obj = this._objects.get(id);
                if (obj) {
                    obj.traverse((child) => {
                        if (!child.material) return;
                        const mats = Array.isArray(child.material) ? child.material : [child.material];
                        for (const mat of mats) { if (mat.color) mat.color.setHex(color); }
                    });
                }
            }
        }
        if (!this._animation.channels || !this._animation.channels.visibility) {
            for (const [id, baselineVisible] of this._baselineVisibility) {
                const obj = this._objects.get(id);
                if (obj) obj.visible = baselineVisible;
            }
        }
        if (frame.visibility) {
            for (const [id, visible] of Object.entries(frame.visibility)) {
                const obj = this._objects.get(id);
                if (obj) obj.visible = visible;
            }
        }
        if (frame.opacity) {
            for (const [id, opacity] of Object.entries(frame.opacity)) {
                const obj = this._objects.get(id);
                if (obj) applyOpacity(obj, opacity);
            }
        }
        if (frame.clip_times) {
            for (const [id, time] of Object.entries(frame.clip_times)) {
                this._setClipTime(id, time);
            }
        }
        if (frame.draw_ranges) {
            for (const [id, value] of Object.entries(frame.draw_ranges)) {
                this._setDrawRange(id, value);
            }
        }

        this._applyCameraTracking(frameIndex);
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

    _applyCameraTracking(frameIndex) {
        if (this._trackMode === 'off') return;

        // Scripted camera channels
        if (this._trackMode === 'scripted' && this._animation?.channels) {
            const ctCh = this._animation.channels.camera_target;
            const cpCh = this._animation.channels.camera_position;
            if (ctCh || cpCh) {
                if (ctCh) {
                    const base = frameIndex * 3;
                    this._controls.target.set(
                        ctCh.data[base], ctCh.data[base + 1], ctCh.data[base + 2]
                    );
                }
                if (cpCh) {
                    const base = frameIndex * 3;
                    this._camera.position.set(
                        cpCh.data[base], cpCh.data[base + 1], cpCh.data[base + 2]
                    );
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

    _getFrameAtTime(time) {
        if (!this._animation || this._animation.frames.length === 0) return 0;
        if (this._animation.uniformDt > 0) {
            const idx = Math.floor(time / this._animation.uniformDt);
            return Math.max(0, Math.min(this._animation.frames.length - 1, idx));
        }
        for (let i = this._animation.frames.length - 1; i >= 0; i--) {
            if (this._animation.frames[i].time <= time) return i;
        }
        return 0;
    }

    _updateAnimationUI() {
        if (!this._animation) return;
        const frameIndex = this._getFrameAtTime(this._animationTime);
        const progress = this._animation.duration > 0 ? (this._animationTime / this._animation.duration) * 100 : 0;
        this._timelineProgressEl.style.width = `${progress}%`;
        this._currentTimeEl.textContent = this._animationTime.toFixed(2);
        this._currentFrameEl.textContent = frameIndex + 1;
        this._btnPlay.textContent = this._animationPlaying ? '\u23F8' : '\u25B6';
    }

    _stepFrames(delta) {
        if (!this._animation) return;
        const currentFrame = this._getFrameAtTime(this._animationTime);
        const newFrame = Math.max(0, Math.min(this._animation.frames.length - 1, currentFrame + delta));
        this._animationTime = this._animation.frames[newFrame].time;
        this._applyFrame(newFrame);
        this._updateAnimationUI();
    }

    _seekToTime(time) {
        if (!this._animation) return;
        this._animationTime = Math.max(0, Math.min(this._animation.duration, time));
        const frameIndex = this._getFrameAtTime(this._animationTime);
        this._applyFrame(frameIndex);
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
        this._pendingFetches--;
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
        // Derive HTTP URL from WS URL for the fetch probe
        const httpUrl = this._wsUrl.replace(/^ws(s?):/, 'http$1:');

        const doConnect = async () => {
            if (this._destroyed) return;
            try {
                await fetch(httpUrl, { mode: 'no-cors' });
            } catch {
                this._reconnectTimeout = setTimeout(doConnect, 1000);
                return;
            }

            this._ws = new WebSocket(this._wsUrl);

            this._ws.onopen = () => {
                this._statusDot.className = 'tjsv-status-dot connected';
                this._statusDot.title = 'Connected';
                this._statusText.textContent = 'Connected';
                this._pendingFetches = 0;
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
                            tree[id] = {
                                type: obj.type,
                                parent: obj.parent?.userData?.id || null,
                                children: obj.children
                                    .filter(c => c.userData?.id)
                                    .map(c => c.userData.id),
                                visible: obj.visible,
                            };
                        }
                        this._ws.send(JSON.stringify({
                            type: 'query_scene_response',
                            requestId: data.requestId,
                            tree: tree,
                        }));
                        break;
                    }
                    case 'load_animation':
                        this._loadAnimation(data.animation);
                        break;
                    case 'load_animation_http':
                        this._onFetchStart();
                        (async () => {
                            try {
                                const t0 = performance.now();
                                const resp = await fetch(data.blob_url);
                                const buffer = await resp.arrayBuffer();
                                const t1 = performance.now();
                                console.log(`HTTP animation fetch: ${(buffer.byteLength / 1024 / 1024).toFixed(1)}MB in ${(t1 - t0).toFixed(0)}ms`);

                                const nFrames = data.frame_count;
                                const DTYPE_INFO = {
                                    float32: { ArrayType: Float32Array, bytes: 4 },
                                    uint32:  { ArrayType: Uint32Array,  bytes: 4 },
                                    uint8:   { ArrayType: Uint8Array,   bytes: 1 },
                                };

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
                    case 'add_model_binary':
                        this._onFetchStart();
                        (async () => {
                            try {
                                const resp = await fetch(data.blob_url);
                                const meshBytes = await resp.arrayBuffer();
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
                    case 'add_polyline_binary':
                        this._onFetchStart();
                        (async () => {
                            try {
                                const resp = await fetch(data.blob_url);
                                const buffer = await resp.arrayBuffer();
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
                            } catch (e) {
                                console.error(`Error creating polyline via HTTP:`, e);
                            } finally {
                                this._onFetchEnd();
                            }
                        })();
                        break;
                    case 'add_mesh_binary':
                        this._onFetchStart();
                        (async () => {
                            try {
                                const resp = await fetch(data.blob_url);
                                const buffer = await resp.arrayBuffer();
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
                                console.log(`Created mesh ${data.id}: ${nv} verts, ${(ni / 3)|0} tris`);
                            } catch (e) {
                                console.error(`Error creating mesh:`, e);
                            } finally {
                                this._onFetchEnd();
                            }
                        })();
                        break;
                    case 'stop_animation':
                        this._stopAnimation();
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
                    case 'mark_assets_complete':
                        this._assetsComplete = true;
                        this._maybeNotifyAssetsLoaded();
                        break;
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

            const frameIndex = this._getFrameAtTime(this._animationTime);
            this._applyFrame(frameIndex);
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
