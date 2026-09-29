// Run inside the bundled viewer: uses its exact Three.js version/import map.
const THREE = await import('three');
const v = window.threejsViewer;
clearTimeout(v._reconnectTimeout);
await v._cubemapsReady;
v.setDisplayQuality('high');
v._applyToneMapping('aces');
v._applyEnvironmentEnabled(true);
v._applyToneMappingExposure(1.9);
v._applyAmbientIntensity(0.5);
v._applyEnvironmentIntensity(1.1);
v.setSun({enabled: true, intensity: 5.7, azimuth: -46, elevation: 45});
const meshes = [];
function add(name, geometry, position, rotation = [0, 0, 0], metalness = 0.35) {
    const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
        color: 0x888888, roughness: 0.38, metalness,
    }));
    mesh.position.set(...position);
    mesh.rotation.set(...rotation);
    v._scene.add(mesh);
    v._registerObject(name, mesh);
    meshes.push(mesh);
    return mesh;
}
add('floor', new THREE.BoxGeometry(20, 20, 0.1), [0, 0, -0.05]);
add('back', new THREE.BoxGeometry(5, 0.12, 3), [0, 0.8, 2.1]);
add('sloped', new THREE.BoxGeometry(2.5, 0.1, 3), [-3, 0.6, 2], [0.2, 0, -0.35]);
add('shelf', new THREE.BoxGeometry(7, 2, 0.12), [0, 0, 1]);
add('hood', new THREE.BoxGeometry(3, 1.5, 0.12), [0, 0, 3.6]);
for (const x of [-2, 2]) {
    add('leg' + x, new THREE.CylinderGeometry(0.14, 0.14, 2, 64), [x, -0.5, 1], [Math.PI / 2, 0, 0], 0.8);
}
add('sphere', new THREE.SphereGeometry(0.65, 64, 32), [3.1, -0.2, 1.72], [0, 0, 0], 0.6);
add('contact', new THREE.BoxGeometry(0.55, 0.55, 0.7), [0.5, -0.4, 1.41]);
// Open CAD panels: unlike thick boxes, both shadow and visible depths coincide.
const panel = add('open-panel', new THREE.PlaneGeometry(2.3, 2.8), [-2.7, -0.1, 2.1], [Math.PI / 2, 0.25, -0.3]);
panel.material.side = THREE.DoubleSide;
v._sceneBoundsDirty = true;
v._camController.updateSceneBounds();
v._camera.near = 0.01;
v._camera.far = 100;
v._camera.updateProjectionMatrix();
v.setCameraPose({position: [4, -7, 4], target: [0, 0, 1.8], up: [0, 0, 1]});
v._updateSun();
const presets = {
    production: {type: v._renderer.shadowMap.type, radius: v._sun.shadow.radius,
        size: v._sun.shadow.mapSize.x, normal: 1.5, filter: 'tent5'},
    baseline: {type: THREE.PCFShadowMap, radius: 4, size: 2048, normal: 1.5},
    narrow: {type: THREE.PCFShadowMap, radius: 1, size: 2048, normal: 1.5},
    medium: {type: THREE.PCFShadowMap, radius: 2, size: 2048, normal: 1.5},
    biased: {type: THREE.PCFShadowMap, radius: 4, size: 2048, normal: 4},
    larger: {type: THREE.PCFShadowMap, radius: 4, size: 4096, normal: 1.5},
    unshadowed: {type: THREE.PCFShadowMap, radius: 1, size: 2048, normal: 1.5, receive: false},
};
let filter = 'stock';
for (const mesh of meshes) {
    const material = mesh.material;
    const compile = material.onBeforeCompile, key = material.customProgramCacheKey;
    material.onBeforeCompile = function (shader, renderer) {
        compile.call(this, shader, renderer);
        if (filter === 'stock') shader.fragmentShader = shader.fragmentShader.replace(
            '#define TJSV_DIRECTIONAL_SHADOW tjsvSunShadow',
            '#define TJSV_DIRECTIONAL_SHADOW getShadow');
    };
    material.customProgramCacheKey = function () { return key.call(this) + '|lab-' + filter; };
}
function apply(name) {
    const p = presets[name], s = v._sun.shadow;
    filter = p.filter || 'stock';
    v._renderer.shadowMap.type = p.type;
    if (s.mapSize.x !== p.size) {
        s.map?.dispose(); s.map = null;
    }
    s.mapSize.set(p.size, p.size);
    s.radius = p.radius;
    s.normalBias = p.normal * (2 * v._sceneSphere.radius / p.size);
    s.bias = -0.0002;
    for (const mesh of meshes) {
        mesh.receiveShadow = p.receive !== false;
        mesh.material.needsUpdate = true;
    }
    v.requestShadowUpdate();
    v._updateSun();
    v._renderer.render(v._scene, v._camera);
    return {name, filter, filterRadius: s.radius, normalBias: s.normalBias};
}
// Timer queries measure GPU work; JS submission times hide map-resolution costs.
async function benchmark(name, dirty = false, count = 60) {
    apply(name);
    const r = v._renderer, gl = r.getContext(), samples = [];
    const ext = gl.getExtension('EXT_disjoint_timer_query_webgl2');
    if (!ext) throw new Error('GPU timer queries are unavailable on this browser');
    for (let i = 0; i < count + 15; i++) {
        await new Promise(requestAnimationFrame);
        const query = gl.createQuery();
        gl.beginQuery(ext.TIME_ELAPSED_EXT, query);
        for (let j = 0; j < 5; j++) {
            if (dirty) r.shadowMap.needsUpdate = true;
            r.render(v._scene, v._camera);
        }
        gl.endQuery(ext.TIME_ELAPSED_EXT);
        while (!gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE)) await new Promise(requestAnimationFrame);
        const disjoint = gl.getParameter(ext.GPU_DISJOINT_EXT);
        const ms = gl.getQueryParameter(query, gl.QUERY_RESULT) / 1e6 / 5;
        gl.deleteQuery(query);
        if (disjoint) throw new Error('GPU clock changed; rerun measurement');
        if (i >= 15) samples.push(ms);
    }
    samples.sort((a, b) => a - b);
    return {name, dirty, medianMs: samples[Math.floor(samples.length / 2)],
        p90Ms: samples[Math.floor(samples.length * 0.9)], samples: samples.length};
}
function acneScore(name, azimuth = -46, elevation = 45) {
    // Isolate the open panel so any darkening is erroneous self-shadowing.
    // Retain the scene's full map coverage and sample away from panel edges.
    const previousSun = v.getSun();
    const visibility = meshes.map(m => m.visible);
    for (const mesh of meshes) mesh.visible = mesh === panel;
    v.setSun({azimuth, elevation}); v._updateSun(); apply(name);
    const r = v._renderer, gl = r.getContext();
    const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
    const read = () => {
        r.render(v._scene, v._camera);
        const data = new Uint8Array(w * h * 4);
        gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, data);
        return data;
    };
    const shadowed = read();
    panel.receiveShadow = false; panel.material.needsUpdate = true;
    const reference = read();
    let total = 0, dark = 0, max = 0, sum = 0;
    for (let y = -0.9; y <= 0.9; y += 0.035) for (let x = -0.85; x <= 0.85; x += 0.035) {
        const p = panel.localToWorld(new THREE.Vector3(x, y, 0)).project(v._camera);
        const px = Math.floor((p.x + 1) * w / 2), py = Math.floor((p.y + 1) * h / 2);
        if (px < 0 || py < 0 || px >= w || py >= h) continue;
        const offset = (py * w + px) * 4;
        const delta = (reference[offset] + reference[offset + 1] + reference[offset + 2]
            - shadowed[offset] - shadowed[offset + 1] - shadowed[offset + 2]) / 3;
        total++; if (delta > 3) dark++; max = Math.max(max, delta); sum += delta;
    }
    meshes.forEach((m, i) => { m.visible = visibility[i]; });
    v.setSun(previousSun); v._updateSun(); apply(name);
    return {name, azimuth, elevation, samples: total, darkFraction: dark / total, meanLoss: sum / total, maxLoss: max};
}
function edgeScore(name) {
    // Mean 20–80% transition width across 17 scanlines on the back panel.
    // This must grow relative to radius 1: clearing acne alone is not enough.
    apply(name);
    const r = v._renderer, gl = r.getContext();
    const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
    r.render(v._scene, v._camera);
    const data = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, data);
    const value = (x, z) => {
        const p = new THREE.Vector3(x, 0.739, z).project(v._camera);
        const i = (Math.floor((p.y + 1) * h / 2) * w + Math.floor((p.x + 1) * w / 2)) * 4;
        return (data[i] + data[i + 1] + data[i + 2]) / 3;
    };
    const widths = [];
    for (let z = 2.2; z < 3.01; z += 0.05) {
        const low = value(0.1, z), high = value(1.4, z);
        let width = 0;
        for (let x = 0.1; x < 1.4; x += 0.001) {
            const a = (value(x, z) - low) / (high - low);
            if (a > 0.2 && a < 0.8) width += 0.001;
        }
        widths.push(width);
    }
    return {name, widths, meanWidth: widths.reduce((a, b) => a + b) / widths.length};
}
window.shadowLab = {apply, benchmark, acneScore, edgeScore, presets, meshes, viewer: v};
apply('baseline');
