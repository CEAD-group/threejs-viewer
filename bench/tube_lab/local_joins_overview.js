// Embedded by local_joins.py into the actual built viewer; one WebGL context.
const labCases = /* CASES */;
const labMessages = /* MESSAGES */;
const labViewer = window.threejsViewer;
const { OrbitControls: LabOrbitControls } = await import('three/addons/controls/OrbitControls.js');
const labStyle = document.createElement('style');
labStyle.textContent = `
body {margin:0;background:#151c24;color:#e7eef7;font:12px system-ui;overflow:hidden}
#viewer-container {visibility:hidden;pointer-events:none}
#board-canvas {position:fixed;inset:0;width:100%;height:100%;z-index:200;pointer-events:none}
#board {position:fixed;inset:78px 8px 8px;display:grid;gap:7px;z-index:201}
#board article {min-width:0;min-height:0;display:flex;flex-direction:column;border:1px solid #405065;border-radius:6px;overflow:hidden}
#board h2 {font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin:0;padding:4px 6px;background:#1a2532;cursor:zoom-in}
#board .pair {flex:1;min-height:0;display:flex}
#board .view {width:50%;position:relative;touch-action:none;cursor:grab}
#board .view+ .view {border-left:1px solid #405065}
#board .label {position:absolute;top:3px;left:5px;color:#c3d3e4;font-size:10px;pointer-events:none}
#board .stats {background:#1a2532;color:#afc1d4;padding:3px 6px;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#board.expanded {grid-template:1fr / 1fr!important}
#board.expanded article:not(.selected) {display:none}
#board.expanded h2 {font-size:16px;cursor:zoom-out}
#board-controls {position:fixed;inset:0 0 auto;z-index:202;min-height:70px;padding:6px 12px;box-sizing:border-box;background:#101820;display:flex;flex-wrap:wrap;gap:7px 16px;align-items:center}
#board-controls button,#board-controls select {background:#26374a;color:#eef;border:1px solid #627487;border-radius:4px;padding:3px 9px}
#board-controls input[type=range] {flex:1;min-width:100px}
#board-controls small {width:100%;color:#aabcce;font-size:11px}
`;
document.head.append(labStyle);
const labPanel = document.createElement('header');
labPanel.id = 'board-controls';
labPanel.innerHTML = `<b>Local joins · all ${labCases.length} examples</b><button id="board-play">Pause</button><button id="board-reset">Restart</button><input aria-label="Print progress" id="board-progress" type="range" min="0" max="100" step=".1" value="0"><output id="board-percent">0%</output><label><input id="board-maps" type="checkbox" checked>Material maps</label><label><input id="board-wire" type="checkbox">Wireframe</label><select id="board-angle" aria-label="Camera view"><option value="perspective">Perspective</option><option value="top">Top</option><option value="side">Side</option></select><button id="board-all">All examples</button><small id="board-status">Loading geometry…</small>`;
document.body.append(labPanel);
const labBoard = document.createElement('main');
labBoard.id = 'board';
document.body.append(labBoard);
for (const m of labMessages) {
    await labViewer.handleMessage(m);
    if (m.id && labViewer._inflightLoads.has(m.id)) await labViewer._inflightLoads.get(m.id).promise;
}
cancelAnimationFrame(labViewer._animationFrameId);
labViewer._controls.enabled = false;
const labRenderer = labViewer._renderer;
labRenderer.domElement.id = 'board-canvas';
document.body.append(labRenderer.domElement);
labRenderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
labRenderer.shadowMap.enabled = false;
labRenderer.autoClear = false;
const lab$ = id => document.getElementById('board-' + id);
const labItems = [];
let labTime = 0, labPlaying = true, labLast = performance.now();
function labLayout() {
    const top = labPanel.getBoundingClientRect().bottom + 8;
    labBoard.style.top = `${top}px`;
    const cols = Math.max(1, Math.ceil(Math.sqrt(labCases.length * innerWidth / Math.max(1, innerHeight - top - 8) / 2.3)));
    labBoard.style.gridTemplateColumns = `repeat(${cols},minmax(0,1fr))`;
    labBoard.style.gridTemplateRows = `repeat(${Math.ceil(labCases.length / cols)},minmax(0,1fr))`;
    labRenderer.setSize(innerWidth, innerHeight);
}
function labCamera(item) {
    const direction = lab$('angle').value === 'top' ? new THREE.Vector3(0,0,1) :
        lab$('angle').value === 'side' ? new THREE.Vector3(0,-1,0) : new THREE.Vector3(.75,-1,.9).normalize();
    item.camera.up.set(0, lab$('angle').value === 'top' ? 1 : 0, lab$('angle').value === 'top' ? 0 : 1);
    item.camera.position.copy(item.center).addScaledVector(direction, item.distance);
    item.camera.lookAt(item.center);
    for (const control of item.controls) control.target.copy(item.center);
}
for (const entry of labCases) {
    const card = document.createElement('article');
    card.dataset.case = entry.name;
    card.innerHTML = `<h2 title="Click to enlarge">${entry.title}</h2><div class="pair"><div class="view"><span class="label">Before · original</span></div><div class="view"><span class="label">After · local joins</span></div></div><div class="stats"></div>`;
    labBoard.append(card);
    card.querySelector('h2').onclick = () => {
        const expand = !card.classList.contains('selected');
        labBoard.querySelectorAll('.selected').forEach(c => c.classList.remove('selected'));
        labBoard.classList.toggle('expanded', expand);
        card.classList.toggle('selected', expand);
    };
    const objects = ['rings','local_join'].map(mode => labViewer._objects.get(entry.name + '_' + mode));
    const bounds = new THREE.Box3();
    for (const object of objects) {object.visible = true;bounds.union(new THREE.Box3().setFromObject(object));}
    const center = bounds.getCenter(new THREE.Vector3());
    const radius = Math.max(1, bounds.getSize(new THREE.Vector3()).length() / 2);
    const item = {name:entry.name, card, center, distance:radius * 3, radius, camera:new THREE.PerspectiveCamera(40,1,.01,radius*100), controls:[], views:[]};
    for (let k = 0; k < 2; k++) {
        const element = card.querySelectorAll('.view')[k], object = objects[k];
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x202b38);
        scene.environment = labViewer._scene.environment;
        scene.environmentIntensity = .6;
        scene.add(new THREE.HemisphereLight(0xe7f3ff,0x485466,2));
        const light = new THREE.DirectionalLight(0xffffff,2.5);
        light.position.copy(center).add(new THREE.Vector3(radius,-radius,radius*2));
        light.target.position.copy(center);scene.add(light,light.target,object);
        const plain = new THREE.MeshStandardMaterial({color:0x7ab8cc,roughness:.8,side:THREE.DoubleSide});
        item.views.push({element,scene,object,mapped:object.material,plain});
        const control = new LabOrbitControls(item.camera,element);
        control.enablePan = false;control.enableDamping = false;
        item.controls.push(control);
    }
    labCamera(item);
    const stats = objects[1].userData.localJoinStats;
    card.querySelector('.stats').textContent = `${stats.originalTriangles.toLocaleString()} → ${(stats.originalTriangles+stats.addedTriangles).toLocaleString()} triangles · ${stats.accepted} repairs`;
    card.querySelector('.stats').title = `Fallbacks: ${JSON.stringify(stats.fallback)}`;
    labItems.push(item);
}
function labMaterials() {
    for (const item of labItems) for (const view of item.views) {
        view.object.material = lab$('maps').checked ? view.mapped : view.plain;
        view.object.material.wireframe = lab$('wire').checked;
        if (view.object.userData.localJoinCompleted) view.object.userData.localJoinCompleted.material = view.object.material;
    }
}
lab$('maps').onchange = labMaterials;lab$('wire').onchange = labMaterials;
lab$('angle').onchange = () => labItems.forEach(labCamera);
function labSetPlaying(value) {labPlaying=value;lab$('play').textContent=value?'Pause':'Play';}
lab$('play').onclick = () => labSetPlaying(!labPlaying);
lab$('reset').onclick = () => {labTime=0;labSetPlaying(true);};
lab$('progress').oninput = () => {labTime=Number(lab$('progress').value)/100*10;labSetPlaying(false);};
lab$('all').onclick = () => {labBoard.classList.remove('expanded');labBoard.querySelectorAll('.selected').forEach(c=>c.classList.remove('selected'));};
addEventListener('keydown',event=>{if(event.key==='Escape')lab$('all').click();});
addEventListener('resize',labLayout);
labLayout();
lab$('status').textContent = 'Shared 0–100% playback · drag either side to orbit both; scroll to zoom · click a title to enlarge · original collapse off; unsupported repairs retain original geometry.';
function labTick(now) {
    if (labPlaying) labTime=(labTime+Math.min(.1,(now-labLast)/1000))%12;
    labLast=now;
    const fraction=Math.min(1,labTime/10);
    lab$('progress').value=String(fraction*100);lab$('percent').textContent=`${Math.round(fraction*100)}%`;
    labRenderer.setScissorTest(false);labRenderer.setClearColor(0x151c24);labRenderer.clear();
    labRenderer.setScissorTest(true);
    for (const item of labItems) {
        for (const view of item.views) view.object.userData.parametricTube.setDrawRange(fraction);
        for (const view of item.views) {
            const rect=view.element.getBoundingClientRect();
            if (rect.width<1 || rect.height<1) continue;
            item.camera.aspect=rect.width/rect.height;
            // Keep the complete path in frame at both narrow and wide tile sizes.
            item.camera.fov=THREE.MathUtils.radToDeg(2*Math.atan(Math.tan(20*Math.PI/180)/Math.min(1,item.camera.aspect)));
            item.camera.updateProjectionMatrix();
            labRenderer.setViewport(rect.left,innerHeight-rect.bottom,rect.width,rect.height);
            labRenderer.setScissor(rect.left,innerHeight-rect.bottom,rect.width,rect.height);
            labRenderer.render(view.scene,item.camera);
        }
    }
    labRenderer.setScissorTest(false);
    window.labProgress=fraction;
    requestAnimationFrame(labTick);
}
window.labItems=labItems;
window.demoReady=true;
requestAnimationFrame(labTick);
