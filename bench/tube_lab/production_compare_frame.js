// Injected after initialization into each unmodified pinned viewer build.
const cmpBuild = "__BUILD__";
const cmpViewer = window.threejsViewer;
cancelAnimationFrame(cmpViewer._animationFrameId);
cmpViewer._controls.enabled = false;
const { OrbitControls: CompareOrbitControls } = await import('three/addons/controls/OrbitControls.js');
const cmpRenderer = cmpViewer._renderer;
cmpRenderer.setPixelRatio(1);
cmpRenderer.shadowMap.enabled = false;
const cmpStyle = document.createElement('style');
cmpStyle.textContent = 'body{margin:0;background:#202b38}#toolbar,#status-bar,.toolbar,.status-bar{display:none!important}body>*:not(#viewer-container):not(script):not(style){display:none!important}#viewer-container{position:fixed!important;inset:0!important;width:100%!important;height:100%!important}';
document.head.append(cmpStyle);
// Hide viewer UI without relying on the different versions' UI selectors.
const cmpCanvas = cmpRenderer.domElement;
document.body.append(cmpCanvas);
cmpCanvas.style.cssText='position:fixed;inset:0;width:100%;height:100%;display:block!important;z-index:100000';
const cmpScene = new THREE.Scene();
cmpScene.background = new THREE.Color(0x202b38);
cmpScene.add(new THREE.HemisphereLight(0xe7f3ff,0x485466,2));
const cmpLight = new THREE.DirectionalLight(0xffffff,2.5);
cmpScene.add(cmpLight, cmpLight.target);
const cmpCamera = cmpViewer._camera;
const cmpControls = new CompareOrbitControls(cmpCamera,cmpCanvas);
cmpControls.enableDamping = false;
let cmpObject = null, cmpMuted = false, cmpResult = null, cmpBounds = null;
function cmpRender() {
    cmpRenderer.setSize(innerWidth, innerHeight, false);
    if(cmpBounds) {
        // Scale depth precision to the actual view. A fixed microscopic near
        // plane made distinct bead surfaces fight in every pinned renderer.
        const distance=cmpCamera.position.distanceTo(cmpControls.target);
        cmpCamera.near=Math.max(1e-8,distance*.01);
        cmpCamera.far=Math.max(cmpCamera.near*100,
            cmpCamera.position.distanceTo(cmpBounds.center)+cmpBounds.radius*1.5);
    }
    cmpCamera.aspect = innerWidth / Math.max(1, innerHeight);
    cmpCamera.updateProjectionMatrix();
    cmpRenderer.render(cmpScene,cmpCamera);
}
function cmpCameraState() {return {position:cmpCamera.position.toArray(),target:cmpControls.target.toArray(),up:cmpCamera.up.toArray()};}
function cmpSetCamera(state) {
    cmpMuted=true;
    cmpCamera.position.fromArray(state.position);cmpCamera.up.fromArray(state.up);
    cmpControls.target.fromArray(state.target);cmpCamera.lookAt(cmpControls.target);cmpControls.update();
    cmpMuted=false;cmpRender();
}
cmpControls.addEventListener('change',()=>{
    cmpRender();
    if(!cmpMuted)parent.postMessage({type:'compare-camera',build:cmpBuild,camera:cmpCameraState()},location.origin);
});
addEventListener('resize',cmpRender);
function cmpPacked(t) {
    const channel=x=>Math.floor(255*Math.max(0,Math.min(1,1.5-Math.abs(x))));
    return (channel(4*t-3)<<16)|(channel(4*t-2)<<8)|channel(4*t-1);
}
async function cmpLoad({entry, mode, shading, lod, camera, wire}) {
    const colored = !['texture','plain'].includes(mode);
    cmpBounds={center:new THREE.Vector3().fromArray(entry.presets[0].center),radius:entry.presets[0].radius};
    cmpSetCamera(camera);
    const response=await fetch(entry.blob);if(!response.ok)throw Error(`Blob HTTP ${response.status}`);
    const original=await response.arrayBuffer(), n=entry.points;
    let payload=original;
    const ranges=entry.ranges[mode]||[0,1];
    let scalars=null;
    if(mode==='layer') {
        if(!entry.scalars)return {unsupported:true,reason:'No layer IDs recorded for this historical example.'};
        const res=await fetch(entry.scalars);if(!res.ok)throw Error('Missing layer metadata');
        scalars=new Float32Array(await res.arrayBuffer(),0,n);
    }
    if(colored) {
        payload=new ArrayBuffer(original.byteLength+4*n);
        const bytes=new Uint8Array(payload),src=new Uint8Array(original);
        bytes.set(src.subarray(0,n*20));bytes.set(src.subarray(n*20),n*24);
        const packed=new Uint32Array(payload,n*20,n);
        const data=mode==='elevation'?new Float32Array(original,0,n*3):mode==='width'?new Float32Array(original,n*12,n):mode==='height'?new Float32Array(original,n*16,n):scalars;
        for(let i=0;i<n;i++) {
            let t;
            if(mode==='contrast')t=i%2; // Diagnostic, deliberately harsher than most data fields.
            else if(mode==='station')t=i/Math.max(1,n-1);
            else {const value=data[mode==='elevation'?i*3+2:i]*(mode==='layer'?1:1000);t=(value-ranges[0])/(ranges[1]-ranges[0]||1);}
            packed[i]=cmpPacked(Math.max(0,Math.min(1,t)));
        }
        // The only modified bytes are the added optional colour block.
        if(!src.subarray(0,n*20).every((b,i)=>bytes[i]===b) || !src.subarray(n*20).every((b,i)=>bytes[n*24+i]===b))throw Error('Geometry preservation check failed');
    }
    const url=URL.createObjectURL(new Blob([payload]));
    const msg={...entry.meta,type:'add_parametric_tube_binary',id:'compare',blob_url:url,hasColors:colored,
        lod:lod==='off'?false:entry.meta.lod,color:colored||mode==='texture'?0xffffff:0x7ab8cc,metalness:0,roughness:1};
    if(cmpBuild==='local')msg.surface='local_join';
    if(cmpBuild==='round')msg.joins='round';
    if(mode==='texture')msg.texture={url:new URL('color.png',location.href).href,normalUrl:new URL('normal.png',location.href).href,roughnessUrl:new URL('roughness.png',location.href).href,length:.025};
    const start=performance.now();
    try {
        await cmpViewer.handleMessage(msg);
        const pending=cmpViewer._inflightLoads.get('compare');if(pending)await pending.promise;
    } finally {URL.revokeObjectURL(url);}
    cmpObject=cmpViewer._objects.get('compare');
    if(!cmpObject)throw Error('Viewer did not create the tube; inspect console');
    const buildMs=performance.now()-start;
    cmpScene.add(cmpObject);
    if(colored && shading==='flat') {
        const basic=new THREE.MeshBasicMaterial({vertexColors:true,color:0xffffff,side:THREE.DoubleSide,toneMapped:false});
        cmpObject.material=Array.isArray(cmpObject.material)?[basic]:basic;
        cmpObject.userData.tubeBaseMaterial=basic;
    }
    cmpObject.traverse(obj=>{
        if(obj.isMesh)for(const mat of Array.isArray(obj.material)?obj.material:[obj.material])mat.wireframe=wire;
    });
    const center=new THREE.Vector3().fromArray(entry.presets[0].center), r=entry.presets[0].radius;
    cmpLight.position.copy(center).add(new THREE.Vector3(r,-r,2*r));cmpLight.target.position.copy(center);
    const buffers=new Set();let bytes=0,triangles=0;
    cmpObject.traverse(obj=>{if(obj.isMesh){const g=obj.geometry;triangles+=(g.index?.count||0)/3;for(const a of [g.index,...Object.values(g.attributes)].filter(Boolean)){if(!buffers.has(a.array.buffer)){buffers.add(a.array.buffer);bytes+=a.array.byteLength;}}}});
    const u=cmpObject.userData;
    cmpResult={build:cmpBuild,buildMs,geometryBytes:bytes,allocatedTriangles:triangles,stations:u.tubeNumSpinePoints,
        joints:u.tubeJointVerts?.length||0,repairs:u.localJoinStats?.accepted||0,repairStats:u.localJoinStats||null,
        colorVertices:cmpObject.geometry.getAttribute('color')?.count||0,
        patchColorVertices:u.localJoinCompleted?.geometry.getAttribute('color')?.count||0,maps:!!u.tubeBaseMaterial.map,
        normalMap:!!u.tubeBaseMaterial.normalMap,roughnessMap:!!u.tubeBaseMaterial.roughnessMap,geometryPreserved:true};
    cmpSetCamera(camera);return cmpResult;
}
window.compareFrame={load:cmpLoad,camera:cmpSetCamera,progress(f){if(cmpObject){cmpObject.userData.parametricTube.setDrawRange(f);cmpRender();}},
    stats(){return {...cmpResult,drawCalls:cmpRenderer.info.render.calls,submittedTriangles:cmpRenderer.info.render.triangles,contextLost:cmpRenderer.getContext().isContextLost(),cameraNear:cmpCamera.near,cameraFar:cmpCamera.far,cameraDistance:cmpCamera.position.distanceTo(cmpControls.target)};}};
window.compareReady=true;
