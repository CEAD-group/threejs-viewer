// Deterministic robustness/performance checks without a browser or Three.js.
// Run: node bench/mesh_cap_review/slice_bench.mjs [path/to/viewer.js]
import fs from 'node:fs';
import assert from 'node:assert/strict';
const factory = fs.readFileSync(new URL('../../src/threejs_viewer/viewer/polygon-clipping.js', import.meta.url), 'utf8');
const createPolygonClipper = new Function(factory.replace('export function ', 'function ') + '\nreturn createPolygonClipper;')();
const path = process.argv[2] || new URL('../../src/threejs_viewer/viewer/viewer.js', import.meta.url);
const source = fs.readFileSync(path, 'utf8');
const names = ['clipCapBasis', 'earcut2D', 'sliceMeshClipCap'];
const code = names.map(name => {
    const start = source.indexOf(`function ${name}(`);
    assert(start >= 0, name);
    return source.slice(start, source.indexOf('\n}', start) + 2);
}).join('\n');
const clipper = createPolygonClipper();
const slice = new Function('meshCapPolygons', `const MESH_CAP_HOLE_RATIO=.25, MESH_CAP_BOUNDARY_RATIO=.05;\n${code}\nreturn sliceMeshClipCap;`)(clipper);
function prism(src, n, x, y, radius=1, angle=0, reversed=false) {
    const base = src.pos.length/3;
    for (const z of [-1,1]) for(let i=0;i<n;i++) {
        const t=angle+2*Math.PI*i/n;
        const px=x+radius*Math.cos(t), py=y+radius*Math.sin(t);
        src.pos.push(px,py,z); src.col?.push((px+5)/10,.5,(py+5)/10);
    }
    const tri=(a,b,c)=>src.idx.push(base+a,base+(reversed?c:b),base+(reversed?b:c));
    for(let i=0;i<n;i++) {const j=(i+1)%n;tri(i,j,n+j);tri(i,n+j,n+i);}
    for(let i=1;i<n-1;i++){tri(0,i+1,i);tri(n,n+i,n+i+1);}
}
const run = src => slice(src,0,src.idx.length,[[0,0,1,.123]]);
const area = cap => {
    let total=0;const p=cap.pos;
    for(let i=0;i<p.length;i+=9)total+=Math.abs((p[i+3]-p[i])*(p[i+7]-p[i+1])-(p[i+6]-p[i])*(p[i+4]-p[i+1]))/2;
    return total;
};
for(const n of [1024,4096]){
    const src={pos:[],idx:[],col:[]};prism(src,n,0,0);prism(src,n,1,0);
    run(src); const start=performance.now();const [cap]=run(src);const ms=performance.now()-start;
    assert(cap.pos.length>0 && Math.abs(area(cap)-(2*Math.PI-2*Math.acos(.5)+Math.sqrt(3)/2))<1e-4);
    let error=0;for(let i=0;i<cap.pos.length;i+=3)error=Math.max(error,Math.abs(cap.col[i]-(cap.pos[i]+5)/10));
    assert(error<1e-5,`color error ${error}`);
    console.log(JSON.stringify({case:'colored overlapping cylinders',segments:n,ms,area:area(cap),error}));
}
{
    const src={pos:[],idx:[],col:null};
    for(let i=0;i<40;i++)prism(src,256,0,0,4-i*.075,0,i%2===1);
    run(src);const start=performance.now();const [cap]=run(src);
    const ms=performance.now()-start;
    let expected=0;for(let i=0;i<40;i++)expected+=(i%2?-1:1)*(4-i*.075)**2*128*Math.sin(2*Math.PI/256);
    assert(Math.abs(area(cap)-expected)<1e-4);
    console.log(JSON.stringify({case:'40 nested rings',ms,area:area(cap)}));
}
// Inject a sweep-line failure: all original regions must survive, with a
// warning that the controller can report in both main-thread and worker paths.
if(!process.argv[2]){
    const original=clipper.difference;clipper.difference=()=>{throw Error('injected sweep-line failure');};
    const src={pos:[],idx:[],col:null};prism(src,4,0,0);prism(src,4,1,0);
    const [cap]=run(src);assert.equal(area(cap),4);assert.match(cap.warning,/unsubtracted/);
    clipper.difference=original;console.log('Fallback retains both caps and reports its warning.');
}
let seed=321;const random=()=>((seed=(1664525*seed+1013904223)>>>0)/2**32);
let warnings=0;
for(let i=0;i<300;i++){
    const src={pos:[],idx:[],col:null};const n=2+Math.floor(random()*9);
    for(let j=0;j<n;j++){
        const jitter=[0,1e-9,1e-7,1e-6][i%4];
        prism(src,4+Math.floor(random()*7),Math.round(random()*24)/8+jitter,Math.round(random()*24)/8-jitter,
              .5+random(),random()*Math.PI);
    }
    const norm=Math.hypot(.13,.27,1);
    const [cap]=slice(src,0,src.idx.length,[[.13/norm,.27/norm,1/norm,.123]]);
    assert(cap.pos.length && cap.pos.every(Number.isFinite),`empty/nonfinite scene ${i}`);
    if(cap.warning)warnings++;
}
console.log(`300 seeded tilted-prism scenes: no lost caps; ${warnings} fallback warnings.`);
