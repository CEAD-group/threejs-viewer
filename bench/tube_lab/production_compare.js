const $=id=>document.getElementById(id);
const manifest=await(await fetch('manifest.json')).json();
const params=new URLSearchParams(location.search);
for(const c of manifest.cases){const o=document.createElement('option');o.value=c.key;o.textContent=`${c.key.replace('captures/','')} · ${c.points.toLocaleString()} pts`;$('case').append(o);}
$('case').value=params.get('case')||'captures/real/boat_v03_mini';
for(const id of ['mode','shading','angle','lod'])if(params.has(id))$(id).value=params.get(id);
let generation=0,frames=[],playing=false,last=0,progress=1,camera=null;
const entry=()=>manifest.cases.find(c=>c.key===$('case').value);
function focusOptions(){ $('focus').replaceChildren();entry().presets.forEach((p,i)=>{const o=document.createElement('option');o.value=String(i);o.textContent=p.name;$('focus').append(o);});}
focusOptions();if(params.has('focus'))$('focus').value=params.get('focus');
function defaultCamera(){const p=entry().presets[Number($('focus').value)],dir=$('angle').value==='top'?[0,0,1]:$('angle').value==='side'?[0,-1,0]:[.75,-1,.9];const len=Math.hypot(...dir);return{position:p.center.map((v,i)=>v+dir[i]/len*p.radius*4),target:p.center,up:$('angle').value==='top'?[0,1,0]:[0,0,1]};}
function legend(){const mode=$('mode').value,range=entry().ranges[mode]||[0,1];$('legend').hidden=['texture','plain'].includes(mode);$('minimum').textContent=range[0].toLocaleString(undefined,{maximumFractionDigits:3});$('maximum').textContent=range[1].toLocaleString(undefined,{maximumFractionDigits:3});}
function setProgress(f){progress=f;$('progress').value=String(f);$('percent').textContent=`${(f*100).toFixed(2)}%`;for(const v of frames)if(v.ready)v.frame.contentWindow.compareFrame.progress(f);}
function pause(){playing=false;$('play').textContent='Play';}
async function rebuild(resetCamera=true){
    const gen=++generation;pause();window.comparisonReady=false;window.comparisonResults=[];
    if(resetCamera)camera=defaultCamera();
    const c=entry(),preset=c.presets[Number($('focus').value)];
    if(resetCamera)progress=preset.progress;
    // Releasing old contexts before loading the next case matters for full hulls.
    for(const v of frames){try{v.frame.contentWindow.threejsViewer?._renderer.dispose();v.frame.contentWindow.threejsViewer?._renderer.forceContextLoss();}catch{}}
    frames=[];$('views').replaceChildren();$('error').textContent='';legend();
    $('description').textContent=`${c.meta.description} · ${c.breaks.toLocaleString()} breaks${preset.point!==undefined?` · focus station ${preset.point}, MPF line ${preset.line??'n/a'}`:''}`;
    $('provenance').textContent=`${c.group==='captures'?'Original production blob':'Historical example'} · metres · SHA-256 ${c.sha256} · geometry unchanged; data modes add a colour block`;
    const query=new URLSearchParams();for(const id of ['case','mode','shading','angle','lod','focus'])query.set(id,$(id).value);history.replaceState(null,'','?'+query);
    // Load sequentially to avoid simultaneous allocation peaks, then animate together.
    for(const b of manifest.builds){
        if(gen!==generation)return;
        const article=document.createElement('article');article.innerHTML=`<h2>${b.label} · ${b.rev.slice(0,7)}</h2><iframe title="${b.label}"></iframe><div class="message">Loading…</div><div class="stats">Waiting for geometry</div>`;$('views').append(article);
        const frame=article.querySelector('iframe'),message=article.querySelector('.message'),stats=article.querySelector('.stats');
        const v={frame,ready:false,build:b.key};frames.push(v);
        try{
            frame.src=b.key+'.html';
            const deadline=performance.now()+120000;
            while(!frame.contentWindow.compareReady){if(gen!==generation)return;if(performance.now()>deadline)throw Error('Viewer initialization timed out');await new Promise(r=>setTimeout(r,40));}
            const result=await frame.contentWindow.compareFrame.load({entry:c,mode:$('mode').value,shading:$('shading').value,lod:$('lod').value,camera,wire:$('wire').checked});
            if(gen!==generation)return;
            window.comparisonResults.push({build:b.key,...result});
            if(result.unsupported){message.textContent=result.reason;stats.textContent='Unsupported · no comparison geometry';}
            else{v.ready=true;message.hidden=true;stats.textContent=`${result.stations.toLocaleString()} stations · ${result.allocatedTriangles.toLocaleString()} allocated triangles · ${(result.geometryBytes/1e6).toFixed(2)} MB buffers · ${result.joints||result.repairs} ${b.key==='round'?'joints':'repairs'}`;stats.title=`Load/build ${result.buildMs.toFixed(1)} ms (not a performance benchmark)`;}
        }catch(error){if(gen!==generation)return;message.textContent=String(error);$('error').textContent='A view failed to load. See its error.';window.comparisonResults.push({build:b.key,error:String(error)});}
    }
    setProgress(progress);window.comparisonReady=true;
}
$('case').onchange=()=>{focusOptions();rebuild();};
for(const id of ['mode','shading','lod','wire'])$(id).onchange=()=>rebuild(false);
$('focus').onchange=()=>rebuild();
$('angle').onchange=()=>{camera=defaultCamera();for(const v of frames)v.frame.contentWindow.compareFrame?.camera(camera);};
$('reload').onclick=()=>rebuild(false);
$('progress').oninput=()=>{pause();setProgress(Number($('progress').value));};
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pause':'Play';if(playing&&progress>=1)progress=0;last=performance.now();};
addEventListener('message',event=>{if(event.origin!==location.origin||event.data.type!=='compare-camera')return;if(!frames.some(v=>v.frame.contentWindow===event.source))return;camera=event.data.camera;for(const v of frames)if(v.frame.contentWindow!==event.source)v.frame.contentWindow.compareFrame?.camera(camera);});
function tick(now){if(playing){setProgress((progress+Math.min(.1,(now-last)/1000)/15)%1);}last=now;requestAnimationFrame(tick);}requestAnimationFrame(tick);
window.comparison={manifest,rebuild,setProgress,pause,getCamera:()=>camera,stats:()=>frames.filter(v=>v.ready).map(v=>v.frame.contentWindow.compareFrame.stats())};
await rebuild();
