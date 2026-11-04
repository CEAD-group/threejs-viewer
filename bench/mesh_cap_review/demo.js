// Development-only probes deliberately inspect cap internals to reproduce the
// cache invalidation and rebaking regressions. Scene loading uses handleMessage.
(async () => {
    const wait = async (predicate) => {
        const deadline = performance.now() + 45000;
        while (!predicate()) {
            if (performance.now() > deadline) throw new Error('Demo setup timed out');
            await new Promise(requestAnimationFrame);
        }
    };
    const frames = async () => {
        await new Promise(requestAnimationFrame);
        await new Promise(requestAnimationFrame);
    };
    await wait(() => window.threejsViewer);
    const v = window.threejsViewer;
    const q = new URLSearchParams(location.search);
    window.__clipCapWorkerTriangles = q.get('worker') === '1' ? 0 : null;
    const [title, description, expected] = DEMO.cases[DEMO.case];
    const style = document.createElement('style');
    style.textContent = `
        #review-header,#review-footer {position:fixed;left:0;right:0;z-index:100;
            background:#141a22ed;color:#e6edf4;font:14px system-ui;padding:18px 28px}
        #review-header {top:0} #review-footer {bottom:0}
        #review-header strong {font-size:24px;display:block;margin:9px 0}
        #review-header p {margin:5px 0;color:#aab7c5}
        #review-header a,#review-footer a {color:#8ac9ff;margin-right:15px;line-height:1.8}
        .review-small {font-size:12px;color:#aab7c5}
    `;
    document.head.appendChild(style);
    const header = document.createElement('header'); header.id = 'review-header';
    header.innerHTML = `<span class="review-small">PR #331 · ${DEMO.variant.toUpperCase()} · ${q.get('worker') === '1' ? 'Worker' : 'Main thread'}</span>
        <strong>${title}</strong><p>${description}</p>`;
    if (!q.has('capture')) {
        const nav = document.createElement('nav');
        for (const [name, [label]] of Object.entries(DEMO.cases)) {
            const a = document.createElement('a');
            const next = new URLSearchParams(q); next.set('case', name);
            a.href = '?' + next; a.textContent = label; nav.appendChild(a);
        }
        header.appendChild(nav);
    }
    document.body.appendChild(header);
    const footer = document.createElement('footer'); footer.id = 'review-footer';
    footer.textContent = 'Loading GLB and building section caps…';
    document.body.appendChild(footer);
    v.handleMessage({type:'set_background', color:'#20252b'});
    v.handleMessage({type:'show_grid', visible:false});
    v.handleMessage({type:'set_sun', enabled:false});
    v.handleMessage({type:'add_object', id:'case', object:{model:`/fixtures/${DEMO.case}.glb`, format:'glb'}});
    await wait(() => v.getObject('case') && v._pendingFetches === 0);
    const root = v.getObject('case');
    const meshes = [];
    root.traverse(n => { if (n.isMesh) meshes.push(n); });
    if (DEMO.case === 'groups') {
        for (const n of meshes) {
            const other = n.material.clone(); other.map = n.material.map.clone();
            const canvas = document.createElement('canvas'); canvas.width = canvas.height = 8;
            const ctx = canvas.getContext('2d'); ctx.fillStyle = '#d0a040'; ctx.fillRect(0,0,8,8);
            other.map.source = new n.material.map.source.constructor(canvas); other.map.needsUpdate = true;
            n.material = [n.material, other];
            n.geometry.clearGroups(); n.geometry.addGroup(0,n.geometry.index.count,0);
        }
    }
    v.handleMessage({type:'set_clipping_caps', enabled:true});
    v.handleMessage({type:'set_clipping_slab',normal:[0,0,1],center:0,thickness:0.8,show_helper:false});
    v.getMenu('clip').close();
    const wide = DEMO.case === 'layouts';
    const distance = wide ? 25 : DEMO.case === 'hollow' ? 17 : 13;
    const x = ['overlap','collinear'].includes(DEMO.case) ? 0.5 : 0;
    v.handleMessage({type:'set_camera',position:[x,-distance*0.36,distance],target:[x,0,0],up:[0,0,1],fov:25});
    await frames(); await wait(() => v._clipCaps.settled()); await frames();
    if (DEMO.case === 'replacement') {
        for (const n of meshes) {
            const p = n.geometry.getAttribute('position').clone();
            for (let i=0; i<p.count; i++) p.setX(i,p.getX(i)*2);
            n.geometry.setAttribute('position',p); n.geometry.computeBoundingSphere();
        }
    }
    if (DEMO.case === 'groups') {
        for (const n of meshes) n.geometry.groups[0].materialIndex = 1;
    }
    await frames(); await wait(() => v._clipCaps.settled()); await frames();
    // Each slab has two caps: sum their areas and divide by two. World-space
    // coordinates account for the mirrored GLB node in the layout fixture.
    let area = 0; let count = 0; const colors = [];
    root.updateWorldMatrix(true,true);
    root.traverse(n => {
        if (!n.userData.__clipCap || !n.visible) return;
        count++;
        const p = n.geometry.getAttribute('position');
        const a = n.position.clone(), b = a.clone(), c = a.clone();
        for (let i=0;i<p.count;i+=3) {
            a.fromBufferAttribute(p,i).applyMatrix4(n.matrixWorld);
            b.fromBufferAttribute(p,i+1).applyMatrix4(n.matrixWorld);
            c.fromBufferAttribute(p,i+2).applyMatrix4(n.matrixWorld);
            area += b.sub(a).cross(c.sub(a)).length()/2;
        }
        const ca = n.geometry.getAttribute('color');
        if (ca) colors.push([ca.getX(0),ca.getY(0),ca.getZ(0)]);
    });
    area /= 2;
    let pass = count > 0 && Math.abs(area-expected)<1e-4;
    const linear = n => ((n/255+0.055)/1.055)**2.4;
    const rgb = DEMO.case === 'texture' ? [190,120,65] : DEMO.case === 'groups' ? [208,160,64] : DEMO.case === 'uv1' ? [224,208,192] : null;
    if (rgb) pass = pass && colors.length > 0 && colors.every(c => c.every((n,i) => Math.abs(n-linear(rgb[i])*(DEMO.case==='texture'?0.5:1))<1e-4));
    window.demoStats = {area, expected, capCount:count, colors, pass};
    footer.innerHTML = `<strong>Measured cap area: ${area.toFixed(2)} · expected ${expected}</strong>
        <span style="float:right;color:${pass?'#81dcad':'#ffb18b'}">${pass?'Checks pass':'Regression reproduced'}</span>`;
    if (!q.has('capture')) {
        const worker = new URLSearchParams(q); worker.set('worker',q.get('worker')==='1'?'0':'1');
        footer.innerHTML += `<br><a href="/fixtures/${DEMO.case}.glb" download>Download GLB</a>
            <a href="?${worker}">Switch execution path</a><span class="review-small">Drag to orbit · C for clipping settings · reload to reset</span>`;
        if (DEMO.before) {
            const variant = new URLSearchParams(q); variant.set('variant',DEMO.variant==='after'?'before':'after');
            footer.innerHTML += `<a href="?${variant}">Compare ${DEMO.variant==='after'?'before':'after'}</a>`;
        }
    } else document.querySelectorAll('.tjsv-rail').forEach(n => n.style.display='none');
    window.demoReady = true;
})().catch(e => {window.demoError=String(e); console.error(e);});
