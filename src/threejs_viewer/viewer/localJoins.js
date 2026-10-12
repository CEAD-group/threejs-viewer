// @ts-check
import * as THREE from 'three';

/**
 * Direct, bounded round joins. No unions and no prefix remeshing. Each accepted
 * corner replaces two strip pairs. Unchanged attributes are shared with the
 * existing renderer; it continues to own the moving frontier and its cap.
 */
export function installLocalJoins(mesh, options) {
    if (!options.up || options.up.length !== 3 || !options.up.every(Number.isFinite) ||
        Math.hypot(...options.up) < 1e-12 || (options.tolerance != null &&
        (!Number.isFinite(options.tolerance) || options.tolerance <= 0))) {
        throw new Error('local_join requires a finite nonzero up vector and positive tolerance');
    }
    const started = performance.now(), ud = mesh.userData, md = ud.tubeMorphData;
    const base = ud.parametricTube, geometry = mesh.geometry;
    const n = ud.tubeNumSpinePoints, cs = ud.tubeNCs, cap = ud.tubeCapIndicesPerCap;
    const pairSize = ud.tubeIndicesPerRingPair, originalIndex = geometry.index.array;
    const attrs = geometry.attributes, added = {};
    for (const name of Object.keys(attrs)) added[name] = [];
    const originalVertices = attrs.position.count;
    // Allocate compact prefix tables only after a repair is accepted. A long
    // path with one defect must not allocate one JS object per original pair.
    /** @type {Uint32Array | null} */
    let indexData = null;
    /** @type {Uint32Array | null} */
    let blockEnds = null;
    /** @type {Uint32Array | null} */
    let blockCounts = null;
    let indexCount = cap, blockCount = 0;
    const stats = { accepted: 0, clusters: 0, planar: 0, spatial: 0, smooth: 0, fallback: {},
        originalTriangles: originalIndex.length / 3, addedTriangles: 0, maxPatchTriangles: 0,
        buildMs: 0, maxWindowSegments: 2, triangleBudget: Math.max(96, Math.ceil(originalIndex.length / 3 * .15)) };
    const reject = reason => { stats.fallback[reason] = (stats.fallback[reason] || 0) + 1; };
    const point = i => new THREE.Vector3().fromArray(md.spine, 3 * i);
    const up = new THREE.Vector3(...options.up).normalize();
    let nextPair = 0;
    const appendOriginal = end => {
        if (!indexData) {
            indexData = new Uint32Array(originalIndex.length - cap + 3 * stats.triangleBudget);
            indexData.set(originalIndex.subarray(0, cap));
            blockEnds = new Uint32Array(n - 1); blockCounts = new Uint32Array(n - 1);
        }
        indexData.set(originalIndex.subarray(cap + nextPair * pairSize, cap + end * pairSize), indexCount);
        while (nextPair < end) {
            indexCount += pairSize;
            blockEnds[blockCount] = ++nextPair; blockCounts[blockCount++] = indexCount;
        }
    };
    const tryCluster = i => {
        const lo = Math.max(nextPair, i - 2);
        let hi = Math.min(n - 2, i + 6);
        let patch = localCluster(mesh, options, lo, hi, originalVertices + added.position.length / 3);
        if (!patch) {
            hi = Math.min(n - 2, i + 8);
            patch = localCluster(mesh, options, lo, hi, originalVertices + added.position.length / 3);
        }
        if (!patch) return false;
        if (stats.addedTriangles + patch.indices.length / 3 - (hi-lo)*pairSize/3 > stats.triangleBudget) {
            reject('total triangle budget'); return false;
        }
        appendOriginal(lo);
        for (const name of Object.keys(attrs)) {
            const attr = attrs[name];
            for (let k = 0; k < patch.positions.length / 3; k++) {
                if (name === 'position' || name === 'normal') {
                    const src = name === 'position' ? patch.positions : patch.normals;
                    added[name].push(...src.slice(k * 3, k * 3 + 3));
                } else if (name === 'uv' || name === 'uvAlt') {
                    const x = patch.coordinates[k], period = md.uvPeriod;
                    added[name].push(period > 0 ? x - period * (name === 'uv' ? Math.round(x / period) : Math.floor(x / period)) : x);
                    if (name === 'uv') added[name].push(attr.array[(lo * cs + k % cs) * 2 + 1]);
                } else for (let d = 0; d < attr.itemSize; d++) added[name].push(attr.array[(lo * cs + k % cs) * attr.itemSize + d]);
            }
        }
        indexData.set(patch.indices, indexCount); indexCount += patch.indices.length;
        nextPair = hi; blockEnds[blockCount] = hi; blockCounts[blockCount++] = indexCount;
        stats.accepted++; stats.clusters++; stats.planar++;
        stats.addedTriangles += patch.indices.length / 3 - (hi-lo)*pairSize/3;
        stats.maxPatchTriangles = Math.max(stats.maxPatchTriangles, patch.indices.length / 3);
        stats.maxWindowSegments = Math.max(stats.maxWindowSegments,hi-lo);
        return true;
    };
    for (let i = 1; i < n - 1; i++) {
        if (i - 1 < nextPair) continue; // Accepted windows never overlap.
        if (options.breaks?.[i] || options.breaks?.[i + 1]) continue;
        const p = point(i), incoming = p.clone().sub(point(i - 1)), outgoing = point(i + 1).sub(p);
        const l0 = incoming.length(), l1 = outgoing.length();
        if (Math.min(l0, l1) < 1e-8) { reject('duplicate'); continue; }
        incoming.divideScalar(l0); outgoing.divideScalar(l1);
        const dot = THREE.MathUtils.clamp(incoming.dot(outgoing), -1, 1), angle = Math.acos(dot);
        const radius = Math.max(md.widths[i], md.heights[i]) / 2;
        const tolerance = options.tolerance || Math.min(md.widths[i], md.widths[i - 1], md.widths[i + 1]) / 100;
        if (radius <= 0 || md.heights[i] <= 0) { reject('zero section'); continue; }
        // A faceted curve already within the chord tolerance needs no new mesh.
        if (radius * (1 / Math.max(.001, Math.cos(angle / 2)) - 1) <= tolerance) { stats.smooth++; continue; }
        if (dot < -.9) { if (!tryCluster(i)) reject('reversal'); continue; }
        const axis = new THREE.Vector3().crossVectors(incoming, outgoing).normalize();
        const inward = new THREE.Vector3().crossVectors(axis, incoming);
        const bisector = inward.clone().applyAxisAngle(axis, angle / 2);
        const U = new THREE.Vector3().crossVectors(up, incoming);
        if (U.lengthSq() < .01) { reject('vertical frame'); continue; }
        U.normalize();
        // Recomputing a frame from up can flip across vertical while the strip's
        // transported frame remains continuous. Stitching equal vertex indices
        // then twists the patch. Retain the strip at incompatible interfaces.
        const exitU = U.clone().applyAxisAngle(axis, angle);
        const startFrame = new THREE.Vector3().fromArray(md.localFrames, (i - 1) * 6);
        const endFrame = new THREE.Vector3().fromArray(md.localFrames, (i + 1) * 6);
        if (U.dot(startFrame) <= 0 || exitU.dot(endFrame) <= 0) {
            reject('frame reversal'); continue;
        }
        const V = new THREE.Vector3().crossVectors(incoming, U);
        // The full path need not be planar. Test only this small neighborhood,
        // with a bead-relative tolerance, and retain its real elevations.
        const planeTolerance = .1 * md.heights[i] + .002 * md.widths[i];
        let deviation = 0;
        for (let k = Math.max(0, i - 2); k <= Math.min(n - 1, i + 2); k++) {
            if (k > 0 && options.breaks?.[k]) continue;
            deviation = Math.max(deviation, Math.abs(point(k).sub(p).dot(axis)));
        }
        const planar = deviation <= planeTolerance;
        // Large section changes need a separate transition algorithm.
        if ([i - 1, i + 1].some(k => Math.abs(md.widths[k] / md.widths[i] - 1) > .05 ||
            Math.abs(md.heights[k] / md.heights[i] - 1) > .05)) { reject('section change'); continue; }
        const w = md.widths[i] / 2, h = md.heights[i] / 2, c = Math.min(w, h);
        const section = w >= h ? [[w, 0], [w-c, h], [-w+c, h], [-w, 0], [-w+c, -h], [w-c, -h]] :
            [[w, -h+c], [w, h-c], [0, h], [-w, h-c], [-w, -h+c], [0, -h]];
        const offsets = section.map(([u, v]) => U.clone().multiplyScalar(u).addScaledVector(V, v));
        const radial = offsets.map(q => q.dot(inward));
        // Trim the inner offsets only if their intersection lies inside both
        // neighboring segments. Tight jogs are explicitly left to the fallback.
        const reach = Math.max(...radial.map(Math.abs)) * Math.tan(angle / 2);
        if (reach > .8 * Math.min(l0, l1)) { if (!tryCluster(i)) reject('short segments'); continue; }
        const steps = Math.ceil(angle / (2 * Math.acos(1 - Math.min(.5, tolerance / radius))));
        if (steps > 6) { reject('triangle budget'); continue; }
        const positions = [], normals = [], coordinates = [], ringIds = [];
        const vOff = md.vOffs?.[i] || 0;
        const anchor = new THREE.Vector3().fromArray(md.localFrames, i * 6 + 3).multiplyScalar(vOff);
        const startVertex = originalVertices + added.position.length / 3;
        for (let k = 0; k <= steps; k++) {
            const theta = angle * k / steps, ids = [];
            for (let j = 0; j < cs; j++) {
                const r = radial[j], a = offsets[j].dot(axis);
                const q = r > 0 ? bisector.clone().multiplyScalar(r / Math.cos(angle / 2)).addScaledVector(axis, a) :
                    offsets[j].clone().applyAxisAngle(axis, theta);
                positions.push(...q.add(p).add(anchor).toArray());
                // Bisect the two cross-section edge normals, then transport.
                const prev = section[(j + cs - 1) % cs], cur = section[j], next = section[(j + 1) % cs];
                const e0 = new THREE.Vector2(cur[1] - prev[1], prev[0] - cur[0]).normalize();
                const e1 = new THREE.Vector2(next[1] - cur[1], cur[0] - next[0]).normalize();
                const sn = e0.add(e1).normalize();
                const normal = U.clone().multiplyScalar(sn.x).addScaledVector(V, sn.y).applyAxisAngle(axis, r > 0 ? angle / 2 : theta);
                normals.push(...normal.toArray());
                // Preserve original path-distance coordinates outside the join;
                // advance over the outer arc, with one shared UV on the inner trim.
                coordinates.push((md.arc?.[i] || 0) + (r < 0 ? -r * (theta - angle / 2) : 0));
                ids.push(startVertex + k * cs + j);
            }
            ringIds.push(ids);
        }
        const strip = [Array.from({ length: cs }, (_, j) => (i - 1) * cs + j), ...ringIds,
            Array.from({ length: cs }, (_, j) => (i + 1) * cs + j)];
        const patch = [], vec = id => new THREE.Vector3().fromArray(id < originalVertices ? attrs.position.array : positions,
            id < originalVertices ? id * 3 : (id - startVertex) * 3);
        const nor = id => new THREE.Vector3().fromArray(id < originalVertices ? attrs.normal.array : normals,
            id < originalVertices ? id * 3 : (id - startVertex) * 3);
        let invalid = false;
        for (let k = 1; k < strip.length; k++) for (let j = 0; j < cs; j++) {
            const a = strip[k - 1][j], b = strip[k - 1][(j + 1) % cs], d = strip[k][j], e = strip[k][(j + 1) % cs];
            for (const tri of [[a, b, d], [b, e, d]]) {
                const [pa, pb, pc] = tri.map(vec), cross = pb.sub(pa).cross(pc.sub(pa));
                if (cross.lengthSq() < 1e-16) continue; // Inner trim collapses to a shared edge.
                const expected = nor(tri[0]).add(nor(tri[1])).add(nor(tri[2]));
                if (cross.dot(expected) < 0) [tri[1], tri[2]] = [tri[2], tri[1]];
                if (!cross.toArray().every(Number.isFinite)) invalid = true;
                patch.push(...tri);
            }
        }
        if (invalid || patch.length / 3 > 96) { reject('invalid or over budget'); continue; }
        if (stats.addedTriangles + patch.length / 3 - 2 * pairSize / 3 > stats.triangleBudget) {
            reject('total triangle budget'); continue;
        }
        appendOriginal(i - 1);
        for (const name of Object.keys(attrs)) {
            const attr = attrs[name];
            for (let k = 0; k < positions.length / 3; k++) {
                if (name === 'position') added[name].push(...positions.slice(k * 3, k * 3 + 3));
                else if (name === 'normal') added[name].push(...normals.slice(k * 3, k * 3 + 3));
                else if (name === 'uv' || name === 'uvAlt') {
                    const x = coordinates[k], period = md.uvPeriod;
                    added[name].push(period > 0 ? x - period * (name === 'uv' ? Math.round(x / period) : Math.floor(x / period)) : x);
                    if (name === 'uv') added[name].push(attr.array[(i * cs + k % cs) * 2 + 1]);
                } else for (let d = 0; d < attr.itemSize; d++) added[name].push(attr.array[(i * cs + k % cs) * attr.itemSize + d]);
            }
        }
        indexData.set(patch, indexCount); indexCount += patch.length;
        nextPair = i + 1;
        blockEnds[blockCount] = nextPair; blockCounts[blockCount++] = indexCount;
        stats.accepted++; stats[planar ? 'planar' : 'spatial']++;
        stats.addedTriangles += patch.length / 3 - 2 * pairSize / 3;
        stats.maxPatchTriangles = Math.max(stats.maxPatchTriangles, patch.length / 3);
    }
    if (!stats.accepted) {
        // A smooth vase adds neither triangles nor a second index buffer.
        ud.localJoinStats = stats;
        stats.buildMs = performance.now() - started;
        return stats;
    }
    appendOriginal(n - 1);
    const completed = new THREE.BufferGeometry();
    for (const name of Object.keys(attrs)) {
        const old = attrs[name], array = new Float32Array(old.array.length + added[name].length);
        array.set(old.array); array.set(added[name], old.array.length);
        const attr = new THREE.BufferAttribute(array, old.itemSize, old.normalized);
        geometry.setAttribute(name, attr); completed.setAttribute(name, attr);
    }
    completed.setIndex(new THREE.BufferAttribute(indexData.slice(0, indexCount), 1));
    indexData = null; // Release the temporary capacity buffer after compaction.
    // Copy bounds once; patch vertices stay within the old ring + bead bounds.
    geometry.computeBoundingBox(); geometry.computeBoundingSphere();
    completed.boundingBox = geometry.boundingBox.clone(); completed.boundingSphere = geometry.boundingSphere.clone();
    const frozen = new THREE.Mesh(completed, mesh.material);
    frozen.name = `${mesh.name}:completed`;
    frozen.userData.localJoinCompleted = true;
    mesh.add(frozen);
    ud.localJoinStats = stats;
    ud.localJoinCompleted = frozen;
    // Keep the original controller and all its update hooks. Only replace the
    // draw-range entry point; both meshes share the original mutable attributes.
    const setBaseRange = base.setDrawRange.bind(base);
    base.setDrawRange = value => {
        if (!Number.isFinite(value)) return;
        const v = THREE.MathUtils.clamp(value, 0, 1);
        setBaseRange(v);
        // A full pair behind the frontier is immutable. Binary search permits
        // instant backward seeks without scanning the already printed mesh.
        const limit = v === 1 ? n - 1 : Math.max(0, Math.floor(v * (n - 1)) - 1);
        let lo = 0, hi = blockCount;
        while (lo < hi) { const mid = (lo + hi) >>> 1; if (blockEnds[mid] <= limit) lo = mid + 1; else hi = mid; }
        const end = lo ? blockEnds[lo - 1] : 0, count = lo ? blockCounts[lo - 1] : 0;
        completed.setDrawRange(0, v < 1e-6 ? 0 : count);
        if (count) {
            const start = cap + end * pairSize;
            geometry.setDrawRange(start, Math.max(0, geometry.drawRange.count - start));
        }
        frozen.material = mesh.material;
        frozen.castShadow = mesh.castShadow; frozen.receiveShadow = mesh.receiveShadow;
        ud.localJoinCommittedPairs = end;
    };
    base.setDrawRange(1);
    stats.buildMs = performance.now() - started;
    return stats;
}

/**
 * A short almost-planar jog can be treated as a single strip if every transverse
 * slice of its swept footprint is connected. Evaluate the analytic capsule
 * envelope at bounded stations, retaining six vertices per station. This is a
 * local approximation, not a Boolean union; holes or disconnected slices fail.
 */
function localCluster(mesh, options, lo, hi, vertexBase) {
    const md = mesh.userData.tubeMorphData, attrs = mesh.geometry.attributes;
    if (hi - lo > 10 || hi - lo < 3) return null;
    if (options.breaks) for (let i = lo + 1; i <= hi; i++) if (options.breaks[i]) return null;
    const up = new THREE.Vector3(...options.up).normalize();
    const origin = new THREE.Vector3().fromArray(md.spine, lo * 3);
    const direction = new THREE.Vector3().fromArray(md.spine, hi * 3).sub(origin);
    const rise = direction.dot(up); direction.addScaledVector(up, -rise);
    const length = direction.length();
    if (length < md.widths[lo]) return null;
    direction.divideScalar(length);
    const across = new THREE.Vector3().crossVectors(up, direction);
    const points = [], tolerance = options.tolerance || md.widths[lo] / 100;
    let minRadius = Infinity, maxRadius = 0, minHeight = Infinity, maxHeight = 0;
    for (let i = lo; i <= hi; i++) {
        const p = new THREE.Vector3().fromArray(md.spine, i * 3).sub(origin);
        const x = p.dot(direction), y = p.dot(across), z = p.dot(up), r = md.widths[i] / 2, h = md.heights[i];
        if (x < -1e-5 || x > length + 1e-5 || r <= h / 2 || h <= 0 ||
            Math.abs(z - rise * x / length) > .1 * h + .002 * md.widths[i]) return null;
        points.push({ x, y, z, r, h, offset: md.vOffs?.[i] || 0 });
        minRadius = Math.min(minRadius, r); maxRadius = Math.max(maxRadius, r);
        minHeight = Math.min(minHeight, h); maxHeight = Math.max(maxHeight, h);
    }
    if (maxRadius / minRadius > 1.05 || maxHeight / minHeight > 1.05) return null;
    // Interfaces use the original ring vertices, including the exact anchor.
    let start = -Infinity, end = Infinity;
    for (const i of [lo, hi]) {
        const u = new THREE.Vector3().fromArray(md.localFrames, i * 6);
        if (u.dot(across) < .8) return null;
        for (let j = 0; j < 6; j++) {
            const p = new THREE.Vector3().fromArray(attrs.position.array, (i * 6 + j) * 3).sub(origin);
            if (i === lo) start = Math.max(start, p.dot(direction));
            else end = Math.min(end, p.dot(direction));
        }
    }
    // This guard keeps the first/last replacement strips from folding across
    // an oblique interface. Interior samples have strictly increasing x.
    start += tolerance; end -= tolerance;
    if (end <= start) return null;
    const steps = Math.max(3, Math.ceil((end - start) / Math.sqrt(8 * tolerance * Math.max(.1, minRadius - maxHeight / 2))));
    if (steps > 14) return null;
    const envelope = (x, top) => {
        const intervals = [];
        for (let i = 1; i < points.length; i++) {
            const a = points[i - 1], b = points[i], ra = a.r - (top ? a.h / 2 : 0), rb = b.r - (top ? b.h / 2 : 0);
            const hits = [];
            for (const [p, r] of [[a, ra], [b, rb]]) {
                const dx = x - p.x;
                if (Math.abs(dx) <= r) { const dy = Math.sqrt(Math.max(0, r * r - dx * dx)); hits.push(p.y - dy, p.y + dy); }
            }
            const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy);
            if (len > Math.abs(rb - ra) + 1e-9) {
                const along = (ra - rb) / len, side = Math.sqrt(1 - along * along);
                for (const sign of [-1, 1]) {
                    const nx = along * dx / len - sign * side * dy / len;
                    const ny = along * dy / len + sign * side * dx / len;
                    const ax = a.x + nx * ra, ay = a.y + ny * ra, bx = b.x + nx * rb, by = b.y + ny * rb;
                    if (Math.abs(bx - ax) > 1e-10) {
                        const t = (x - ax) / (bx - ax);
                        if (t >= 0 && t <= 1) hits.push(ay + t * (by - ay));
                    }
                }
            }
            if (hits.length) intervals.push([Math.min(...hits), Math.max(...hits)]);
        }
        intervals.sort((a, b) => a[0] - b[0]);
        if (!intervals.length) return null;
        let upper = intervals[0][1];
        for (let i = 1; i < intervals.length; i++) {
            if (intervals[i][0] > upper + 1e-6) return null;
            upper = Math.max(upper, intervals[i][1]);
        }
        return [intervals[0][0], upper];
    };
    const positions = [], coordinates = [];
    for (let k = 0; k <= steps; k++) {
        const x = start + (end - start) * k / steps, mid = envelope(x, false), top = envelope(x, true);
        if (!mid || !top || top[0] < mid[0] - 1e-6 || top[1] > mid[1] + 1e-6) return null;
        // Recover actual height/rise from the nearest centreline segment.
        let best = Infinity, z = 0, h = 0, offset = 0;
        const centre = (mid[0] + mid[1]) / 2;
        for (let j = 1; j < points.length; j++) {
            const a = points[j - 1], b = points[j], dx = b.x - a.x, dy = b.y - a.y;
            const t = THREE.MathUtils.clamp(((x - a.x) * dx + (centre - a.y) * dy) / Math.max(1e-12, dx*dx + dy*dy), 0, 1);
            const d = (x-a.x-t*dx)**2 + (centre-a.y-t*dy)**2;
            if (d < best) { best = d; z = a.z + t*(b.z-a.z); h = a.h+t*(b.h-a.h); offset = a.offset+t*(b.offset-a.offset); }
        }
        const yz = [[mid[1], 0], [top[1], h/2], [top[0], h/2], [mid[0], 0], [top[0], -h/2], [top[1], -h/2]];
        for (let j = 0; j < 6; j++) {
            positions.push(...origin.clone().addScaledVector(direction,x).addScaledVector(across,yz[j][0]).addScaledVector(up,z+offset+yz[j][1]).toArray());
            coordinates.push((md.arc?.[lo] || 0) + x/length*((md.arc?.[hi] || 0)-(md.arc?.[lo] || 0)));
        }
    }
    const rings = [Array.from({ length: 6 }, (_, j) => lo*6+j)];
    for (let k = 0; k <= steps; k++) rings.push(Array.from({ length: 6 }, (_, j) => vertexBase+k*6+j));
    rings.push(Array.from({ length: 6 }, (_, j) => hi*6+j));
    const indices = [], normals = new Array(positions.length).fill(0);
    const get = id => new THREE.Vector3().fromArray(id < vertexBase ? attrs.position.array : positions, id < vertexBase ? id*3 : (id-vertexBase)*3);
    for (let k = 1; k < rings.length; k++) for (let j = 0; j < 6; j++) {
        const a=rings[k-1][j], b=rings[k-1][(j+1)%6], c=rings[k][j], d=rings[k][(j+1)%6];
        for (const tri of [[a,b,c],[b,d,c]]) {
            const [p,q,r]=tri.map(get), normal=q.sub(p).cross(r.sub(p));
            if (normal.lengthSq() < 1e-16) return null;
            // Section rings are CCW viewed along the path; the strip has this winding.
            indices.push(...tri);
            for (const id of tri) if (id >= vertexBase) for (let d=0;d<3;d++) normals[(id-vertexBase)*3+d]+=normal.getComponent(d);
        }
    }
    if (indices.length / 3 > Math.min(192, (hi-lo)*12+96)) return null;
    for (let k=0;k<normals.length;k+=3) {
        const len=Math.hypot(normals[k],normals[k+1],normals[k+2]);
        if (len < 1e-12) return null;
        for(let d=0;d<3;d++) normals[k+d]/=len;
    }
    return {positions,normals,coordinates,indices,lo,hi};
}
