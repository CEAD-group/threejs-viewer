// @ts-check
import * as THREE from 'three';
import { POLYGON_CLIPPING_SOURCE } from './static/polygon-clipping/source.js';

// This function is serialized into a dedicated worker. Keep it self-contained.
// The moving kernel is the existing flat-topped bead revolved about its up axis:
// at height z its footprint is a disk of radius width/2 - abs(z).
function planarUnionWorker() {
    const clip = self['polygonClipping'];
    let points, widths, breaks, height, tolerance;
    let segments = [], caches = [], completed = 0;
    const levels = [0, 1];
    let sides = 48;
    function hull(points) {
        points.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
        const cross = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        const lower = [], upper = [];
        for (const p of points) {
            while (lower.length > 1 && cross(lower.at(-2), lower.at(-1), p) <= 0) lower.pop();
            lower.push(p);
        }
        for (let i = points.length - 1; i >= 0; i--) {
            const p = points[i];
            while (upper.length > 1 && cross(upper.at(-2), upper.at(-1), p) <= 0) upper.pop();
            upper.push(p);
        }
        const ring = lower.slice(0, -1).concat(upper.slice(0, -1));
        ring.push(ring[0]);
        return [ring];
    }
    function capsule(a, b, wa, wb, z) {
        const vertices = [];
        for (let j = 0; j < sides; j++) {
            const theta = j * 2 * Math.PI / sides, x = Math.cos(theta), y = Math.sin(theta);
            vertices.push([a[0] + (wa / 2 - z) * x, a[1] + (wa / 2 - z) * y]);
            vertices.push([b[0] + (wb / 2 - z) * x, b[1] + (wb / 2 - z) * y]);
        }
        // Convex hull of the endpoint disks = sweep of a linearly varying disk.
        return hull(vertices);
    }
    function unionBatch(existing, additions) {
        // Bound argument count and the clipping library's event queue.
        for (let i = 0; i < additions.length; i += 64) {
            const batch = additions.slice(i, i + 64);
            existing = existing.length ? clip.union(existing, ...batch) : clip.union(...batch);
        }
        return existing;
    }
    self.onmessage = ({ data }) => {
        try {
            const started = performance.now();
            if (data.init) {
                ({ points, widths, breaks, height, tolerance } = data.init);
                let maxRadius = 0;
                for (const w of widths) maxRadius = Math.max(maxRadius, w / 2);
                sides = Math.max(12, Math.ceil(Math.PI / Math.acos(1 - Math.min(tolerance / maxRadius, 0.25))));
                if (sides > 512) throw new Error('surface_tolerance requires more than 512 circle sides');
                // Remove redundant stations geometrically, retaining original indices
                // for playback. Dense straight runs then cost one capsule, not N rings.
                segments = [];
                for (let i = 1; i < points.length; i++) {
                    if (breaks && breaks[i]) continue;
                    const prev = segments.at(-1);
                    if (prev && prev[1] === i - 1) {
                        const a = points[prev[0]], b = points[i - 1], c = points[i];
                        const dx = c[0] - a[0], dy = c[1] - a[1], d2 = dx * dx + dy * dy;
                        const t = d2 ? ((b[0] - a[0]) * dx + (b[1] - a[1]) * dy) / d2 : -1;
                        if (t >= 0 && t <= 1 && Math.abs((b[0] - a[0]) * dy - (b[1] - a[1]) * dx) < 1e-10 &&
                            Math.abs(widths[i - 1] - (widths[prev[0]] * (1 - t) + widths[i] * t)) < 1e-10) {
                            prev[1] = i;
                            continue;
                        }
                    }
                    segments.push([i - 1, i]);
                }
                caches = levels.map(() => []);
                completed = 0;
            }
            const value = data.value, at = value * (points.length - 1);
            let count = 0;
            while (count < segments.length && segments[count][1] <= at) count++;
            if (count < completed) { caches = levels.map(() => []); completed = 0; }
            const footprints = levels.map((level, k) => {
                const z = level * height / 2, additions = [];
                for (let j = completed; j < count; j++) {
                    const [a, b] = segments[j];
                    additions.push(capsule(points[a], points[b], widths[a], widths[b], z));
                }
                caches[k] = unionBatch(caches[k], additions);
                let result = caches[k];
                const next = segments[count];
                if (value > 0 && next && at > next[0]) {
                    const i = Math.floor(at), t = at - i;
                    const p = [points[i][0] * (1 - t) + points[i + 1][0] * t,
                        points[i][1] * (1 - t) + points[i + 1][1] * t];
                    const w = widths[i] * (1 - t) + widths[i + 1] * t;
                    result = unionBatch(result, [capsule(points[next[0]], p, widths[next[0]], w, z)]);
                }
                return result;
            });
            completed = count;
            const bands = footprints.slice(0, -1).map((outer, k) => outer.length
                ? clip.difference(outer, footprints[k + 1]) : []);
            const active = segments.slice(0, count).map(([a, b]) => [points[a], points[b], widths[a] / 2, widths[b] / 2]);
            const next = segments[count];
            if (value > 0 && next && at > next[0]) {
                const i = Math.floor(at), t = at - i;
                active.push([points[next[0]], [points[i][0] * (1 - t) + points[i + 1][0] * t,
                    points[i][1] * (1 - t) + points[i + 1][1] * t], widths[next[0]] / 2,
                    (widths[i] * (1 - t) + widths[i + 1] * t) / 2]);
            }
            self.postMessage({ value, footprints, bands, levels, active, segmentCount: segments.length, workerMs: performance.now() - started });
        } catch (error) {
            self.postMessage({ value: data.value, error: String(error) });
        }
    };
}

/**
 * Compact planar union surface with a worker-owned prefix cache. The scene keeps
 * one ordinary mesh: no voxel grid, duplicated internal walls, or per-input-ring
 * triangles. New prefixes replace topology, including holes and retreating walls.
 */
export class PlanarUnionTube {
    /** @param {THREE.Mesh} mesh @param {any} input */
    constructor(mesh, input) {
        if (!Number.isFinite(input.height) || input.height <= 0 ||
            !Number.isFinite(input.tolerance) || input.tolerance <= 0 ||
            input.up.length !== 3 || !input.up.every(Number.isFinite) || Math.hypot(...input.up) < 1e-12 ||
            !input.widths.every(w => Number.isFinite(w) && w > input.height) ||
            !input.spine.every(Number.isFinite)) {
            throw new Error('planar_union requires finite planar input, widths > height > 0, and a positive tolerance');
        }
        this.mesh = mesh;
        this.input = input;
        this.disposed = false;
        this.busy = false;
        this.requested = 1;
        this.rendered = -1;
        this.up = new THREE.Vector3(...input.up).normalize();
        this.origin = new THREE.Vector3().fromArray(input.spine);
        this.u = new THREE.Vector3(Math.abs(this.up.x) < 0.9 ? 1 : 0, Math.abs(this.up.x) < 0.9 ? 0 : 1, 0);
        this.u.addScaledVector(this.up, -this.u.dot(this.up)).normalize();
        this.v = new THREE.Vector3().crossVectors(this.up, this.u);
        const points = [];
        const p = new THREE.Vector3();
        let magnitude = 0;
        for (const x of input.spine) magnitude = Math.max(magnitude, Math.abs(x));
        const planeTolerance = Math.max(input.height * 1e-5, 4 * 2 ** -23 * magnitude);
        for (let i = 0; i < input.widths.length; i++) {
            p.fromArray(input.spine, i * 3).sub(this.origin);
            if (Math.abs(p.dot(this.up)) > planeTolerance) throw new Error('planar_union requires a planar spine');
            points.push([p.dot(this.u), p.dot(this.v)]);
        }
        this.points = points;
        this.ready = new Promise((resolve, reject) => { this.resolve = resolve; this.reject = reject; });
        const url = URL.createObjectURL(new Blob([POLYGON_CLIPPING_SOURCE,
            `\n(${planarUnionWorker.toString()})();`], { type: 'application/javascript' }));
        this.worker = new Worker(url);
        URL.revokeObjectURL(url);
        this.worker.onerror = (e) => { this.fail(e.message); };
        this.worker.onmessage = ({ data }) => {
            if (this.disposed) return;
            this.busy = false;
            if (data.error) { this.fail(data.error); return; }
            // Forward playback may display a slightly older prefix while the worker
            // catches up. Never display future material after a backward seek.
            if (data.value <= this.requested && data.value >= this.rendered) {
                const started = performance.now();
                let geometry;
                try { geometry = this.buildGeometry(data); }
                catch (error) { this.fail(String(error)); return; }
                this.mesh.userData.unionMeshingMs = performance.now() - started;
                this.mesh.userData.unionWorkerMs = data.workerMs;
                this.mesh.geometry.dispose();
                this.mesh.geometry = geometry;
                if (this.mesh.userData.wireframeOverlay) this.mesh.userData.wireframeOverlay.geometry = geometry;
                this.input.onGeometry?.();
                this.rendered = data.value;
                this.mesh.userData.unionRenderedValue = data.value;
                this.mesh.userData.unionTriangleCount = geometry.index.count / 3;
                this.mesh.userData.unionSegmentCount = data.segmentCount;
                this.resolve();
            }
            if (this.requested !== data.value) this.send();
        };
        this.busy = true;
        this.worker.postMessage({ init: { points, widths: input.widths, breaks: input.breaks,
            height: input.height, tolerance: input.tolerance }, value: 1 });
    }
    fail(message) {
        this.mesh.userData.unionError = message;
        console.error(`planar_union '${this.mesh.name}': ${message}`);
        this.reject(new Error(message));
        this.worker.terminate();
        this.disposed = true;
    }
    send() {
        if (this.disposed || this.busy) return;
        this.busy = true;
        this.worker.postMessage({ value: this.requested });
    }
    setDrawRange(value) {
        if (!Number.isFinite(value)) return;
        value = Math.max(0, Math.min(1, value));
        if (value === this.requested) return;
        this.requested = value;
        if (value < this.rendered || value === 0) {
            this.mesh.geometry.setDrawRange(0, 0);
            this.input.onGeometry?.();
            this.rendered = -1;
            this.mesh.userData.unionRenderedValue = -1;
        }
        this.send();
    }
    dispose() {
        this.disposed = true;
        this.worker.terminate();
        this.reject(new Error('deleted'));
    }
    buildGeometry(data) {
        const positions = [], indices = [], normals = [];
        const height = this.input.height, zOffset = this.input.heightOffset * height;
        const key = (p) => `${Math.round(p[0] * 1e7)},${Math.round(p[1] * 1e7)}`;
        // Height of the union of variable-radius disk sweeps. The stationary
        // point minimizes distance-to-segment minus linearly interpolated radius.
        const field = (p) => {
            let h = -Infinity, nx = 0, ny = 0;
            for (const [a, b, ra, rb] of data.active) {
                const dx = b[0] - a[0], dy = b[1] - a[1], length = Math.hypot(dx, dy);
                const px = p[0] - a[0], py = p[1] - a[1], dr = rb - ra;
                let t = dr > 0 ? 1 : 0;
                if (length > Math.abs(dr)) {
                    const along = (px * dx + py * dy) / length;
                    const across = Math.abs(px * dy - py * dx) / length;
                    t = Math.max(0, Math.min(1, (along + dr * across / Math.sqrt(length * length - dr * dr)) / length));
                }
                const x = px - t * dx, y = py - t * dy, distance = Math.hypot(x, y);
                const candidate = ra + dr * t - distance;
                if (candidate > h) {
                    h = candidate; nx = distance ? x / distance : 0; ny = distance ? y / distance : 0;
                }
            }
            return [Math.max(0, Math.min(height / 2, h)), nx, ny];
        };
        const surfaceHeight = p => field(p)[0];
        const addPatch = (polygons, zFor, sign, band = false) => {
            for (const polygon of polygons) {
                const rings = polygon.map(ring => ring.slice(0, -1).map(p => new THREE.Vector2(p[0], p[1])));
                const triangles = THREE.ShapeUtils.triangulateShape(rings[0], rings.slice(1));
                const flat = rings.flat().map(p => [p.x, p.y]);
                const zs = flat.map(zFor), base = positions.length / 3;
                const edgeKey = (a, b) => a < b ? `${a},${b}` : `${b},${a}`;
                const edges = new Map();
                if (band) {
                    // Prefer diagonals between different contour levels. Earcut's
                    // unconstrained ears otherwise create flat terraces and force
                    // extra interior vertices. Each flip removes one such diagonal.
                    const cross = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
                    let changed = true;
                    while (changed) {
                        changed = false;
                        const adjacency = new Map(), touched = new Set();
                        for (let ti = 0; ti < triangles.length; ti++) {
                            const tri = triangles[ti];
                            for (let j = 0; j < 3; j++) {
                                const a = tri[j], b = tri[(j + 1) % 3], c = tri[(j + 2) % 3];
                                const key = edgeKey(a, b), other = adjacency.get(key);
                                if (!other) { adjacency.set(key, { ti, c }); continue; }
                                const d = other.c;
                                if (touched.has(ti) || touched.has(other.ti) || zs[a] !== zs[b] || zs[c] === zs[d]) continue;
                                if (cross(flat[c], flat[d], flat[a]) * cross(flat[c], flat[d], flat[b]) >= -1e-20) continue;
                                triangles[ti] = [c, d, a]; triangles[other.ti] = [d, c, b];
                                touched.add(ti); touched.add(other.ti); changed = true;
                            }
                        }
                    }

                    for (const tri of triangles) for (let j = 0; j < 3; j++) {
                        const a = tri[j], b = tri[(j + 1) % 3], key = edgeKey(a, b);
                        const edge = edges.get(key);
                        if (edge) edge.count++;
                        else edges.set(key, { a, b, count: 1, mid: -1 });
                    }
                    for (const edge of edges.values()) {
                        if (edge.count !== 2 || zs[edge.a] !== zs[edge.b]) continue;
                        const a = flat[edge.a], b = flat[edge.b], p = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
                        edge.mid = flat.length;
                        flat.push(p); zs.push(surfaceHeight(p));
                    }
                }
                const emit = (tri) => {
                    const [a, b, c] = tri.map(i => flat[i]);
                    const area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
                    if (Math.abs(area) < 1e-16) return;
                    if (area * sign < 0) [tri[1], tri[2]] = [tri[2], tri[1]];
                    indices.push(...tri.map(i => base + i));
                };
                for (const tri of triangles) {
                    const boundary = [];
                    for (let j = 0; j < 3; j++) {
                        const a = tri[j], b = tri[(j + 1) % 3];
                        boundary.push(a);
                        const edge = edges.get(edgeKey(a, b));
                        if (edge && edge.mid >= 0) boundary.push(edge.mid);
                    }
                    const sameLevel = zs[tri[0]] === zs[tri[1]] && zs[tri[1]] === zs[tri[2]];
                    if (boundary.length === 3 && (!band || !sameLevel)) { emit(tri); continue; }
                    const p = [0, 0];
                    for (const i of tri) { p[0] += flat[i][0] / 3; p[1] += flat[i][1] / 3; }
                    const centre = flat.length;
                    flat.push(p); zs.push(surfaceHeight(p));
                    for (let j = 0; j < boundary.length; j++) emit([boundary[j], boundary[(j + 1) % boundary.length], centre]);
                }
                for (let i = 0; i < flat.length; i++) {
                    const [x, y] = flat[i], z = sign * zs[i] + zOffset;
                    const f = band ? field([x, y]) : [0, 0, 0];
                    const nx = f[1], ny = f[2], nz = sign;
                    const inv = 1 / Math.hypot(nx, ny, nz);
                    normals.push((this.u.x * nx + this.v.x * ny + this.up.x * nz) * inv,
                        (this.u.y * nx + this.v.y * ny + this.up.y * nz) * inv,
                        (this.u.z * nx + this.v.z * ny + this.up.z * nz) * inv);
                    positions.push(this.origin.x + this.u.x * x + this.v.x * y + this.up.x * z,
                        this.origin.y + this.u.y * x + this.v.y * y + this.up.y * z,
                        this.origin.z + this.u.z * x + this.v.z * y + this.up.z * z);
                }
            }
        };
        for (const sign of [1, -1]) {
            addPatch(data.footprints.at(-1), () => height / 2, sign);
            for (let k = 0; k < data.bands.length; k++) {
                const innerKeys = new Set();
                for (const poly of data.footprints[k + 1]) for (const ring of poly) for (const p of ring) innerKeys.add(key(p));
                addPatch(data.bands[k], p => height / 2 * data.levels[k + (innerKeys.has(key(p)) ? 1 : 0)], sign, true);
            }
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setIndex(indices);
        geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
        geometry.computeBoundingBox();
        geometry.computeBoundingSphere();
        return geometry;
    }
}
