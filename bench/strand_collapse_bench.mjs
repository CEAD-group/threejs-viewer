// Benchmark for collapseTubeStrandFolds optimizations.
// Run with: node bench/strand_collapse_bench.mjs
//
// V0 — current ship: forward inner scan, log all hits.
// V1 — first-hit early-exit (backwards inner scan, break on first hit).
// V2 — V1 + per-i turn-sum gate (skip strand scan when local cumulative turn < π/2).
// V3 — V2 + spine-distance pre-filter on each (i, k) pair.

const N_CS = 6;
const WIN = 30;
const TURN_GATE = Math.PI * 0.125; // 22.5°: lower than the smallest cumulative turn that can produce a fold for our half-width

// ---- Test spine: same blobby Catmull-Rom shape as example 21 ----

function blobbyControlPoints(n, baseR, jitter, seed) {
    let state = seed | 0;
    const rng = () => {
        state = (state * 1664525 + 1013904223) | 0;
        return ((state >>> 0) / 4294967296);
    };
    const pts = [];
    for (let i = 0; i < n; i++) {
        const baseTheta = (i / n) * 2 * Math.PI;
        const theta = baseTheta + (rng() - 0.5) * 2 * 0.35 * (2 * Math.PI / n);
        const r = baseR * (1 + (rng() - 0.5) * 2 * jitter);
        pts.push([r * Math.cos(theta), r * Math.sin(theta)]);
    }
    return pts;
}

function catmullRomEval(t, p0, p1, p2, p3, t0, t1, t2, t3) {
    const w01 = (t1 - t) / (t1 - t0), w11 = (t - t0) / (t1 - t0);
    const w02 = (t2 - t) / (t2 - t1), w12 = (t - t1) / (t2 - t1);
    const w03 = (t3 - t) / (t3 - t2), w13 = (t - t2) / (t3 - t2);
    const a1x = w01 * p0[0] + w11 * p1[0], a1y = w01 * p0[1] + w11 * p1[1];
    const a2x = w02 * p1[0] + w12 * p2[0], a2y = w02 * p1[1] + w12 * p2[1];
    const a3x = w03 * p2[0] + w13 * p3[0], a3y = w03 * p2[1] + w13 * p3[1];
    const wb01 = (t2 - t) / (t2 - t0), wb11 = (t - t0) / (t2 - t0);
    const wb02 = (t3 - t) / (t3 - t1), wb12 = (t - t1) / (t3 - t1);
    const b1x = wb01 * a1x + wb11 * a2x, b1y = wb01 * a1y + wb11 * a2y;
    const b2x = wb02 * a2x + wb12 * a3x, b2y = wb02 * a2y + wb12 * a3y;
    const wc0 = (t2 - t) / (t2 - t1), wc1 = (t - t1) / (t2 - t1);
    return [wc0 * b1x + wc1 * b2x, wc0 * b1y + wc1 * b2y];
}

// Mirrors `catmull_rom_periodic` from examples/21_tube_corner_zfight.py:
// returns a periodic-but-open polyline (last sample one cadence step before
// the start) so behaviour matches what the demo actually feeds the algorithm.
function catmullRomPeriodic(points, samplesPerSegment, alpha = 0.5) {
    const n = points.length;
    const knot = (ti, a, b) => {
        const dx = b[0] - a[0], dy = b[1] - a[1];
        return ti + Math.pow(Math.max(Math.sqrt(dx * dx + dy * dy), 1e-9), alpha);
    };
    const out = [];
    for (let i = 0; i < n; i++) {
        const p0 = points[(i - 1 + n) % n];
        const p1 = points[i];
        const p2 = points[(i + 1) % n];
        const p3 = points[(i + 2) % n];
        const t0 = 0, t1 = knot(t0, p0, p1), t2 = knot(t1, p1, p2), t3 = knot(t2, p2, p3);
        for (let s = 0; s < samplesPerSegment; s++) {
            const t = t1 + (t2 - t1) * (s / samplesPerSegment);
            out.push(catmullRomEval(t, p0, p1, p2, p3, t0, t1, t2, t3));
        }
    }
    return out;
}

function liftXY(xy) {
    const out = new Float32Array(xy.length * 3);
    for (let i = 0; i < xy.length; i++) {
        out[i * 3] = xy[i][0]; out[i * 3 + 1] = xy[i][1]; out[i * 3 + 2] = 0;
    }
    return out;
}

// Constant-up, regular-hex offset positions — minimal mock of writeRingVerts.
// Topology and fold behaviour identical to the chamfered hex at this scale.
function buildTestPositions(spine, nCs, radius) {
    const nSpine = spine.length / 3;
    const positions = new Float32Array(nSpine * nCs * 3);
    for (let i = 0; i < nSpine; i++) {
        let tx, ty, tz;
        if (i < nSpine - 1) {
            tx = spine[(i + 1) * 3] - spine[i * 3];
            ty = spine[(i + 1) * 3 + 1] - spine[i * 3 + 1];
            tz = spine[(i + 1) * 3 + 2] - spine[i * 3 + 2];
        } else {
            tx = spine[i * 3] - spine[(i - 1) * 3];
            ty = spine[i * 3 + 1] - spine[(i - 1) * 3 + 1];
            tz = spine[i * 3 + 2] - spine[(i - 1) * 3 + 2];
        }
        const tlen = Math.hypot(tx, ty, tz);
        if (tlen > 1e-12) { tx /= tlen; ty /= tlen; tz /= tlen; }
        const dot = tz; // up = (0,0,1)
        let vx = -dot * tx, vy = -dot * ty, vz = 1 - dot * tz;
        const vlen = Math.hypot(vx, vy, vz);
        if (vlen > 1e-12) { vx /= vlen; vy /= vlen; vz /= vlen; }
        const ux = vy * tz - vz * ty;
        const uy = vz * tx - vx * tz;
        const uz = vx * ty - vy * tx;
        const sx = spine[i * 3], sy = spine[i * 3 + 1], sz = spine[i * 3 + 2];
        for (let j = 0; j < nCs; j++) {
            const a = (2 * Math.PI / nCs) * j;
            const cu = radius * Math.cos(a), cv = radius * Math.sin(a);
            const off = (i * nCs + j) * 3;
            positions[off]     = sx + cu * ux + cv * vx;
            positions[off + 1] = sy + cu * uy + cv * vy;
            positions[off + 2] = sz + cu * uz + cv * vz;
        }
    }
    return positions;
}

// ---- Variant V0: current ship ----

function collapseV0(positions, widths, heights, nSpine, nCs, _spine) {
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
    const rangeStart = [], rangeEnd = [];
    for (let j = 0; j < nCs; j++) {
        rangeStart.length = 0; rangeEnd.length = 0;
        for (let i = 0; i < nSpine; i++) {
            const ip = i * ringStride + j * 3;
            const px = positions[ip], py = positions[ip + 1], pz = positions[ip + 2];
            const kHi = Math.min(nSpine - 2, i + WIN);
            for (let k = i + 2; k <= kHi; k++) {
                const ka = k * ringStride + j * 3;
                const kb = (k + 1) * ringStride + j * 3;
                const ax = positions[ka], ay = positions[ka + 1], az = positions[ka + 2];
                const ex = positions[kb] - ax, ey = positions[kb + 1] - ay, ez = positions[kb + 2] - az;
                const eL2 = ex * ex + ey * ey + ez * ez;
                if (eL2 < 1e-24) continue;
                const t = ((px - ax) * ex + (py - ay) * ey + (pz - az) * ez) / eL2;
                if (t < 0 || t > 1) continue;
                const dx = px - (ax + t * ex);
                const dy = py - (ay + t * ey);
                const dz = pz - (az + t * ez);
                if (dx * dx + dy * dy + dz * dz < tolSq) {
                    rangeStart.push(i); rangeEnd.push(k + 1);
                }
            }
        }
        applyMergedCollapses(positions, ringStride, j, nSpine, rangeStart, rangeEnd);
    }
}

// ---- V1: first-hit early-exit (backwards inner scan) ----

function collapseV1(positions, widths, heights, nSpine, nCs, _spine) {
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
    const rangeStart = [], rangeEnd = [];
    for (let j = 0; j < nCs; j++) {
        rangeStart.length = 0; rangeEnd.length = 0;
        for (let i = 0; i < nSpine; i++) {
            const ip = i * ringStride + j * 3;
            const px = positions[ip], py = positions[ip + 1], pz = positions[ip + 2];
            const kHi = Math.min(nSpine - 2, i + WIN);
            for (let k = kHi; k >= i + 2; k--) {
                const ka = k * ringStride + j * 3;
                const kb = (k + 1) * ringStride + j * 3;
                const ax = positions[ka], ay = positions[ka + 1], az = positions[ka + 2];
                const ex = positions[kb] - ax, ey = positions[kb + 1] - ay, ez = positions[kb + 2] - az;
                const eL2 = ex * ex + ey * ey + ez * ez;
                if (eL2 < 1e-24) continue;
                const t = ((px - ax) * ex + (py - ay) * ey + (pz - az) * ez) / eL2;
                if (t < 0 || t > 1) continue;
                const dx = px - (ax + t * ex);
                const dy = py - (ay + t * ey);
                const dz = pz - (az + t * ez);
                if (dx * dx + dy * dy + dz * dz < tolSq) {
                    rangeStart.push(i); rangeEnd.push(k + 1);
                    break;
                }
            }
        }
        applyMergedCollapses(positions, ringStride, j, nSpine, rangeStart, rangeEnd);
    }
}

// ---- V2: V1 + per-i turn-sum gate ----

function precomputeTurnSum(spine, nSpine) {
    const turn = new Float32Array(nSpine);
    for (let i = 1; i < nSpine - 1; i++) {
        const ax = spine[i * 3] - spine[(i - 1) * 3];
        const ay = spine[i * 3 + 1] - spine[(i - 1) * 3 + 1];
        const az = spine[i * 3 + 2] - spine[(i - 1) * 3 + 2];
        const bx = spine[(i + 1) * 3] - spine[i * 3];
        const by = spine[(i + 1) * 3 + 1] - spine[i * 3 + 1];
        const bz = spine[(i + 1) * 3 + 2] - spine[i * 3 + 2];
        const aL = Math.hypot(ax, ay, az);
        const bL = Math.hypot(bx, by, bz);
        if (aL < 1e-12 || bL < 1e-12) continue;
        let cosT = (ax * bx + ay * by + az * bz) / (aL * bL);
        if (cosT > 1) cosT = 1; else if (cosT < -1) cosT = -1;
        turn[i] = Math.acos(cosT);
    }
    const turnSum = new Float32Array(nSpine);
    let acc = 0;
    // Sum of turn[i+1..i+WIN]
    for (let k = 1; k <= Math.min(WIN, nSpine - 1); k++) acc += turn[k];
    turnSum[0] = acc;
    for (let i = 1; i < nSpine; i++) {
        const removeIdx = i; // turn[i] leaves the window
        const addIdx = i + WIN;
        acc -= turn[removeIdx];
        if (addIdx < nSpine) acc += turn[addIdx];
        turnSum[i] = acc;
    }
    return turnSum;
}

function collapseV2(positions, widths, heights, nSpine, nCs, spine) {
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
    const turnSum = precomputeTurnSum(spine, nSpine);
    const rangeStart = [], rangeEnd = [];
    for (let j = 0; j < nCs; j++) {
        rangeStart.length = 0; rangeEnd.length = 0;
        for (let i = 0; i < nSpine; i++) {
            if (turnSum[i] < TURN_GATE) continue;
            const ip = i * ringStride + j * 3;
            const px = positions[ip], py = positions[ip + 1], pz = positions[ip + 2];
            const kHi = Math.min(nSpine - 2, i + WIN);
            for (let k = kHi; k >= i + 2; k--) {
                const ka = k * ringStride + j * 3;
                const kb = (k + 1) * ringStride + j * 3;
                const ax = positions[ka], ay = positions[ka + 1], az = positions[ka + 2];
                const ex = positions[kb] - ax, ey = positions[kb + 1] - ay, ez = positions[kb + 2] - az;
                const eL2 = ex * ex + ey * ey + ez * ez;
                if (eL2 < 1e-24) continue;
                const t = ((px - ax) * ex + (py - ay) * ey + (pz - az) * ez) / eL2;
                if (t < 0 || t > 1) continue;
                const dx = px - (ax + t * ex);
                const dy = py - (ay + t * ey);
                const dz = pz - (az + t * ez);
                if (dx * dx + dy * dy + dz * dz < tolSq) {
                    rangeStart.push(i); rangeEnd.push(k + 1);
                    break;
                }
            }
        }
        applyMergedCollapses(positions, ringStride, j, nSpine, rangeStart, rangeEnd);
    }
}

// ---- V3: V1 + tight spine-distance pre-filter (no turn-sum gate) ----
//
// Correctness bound: strand vertex i is at most (max bead half-width = maxDim/2)
// from spine[i]. The closest point on segment(p_k, p_{k+1}) is at most
// maxDim/2 + max(‖spine[k]-spine[k']‖) from spine[k] (worst case at endpoint),
// so it's at most (maxDim/2 + maxSegLen) from spine[k]. Hence if ‖spine[k] -
// spine[i]‖ > maxDim + maxSegLen + tol, no fold possible.

function collapseV3(positions, widths, heights, nSpine, nCs, spine) {
    if (nSpine < 4) return;
    let maxDim = 0;
    let maxSegLen2 = 0;
    for (let i = 0; i < nSpine; i++) {
        if (widths[i] > maxDim) maxDim = widths[i];
        if (heights[i] > maxDim) maxDim = heights[i];
    }
    for (let i = 0; i < nSpine - 1; i++) {
        const dx = spine[(i + 1) * 3] - spine[i * 3];
        const dy = spine[(i + 1) * 3 + 1] - spine[i * 3 + 1];
        const dz = spine[(i + 1) * 3 + 2] - spine[i * 3 + 2];
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > maxSegLen2) maxSegLen2 = d2;
    }
    if (maxDim <= 0) return;
    const tol = 0.05 * maxDim;
    const tolSq = tol * tol;
    const rejectDist = maxDim + Math.sqrt(maxSegLen2) + tol;
    const rejectDist2 = rejectDist * rejectDist;
    const ringStride = nCs * 3;
    const rangeStart = [], rangeEnd = [];
    for (let j = 0; j < nCs; j++) {
        rangeStart.length = 0; rangeEnd.length = 0;
        for (let i = 0; i < nSpine; i++) {
            const ip = i * ringStride + j * 3;
            const px = positions[ip], py = positions[ip + 1], pz = positions[ip + 2];
            const isx = spine[i * 3], isy = spine[i * 3 + 1], isz = spine[i * 3 + 2];
            const kHi = Math.min(nSpine - 2, i + WIN);
            for (let k = kHi; k >= i + 2; k--) {
                const sdx = spine[k * 3] - isx;
                const sdy = spine[k * 3 + 1] - isy;
                const sdz = spine[k * 3 + 2] - isz;
                if (sdx * sdx + sdy * sdy + sdz * sdz > rejectDist2) continue;
                const ka = k * ringStride + j * 3;
                const kb = (k + 1) * ringStride + j * 3;
                const ax = positions[ka], ay = positions[ka + 1], az = positions[ka + 2];
                const ex = positions[kb] - ax, ey = positions[kb + 1] - ay, ez = positions[kb + 2] - az;
                const eL2 = ex * ex + ey * ey + ez * ez;
                if (eL2 < 1e-24) continue;
                const t = ((px - ax) * ex + (py - ay) * ey + (pz - az) * ez) / eL2;
                if (t < 0 || t > 1) continue;
                const dx = px - (ax + t * ex);
                const dy = py - (ay + t * ey);
                const dz = pz - (az + t * ez);
                if (dx * dx + dy * dy + dz * dz < tolSq) {
                    rangeStart.push(i); rangeEnd.push(k + 1);
                    break;
                }
            }
        }
        applyMergedCollapses(positions, ringStride, j, nSpine, rangeStart, rangeEnd);
    }
}

// ---- V4: V1 + per-i kMax precompute (spine-distance pre-filter, hoisted) ----
//
// One pass over the spine computes, per i, the highest k ≤ i+WIN such that
// ‖spine[k] - spine[i]‖ ≤ maxDim + maxSegLen + tol. Folds further away are
// geometrically impossible (strand offset ≤ maxDim/2 per side, segment length
// adds maxSegLen). The inner strand scan then runs only up to kMax[i],
// amortizing the O(N·W) precompute over all 6 strands.

function collapseV4(positions, widths, heights, nSpine, nCs, spine) {
    if (nSpine < 4) return;
    let maxDim = 0;
    let maxSegLen2 = 0;
    for (let i = 0; i < nSpine; i++) {
        if (widths[i] > maxDim) maxDim = widths[i];
        if (heights[i] > maxDim) maxDim = heights[i];
    }
    for (let i = 0; i < nSpine - 1; i++) {
        const dx = spine[(i + 1) * 3] - spine[i * 3];
        const dy = spine[(i + 1) * 3 + 1] - spine[i * 3 + 1];
        const dz = spine[(i + 1) * 3 + 2] - spine[i * 3 + 2];
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > maxSegLen2) maxSegLen2 = d2;
    }
    if (maxDim <= 0) return;
    const tol = 0.05 * maxDim;
    const tolSq = tol * tol;
    const rejectDist = maxDim + Math.sqrt(maxSegLen2) + tol;
    const rejectDist2 = rejectDist * rejectDist;
    const ringStride = nCs * 3;
    // Precompute per-i highest qualifying k. -1 means no candidate.
    const kMax = new Int32Array(nSpine);
    for (let i = 0; i < nSpine; i++) {
        const isx = spine[i * 3], isy = spine[i * 3 + 1], isz = spine[i * 3 + 2];
        const kHi = Math.min(nSpine - 2, i + WIN);
        let best = -1;
        for (let k = kHi; k >= i + 2; k--) {
            const dx = spine[k * 3] - isx;
            const dy = spine[k * 3 + 1] - isy;
            const dz = spine[k * 3 + 2] - isz;
            if (dx * dx + dy * dy + dz * dz <= rejectDist2) {
                best = k;
                break;
            }
        }
        kMax[i] = best;
    }
    const rangeStart = [], rangeEnd = [];
    for (let j = 0; j < nCs; j++) {
        rangeStart.length = 0; rangeEnd.length = 0;
        for (let i = 0; i < nSpine; i++) {
            const kHi = kMax[i];
            if (kHi < i + 2) continue;
            const ip = i * ringStride + j * 3;
            const px = positions[ip], py = positions[ip + 1], pz = positions[ip + 2];
            for (let k = kHi; k >= i + 2; k--) {
                const ka = k * ringStride + j * 3;
                const kb = (k + 1) * ringStride + j * 3;
                const ax = positions[ka], ay = positions[ka + 1], az = positions[ka + 2];
                const ex = positions[kb] - ax, ey = positions[kb + 1] - ay, ez = positions[kb + 2] - az;
                const eL2 = ex * ex + ey * ey + ez * ez;
                if (eL2 < 1e-24) continue;
                const t = ((px - ax) * ex + (py - ay) * ey + (pz - az) * ez) / eL2;
                if (t < 0 || t > 1) continue;
                const dx = px - (ax + t * ex);
                const dy = py - (ay + t * ey);
                const dz = pz - (az + t * ez);
                if (dx * dx + dy * dy + dz * dz < tolSq) {
                    rangeStart.push(i); rangeEnd.push(k + 1);
                    break;
                }
            }
        }
        applyMergedCollapses(positions, ringStride, j, nSpine, rangeStart, rangeEnd);
    }
}

// ---- V5: bidirectional V1 + collapse to segment-segment closest-pair
// midpoint (current ship). For each detection (i, k), the per-detection
// fold target is the midpoint of the closest point on segment(p_a, p_{a+1})
// and the closest point on segment(p_k, p_{k+1}), where a = i for forward
// detection and a = i-1 for backward. Sits in the gap between the two
// folding segments, so averaging across the merged range stays at the
// geometric fold location instead of drifting toward the spine.

function _segSegMidpoint(positions, ringStride, j, aIdx, bIdx, out) {
    const a0 = aIdx * ringStride + j * 3;
    const a1 = (aIdx + 1) * ringStride + j * 3;
    const b0 = bIdx * ringStride + j * 3;
    const b1 = (bIdx + 1) * ringStride + j * 3;
    const p0x = positions[a0], p0y = positions[a0 + 1], p0z = positions[a0 + 2];
    const p1x = positions[a1], p1y = positions[a1 + 1], p1z = positions[a1 + 2];
    const q0x = positions[b0], q0y = positions[b0 + 1], q0z = positions[b0 + 2];
    const q1x = positions[b1], q1y = positions[b1 + 1], q1z = positions[b1 + 2];
    const dx = p1x - p0x, dy = p1y - p0y, dz = p1z - p0z;
    const ex = q1x - q0x, ey = q1y - q0y, ez = q1z - q0z;
    const rx = p0x - q0x, ry = p0y - q0y, rz = p0z - q0z;
    const dL2 = dx * dx + dy * dy + dz * dz;
    const eL2 = ex * ex + ey * ey + ez * ez;
    const f = ex * rx + ey * ry + ez * rz;
    let s = 0, t = 0;
    if (dL2 > 1e-24 && eL2 > 1e-24) {
        const b = dx * ex + dy * ey + dz * ez;
        const c = dx * rx + dy * ry + dz * rz;
        const denom = dL2 * eL2 - b * b;
        s = denom !== 0 ? (b * f - c * eL2) / denom : 0;
        if (s < 0) s = 0; else if (s > 1) s = 1;
        t = (b * s + f) / eL2;
        if (t < 0) {
            t = 0;
            s = -c / dL2;
            if (s < 0) s = 0; else if (s > 1) s = 1;
        } else if (t > 1) {
            t = 1;
            s = (b - c) / dL2;
            if (s < 0) s = 0; else if (s > 1) s = 1;
        }
    } else if (eL2 > 1e-24) {
        t = f / eL2;
        if (t < 0) t = 0; else if (t > 1) t = 1;
    }
    out[0] = 0.5 * (p0x + s * dx + q0x + t * ex);
    out[1] = 0.5 * (p0y + s * dy + q0y + t * ey);
    out[2] = 0.5 * (p0z + s * dz + q0z + t * ez);
}

function collapseV5(positions, widths, heights, nSpine, nCs, _spine) {
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
    const rangeStart = [], rangeEnd = [];
    const foldX = [], foldY = [], foldZ = [];
    const foldOut = new Float64Array(3);
    for (let j = 0; j < nCs; j++) {
        rangeStart.length = 0; rangeEnd.length = 0;
        foldX.length = 0; foldY.length = 0; foldZ.length = 0;
        for (let i = 0; i < nSpine; i++) {
            const ip = i * ringStride + j * 3;
            const px = positions[ip], py = positions[ip + 1], pz = positions[ip + 2];
            const kHi = Math.min(nSpine - 2, i + WIN);
            for (let k = kHi; k >= i + 2; k--) {
                const ka = k * ringStride + j * 3;
                const kb = (k + 1) * ringStride + j * 3;
                const ax = positions[ka], ay = positions[ka + 1], az = positions[ka + 2];
                const ex = positions[kb] - ax, ey = positions[kb + 1] - ay, ez = positions[kb + 2] - az;
                const eL2 = ex * ex + ey * ey + ez * ez;
                if (eL2 < 1e-24) continue;
                const t = ((px - ax) * ex + (py - ay) * ey + (pz - az) * ez) / eL2;
                if (t < 0 || t > 1) continue;
                const dx = px - (ax + t * ex);
                const dy = py - (ay + t * ey);
                const dz = pz - (az + t * ez);
                if (dx * dx + dy * dy + dz * dz < tolSq) {
                    rangeStart.push(i); rangeEnd.push(k + 1);
                    _segSegMidpoint(positions, ringStride, j, i, k, foldOut);
                    foldX.push(foldOut[0]); foldY.push(foldOut[1]); foldZ.push(foldOut[2]);
                    break;
                }
            }
            const kLo = Math.max(0, i - WIN);
            for (let k = kLo; k <= i - 2; k++) {
                const ka = k * ringStride + j * 3;
                const kb = (k + 1) * ringStride + j * 3;
                const ax = positions[ka], ay = positions[ka + 1], az = positions[ka + 2];
                const ex = positions[kb] - ax, ey = positions[kb + 1] - ay, ez = positions[kb + 2] - az;
                const eL2 = ex * ex + ey * ey + ez * ez;
                if (eL2 < 1e-24) continue;
                const t = ((px - ax) * ex + (py - ay) * ey + (pz - az) * ez) / eL2;
                if (t < 0 || t > 1) continue;
                const dx = px - (ax + t * ex);
                const dy = py - (ay + t * ey);
                const dz = pz - (az + t * ez);
                if (dx * dx + dy * dy + dz * dz < tolSq) {
                    rangeStart.push(k); rangeEnd.push(i);
                    _segSegMidpoint(positions, ringStride, j, i - 1, k, foldOut);
                    foldX.push(foldOut[0]); foldY.push(foldOut[1]); foldZ.push(foldOut[2]);
                    break;
                }
            }
        }
        if (rangeStart.length === 0) continue;
        const order = new Int32Array(rangeStart.length);
        for (let m = 0; m < order.length; m++) order[m] = m;
        const _s = rangeStart.slice();
        const _e = rangeEnd.slice();
        const _fx = foldX.slice();
        const _fy = foldY.slice();
        const _fz = foldZ.slice();
        Array.prototype.sort.call(order, (a, b) => _s[a] - _s[b]);
        const o0 = order[0];
        const mS = [_s[o0]], mE = [_e[o0]];
        const mFx = [_fx[o0]], mFy = [_fy[o0]], mFz = [_fz[o0]], mCnt = [1];
        for (let m = 1; m < order.length; m++) {
            const o = order[m];
            const s = _s[o], e = _e[o];
            const last = mE.length - 1;
            if (s <= mE[last]) {
                if (e > mE[last]) mE[last] = e;
                mFx[last] += _fx[o]; mFy[last] += _fy[o]; mFz[last] += _fz[o];
                mCnt[last]++;
            } else {
                mS.push(s); mE.push(e);
                mFx.push(_fx[o]); mFy.push(_fy[o]); mFz.push(_fz[o]);
                mCnt.push(1);
            }
        }
        for (let m = 0; m < mS.length; m++) {
            const s = Math.max(1, mS[m]);
            const e = Math.min(nSpine - 2, mE[m]);
            if (e < s) continue;
            const inv = 1 / mCnt[m];
            const cx = mFx[m] * inv, cy = mFy[m] * inv, cz = mFz[m] * inv;
            for (let i = s; i <= e; i++) {
                const ip = i * ringStride + j * 3;
                positions[ip] = cx; positions[ip + 1] = cy; positions[ip + 2] = cz;
            }
        }
    }
}

// ---- Shared merge + collapse application ----

function applyMergedCollapses(positions, ringStride, j, nSpine, rangeStart, rangeEnd) {
    if (rangeStart.length === 0) return;
    const mStart = [rangeStart[0]];
    const mEnd = [rangeEnd[0]];
    for (let m = 1; m < rangeStart.length; m++) {
        const s = rangeStart[m], e = rangeEnd[m];
        const last = mEnd.length - 1;
        if (s <= mEnd[last]) {
            if (e > mEnd[last]) mEnd[last] = e;
        } else {
            mStart.push(s); mEnd.push(e);
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

// ---- Bench harness ----

function hashPositions(p) {
    let h = 0x9e3779b9;
    for (let i = 0; i < p.length; i++) {
        const v = Math.round(p[i] * 1e6) | 0;
        h = ((h ^ v) * 16777619) | 0;
    }
    return h >>> 0;
}

function run(label, fn, scratch, positions0, widths, heights, nSpine, spine, iters) {
    // Warmup
    for (let i = 0; i < 3; i++) {
        scratch.set(positions0);
        fn(scratch, widths, heights, nSpine, N_CS, spine);
    }
    const start = performance.now();
    for (let i = 0; i < iters; i++) {
        scratch.set(positions0);
        fn(scratch, widths, heights, nSpine, N_CS, spine);
    }
    return (performance.now() - start) / iters;
}

const ctrl = blobbyControlPoints(13, 1.10, 0.45, 7);
const RADIUS = 0.35;

// Sizes: vary samples-per-segment to hit ~target spine lengths.
const targets = [130, 1300, 13000, 130000, 1300000];
const itersFor = (n) => {
    if (n < 200) return 2000;
    if (n < 2000) return 200;
    if (n < 20000) return 30;
    if (n < 200000) return 5;
    return 2;
};

console.log("collapseTubeStrandFolds — variant comparison\n");
console.log("V0 = forward-only, log all hits (original)");
console.log("V1 = forward-only, backwards inner scan + first-hit break");
console.log("V2 = V1 + cumulative-spine-turn gate (UNSAFE: misses folds)");
console.log("V3 = V1 + per-(i,k) spine-distance pre-filter");
console.log("V4 = V1 + V3's pre-filter hoisted into one per-i precompute pass");
console.log("V5 = bidirectional V1 — current ship after PR review (correctness fix; output differs from V0/V1 by design)\n");
console.log("N         | V0 ms     | V1 (vs V0)      | V2 (vs V0)      | V3 (vs V0)      | V4 (vs V0)      | V5 (vs V0)      | hits");
console.log("----------+-----------+-----------------+-----------------+-----------------+-----------------+-----------------+-----");

for (const target of targets) {
    const samplesPerSeg = Math.max(1, Math.round(target / 13));
    const spineXY = catmullRomPeriodic(ctrl, samplesPerSeg);
    const spine = liftXY(spineXY);
    const nSpine = spine.length / 3;
    const widths = new Float32Array(nSpine).fill(2 * RADIUS);
    const heights = new Float32Array(nSpine).fill(2 * RADIUS);
    const positions0 = buildTestPositions(spine, N_CS, RADIUS);
    const scratch = new Float32Array(positions0.length);
    const iters = itersFor(nSpine);

    // Correctness check: all variants must produce identical final positions
    // (within Float32 precision).
    const v0out = new Float32Array(positions0.length);
    v0out.set(positions0);
    collapseV0(v0out, widths, heights, nSpine, N_CS, spine);
    // V5 is intentionally NOT compared element-wise to V0: bidirectional scan
    // catches asymmetric folds V0 misses, so outputs differ by design.
    const labels = ["V1", "V2", "V3", "V4"];
    const others = [collapseV1, collapseV2, collapseV3, collapseV4];
    const mismatches = [];
    for (let f = 0; f < others.length; f++) {
        const cmp = new Float32Array(positions0.length);
        cmp.set(positions0);
        others[f](cmp, widths, heights, nSpine, N_CS, spine);
        let maxAbs = 0;
        let firstIdx = -1;
        for (let i = 0; i < cmp.length; i++) {
            const d = Math.abs(cmp[i] - v0out[i]);
            if (d > maxAbs) maxAbs = d;
            if (d > 1e-4 && firstIdx < 0) firstIdx = i;
        }
        if (firstIdx >= 0) mismatches.push(`${labels[f]} maxΔ=${maxAbs.toExponential(2)} @${firstIdx}`);
    }
    const allSame = mismatches.length === 0;

    // Count hits on the unmodified positions (no collapses applied) for context.
    const check = new Float32Array(positions0.length);
    check.set(positions0);
    let hitCount = 0;
    {
        const ringStride = N_CS * 3;
        let maxDim = 0;
        for (let i = 0; i < nSpine; i++) {
            if (widths[i] > maxDim) maxDim = widths[i];
            if (heights[i] > maxDim) maxDim = heights[i];
        }
        const tol = 0.05 * maxDim;
        const tolSq = tol * tol;
        for (let j = 0; j < N_CS; j++) {
            for (let i = 0; i < nSpine; i++) {
                const ip = i * ringStride + j * 3;
                const px = check[ip], py = check[ip + 1], pz = check[ip + 2];
                const kHi = Math.min(nSpine - 2, i + WIN);
                for (let k = i + 2; k <= kHi; k++) {
                    const ka = k * ringStride + j * 3;
                    const kb = (k + 1) * ringStride + j * 3;
                    const ax = check[ka], ay = check[ka + 1], az = check[ka + 2];
                    const ex = check[kb] - ax, ey = check[kb + 1] - ay, ez = check[kb + 2] - az;
                    const eL2 = ex * ex + ey * ey + ez * ez;
                    if (eL2 < 1e-24) continue;
                    const t = ((px - ax) * ex + (py - ay) * ey + (pz - az) * ez) / eL2;
                    if (t < 0 || t > 1) continue;
                    const dx = px - (ax + t * ex);
                    const dy = py - (ay + t * ey);
                    const dz = pz - (az + t * ez);
                    if (dx * dx + dy * dy + dz * dz < tolSq) hitCount++;
                }
            }
        }
    }

    const t0 = run("V0", collapseV0, scratch, positions0, widths, heights, nSpine, spine, iters);
    const t1 = run("V1", collapseV1, scratch, positions0, widths, heights, nSpine, spine, iters);
    const t2 = run("V2", collapseV2, scratch, positions0, widths, heights, nSpine, spine, iters);
    const t3 = run("V3", collapseV3, scratch, positions0, widths, heights, nSpine, spine, iters);
    const t4 = run("V4", collapseV4, scratch, positions0, widths, heights, nSpine, spine, iters);
    const t5 = run("V5", collapseV5, scratch, positions0, widths, heights, nSpine, spine, iters);

    const fmt = (x) => (x < 1 ? x.toFixed(3) : x.toFixed(2)).padStart(7);
    const ratio = (a, b) => `${(a / b).toFixed(2)}×`.padStart(7);
    console.log(
        `${String(nSpine).padStart(9)} | ${fmt(t0)} ms | ${fmt(t1)} ${ratio(t0, t1)} | ${fmt(t2)} ${ratio(t0, t2)} | ${fmt(t3)} ${ratio(t0, t3)} | ${fmt(t4)} ${ratio(t0, t4)} | ${fmt(t5)} ${ratio(t0, t5)} | ${hitCount}` + (allSame ? "" : ` [${mismatches.join(", ")}]`)
    );
}
console.log("\n(speedups vs V0; higher = faster; 'MISMATCH' = variant produced different output)");
