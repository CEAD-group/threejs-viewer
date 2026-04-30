// Benchmark for the local-min strand-collapse algorithm.
// Run with: node bench/strand_collapse_bench.mjs
//
// V0 — current ship (un-mitered strand polyline build per strand,
//      full (i, j) seg-seg dist² fill, 8-neighbour local-min sweep,
//      mitered-mesh midpoint snap).
// V1 — V0 + per-i spine-distance pre-filter (kMax). Pairs whose spine
//      endpoints sit further apart than the worst-case fold radius
//      can't possibly fold — skipped before the dist² compute.

const N_CS = 6;
const WIN = 50;
const MIN_GAP = 4;
const TOL_FRAC = 0.04;

// ---- Mocks of the production helpers ----

function sampleChamferedRect(out, width, height) {
    const hw = width * 0.5, hh = height * 0.5;
    const c = Math.min(hw, hh);
    if (width >= height) {
        out[0]  = +hw;        out[1]  = 0;
        out[2]  = +(hw - c);  out[3]  = +hh;
        out[4]  = -(hw - c);  out[5]  = +hh;
        out[6]  = -hw;        out[7]  = 0;
        out[8]  = -(hw - c);  out[9]  = -hh;
        out[10] = +(hw - c);  out[11] = -hh;
    } else {
        out[0]  = +hw;        out[1]  = -(hh - c);
        out[2]  = +hw;        out[3]  = +(hh - c);
        out[4]  = 0;          out[5]  = +hh;
        out[6]  = -hw;        out[7]  = +(hh - c);
        out[8]  = -hw;        out[9]  = -(hh - c);
        out[10] = 0;          out[11] = -hh;
    }
}

function segSegMidpoint(p0x, p0y, p0z, p1x, p1y, p1z,
                       q0x, q0y, q0z, q1x, q1y, q1z, out) {
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
    const Psx = p0x + s * dx, Psy = p0y + s * dy, Psz = p0z + s * dz;
    const Qtx = q0x + t * ex, Qty = q0y + t * ey, Qtz = q0z + t * ez;
    out[0] = 0.5 * (Psx + Qtx);
    out[1] = 0.5 * (Psy + Qty);
    out[2] = 0.5 * (Psz + Qtz);
    const gx = Psx - Qtx, gy = Psy - Qty, gz = Psz - Qtz;
    out[3] = gx * gx + gy * gy + gz * gz;
}

function polySegSeg(poly, i, j, out) {
    const a0 = i * 3, a1 = (i + 1) * 3;
    const b0 = j * 3, b1 = (j + 1) * 3;
    segSegMidpoint(
        poly[a0], poly[a0+1], poly[a0+2],
        poly[a1], poly[a1+1], poly[a1+2],
        poly[b0], poly[b0+1], poly[b0+2],
        poly[b1], poly[b1+1], poly[b1+2],
        out,
    );
}

function meshSegSeg(positions, ringStride, k, i, j, out) {
    const a0 = i * ringStride + k * 3;
    const a1 = (i + 1) * ringStride + k * 3;
    const b0 = j * ringStride + k * 3;
    const b1 = (j + 1) * ringStride + k * 3;
    segSegMidpoint(
        positions[a0], positions[a0+1], positions[a0+2],
        positions[a1], positions[a1+1], positions[a1+2],
        positions[b0], positions[b0+1], positions[b0+2],
        positions[b1], positions[b1+1], positions[b1+2],
        out,
    );
}

// ---- V0: current ship ----

function collapseV0(positions, spine, widths, heights, localFrames, nSpine, nCs) {
    const stride = WIN - MIN_GAP + 1;
    const nSeg = nSpine - 1;
    if (stride <= 0 || nSeg < MIN_GAP + 1) return;
    let maxDim = 0;
    for (let i = 0; i < nSpine; i++) {
        if (widths[i] > maxDim) maxDim = widths[i];
        if (heights[i] > maxDim) maxDim = heights[i];
    }
    if (maxDim <= 0) return;
    const tol = TOL_FRAC * maxDim;
    const tolSq = tol * tol;
    const ringStride = nCs * 3;
    const dist2 = new Float32Array(nSeg * stride);
    const strandPoly = new Float32Array(nSpine * 3);
    const sec = new Float32Array(nCs * 2);
    const segOut = new Float64Array(4);
    const foldOut = new Float64Array(4);

    for (let k = 0; k < nCs; k++) {
        for (let i = 0; i < nSpine; i++) {
            sampleChamferedRect(sec, widths[i], heights[i]);
            const u = sec[k * 2], v = sec[k * 2 + 1];
            const Ux = localFrames[i*6],   Uy = localFrames[i*6+1], Uz = localFrames[i*6+2];
            const Vx = localFrames[i*6+3], Vy = localFrames[i*6+4], Vz = localFrames[i*6+5];
            strandPoly[i*3]     = spine[i*3]     + u*Ux + v*Vx;
            strandPoly[i*3 + 1] = spine[i*3 + 1] + u*Uy + v*Vy;
            strandPoly[i*3 + 2] = spine[i*3 + 2] + u*Uz + v*Vz;
        }
        for (let i = 0; i < nSeg; i++) {
            const rb = i * stride;
            for (let off = 0; off < stride; off++) {
                const j = i + MIN_GAP + off;
                if (j >= nSeg) {
                    dist2[rb + off] = Infinity;
                } else {
                    polySegSeg(strandPoly, i, j, segOut);
                    dist2[rb + off] = segOut[3];
                }
            }
        }
        for (let i = 0; i < nSeg; i++) {
            const rb = i * stride;
            for (let off = 0; off < stride; off++) {
                const j = i + MIN_GAP + off;
                if (j >= nSeg) break;
                const d = dist2[rb + off];
                if (!(d < tolSq)) continue;
                let isMin = true;
                outer:
                for (let di = -1; di <= 1; di++) {
                    const ni = i + di;
                    if (ni < 0 || ni >= nSeg) continue;
                    const niRow = ni * stride;
                    for (let dj = -1; dj <= 1; dj++) {
                        if (di === 0 && dj === 0) continue;
                        const noff = off + (dj - di);
                        if (noff < 0 || noff >= stride) continue;
                        if (dist2[niRow + noff] <= d) { isMin = false; break outer; }
                    }
                }
                if (!isMin) continue;
                meshSegSeg(positions, ringStride, k, i, j, foldOut);
                const cx = foldOut[0], cy = foldOut[1], cz = foldOut[2];
                const sLo = Math.max(1, i + 1);
                const sHi = Math.min(nSpine - 2, j);
                for (let r = sLo; r <= sHi; r++) {
                    const ip = r * ringStride + k * 3;
                    positions[ip] = cx;
                    positions[ip + 1] = cy;
                    positions[ip + 2] = cz;
                }
            }
        }
    }
}

// ---- V3: V2 + cache-tiled streaming dist² (3-row sliding buffer) ----
//
// V0/V1/V2 allocate dist²[nSeg × stride] (24 MB at N=130k), way bigger
// than L2. Sequential writes are cheap but the local-min sweep's 8-
// neighbour reads cross row boundaries — every read is a potential
// cache miss. V3 keeps only three rows of dist² alive at once (~564 B),
// fully L1-resident. On every iteration we fill the row two ahead, then
// check the centre row using its three-row neighbourhood.
//
// strandPoly stays the same (1.6 MB at N=130k, sequential, fits L2).
// Sentinel rows (full-Infinity) act as out-of-bounds neighbours so the
// loop body has no conditionals on prev/next existence.

function collapseV3(positions, spine, widths, heights, localFrames, nSpine, nCs) {
    const stride = WIN - MIN_GAP + 1;
    const nSeg = nSpine - 1;
    if (stride <= 0 || nSeg < MIN_GAP + 1) return;
    let maxDim = 0;
    let maxSegLen2 = 0;
    for (let i = 0; i < nSpine; i++) {
        if (widths[i] > maxDim) maxDim = widths[i];
        if (heights[i] > maxDim) maxDim = heights[i];
    }
    for (let i = 0; i < nSpine - 1; i++) {
        const dx = spine[(i+1)*3] - spine[i*3];
        const dy = spine[(i+1)*3+1] - spine[i*3+1];
        const dz = spine[(i+1)*3+2] - spine[i*3+2];
        const d2 = dx*dx + dy*dy + dz*dz;
        if (d2 > maxSegLen2) maxSegLen2 = d2;
    }
    if (maxDim <= 0) return;
    const tol = TOL_FRAC * maxDim;
    const tolSq = tol * tol;
    const ringStride = nCs * 3;
    const reject = maxDim + 2 * Math.sqrt(maxSegLen2) + tol;
    const rejectSq = reject * reject;
    const kMax = new Int32Array(nSeg);
    for (let i = 0; i < nSeg; i++) {
        const isx = spine[i*3], isy = spine[i*3+1], isz = spine[i*3+2];
        const jHi = Math.min(nSeg - 1, i + WIN);
        let best = -1;
        for (let j = jHi; j >= i + MIN_GAP; j--) {
            const dx = spine[j*3] - isx;
            const dy = spine[j*3+1] - isy;
            const dz = spine[j*3+2] - isz;
            if (dx*dx + dy*dy + dz*dz <= rejectSq) { best = j; break; }
        }
        kMax[i] = best;
    }

    const strandPoly = new Float32Array(nSpine * 3);
    const sec = new Float32Array(nCs * 2);
    const segOut = new Float64Array(4);
    const foldOut = new Float64Array(4);
    // Three-row sliding dist² buffer + one sentinel for out-of-bounds.
    const rowBuf = [
        new Float32Array(stride),
        new Float32Array(stride),
        new Float32Array(stride),
    ];
    const infRow = new Float32Array(stride).fill(Infinity);

    function fillRow(buf, i) {
        const jLim = kMax[i];
        if (jLim < 0) {
            for (let off = 0; off < stride; off++) buf[off] = Infinity;
            return;
        }
        const offMax = Math.min(jLim - i - MIN_GAP, stride - 1);
        let off = 0;
        for (; off <= offMax; off++) {
            polySegSeg(strandPoly, i, i + MIN_GAP + off, segOut);
            buf[off] = segOut[3];
        }
        for (; off < stride; off++) buf[off] = Infinity;
    }

    for (let k = 0; k < nCs; k++) {
        for (let i = 0; i < nSpine; i++) {
            sampleChamferedRect(sec, widths[i], heights[i]);
            const u = sec[k * 2], v = sec[k * 2 + 1];
            const Ux = localFrames[i*6],   Uy = localFrames[i*6+1], Uz = localFrames[i*6+2];
            const Vx = localFrames[i*6+3], Vy = localFrames[i*6+4], Vz = localFrames[i*6+5];
            strandPoly[i*3]     = spine[i*3]     + u*Ux + v*Vx;
            strandPoly[i*3 + 1] = spine[i*3 + 1] + u*Uy + v*Vy;
            strandPoly[i*3 + 2] = spine[i*3 + 2] + u*Uz + v*Vz;
        }

        // Pre-fill row 0 and row 1 (when present) so the loop can read
        // a valid `next` at i=0.
        fillRow(rowBuf[0], 0);
        if (nSeg > 1) fillRow(rowBuf[1], 1);

        for (let i = 0; i < nSeg; i++) {
            const prev = i > 0 ? rowBuf[(i - 1) % 3] : infRow;
            const curr = rowBuf[i % 3];
            const next = i + 1 < nSeg ? rowBuf[(i + 1) % 3] : infRow;
            const jLim = kMax[i];
            if (jLim >= 0) {
                const offMax = Math.min(jLim - i - MIN_GAP, stride - 1);
                for (let off = 0; off <= offMax; off++) {
                    const j = i + MIN_GAP + off;
                    if (j >= nSeg) break;
                    const d = curr[off];
                    if (!(d < tolSq)) continue;
                    let isMin = true;
                    // Inlined 3-row × 3-col neighbour check (8 neighbours).
                    // prev row neighbours (di = -1): nooff = off + (dj + 1)
                    // curr row neighbours (di =  0): noff  = off + dj   (skip dj=0)
                    // next row neighbours (di = +1): noff  = off + (dj - 1)
                    if (off + 0 >= 0 && off + 0 < stride && prev[off + 0] <= d) isMin = false;
                    else if (off + 1 >= 0 && off + 1 < stride && prev[off + 1] <= d) isMin = false;
                    else if (off + 2 >= 0 && off + 2 < stride && prev[off + 2] <= d) isMin = false;
                    else if (off - 1 >= 0 && off - 1 < stride && curr[off - 1] <= d) isMin = false;
                    else if (off + 1 >= 0 && off + 1 < stride && curr[off + 1] <= d) isMin = false;
                    else if (off - 2 >= 0 && off - 2 < stride && next[off - 2] <= d) isMin = false;
                    else if (off - 1 >= 0 && off - 1 < stride && next[off - 1] <= d) isMin = false;
                    else if (off + 0 >= 0 && off + 0 < stride && next[off + 0] <= d) isMin = false;
                    if (!isMin) continue;
                    meshSegSeg(positions, ringStride, k, i, j, foldOut);
                    const cx = foldOut[0], cy = foldOut[1], cz = foldOut[2];
                    const sLo = Math.max(1, i + 1);
                    const sHi = Math.min(nSpine - 2, j);
                    for (let r = sLo; r <= sHi; r++) {
                        const ip = r * ringStride + k * 3;
                        positions[ip] = cx;
                        positions[ip + 1] = cy;
                        positions[ip + 2] = cz;
                    }
                }
            }
            // Pre-fill row i+2 (overwrites slot (i-1)%3, no longer needed).
            if (i + 2 < nSeg) {
                fillRow(rowBuf[(i + 2) % 3], i + 2);
            }
        }
    }
}

// ---- V2: V1 with the kMax check hoisted out of the inner loop ----
//
// V1's branch inside polySegSeg's inner loop confused V8's optimizer and
// gave only ~1.1×. Splitting fill into two consecutive runs (compute up
// to offMax, then fill the tail with Infinity) removes the per-iteration
// branch and lets V8 keep tight code on the hot path.

function collapseV2(positions, spine, widths, heights, localFrames, nSpine, nCs) {
    const stride = WIN - MIN_GAP + 1;
    const nSeg = nSpine - 1;
    if (stride <= 0 || nSeg < MIN_GAP + 1) return;
    let maxDim = 0;
    let maxSegLen2 = 0;
    for (let i = 0; i < nSpine; i++) {
        if (widths[i] > maxDim) maxDim = widths[i];
        if (heights[i] > maxDim) maxDim = heights[i];
    }
    for (let i = 0; i < nSpine - 1; i++) {
        const dx = spine[(i+1)*3] - spine[i*3];
        const dy = spine[(i+1)*3+1] - spine[i*3+1];
        const dz = spine[(i+1)*3+2] - spine[i*3+2];
        const d2 = dx*dx + dy*dy + dz*dz;
        if (d2 > maxSegLen2) maxSegLen2 = d2;
    }
    if (maxDim <= 0) return;
    const tol = TOL_FRAC * maxDim;
    const tolSq = tol * tol;
    const ringStride = nCs * 3;
    const reject = maxDim + 2 * Math.sqrt(maxSegLen2) + tol;
    const rejectSq = reject * reject;
    const kMax = new Int32Array(nSeg);
    for (let i = 0; i < nSeg; i++) {
        const isx = spine[i*3], isy = spine[i*3+1], isz = spine[i*3+2];
        const jHi = Math.min(nSeg - 1, i + WIN);
        let best = -1;
        for (let j = jHi; j >= i + MIN_GAP; j--) {
            const dx = spine[j*3] - isx;
            const dy = spine[j*3+1] - isy;
            const dz = spine[j*3+2] - isz;
            if (dx*dx + dy*dy + dz*dz <= rejectSq) { best = j; break; }
        }
        kMax[i] = best;
    }
    const dist2 = new Float32Array(nSeg * stride);
    const strandPoly = new Float32Array(nSpine * 3);
    const sec = new Float32Array(nCs * 2);
    const segOut = new Float64Array(4);
    const foldOut = new Float64Array(4);

    for (let k = 0; k < nCs; k++) {
        for (let i = 0; i < nSpine; i++) {
            sampleChamferedRect(sec, widths[i], heights[i]);
            const u = sec[k * 2], v = sec[k * 2 + 1];
            const Ux = localFrames[i*6],   Uy = localFrames[i*6+1], Uz = localFrames[i*6+2];
            const Vx = localFrames[i*6+3], Vy = localFrames[i*6+4], Vz = localFrames[i*6+5];
            strandPoly[i*3]     = spine[i*3]     + u*Ux + v*Vx;
            strandPoly[i*3 + 1] = spine[i*3 + 1] + u*Uy + v*Vy;
            strandPoly[i*3 + 2] = spine[i*3 + 2] + u*Uz + v*Vz;
        }
        for (let i = 0; i < nSeg; i++) {
            const rb = i * stride;
            const jLim = kMax[i];
            if (jLim < 0) {
                for (let off = 0; off < stride; off++) dist2[rb + off] = Infinity;
                continue;
            }
            const offMax = Math.min(jLim - i - MIN_GAP, stride - 1);
            let off = 0;
            for (; off <= offMax; off++) {
                polySegSeg(strandPoly, i, i + MIN_GAP + off, segOut);
                dist2[rb + off] = segOut[3];
            }
            for (; off < stride; off++) dist2[rb + off] = Infinity;
        }
        for (let i = 0; i < nSeg; i++) {
            const rb = i * stride;
            const jLim = kMax[i];
            if (jLim < 0) continue;
            const offMax = Math.min(jLim - i - MIN_GAP, stride - 1);
            for (let off = 0; off <= offMax; off++) {
                const j = i + MIN_GAP + off;
                if (j >= nSeg) break;
                const d = dist2[rb + off];
                if (!(d < tolSq)) continue;
                let isMin = true;
                outer:
                for (let di = -1; di <= 1; di++) {
                    const ni = i + di;
                    if (ni < 0 || ni >= nSeg) continue;
                    const niRow = ni * stride;
                    for (let dj = -1; dj <= 1; dj++) {
                        if (di === 0 && dj === 0) continue;
                        const noff = off + (dj - di);
                        if (noff < 0 || noff >= stride) continue;
                        if (dist2[niRow + noff] <= d) { isMin = false; break outer; }
                    }
                }
                if (!isMin) continue;
                meshSegSeg(positions, ringStride, k, i, j, foldOut);
                const cx = foldOut[0], cy = foldOut[1], cz = foldOut[2];
                const sLo = Math.max(1, i + 1);
                const sHi = Math.min(nSpine - 2, j);
                for (let r = sLo; r <= sHi; r++) {
                    const ip = r * ringStride + k * 3;
                    positions[ip] = cx;
                    positions[ip + 1] = cy;
                    positions[ip + 2] = cz;
                }
            }
        }
    }
}

// ---- V1: V0 + per-i spine-distance pre-filter (kMax precompute) ----
//
// Bound: seg-seg distance ≥ ‖spine[i] - spine[j]‖ − 2·maxOffset − 2·maxSegLen,
// where maxOffset = max(W, H) / 2 (worst-case strand offset from spine).
// If that's > tol, no fold possible. Precompute, per i, the highest j within
// the window where the spine endpoints are still close enough; cells beyond
// stay at +Infinity so they're skipped in dist² fill and local-min sweep.

function collapseV1(positions, spine, widths, heights, localFrames, nSpine, nCs) {
    const stride = WIN - MIN_GAP + 1;
    const nSeg = nSpine - 1;
    if (stride <= 0 || nSeg < MIN_GAP + 1) return;
    let maxDim = 0;
    let maxSegLen2 = 0;
    for (let i = 0; i < nSpine; i++) {
        if (widths[i] > maxDim) maxDim = widths[i];
        if (heights[i] > maxDim) maxDim = heights[i];
    }
    for (let i = 0; i < nSpine - 1; i++) {
        const dx = spine[(i+1)*3] - spine[i*3];
        const dy = spine[(i+1)*3+1] - spine[i*3+1];
        const dz = spine[(i+1)*3+2] - spine[i*3+2];
        const d2 = dx*dx + dy*dy + dz*dz;
        if (d2 > maxSegLen2) maxSegLen2 = d2;
    }
    if (maxDim <= 0) return;
    const tol = TOL_FRAC * maxDim;
    const tolSq = tol * tol;
    const ringStride = nCs * 3;
    const reject = maxDim + 2 * Math.sqrt(maxSegLen2) + tol;
    const rejectSq = reject * reject;
    // kMax[i] = highest j in [i+MIN_GAP, i+WIN] with ‖spine[i]−spine[j]‖² ≤ rejectSq.
    // -1 if none. The dist² grid skips cells beyond kMax.
    const kMax = new Int32Array(nSeg);
    for (let i = 0; i < nSeg; i++) {
        const isx = spine[i*3], isy = spine[i*3+1], isz = spine[i*3+2];
        const jHi = Math.min(nSeg - 1, i + WIN);
        let best = -1;
        for (let j = jHi; j >= i + MIN_GAP; j--) {
            const dx = spine[j*3] - isx;
            const dy = spine[j*3+1] - isy;
            const dz = spine[j*3+2] - isz;
            if (dx*dx + dy*dy + dz*dz <= rejectSq) { best = j; break; }
        }
        kMax[i] = best;
    }
    const dist2 = new Float32Array(nSeg * stride);
    const strandPoly = new Float32Array(nSpine * 3);
    const sec = new Float32Array(nCs * 2);
    const segOut = new Float64Array(4);
    const foldOut = new Float64Array(4);

    for (let k = 0; k < nCs; k++) {
        for (let i = 0; i < nSpine; i++) {
            sampleChamferedRect(sec, widths[i], heights[i]);
            const u = sec[k * 2], v = sec[k * 2 + 1];
            const Ux = localFrames[i*6],   Uy = localFrames[i*6+1], Uz = localFrames[i*6+2];
            const Vx = localFrames[i*6+3], Vy = localFrames[i*6+4], Vz = localFrames[i*6+5];
            strandPoly[i*3]     = spine[i*3]     + u*Ux + v*Vx;
            strandPoly[i*3 + 1] = spine[i*3 + 1] + u*Uy + v*Vy;
            strandPoly[i*3 + 2] = spine[i*3 + 2] + u*Uz + v*Vz;
        }
        for (let i = 0; i < nSeg; i++) {
            const rb = i * stride;
            const jLim = kMax[i];
            const offMax = jLim < 0 ? -1 : jLim - i - MIN_GAP;
            for (let off = 0; off < stride; off++) {
                const j = i + MIN_GAP + off;
                if (j >= nSeg || off > offMax) {
                    dist2[rb + off] = Infinity;
                } else {
                    polySegSeg(strandPoly, i, j, segOut);
                    dist2[rb + off] = segOut[3];
                }
            }
        }
        for (let i = 0; i < nSeg; i++) {
            const rb = i * stride;
            const jLim = kMax[i];
            const offMax = jLim < 0 ? -1 : jLim - i - MIN_GAP;
            for (let off = 0; off <= offMax; off++) {
                const j = i + MIN_GAP + off;
                if (j >= nSeg) break;
                const d = dist2[rb + off];
                if (!(d < tolSq)) continue;
                let isMin = true;
                outer:
                for (let di = -1; di <= 1; di++) {
                    const ni = i + di;
                    if (ni < 0 || ni >= nSeg) continue;
                    const niRow = ni * stride;
                    for (let dj = -1; dj <= 1; dj++) {
                        if (di === 0 && dj === 0) continue;
                        const noff = off + (dj - di);
                        if (noff < 0 || noff >= stride) continue;
                        if (dist2[niRow + noff] <= d) { isMin = false; break outer; }
                    }
                }
                if (!isMin) continue;
                meshSegSeg(positions, ringStride, k, i, j, foldOut);
                const cx = foldOut[0], cy = foldOut[1], cz = foldOut[2];
                const sLo = Math.max(1, i + 1);
                const sHi = Math.min(nSpine - 2, j);
                for (let r = sLo; r <= sHi; r++) {
                    const ip = r * ringStride + k * 3;
                    positions[ip] = cx;
                    positions[ip + 1] = cy;
                    positions[ip + 2] = cz;
                }
            }
        }
    }
}

// ---- Test data: blobby Catmull-Rom + chamfered hex + constant-up frames ----

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

function catmullRomPeriodic(points, samplesPerSegment, alpha = 0.5) {
    const n = points.length;
    const knot = (ti, a, b) => {
        const dx = b[0] - a[0], dy = b[1] - a[1];
        return ti + Math.pow(Math.max(Math.sqrt(dx*dx + dy*dy), 1e-9), alpha);
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

function buildTestData(spineXY, beadW, beadH) {
    const nSpine = spineXY.length;
    const spine = new Float32Array(nSpine * 3);
    for (let i = 0; i < nSpine; i++) {
        spine[i*3] = spineXY[i][0];
        spine[i*3+1] = spineXY[i][1];
        spine[i*3+2] = 0;
    }
    const widths = new Float32Array(nSpine).fill(beadW);
    const heights = new Float32Array(nSpine).fill(beadH);
    // Constant-up frames (V = +Z projected ⟂ T, U = V × T, V = T × U).
    const localFrames = new Float32Array(nSpine * 6);
    for (let i = 0; i < nSpine; i++) {
        let tx, ty, tz;
        if (i === 0) {
            tx = spine[3] - spine[0]; ty = spine[4] - spine[1]; tz = spine[5] - spine[2];
        } else if (i === nSpine - 1) {
            tx = spine[i*3] - spine[(i-1)*3];
            ty = spine[i*3+1] - spine[(i-1)*3+1];
            tz = spine[i*3+2] - spine[(i-1)*3+2];
        } else {
            tx = spine[(i+1)*3] - spine[(i-1)*3];
            ty = spine[(i+1)*3+1] - spine[(i-1)*3+1];
            tz = spine[(i+1)*3+2] - spine[(i-1)*3+2];
        }
        const tlen = Math.hypot(tx, ty, tz) || 1;
        tx /= tlen; ty /= tlen; tz /= tlen;
        const dot = tz;
        let vx = -dot * tx, vy = -dot * ty, vz = 1 - dot * tz;
        const vlen = Math.hypot(vx, vy, vz) || 1;
        vx /= vlen; vy /= vlen; vz /= vlen;
        let ux = vy*tz - vz*ty;
        let uy = vz*tx - vx*tz;
        let uz = vx*ty - vy*tx;
        const ulen = Math.hypot(ux, uy, uz) || 1;
        ux /= ulen; uy /= ulen; uz /= ulen;
        // V = T × U
        vx = ty*uz - tz*uy;
        vy = tz*ux - tx*uz;
        vz = tx*uy - ty*ux;
        localFrames[i*6] = ux; localFrames[i*6+1] = uy; localFrames[i*6+2] = uz;
        localFrames[i*6+3] = vx; localFrames[i*6+4] = vy; localFrames[i*6+5] = vz;
    }
    // Mesh positions from chamfered hex (no miter — bench keeps the snap
    // target equal to the un-mitered detection so V0/V1 outputs are
    // bitwise identical and we measure perf only).
    const sec = new Float32Array(N_CS * 2);
    const positions = new Float32Array(nSpine * N_CS * 3);
    for (let i = 0; i < nSpine; i++) {
        sampleChamferedRect(sec, widths[i], heights[i]);
        const Ux = localFrames[i*6], Uy = localFrames[i*6+1], Uz = localFrames[i*6+2];
        const Vx = localFrames[i*6+3], Vy = localFrames[i*6+4], Vz = localFrames[i*6+5];
        const sx = spine[i*3], sy = spine[i*3+1], sz = spine[i*3+2];
        for (let k = 0; k < N_CS; k++) {
            const u = sec[k*2], v = sec[k*2+1];
            const off = (i * N_CS + k) * 3;
            positions[off]     = sx + u*Ux + v*Vx;
            positions[off + 1] = sy + u*Uy + v*Vy;
            positions[off + 2] = sz + u*Uz + v*Vz;
        }
    }
    return { spine, widths, heights, localFrames, positions };
}

// ---- Bench harness ----

function run(label, fn, scratch, positions0, td, iters) {
    for (let i = 0; i < 3; i++) {
        scratch.set(positions0);
        fn(scratch, td.spine, td.widths, td.heights, td.localFrames, td.spine.length / 3, N_CS);
    }
    const start = performance.now();
    for (let i = 0; i < iters; i++) {
        scratch.set(positions0);
        fn(scratch, td.spine, td.widths, td.heights, td.localFrames, td.spine.length / 3, N_CS);
    }
    return (performance.now() - start) / iters;
}

// Bench grid: vary the number of control points so the loop SIZE grows
// with N while sample density (samplesPerSeg = 10) stays realistic for a
// 3D-printer toolpath. base_r scales linearly with nCtrl so chord length
// per spine sample stays ~0.054 ≈ bead height regardless of N. This
// keeps the kMax pre-filter measurably useful (long window = long arc =
// pairs far enough apart in 3D to reject).
const SAMPLES_PER_SEG = 10;
const ARC_PER_CTRL = 0.54;
const BEAD_W = 0.70;
const BEAD_H = 0.30;

const targets = [130, 1300, 13000, 130000];
const itersFor = (n) => {
    if (n < 200) return 1000;
    if (n < 2000) return 200;
    if (n < 20000) return 30;
    if (n < 200000) return 5;
    return 2;
};

console.log("collapseTubeStrandFolds — local-min algorithm\n");
console.log("V0 = un-mitered strand build, full dist² grid, local-min sweep (no pre-filter)");
console.log("V1 = V0 + per-i spine-distance pre-filter (kMax) — skip pairs that can't fold");
console.log("V2 = V1 with kMax check hoisted out of the inner loop (two-phase fill) — current ship");
console.log("V3 = V2 + cache-tiled streaming dist² (3-row sliding buffer, ~600 B working set)\n");
console.log(`grid: SAMPLES_PER_SEG=${SAMPLES_PER_SEG}, W=${BEAD_W}, H=${BEAD_H}, WIN=${WIN}, MIN_GAP=${MIN_GAP}, TOL_FRAC=${TOL_FRAC}\n`);
console.log("N         | nCtrl  | V0 ms     | V1 (vs V0)      | V2 (vs V0)      | V3 (vs V0)      | folds | output");
console.log("----------+--------+-----------+-----------------+-----------------+-----------------+-------+--------");

for (const target of targets) {
    const nCtrl = Math.max(13, Math.round(target / SAMPLES_PER_SEG));
    const baseR = ARC_PER_CTRL * nCtrl / (2 * Math.PI);
    const ctrl = blobbyControlPoints(nCtrl, baseR, 0.45, 7);
    const spineXY = catmullRomPeriodic(ctrl, SAMPLES_PER_SEG);
    const td = buildTestData(spineXY, BEAD_W, BEAD_H);
    const nSpine = td.spine.length / 3;
    const positions0 = td.positions;
    const scratch = new Float32Array(positions0.length);
    const iters = itersFor(nSpine);

    // Correctness: V1, V2, V3 should produce identical positions to V0.
    const v0out = new Float32Array(positions0.length);
    v0out.set(positions0);
    collapseV0(v0out, td.spine, td.widths, td.heights, td.localFrames, nSpine, N_CS);
    const v1out = new Float32Array(positions0.length);
    v1out.set(positions0);
    collapseV1(v1out, td.spine, td.widths, td.heights, td.localFrames, nSpine, N_CS);
    const v2out = new Float32Array(positions0.length);
    v2out.set(positions0);
    collapseV2(v2out, td.spine, td.widths, td.heights, td.localFrames, nSpine, N_CS);
    const v3out = new Float32Array(positions0.length);
    v3out.set(positions0);
    collapseV3(v3out, td.spine, td.widths, td.heights, td.localFrames, nSpine, N_CS);
    let maxDelta = 0;
    for (let i = 0; i < v0out.length; i++) {
        const d1 = Math.abs(v0out[i] - v1out[i]);
        const d2 = Math.abs(v0out[i] - v2out[i]);
        const d3 = Math.abs(v0out[i] - v3out[i]);
        if (d1 > maxDelta) maxDelta = d1;
        if (d2 > maxDelta) maxDelta = d2;
        if (d3 > maxDelta) maxDelta = d3;
    }
    // Count snapped vertices in V0 output for context.
    let snapped = 0;
    for (let i = 0; i < positions0.length; i += 3) {
        if (positions0[i] !== v0out[i] || positions0[i+1] !== v0out[i+1] || positions0[i+2] !== v0out[i+2]) {
            snapped++;
        }
    }

    const t0 = run("V0", collapseV0, scratch, positions0, td, iters);
    const t1 = run("V1", collapseV1, scratch, positions0, td, iters);
    const t2 = run("V2", collapseV2, scratch, positions0, td, iters);
    const t3 = run("V3", collapseV3, scratch, positions0, td, iters);
    const fmt = (x) => (x < 1 ? x.toFixed(3) : x.toFixed(2)).padStart(7);
    const ratio = (a, b) => `${(a / b).toFixed(2)}×`.padStart(7);
    const status = maxDelta < 1e-4 ? "match" : `Δ=${maxDelta.toExponential(2)}`;
    console.log(
        `${String(nSpine).padStart(9)} | ${String(nCtrl).padStart(6)} | ${fmt(t0)} ms | ${fmt(t1)} ${ratio(t0, t1)} | ${fmt(t2)} ${ratio(t0, t2)} | ${fmt(t3)} ${ratio(t0, t3)} | ${String(snapped).padStart(5)} | ${status}`
    );
}
console.log("\n(speedups vs V0; higher = faster)");
