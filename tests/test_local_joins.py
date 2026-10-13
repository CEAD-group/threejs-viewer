"""Geometry and playback contracts for bounded local round joins."""

import numpy as np
import pytest

from threejs_viewer import ViewerClient
from test_parametric_tube import _png_bytes


def _assert_closed(p, indices):
    assert np.isfinite(p).all()
    # Weld geometric seams (top/chamfer and upper/lower need distinct normals).
    _, welded = np.unique(np.round(p, 5), axis=0, return_inverse=True)
    tri = welded[indices]
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2), np.unique(counts, return_counts=True)
    a, b, c = p[indices].transpose(1, 0, 2)
    volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    assert volume > 0


def test_local_join_validation():
    client = ViewerClient(port=0, open_browser=False)
    args = dict(
        id="t",
        spine=[[0, 0, 0], [10, 0, 0]],
        widths=[10, 10],
        heights=[2, 2],
        surface="local_join",
    )
    for patch, message in [
        ({"surface_tolerance": 0}, "positive"),
        ({"surface_tolerance": float("nan")}, "positive"),
        ({"up_vector": [0, 0, 0]}, "up_vector"),
        ({"orientations": [[0, 0, 0, 1]] * 2}, "orientations"),
    ]:
        with pytest.raises(ValueError, match=message):
            client.add_parametric_tube(**(args | patch))


def snapshot(page):
    return page.evaluate("""() => {
        const o=window.threejsViewer._objects.get('t'), f=o.userData.localJoinCompleted;
        const g=f?.geometry, b=o.geometry, offset=b.attributes.position.count;
        const visible=g ? Array.from(g.index.array.slice(g.drawRange.start,g.drawRange.start+g.drawRange.count),i=>i+offset) : [];
        const frontier=b.index.array.slice(b.drawRange.start,b.drawRange.start+b.drawRange.count);
        return {p:[...b.attributes.position.array,...(g?.attributes.position.array||[])], i:[...visible,...frontier],
            stats:o.userData.localJoinStats, committed:o.userData.localJoinCommittedPairs,
            uv:[...b.attributes.uv.array,...(g?.attributes.uv.array||[])], shared:g?.attributes.position===b.attributes.position};
    }""")


@pytest.mark.browser
@pytest.mark.parametrize("kind", ["planar", "tilted", "spatial", "elbow"])
def test_local_join_closed_and_scrubbing(viewer_client, viewer_page, kind):
    p = np.array(
        [
            [-30, 0, 0],
            [-20, 0, 0],
            [-10, 0, 0],
            [0, 0, 0],
            [0, 10, 0],
            [0, 20, 0],
            [0, 30, 0],
        ],
        dtype=float,
    )
    up = [0, 0, 1]
    if kind == "tilted":
        up = np.array([1, 2, 3], dtype=float)
        up /= np.linalg.norm(up)
        u = np.cross(up, [0, 1, 0])
        u /= np.linalg.norm(u)
        v = np.cross(up, u)
        p = p[:, 0, None] * u + p[:, 1, None] * v
        up = up.tolist()
    elif kind == "spatial":
        p[:, 2] = [0, 0.1, 0.2, 0.3, 1, 3, 5]
    elif kind == "elbow":
        p[:, 2] = p[:, 1]
        p[:, 1] = 0
    viewer_client.add_parametric_tube(
        "t",
        p,
        [10] * len(p),
        [2] * len(p),
        up_vector=up,
        surface="local_join",
        texture=_png_bytes(),
    )
    viewer_page.wait_for_function(
        "() => !!window.threejsViewer._objects.get('t')?.userData.localJoinStats"
    )
    s = snapshot(viewer_page)
    assert s["stats"]["accepted"] >= 1
    assert s["stats"]["maxPatchTriangles"] <= 96
    assert not s["shared"]
    assert np.isfinite(s["uv"]).all()
    positions = np.array(s["p"]).reshape(-1, 3)
    triangles = np.array(s["i"]).reshape(-1, 3)
    # The existing revolution cap intentionally collapses its terminal ring.
    vertices = positions[triangles]
    area = np.linalg.norm(
        np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
        axis=1,
    )
    _assert_closed(np.round(positions, 4), triangles[area > 1e-6])
    full = s
    for value in [0.05, 0.48, 0.7, 0.99, 0.2, 0, 1]:
        viewer_page.evaluate(
            "v=>window.threejsViewer._objects.get('t').userData.parametricTube.setDrawRange(v)",
            value,
        )
        s = snapshot(viewer_page)
        assert np.isfinite(s["p"]).all()
        if value in (0.7, 0.99):
            pp = np.round(np.asarray(s["p"]).reshape(-1, 3), 4)
            tt = np.asarray(s["i"]).reshape(-1, 3)
            vv = pp[tt]
            aa = np.linalg.norm(
                np.cross(vv[:, 1] - vv[:, 0], vv[:, 2] - vv[:, 0]), axis=1
            )
            _assert_closed(pp, tt[aa > 1e-6])
        if value == 0:
            assert s["i"] == []
        if value == 1:
            assert s["i"] == full["i"]
            assert np.allclose(s["p"], full["p"], atol=1e-5)


@pytest.mark.browser
def test_spiral_stays_compact(viewer_client, viewer_page):
    theta = np.linspace(0, 80 * np.pi, 20001)
    p = np.column_stack([100 * np.cos(theta), 100 * np.sin(theta), theta * 0.4])
    viewer_client.add_parametric_tube(
        "t",
        p,
        np.full(len(p), 10),
        np.full(len(p), 2),
        surface="local_join",
        texture=_png_bytes(),
    )
    viewer_page.wait_for_function(
        "() => !!window.threejsViewer._objects.get('t')?.userData.localJoinStats"
    )
    s = snapshot(viewer_page)
    assert s["stats"]["addedTriangles"] == 0
    assert s["stats"]["smooth"] == len(p) - 2
    assert s["stats"]["originalTriangles"] == len(s["i"]) / 3


@pytest.mark.browser
@pytest.mark.parametrize(
    "issue,w,h,rise", [(277, 11.83, 2, 0), (278, 9.46, 2.5, 0), (277, 11.83, 2, 0.01)]
)
def test_tight_cluster_is_closed(viewer_client, viewer_page, issue, w, h, rise):
    import json
    from pathlib import Path

    p = np.array(
        json.loads(
            (Path(__file__).parent / f"data/tube_issue_{issue}.json").read_text()
        )
    )
    p -= p[len(p) // 2]
    p[:, 2] += np.arange(len(p)) * rise
    viewer_client.add_parametric_tube(
        "t", p, [w] * len(p), [h] * len(p), surface="local_join", texture=_png_bytes()
    )
    viewer_page.wait_for_function(
        "() => !!window.threejsViewer._objects.get('t')?.userData.localJoinStats"
    )
    s = snapshot(viewer_page)
    assert s["stats"]["clusters"] >= 1, s["stats"]
    assert s["stats"]["maxPatchTriangles"] <= 192
    assert s["stats"]["maxWindowSegments"] <= 10
    positions = np.round(np.array(s["p"]).reshape(-1, 3), 4)
    indices = np.array(s["i"]).reshape(-1, 3)
    v = positions[indices]
    area = np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    _assert_closed(np.round(positions, 4), indices[area > 1e-6])
    if rise == 0:
        cross = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        top = v[
            (v[:, :, 2].mean(axis=1) > h * 0.25)
            & (np.linalg.norm(cross[:, :2], axis=1) < np.abs(cross[:, 2]) * 0.01)
        ]
        x, y = np.meshgrid(
            np.linspace(positions[:, 0].min(), positions[:, 0].max(), 101),
            np.linspace(positions[:, 1].min(), positions[:, 1].max(), 103),
        )
        samples = np.column_stack([x.ravel(), y.ravel()])
        coverage = np.zeros(len(samples), dtype=int)
        for tri in top:
            signs = []
            for a, b in zip(tri[:, :2], np.roll(tri[:, :2], -1, axis=0)):
                signs.append(
                    (b[0] - a[0]) * (samples[:, 1] - a[1])
                    - (b[1] - a[1]) * (samples[:, 0] - a[0])
                )
            signs = np.array(signs)
            coverage += np.all(signs > 1e-7, axis=0) | np.all(signs < -1e-7, axis=0)
        assert coverage.max() == 1


@pytest.mark.browser
def test_maps_preserved_and_completed_buffers_stay_fixed(viewer_client, viewer_page):
    p = np.array(
        [
            [-30, 0, 0],
            [-20, 0, 0],
            [-10, 0, 0],
            [0, 0, 0],
            [0, 10, 0],
            [0, 20, 0],
            [0, 30, 0],
        ],
        dtype=float,
    )
    for mode in ("rings", "local_join"):
        viewer_client.add_parametric_tube(
            mode,
            p,
            [10] * len(p),
            [2] * len(p),
            surface=mode,
            lod=False,
            texture=_png_bytes(),
            normal_map=_png_bytes(),
            roughness_map=_png_bytes(),
        )
        viewer_page.wait_for_function(
            "id=>window.threejsViewer._objects.has(id)", arg=mode
        )
    result = viewer_page.evaluate("""() => {
        const v=window.threejsViewer,a=v._objects.get('rings'),b=v._objects.get('local_join'),f=b.userData.localJoinCompleted;
        const original=a.geometry.attributes,joined=b.geometry.attributes;
        const untouched=['position','uv','uvAlt'].every(name=>Array.from(original[name].array.slice(0,7*6*original[name].itemSize)).every((x,i)=>x===joined[name].array[i]));
        const index=f.geometry.index, before=Array.from(index.array),positions=joined.position,attrs=Object.keys(joined).map(k=>joined[k]);
        for(let i=0;i<100;i++)b.userData.parametricTube.setDrawRange((i%17)/16);
        b.userData.parametricTube.setDrawRange(1);
        return {untouched,indexSame:index===f.geometry.index&&before.every((x,i)=>x===index.array[i]),
            attributesSame:attrs.every((a,i)=>a===b.geometry.attributes[Object.keys(joined)[i]]),
            maps:!!(b.material.map&&b.material.normalMap&&b.material.roughnessMap),
            materialSame:f.material===b.material,lod:!!b.userData.tubeLOD};
    }""")
    assert result == dict(
        untouched=True,
        indexSame=True,
        attributesSame=True,
        maps=True,
        materialSame=True,
        lod=False,
    )


@pytest.mark.browser
def test_many_corners_obey_total_budget(viewer_client, viewer_page):
    p = [[0, 0, 0]]
    for i in range(150):
        d = [20, 0, 0] if i % 2 == 0 else [0, 20, 0]
        p.append((np.array(p[-1]) + d).tolist())
    viewer_client.add_parametric_tube(
        "t", p, [10] * len(p), [2] * len(p), surface="local_join"
    )
    viewer_page.wait_for_function(
        "() => !!window.threejsViewer._objects.get('t')?.userData.localJoinStats"
    )
    s = viewer_page.evaluate(
        "() => window.threejsViewer._objects.get('t').userData.localJoinStats"
    )
    assert s["addedTriangles"] <= s["triangleBudget"]
    assert s["fallback"]["total triangle budget"] > 0


@pytest.mark.browser
def test_breaks_and_top_anchor(viewer_client, viewer_page):
    p = np.array(
        [
            [-30, 0, 0],
            [-20, 0, 0],
            [-10, 0, 0],
            [0, 0, 0],
            [0, 10, 0],
            [0, 20, 0],
            [0, 30, 0],
        ],
        dtype=float,
    )
    breaks = np.zeros(len(p), dtype=np.uint8)
    breaks[4] = 1
    viewer_client.add_parametric_tube(
        "t",
        p,
        [10] * len(p),
        [2] * len(p),
        surface="local_join",
        anchor="top",
        break_before=breaks,
    )
    viewer_page.wait_for_function(
        "() => !!window.threejsViewer._objects.get('t')?.userData.localJoinStats"
    )
    assert (
        viewer_page.evaluate(
            "() => window.threejsViewer._objects.get('t').userData.localJoinStats.accepted"
        )
        == 0
    )
    viewer_client.delete("t")
    viewer_page.wait_for_function("() => !window.threejsViewer._objects.has('t')")
    viewer_client.add_parametric_tube(
        "t",
        p,
        [10] * len(p),
        [2] * len(p),
        surface="local_join",
        anchor="top",
        texture=_png_bytes(),
    )
    viewer_page.wait_for_function(
        "() => !!window.threejsViewer._objects.get('t')?.userData.localJoinStats"
    )
    s = snapshot(viewer_page)
    assert s["stats"]["accepted"] == 1
    z = np.array(s["p"]).reshape(-1, 3)[:, 2]
    assert z.max() < 0.002 and z.min() > -2.002
    viewer_client.set_draw_range("t", 0)
    viewer_page.wait_for_function(
        "() => window.threejsViewer._objects.get('t').geometry.drawRange.count===0"
    )
    s = snapshot(viewer_page)
    assert s["i"] == []
    viewer_client.set_draw_range("t", 1)
    viewer_page.wait_for_function(
        "() => window.threejsViewer._objects.get('t').userData.localJoinCommittedPairs===6"
    )


@pytest.mark.browser
def test_out_of_plane_return_keeps_incompatible_frame_unpatched(
    viewer_client, viewer_page
):
    # The original transported U stays +Y across vertical. Recomputing U from
    # up x tangent on the upper return gives -Y and twists corresponding seams.
    angles = np.linspace(0, np.pi, 7)
    arc = np.column_stack(
        [6 * np.sin(angles), np.zeros(len(angles)), 6 * (1 - np.cos(angles))]
    )
    p = np.vstack(
        [
            [[-30, 0, 0], [-20, 0, 0], [-10, 0, 0]],
            arc,
            [[-10, 0, 12], [-20, 0, 12], [-30, 0, 12]],
        ]
    )
    viewer_client.add_parametric_tube(
        "reference", p, np.full(len(p), 8), np.full(len(p), 2), lod=False
    )
    viewer_page.wait_for_function(
        "() => window.threejsViewer._objects.has('reference')"
    )
    viewer_client.add_parametric_tube(
        "t", p, np.full(len(p), 8), np.full(len(p), 2), surface="local_join", lod=False
    )
    viewer_page.wait_for_function(
        "() => !!window.threejsViewer._objects.get('t')?.userData.localJoinStats"
    )
    result = viewer_page.evaluate("""() => {
        const o=window.threejsViewer._objects.get('t'), u=o.userData;
        const cap=u.tubeCapIndicesPerCap, pair=u.tubeIndicesPerRingPair;
        const reference=window.threejsViewer._objects.get('reference');
        const original=Array.from(reference.geometry.index.array.slice(cap+7*pair,cap+12*pair));
        // The unrepaired upper return stays in the original strip; the child
        // geometry now contains only actual patch triangles.
        const suffix=Array.from(o.geometry.index.array.slice(cap+7*pair,cap+12*pair));
        return {stats:u.localJoinStats,original,suffix};
    }""")
    assert result["stats"]["fallback"]["frame reversal"] >= 1
    assert result["suffix"] == result["original"]


@pytest.mark.browser
@pytest.mark.parametrize("n", [1000, 100000])
def test_sparse_repairs_store_only_patch_indices(viewer_client, viewer_page, n):
    lead = np.array([[-30, 0, 0], [-20, 0, 0], [-10, 0, 0], [0, 0, 0]])
    tail = np.column_stack([np.zeros(n), np.arange(1, n + 1) * 10, np.zeros(n)])
    p = np.vstack([lead, tail])
    viewer_client.add_parametric_tube(
        "t",
        p,
        np.full(len(p), 10),
        np.full(len(p), 2),
        surface="local_join",
        lod=False,
        texture=_png_bytes(),
    )
    viewer_page.wait_for_function(
        "() => !!window.threejsViewer._objects.get('t')?.userData.localJoinCompleted"
    )
    result = viewer_page.evaluate("""() => {
        const o=window.threejsViewer._objects.get('t'), u=o.userData;
        const g=u.localJoinCompleted.geometry;
        return {stats:u.localJoinStats, bytes:g.index.array.byteLength,
            vertices:g.attributes.position.count, baseVertices:o.geometry.attributes.position.count};
    }""")
    assert result["stats"]["accepted"] == 1
    assert result["bytes"] == 48 * 3 * 4
    assert result["vertices"] <= 48  # independent of the untouched tail length
    assert result["baseVertices"] > n * 6
    assert result["stats"]["savedIndexBytes"] <= 2 * 36 * 4
    assert result["stats"]["patchTriangles"] <= result["stats"]["patchTriangleBudget"]


@pytest.mark.browser
def test_lod_rebuilds_local_patches_preserving_playback_and_maps(
    viewer_client, viewer_page
):
    # Fine helix plus a genuine, well-spaced corner. Worker LOD must replace
    # patch buffers instead of retaining closures over the previous geometry.
    t = np.linspace(0, 8 * np.pi, 4000)
    helix = np.column_stack([40 * np.cos(t), 40 * np.sin(t), t])
    p = np.vstack([[[0, -30, 0], [0, -20, 0], [40, -20, 0]], helix])
    viewer_client.add_parametric_tube(
        "t",
        p,
        np.full(len(p), 4),
        np.full(len(p), 2),
        surface="local_join",
        lod={"threshold": 0},
        texture=_png_bytes(),
        normal_map=_png_bytes(),
        roughness_map=_png_bytes(),
        anchor="top",
    )
    viewer_page.wait_for_function(
        "() => !!window.threejsViewer._objects.get('t')?.userData.tubeLOD"
    )
    viewer_page.evaluate("""() => {
        const v=window.threejsViewer,o=v._objects.get('t');
        o.userData.parametricTube.setDrawRange(.37);
        window.oldJoinGeometry=o.geometry;
        window.oldPatch=o.userData.localJoinCompleted;
        window.oldPatchDisposed=false;
        window.oldWireDisposed=false;
        v._shading.wireframeMode=2;
        if(oldPatch) {
            v._shading._applyWireframeTo(oldPatch);
            oldPatch.userData.wireframeOverlay.material.addEventListener('dispose',()=>window.oldWireDisposed=true);
        }
        oldPatch?.geometry.addEventListener('dispose',()=>window.oldPatchDisposed=true);
        v._camera.position.set(4000,-4000,4000);v._lodDirty=true;
    }""")
    viewer_page.wait_for_function(
        "() => window.threejsViewer._objects.get('t').geometry!==window.oldJoinGeometry",
        timeout=20000,
    )
    result = viewer_page.evaluate("""() => {
        const o=window.threejsViewer._objects.get('t'),u=o.userData,lod=u.tubeLOD;
        const m=o.material, old=window.oldPatch;
        const fraction=lod.currentDrawRange;
        const children=o.children.filter(c=>c.userData.localJoinCompleted);
        const full=[];
        for(const f of [1, .005, .7, 0, .37, 1]) {
            u.parametricTube.setDrawRange(f);
            const g=o.geometry, c=u.localJoinCompleted?.geometry;
            const count=g.drawRange.count+(c?.drawRange.count||0);
            if(f===0 && count!==0)throw Error('nonempty zero');
            if(f===1)full.push(Array.from(g.index.array));
        }
        return {fraction, reduced:lod.keptIndices.length, original:lod.originalCount,
            maps:!!(m.map&&m.normalMap&&m.roughnessMap), disposed:!old||window.oldPatchDisposed,
            wireDisposed:!old||window.oldWireDisposed,
            detached:!old||old.parent===null, children:children.length,
            stable:full[0].every((x,i)=>x===full[1][i]),
            patchTriangles:u.localJoinStats.patchTriangles, budget:u.localJoinStats.patchTriangleBudget};
    }""")
    assert result["fraction"] == pytest.approx(0.37)
    assert result["reduced"] < result["original"] / 2
    assert result["maps"] and result["disposed"] and result["detached"]
    assert result["wireDisposed"]
    assert result["children"] <= 1 and result["stable"]
    assert result["patchTriangles"] <= result["budget"]
    viewer_page.evaluate("""() => {
        const v=window.threejsViewer,o=v._objects.get('t');
        o.userData.parametricTube.setDrawRange(.37);
        window.farJoinGeometry=o.geometry;
        v._camera.position.set(50,-50,30);v._lodDirty=true;
    }""")
    viewer_page.wait_for_function(
        "() => window.threejsViewer._objects.get('t').geometry!==window.farJoinGeometry",
        timeout=20000,
    )
    near = viewer_page.evaluate("""() => {
        const o=window.threejsViewer._objects.get('t'),u=o.userData;
        return {count:u.tubeNumSpinePoints, fraction:u.tubeLOD.currentDrawRange,
            children:o.children.filter(c=>c.userData.localJoinCompleted).length};
    }""")
    assert near["count"] > result["reduced"]
    assert near["fraction"] == pytest.approx(0.37)
    assert near["children"] <= 1


def _colour_spine():
    return np.array(
        [
            [-30, 0, 0],
            [-20, 0, 0],
            [-10, 0, 0],
            [0, 0, 0],
            [0, 10, 0],
            [80, 0, 0],
            [90, 0, 0],
            [100, 0, 0],
        ],
        dtype=np.float32,
    )


@pytest.mark.browser
@pytest.mark.parametrize("kind", ["corner", "cluster"])
def test_local_join_colour_provenance_and_unchanged_topology(
    viewer_client, viewer_page, kind
):
    import json
    from pathlib import Path

    p = (
        _colour_spine()[:5]
        if kind == "corner"
        else np.array(
            json.loads(
                (Path(__file__).parent / "data/tube_issue_277.json").read_text()
            ),
            dtype=np.float32,
        )
    )
    p = p - p[0]
    n = len(p)
    packed = (
        (np.arange(n, dtype=np.uint32) * (255 // n) << 16)
        | (np.arange(n, dtype=np.uint32) % 2 * 255 << 8)
        | 93
    )
    for name, colors in [("plain", None), ("t", packed)]:
        viewer_client.add_parametric_tube(
            name,
            p,
            np.full(n, 11.83),
            np.full(n, 2),
            colors=colors,
            surface="local_join",
            lod=False,
        )
    viewer_page.wait_for_function(
        "() => !!window.threejsViewer._objects.get('t')?.userData.localJoinCompleted"
    )
    data = viewer_page.evaluate("""() => {
        const v=threejsViewer,a=v._objects.get('plain'),b=v._objects.get('t');
        const g=b.userData.localJoinCompleted.geometry,h=a.userData.localJoinCompleted.geometry;
        const same=(x,y)=>x.length===y.length&&x.every((v,i)=>v===y[i]);
        return {positions:Array.from(g.attributes.position.array),colors:Array.from(g.attributes.color.array),
            base:Array.from(b.geometry.attributes.position.array.slice(0,b.userData.tubeNumSpinePoints*18)),
            unchanged:same(g.attributes.position.array,h.attributes.position.array)&&same(g.index.array,h.index.array)&&same(a.geometry.index.array,b.geometry.index.array),
            widths:Array.from(b.userData.tubeMorphData.widths),heights:Array.from(b.userData.tubeMorphData.heights),
            stats:b.userData.localJoinStats};
    }""")
    assert data["unchanged"]
    assert data["stats"]["accepted"] > 0
    assert (data["stats"]["clusters"] > 0) == (kind == "cluster")
    positions = np.asarray(data["positions"]).reshape(-1, 3)
    actual = np.asarray(data["colors"]).reshape(-1, 3)
    base = np.asarray(data["base"]).reshape(n, 6, 3)
    rgb = (
        np.column_stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255]) / 255
    )
    expected = []
    for point in positions:
        # Interface vertices must retain their original station's data.
        matches = np.argwhere(np.linalg.norm(base - point, axis=2) < 1e-5)
        if len(matches):
            expected.append(rgb[matches[0, 0]])
        elif kind == "corner":
            expected.append(rgb[3])
        else:
            # Independent tapered-capsule oracle. Minimize distance to the
            # centreline minus interpolated radius, accounting for the viewer's
            # tiny deposition-order width/height bias as well as the chamfer.
            ab = np.diff(p[:, :2], axis=0).astype(float)
            length = np.linalg.norm(ab, axis=1)
            direction = ab / np.maximum(length[:, None], 1e-20)
            relative = point[:2] - p[:-1, :2]
            along = (relative * direction).sum(axis=1)
            perpendicular = np.linalg.norm(
                relative - along[:, None] * direction, axis=1
            )
            radius = np.asarray(data["widths"]) / 2
            if abs(point[2] - p[0, 2]) > min(data["heights"]) / 4:
                radius -= np.asarray(data["heights"]) / 2
            slope = np.diff(radius) / np.maximum(length, 1e-20)
            t = np.clip(
                (
                    along
                    + slope
                    * perpendicular
                    / np.sqrt(np.maximum(1 - slope * slope, 1e-20))
                )
                / np.maximum(length, 1e-20),
                0,
                1,
            )
            distance = np.linalg.norm(relative - t[:, None] * ab, axis=1) - (
                radius[:-1] + t * np.diff(radius)
            )
            i = int(np.argmin(distance))
            expected.append(rgb[i] + t[i] * (rgb[i + 1] - rgb[i]))
    np.testing.assert_allclose(actual, expected, atol=2e-5)
    assert data["stats"]["colorSourceBytes"] == len(positions) * 8


@pytest.mark.browser
@pytest.mark.parametrize("initial_colors", [False, True])
def test_local_join_live_colors_reset_and_break_caps(
    viewer_client, viewer_page, initial_colors
):
    p = _colour_spine()
    n = len(p)
    breaks = np.zeros(n, dtype=np.uint8)
    breaks[5] = 1
    viewer_page.evaluate("window.__tubeDrawCapOverride=144")
    viewer_client.add_parametric_tube(
        "t",
        p,
        np.full(n, 8),
        np.full(n, 2),
        colors=np.full(n, 0xFF0000, dtype=np.uint32) if initial_colors else None,
        break_before=breaks,
        surface="local_join",
        lod=False,
        texture=_png_bytes(),
        normal_map=_png_bytes(),
        roughness_map=_png_bytes(),
    )
    viewer_page.wait_for_function(
        "() => !!threejsViewer._objects.get('t')?.userData.localJoinCompleted"
    )
    viewer_page.evaluate("""() => {
        const o=threejsViewer._objects.get('t');o.userData.parametricTube.setDrawRange(.63);
        window.beforeColour={position:Array.from(o.geometry.attributes.position.array),
            patch:o.userData.localJoinCompleted.geometry,indices:Array.from(o.geometry.index.array)};
    }""")
    for packed in (0x00FF00, 0x0000FF):
        viewer_client.update_parametric_tube_colors(
            "t", np.full(n, packed, dtype=np.uint32)
        )
        viewer_page.wait_for_function(
            """(c) => {
            const o=threejsViewer._objects.get('t'),g=o.userData.localJoinCompleted.geometry;
            const expected=[(c>>16&255)/255,(c>>8&255)/255,(c&255)/255];
            return g.attributes.color?.array.every((v,i)=>Math.abs(v-expected[i%3])<1e-6);
        }""",
            arg=packed,
        )
    state = viewer_page.evaluate("""() => {
        const o=threejsViewer._objects.get('t'),u=o.userData,g=u.localJoinCompleted.geometry;
        return {sameGeometry:g===beforeColour.patch,grouped:Array.isArray(o.material),maps:!!(u.tubeBaseMaterial.map&&u.tubeBaseMaterial.normalMap&&u.tubeBaseMaterial.roughnessMap),
            flags:u.tubeBaseMaterial.vertexColors,position:Array.from(o.geometry.attributes.position.array),
            original:beforeColour.position,indices:Array.from(o.geometry.index.array),oldIndices:beforeColour.indices};
    }""")
    assert (
        state["sameGeometry"] and state["grouped"] and state["maps"] and state["flags"]
    )
    np.testing.assert_array_equal(state["position"], state["original"])
    assert state["indices"] == state["oldIndices"]
    viewer_client.update_parametric_tube_colors("t", None, base_color=0x123456)
    viewer_page.wait_for_function("""() => {
        const o=threejsViewer._objects.get('t'),u=o.userData;
        return !o.geometry.attributes.color&&!u.localJoinCompleted.geometry.attributes.color&&!u.tubeBaseMaterial.vertexColors&&u.tubeBaseMaterial.color.getHex()===0x123456;
    }""")
    colors = np.array(
        [
            0x110000,
            0x220000,
            0x330000,
            0x440000,
            0x550000,
            0x660000,
            0x770000,
            0x880000,
        ],
        dtype=np.uint32,
    )
    viewer_client.update_parametric_tube_colors("t", colors)
    viewer_page.wait_for_function(
        "() => !!threejsViewer._objects.get('t').userData.localJoinCompleted.geometry.attributes.color"
    )
    caps = viewer_page.evaluate("""() => {
        const o=threejsViewer._objects.get('t'),u=o.userData;
        for(const f of [1,.1,.75,0,1])u.parametricTube.setDrawRange(f);
        const c=o.geometry.attributes.color.array,n=u.tubeNumSpinePoints,cs=u.tubeNCs,k=u.tubeMorphData.capAngles.length*cs;
        return {start:Array.from(c.slice(n*cs*3,(n*cs+k)*3)),end:Array.from(c.slice((n*cs+k)*3,(n*cs+2*k)*3)),
            rims:Array.from(c.slice((n*cs+2*k)*3)),patch:Array.from(u.localJoinCompleted.geometry.attributes.color.array)};
    }""")
    np.testing.assert_allclose(
        np.asarray(caps["start"]).reshape(-1, 3),
        np.tile([0x11 / 255, 0, 0], (len(caps["start"]) // 3, 1)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(caps["end"]).reshape(-1, 3),
        np.tile([0x88 / 255, 0, 0], (len(caps["end"]) // 3, 1)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(caps["rims"]).reshape(-1, 3),
        np.repeat([[0x55 / 255, 0, 0], [0x66 / 255, 0, 0]], 6, axis=0),
        atol=1e-6,
    )


@pytest.mark.browser
@pytest.mark.parametrize("change", ["add", "replace", "reset"])
def test_local_join_colors_win_over_stale_lod_response(
    viewer_client, viewer_page, change
):
    t = np.linspace(0, 8 * np.pi, 4000)
    p = np.vstack(
        [
            [[0, -30, 0], [0, -20, 0], [40, -20, 0]],
            np.column_stack([40 * np.cos(t), 40 * np.sin(t), t]),
        ]
    )
    n = len(p)
    breaks = np.zeros(n, dtype=np.uint8)
    breaks[n // 2] = 1
    viewer_client.add_parametric_tube(
        "t",
        p,
        np.full(n, 4),
        np.full(n, 2),
        colors=None if change == "add" else np.full(n, 0xFF0000, dtype=np.uint32),
        break_before=breaks,
        surface="local_join",
        lod={"threshold": 0},
    )
    viewer_page.wait_for_function(
        "() => !!threejsViewer._objects.get('t')?.userData.localJoinCompleted"
    )
    viewer_page.evaluate("""() => {
        const v=threejsViewer,o=v._objects.get('t');o.userData.parametricTube.setDrawRange(.37);
        window.colourLODHandler=v._lodWorker.onmessage;
        window.heldColourLOD=null;
        v._lodWorker.onmessage=e=>{
            if(e.data.positions&&!e.data.allReused)window.heldColourLOD=e;
            else window.colourLODHandler(e);
        };
        v._camera.position.set(4000,-4000,4000);v._lodDirty=true;
    }""")
    viewer_page.wait_for_function("() => !!window.heldColourLOD", timeout=20000)
    viewer_client.update_parametric_tube_colors(
        "t", None if change == "reset" else np.full(n, 0x00FF00, dtype=np.uint32)
    )
    viewer_page.wait_for_function(
        """(reset) => {
        const u=threejsViewer._objects.get('t').userData;
        return reset ? !u.tubeLOD.originalRingColors : u.tubeLOD.originalRingColors?.[1]===1;
    }""",
        arg=change == "reset",
    )
    result = viewer_page.evaluate("""() => {
        const v=threejsViewer;v._lodWorker.onmessage=window.colourLODHandler;
        window.colourLODHandler(window.heldColourLOD);
        const o=v._objects.get('t'),u=o.userData,patch=u.localJoinCompleted;
        const fraction=u.tubeLOD.currentDrawRange;
        for(const f of [1,.7,0,.01,.9,1])u.parametricTube.setDrawRange(f);
        const arrays=[o.geometry,patch?.geometry].filter(Boolean).map(g=>g.attributes.color?.array);
        return {fraction,patch:!!patch,n:u.tubeNumSpinePoints,
            noColors:arrays.every(a=>!a),green:arrays.every(a=>a&&a.every((x,i)=>Math.abs(x-(i%3===1?1:0))<1e-6)),
            material:u.tubeBaseMaterial.vertexColors,breaks:u.tubeMorphData.breaks.some(x=>x)};
    }""")
    assert result["fraction"] == pytest.approx(0.37)
    assert result["n"] < n // 2
    assert result["patch"] and result["breaks"]
    assert result["noColors"] if change == "reset" else result["green"]
    assert result["material"] == (change != "reset")
