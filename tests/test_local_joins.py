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
        ({"colors": [0, 0]}, "uniform"),
    ]:
        with pytest.raises(ValueError, match=message):
            client.add_parametric_tube(**(args | patch))


def snapshot(page):
    return page.evaluate("""() => {
        const o=window.threejsViewer._objects.get('t'), f=o.userData.localJoinCompleted;
        const g=f?.geometry || o.geometry, b=o.geometry, visible=f ? g.index.array.slice(g.drawRange.start,g.drawRange.start+g.drawRange.count) : [];
        const frontier=b.index.array.slice(b.drawRange.start,b.drawRange.start+b.drawRange.count);
        return {p:Array.from(g.attributes.position.array), i:[...visible,...frontier],
            stats:o.userData.localJoinStats, committed:o.userData.localJoinCommittedPairs,
            uv:Array.from(g.attributes.uv.array), shared:g.attributes.position===b.attributes.position};
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
    assert s["shared"]
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
