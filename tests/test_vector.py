"""Vector entities: builders, protocol, reconnect replay and browser rendering."""

import json
import math

import numpy as np
import pytest
from websockets.sync.client import connect as ws_connect

from conftest import frames, settle
from threejs_viewer import ViewerClient, vector


@pytest.fixture
def client():
    c = ViewerClient()
    c._messages = []
    c._send = c._messages.append
    return c


# === Builders ===


def test_circle_and_style_keys_are_camel_case():
    e = vector.circle(
        np.array([1, 2, 3]),
        0.5,
        normal=[0, 1, 0],
        layer="CUT",
        color=0xFF0000,
        linetype="dashed",
        line_width=3,
        linetype_scale=2,
    )
    assert e == {
        "type": "circle",
        "center": [1.0, 2.0, 3.0],
        "radius": 0.5,
        "normal": [0.0, 1.0, 0.0],
        "layer": "CUT",
        "color": 0xFF0000,
        "linetype": "dashed",
        "lineWidth": 3.0,
        "linetypeScale": 2.0,
    }


def test_polyline_flattens_vertices_and_trims_bulges():
    e = vector.polyline([[0, 0, 0], [1, 0, 0], [1, 1, 0]], bulges=[0.5, 0, 9])
    assert e["vertices"] == [0, 0, 0, 1, 0, 0, 1, 1, 0]
    assert e["bulges"] == [0.5, 0.0]
    closed = vector.polyline(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0]], bulges=[0, 0, 1], closed=True
    )
    assert closed["closed"] is True and closed["bulges"] == [0.0, 0.0, 1.0]


def test_ellipse_omits_default_params():
    e = vector.ellipse([0, 0, 0], [2, 0, 0], 0.5)
    assert "startParam" not in e and "endParam" not in e
    arc = vector.ellipse([0, 0, 0], [2, 0, 0], 0.5, start_param=0.1, end_param=1.0)
    assert arc["startParam"] == 0.1 and arc["endParam"] == 1.0


def test_spline_accepts_knots_and_weights():
    cps = [[0, 0, 0], [1, 1, 0], [2, 0, 0]]
    e = vector.spline(
        cps, degree=2, knots=[0, 0, 0, 1, 1, 1], weights=[1, math.sqrt(0.5), 1]
    )
    assert e["degree"] == 2 and len(e["knots"]) == 6 and len(e["weights"]) == 3


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: vector.circle([0, 0, 0], 0), "radius"),
        (lambda: vector.circle([0, 0, float("nan")], 1), "center"),
        (lambda: vector.circle([0, 0, 0], 1, normal=[0, 0, 0]), "normal"),
        (lambda: vector.arc([0, 0, 0], 1, 0, float("inf")), "end_angle"),
        (lambda: vector.ellipse([0, 0, 0], [1, 0, 0], 1.5), "ratio"),
        (lambda: vector.ellipse([0, 0, 0], [0, 0, 0], 0.5), "major_axis"),
        (lambda: vector.polyline([[0, 0, 0]]), "vertices"),
        (lambda: vector.polyline([[0, 0, 0], [1, 0, 0]], bulges=[]), "bulges"),
        (lambda: vector.spline([[0, 0, 0], [1, 0, 0]], degree=3), "control_points"),
        (lambda: vector.spline([[0, 0, 0]] * 4, knots=[0, 1, 2]), "knots"),
        (
            lambda: vector.spline([[0, 0, 0]] * 4, knots=[0, 0, 0, 0, 1, 1, 1, 0.5]),
            "knots",
        ),
        (lambda: vector.spline([[0, 0, 0]] * 4, weights=[1, 1, 0, 1]), "weights"),
        (lambda: vector.line([0, 0, 0], [1, 0, 0], line_width=0), "line_width"),
        (
            lambda: vector.line([0, 0, 0], [1, 0, 0], linetype_scale=-1),
            "linetype_scale",
        ),
    ],
)
def test_builders_validate(build, match):
    with pytest.raises(ValueError, match=match):
        build()


# === Protocol ===


def test_add_circle_sends_vector_object(client):
    client.add_circle("c", [1, 0, 0], 2.0, layer="CUT", parent="g", position=[0, 0, 1])
    assert client._messages == [
        {
            "type": "add_object",
            "id": "c",
            "parent": "g",
            "object": {
                "vector": {
                    "entities": [
                        {
                            "type": "circle",
                            "center": [1.0, 0.0, 0.0],
                            "radius": 2.0,
                            "layer": "CUT",
                        }
                    ]
                },
                "transform": {"position": [0, 0, 1]},
            },
        }
    ]


def test_add_vector_entities_defaults_matrix_and_flags(client):
    m = np.eye(4)
    client.add_vector_entities(
        "d",
        [vector.line([0, 0, 0], [1, 0, 0])],
        layer="L",
        color=0x00FF00,
        line_width=2,
        matrix=m,
        visible=False,
        pickable=False,
    )
    msg = client._messages[0]
    spec = msg["object"]["vector"]
    assert (
        spec["layer"] == "L" and spec["color"] == 0x00FF00 and spec["lineWidth"] == 2.0
    )
    assert spec["pickable"] is False
    assert msg["object"]["transform"] == {"matrix": m.reshape(-1).tolist()}
    assert msg["visible"] is False and msg["object"]["visible"] is False


def test_add_vector_entities_rejects_non_entities(client):
    with pytest.raises(ValueError, match="entities\\[1\\]"):
        client.add_vector_entities("d", [vector.point([0, 0, 0]), {"radius": 1}])
    with pytest.raises(ValueError, match="matrix"):
        client.add_vector_entities("d", [], matrix=[1, 2, 3])
    assert client._messages == []


def test_convenience_methods_build_one_entity(client):
    client.add_line("l", [0, 0, 0], [1, 0, 0])
    client.add_arc("a", [0, 0, 0], 1, 0, math.pi)
    client.add_ellipse("e", [0, 0, 0], [2, 0, 0], 0.5)
    client.add_spline("s", [[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0]], visible=False)
    types = [m["object"]["vector"]["entities"][0]["type"] for m in client._messages]
    assert types == ["line", "arc", "ellipse", "spline"]
    assert client._messages[-1]["visible"] is False


def test_layers_and_linetypes_merge_and_record_without_connection(client):
    client._ws = None
    client.set_layer("CUT", color=0xFF0000)
    client.set_layer("CUT", visible=False)
    client.set_linetype("dashdot2", [0.01, -0.005, 0, -0.005])
    assert client._messages == []  # recorded only; replayed on connect
    assert client._vector_layers["CUT"] == {
        "type": "set_vector_layer",
        "name": "CUT",
        "color": 0xFF0000,
        "visible": False,
    }
    assert client._vector_linetypes["DASHDOT2"]["pattern"] == [
        0.01,
        -0.005,
        0.0,
        -0.005,
    ]
    with pytest.raises(ValueError, match="12"):
        client.set_linetype("long", [0.1, -0.1] * 7)
    with pytest.raises(ValueError, match="name"):
        client.set_layer("")


def test_set_layer_sends_only_changed_fields_when_connected(client):
    client._ws = object()
    client.set_layer("A", line_width=4)
    assert client._messages == [
        {"type": "set_vector_layer", "name": "A", "lineWidth": 4.0}
    ]


def test_reconnect_replays_linetypes_then_layers():
    client = ViewerClient(port=0, open_browser=False)
    client._start_servers(http_port=0)
    try:
        client.set_layer("CUT", linetype="MY")
        client.set_linetype("my", [0.1, -0.1])
        seen = []
        with ws_connect(f"ws://127.0.0.1:{client.port}", open_timeout=5) as ws:
            while len([m for m in seen if m["type"].startswith("set_vector_")]) < 2:
                seen.append(json.loads(ws.recv(timeout=5)))
        vector_msgs = [m for m in seen if m["type"].startswith("set_vector_")]
        assert [m["type"] for m in vector_msgs] == [
            "set_vector_linetype",
            "set_vector_layer",
        ]
        assert vector_msgs[1] == {
            "type": "set_vector_layer",
            "name": "CUT",
            "linetype": "MY",
        }
    finally:
        client.disconnect()


# === Browser ===

_STATS_JS = """(id) => {
    const obj = window.threejsViewer._objects.get(id);
    if (!obj) return null;
    const st = obj.userData.vectorEntities;
    const buckets = [...st.buckets.values()].map((b) => ({
        layer: b.layer, kind: b.kind, linetype: b.linetype, visible: b.obj.visible,
        segments: b.segments, dashed: !!b.obj.material.dashed,
    }));
    return { segments: st.segmentCount, skipped: st.skipped,
             levels: Array.from(st.levels), buckets,
             tracked: window.threejsViewer._vectors.groups.has(obj) };
}"""


def _stats(page, id):
    return page.evaluate(_STATS_JS, id)


@pytest.mark.browser
def test_circle_refines_when_zooming_in_and_coarsens_back(viewer_client, viewer_page):
    viewer_client.add_circle("c", [0, 0, 0], 1.0)
    viewer_client.set_camera(position=[0, 0, 10], target=[0, 0, 0])
    settle(viewer_client)
    frames(viewer_page, 3)
    far = _stats(viewer_page, "c")
    assert far["skipped"] == 0 and far["segments"] >= 8

    viewer_client.set_camera(position=[1, 0, 0.05], target=[1, 0, 0])
    settle(viewer_client)
    frames(viewer_page, 3)
    near = _stats(viewer_page, "c")
    assert near["segments"] > 8 * far["segments"]

    # A small move must not rebuild (hysteresis); a big step back coarsens.
    # `far` may be finer than a fresh pick at that distance: its level was
    # chosen at the default camera, before set_camera, and hysteresis kept it.
    viewer_client.set_camera(position=[1, 0, 0.06], target=[1, 0, 0])
    settle(viewer_client)
    frames(viewer_page, 3)
    assert _stats(viewer_page, "c")["segments"] == near["segments"]
    viewer_client.set_camera(position=[0, 0, 10], target=[0, 0, 0])
    settle(viewer_client)
    frames(viewer_page, 3)
    back = _stats(viewer_page, "c")["segments"]
    assert back <= far["segments"] and back * 8 < near["segments"]


@pytest.mark.browser
def test_orthographic_zoom_refines(viewer_client, viewer_page):
    viewer_client.add_circle("c", [0, 0, 0], 1.0)
    viewer_client.set_camera(position=[0, 0, 10], target=[0, 0, 0])
    settle(viewer_client)
    viewer_page.evaluate("() => window.threejsViewer._camController.switch(true)")
    frames(viewer_page, 3)
    assert viewer_page.evaluate(
        "() => !!window.threejsViewer._camera.isOrthographicCamera"
    )
    before = _stats(viewer_page, "c")["segments"]
    # Segments grow with the square root of the zoom; the switch starts near
    # zoom 2.7, so 200 is ~70x (~8x segments, at least ~4x after quantising).
    viewer_client.set_camera(zoom=200)
    settle(viewer_client)
    frames(viewer_page, 3)
    assert _stats(viewer_page, "c")["segments"] > 4 * before


@pytest.mark.browser
def test_entities_batch_by_layer_and_linetype(viewer_client, viewer_page):
    viewer_client.set_layer("AXES", color=0x00FFFF, linetype="center")
    viewer_client.add_vector_entities(
        "d",
        [
            vector.line([-1, 0, 0], [1, 0, 0], layer="AXES"),
            vector.line([0, -1, 0], [0, 1, 0], layer="AXES"),
            vector.circle([0, 0, 0], 0.5, color=0xFF0000),
            vector.arc([0, 0, 0], 0.7, 0, 1),
            vector.points([[0, 0, 0], [1, 1, 0]]),
            vector.polyline([[0, 0, 0], [1, 0, 0]], bulges=[1]),
            vector.ellipse([0, 0, 0], [1, 0, 0], 0.3),
            vector.spline(
                [[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0]], weights=[1, 3, 1, 1]
            ),
        ],
    )
    settle(viewer_client)
    frames(viewer_page)
    st = _stats(viewer_page, "d")
    assert st["skipped"] == 0 and st["tracked"]
    by_key = {(b["kind"], b["layer"], b["linetype"]): b for b in st["buckets"]}
    assert set(by_key) == {
        ("lines", "AXES", "CENTER"),
        ("lines", "0", "CONTINUOUS"),
        ("points", "0", "CONTINUOUS"),
    }
    assert by_key[("lines", "AXES", "CENTER")]["dashed"] is True
    assert by_key[("lines", "AXES", "CENTER")]["segments"] == 2
    assert by_key[("lines", "0", "CONTINUOUS")]["dashed"] is False


@pytest.mark.browser
def test_layer_toggle_hides_buckets_and_restyle_rebuckets(viewer_client, viewer_page):
    viewer_client.add_vector_entities(
        "d",
        [
            vector.circle([0, 0, 0], 1, layer="A"),
            vector.circle([0, 0, 0], 2, layer="B"),
        ],
    )
    viewer_client.set_layer("A", visible=False)
    settle(viewer_client)
    vis = {b["layer"]: b["visible"] for b in _stats(viewer_page, "d")["buckets"]}
    assert vis == {"A": False, "B": True}

    viewer_client.set_layer("A", visible=True, linetype="DASHED")
    settle(viewer_client)
    buckets = {b["layer"]: b for b in _stats(viewer_page, "d")["buckets"]}
    assert buckets["A"]["visible"] and buckets["A"]["linetype"] == "DASHED"
    assert buckets["A"]["dashed"] and not buckets["B"]["dashed"]


@pytest.mark.browser
def test_invalid_entities_are_skipped(viewer_client, viewer_page):
    viewer_client.add_vector_entities(
        "d",
        [
            vector.circle([0, 0, 0], 1),
            {"type": "circle", "center": [0, 0, 0], "radius": -1},
            {"type": "spline", "degree": 3, "controlPoints": [0, 0, 0, 1, 0, 0]},
            {"type": "nonsense"},
        ],
    )
    settle(viewer_client)
    st = _stats(viewer_page, "d")
    assert st["skipped"] == 3 and len(st["levels"]) == 1 and st["segments"] > 0


@pytest.mark.browser
def test_replace_and_delete_stop_tracking(viewer_client, viewer_page):
    viewer_client.add_circle("c", [0, 0, 0], 1)
    settle(viewer_client)
    viewer_page.evaluate(
        "() => { window.__oldVec = window.threejsViewer._objects.get('c'); }"
    )
    viewer_client.add_circle("c", [0, 0, 0], 2)
    settle(viewer_client)
    counts = viewer_page.evaluate(
        """() => { const v = window.threejsViewer;
            return [v._vectors.groups.size, v._vectors.groups.has(window.__oldVec)]; }"""
    )
    assert counts == [1, False]
    viewer_client.delete("c")
    settle(viewer_client)
    assert viewer_page.evaluate("() => window.threejsViewer._vectors.groups.size") == 0


@pytest.mark.browser
def test_linetype_pattern_leaves_gaps_on_screen(viewer_client, viewer_page):
    """A dashed line paints fewer pixels than the same line continuous."""
    viewer_client.set_background(0x000000)
    viewer_client.set_linetype("BIG", [0.2, -0.2])
    viewer_client.set_camera(position=[0, 0, 5], target=[0, 0, 0])
    viewer_client.add_line("l", [-1.5, 0, 0], [1.5, 0, 0], color=0xFFFFFF, line_width=6)
    settle(viewer_client)
    frames(viewer_page, 3)

    count_js = """() => new Promise((resolve) => requestAnimationFrame(() => {
        const v = window.threejsViewer;
        v._renderer.render(v._scene, v._camera);
        const gl = v._renderer.getContext();
        const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
        const px = new Uint8Array(w * h * 4);
        gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
        let n = 0;
        for (let i = 0; i < px.length; i += 4) if (px[i] > 128) n++;
        resolve(n);
    }))"""
    solid = viewer_page.evaluate(count_js)
    viewer_client.add_line(
        "l", [-1.5, 0, 0], [1.5, 0, 0], color=0xFFFFFF, line_width=6, linetype="BIG"
    )
    settle(viewer_client)
    frames(viewer_page, 3)
    dashed = viewer_page.evaluate(count_js)
    assert solid > 0
    assert 0.3 * solid < dashed < 0.7 * solid


# === Blocks and inserts ===


def test_insert_builder_fields():
    e = vector.insert(
        "HOLE",
        [[0, 0, 0], [1, 0, 0]],
        rotations=[0, 1.5],
        scale=2,
        scales=[1, 0.5],
        rows=2,
        columns=3,
        row_spacing=0.1,
        column_spacing=0.2,
        layer="HOLES",
    )
    assert e == {
        "type": "insert",
        "block": "HOLE",
        "positions": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "rotations": [0.0, 1.5],
        "scale": 2.0,
        "scales": [1.0, 0.5],
        "rows": 2,
        "columns": 3,
        "rowSpacing": 0.1,
        "columnSpacing": 0.2,
        "layer": "HOLES",
    }
    single = vector.insert("HOLE", [1, 2, 3], rotation=0.5, scale=[1, 2, 1])
    assert single["positions"] == [1.0, 2.0, 3.0]
    assert single["rotation"] == 0.5 and single["scale"] == [1.0, 2.0, 1.0]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"block": ""}, "block"),
        ({"rotations": [0]}, "rotations"),
        ({"scale": 0}, "scale"),
        ({"scales": [1, 0]}, "scales"),
        ({"rows": 0}, "rows"),
        ({"columns": 1.5}, "columns"),
    ],
)
def test_insert_builder_validates(kwargs, match):
    args = {"block": "B", "positions": [[0, 0, 0], [1, 0, 0]], **kwargs}
    with pytest.raises(ValueError, match=match):
        vector.insert(**args)


def test_blocks_are_sent_and_checked(client):
    hole = [vector.circle([0, 0, 0], 0.01)]
    client.add_vector_entities(
        "p", [vector.insert("HOLE", [[0, 0, 0], [1, 0, 0]])], blocks={"HOLE": hole}
    )
    assert client._messages[0]["object"]["vector"]["blocks"] == {"HOLE": hole}

    with pytest.raises(ValueError, match="unknown block 'NOPE'"):
        client.add_vector_entities("q", [vector.insert("NOPE", [0, 0, 0])])
    with pytest.raises(ValueError, match="inserts unknown block 'C'"):
        client.add_vector_entities(
            "q", [], blocks={"A": [vector.insert("C", [0, 0, 0])]}
        )
    with pytest.raises(ValueError, match="cycle: A -> B -> A"):
        client.add_vector_entities(
            "q",
            [vector.insert("A", [0, 0, 0])],
            blocks={
                "A": [vector.insert("B", [0, 0, 0])],
                "B": [vector.insert("A", [0, 0, 0])],
            },
        )
    assert len(client._messages) == 1


_RECORDS_JS = """(id) => {
    const st = window.threejsViewer._objects.get(id).userData.vectorEntities;
    return st.records.map((r) => ({
        center: r.bound.center.toArray().map((v) => Math.round(v * 1e6) / 1e6),
        radius: Math.round(r.bound.radius * 1e6) / 1e6,
        layer: r.layer ?? null, color: r.color ?? null,
        shared: !!r.base, instScale: r.instScale ?? 1,
    }));
}"""


@pytest.mark.browser
def test_insert_pattern_shares_block_tessellation(viewer_client, viewer_page):
    """A pattern of n holes is n instance records over one shared block
    entity, drawn as one batch whose size is n times one hole."""
    viewer_client.add_circle("one", [0, 0, 0], 0.01)
    pts = [[x * 0.05, y * 0.05, 0] for x in range(10) for y in range(10)]
    viewer_client.add_vector_entities(
        "pattern",
        [vector.insert("HOLE", pts)],
        blocks={"HOLE": [vector.circle([0, 0, 0], 0.01)]},
    )
    viewer_client.set_camera(position=[0.2, 0.2, 2], target=[0.2, 0.2, 0])
    settle(viewer_client)
    frames(viewer_page, 3)
    st = _stats(viewer_page, "pattern")
    assert st["skipped"] == 0 and len(st["levels"]) == 100 and len(st["buckets"]) == 1
    shared = viewer_page.evaluate(
        """() => { const recs = window.threejsViewer._objects.get('pattern')
              .userData.vectorEntities.records;
            return new Set(recs.map((r) => r.base)).size; }"""
    )
    assert shared == 1
    recs = viewer_page.evaluate(_RECORDS_JS, "pattern")
    assert sorted(tuple(r["center"]) for r in recs) == sorted(
        tuple(round(v, 6) for v in p) for p in pts
    )
    # Same view distance class, so every copy matches the standalone circle.
    one = _stats(viewer_page, "one")["segments"]
    assert one <= st["segments"] / 100 <= 2 * one


@pytest.mark.browser
def test_insert_grid_rotation_scale_and_inheritance(viewer_client, viewer_page):
    viewer_client.set_layer("HOLES", color=0x00FF00)
    viewer_client.add_vector_entities(
        "grid",
        [
            vector.insert(
                "MARK",
                [1, 0, 0],
                rotation=math.pi / 2,
                scale=2,
                rows=2,
                columns=3,
                row_spacing=1,
                column_spacing=1,
                layer="HOLES",
                color=0xFF0000,
            )
        ],
        blocks={
            "MARK": [
                vector.circle([0, 0, 0], 0.1),  # inherits layer and colour
                vector.insert("DOT", [0.5, 0, 0]),  # nested block
            ],
            "DOT": [vector.circle([0, 0, 0], 0.05, layer="OWN", color=0x0000FF)],
        },
    )
    settle(viewer_client)
    recs = viewer_page.evaluate(_RECORDS_JS, "grid")
    assert len(recs) == 12
    marks = [r for r in recs if r["layer"] == "HOLES"]
    dots = [r for r in recs if r["layer"] == "OWN"]
    assert len(marks) == 6 and len(dots) == 6
    assert all(r["color"] == 0xFF0000 and r["instScale"] == 2 for r in marks)
    assert all(r["color"] == 0x0000FF for r in dots)
    # Rotated 90 deg: columns step along +Y, rows along -X, from (1, 0, 0).
    centers = sorted(tuple(r["center"]) for r in marks)
    want = sorted((1.0 - r, float(c), 0.0) for r in range(2) for c in range(3))
    assert centers == want
    # The nested dot sits 0.5 along the rotated, scaled block X axis: +Y * 2.
    dot_centers = sorted(tuple(r["center"]) for r in dots)
    assert dot_centers == sorted((x, y + 1.0, 0.0) for x, y, _ in want)
    assert all(abs(r["radius"] - 0.1 * math.sqrt(3)) < 1e-6 for r in dots)


@pytest.mark.browser
def test_bad_inserts_are_skipped(viewer_client, viewer_page):
    """Messages sent without the Python checks (embedders, glTF) still
    degrade gracefully: unknown blocks and cycles skip only that insert."""
    viewer_client._send(
        {
            "type": "add_object",
            "id": "bad",
            "object": {
                "vector": {
                    "entities": [
                        {"type": "insert", "block": "NOPE", "positions": [0, 0, 0]},
                        {"type": "insert", "block": "A", "positions": [0, 0, 0]},
                        {"type": "circle", "center": [0, 0, 0], "radius": 1},
                    ],
                    "blocks": {
                        "A": [{"type": "insert", "block": "A", "positions": [0, 0, 0]}]
                    },
                }
            },
        }
    )
    settle(viewer_client)
    st = _stats(viewer_page, "bad")
    assert len(st["levels"]) == 1 and st["skipped"] >= 2 and st["segments"] > 0


# === glTF loading ===

_GLTF_KEY = "TJSV_vector_entities"


def _glb_with_vectors(payload=None, fallback=True, marker=True) -> bytes:
    """A one-triangle GLB, optionally with a fallback LINES primitive (flagged
    through its material's extras) and the vector payload in root extras."""
    import struct

    tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    seg = np.array([[0, 0, 0], [1, 1, 0]], np.float32)
    blob = tri.tobytes() + seg.tobytes()
    views = [
        {"buffer": 0, "byteOffset": 0, "byteLength": 36},
        {"buffer": 0, "byteOffset": 36, "byteLength": 24},
    ]
    accessors = [
        {
            "bufferView": 0,
            "componentType": 5126,
            "count": 3,
            "type": "VEC3",
            "min": [0, 0, 0],
            "max": [1, 1, 0],
        },
        {
            "bufferView": 1,
            "componentType": 5126,
            "count": 2,
            "type": "VEC3",
            "min": [0, 0, 0],
            "max": [1, 1, 0],
        },
    ]
    materials = [{"name": "body"}]
    prims = [{"attributes": {"POSITION": 0}, "material": 0}]
    if fallback:
        materials.append(
            {"name": "TJSV layer:CUT", "extras": {_GLTF_KEY: {"fallback": True}}}
        )
        prims.append({"attributes": {"POSITION": 1}, "mode": 1, "material": 1})
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "part", "mesh": 0}],
        "meshes": [{"primitives": prims}],
        "materials": materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }
    if payload is not None:
        doc["extras"] = {_GLTF_KEY: payload}
        if marker:
            doc["extensionsUsed"] = [_GLTF_KEY]
            doc["extensions"] = {_GLTF_KEY: {"version": 1, "storage": "extras"}}
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    blob += b"\0" * (-len(blob) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    return (
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(js), 0x4E4F534A)
        + js
        + struct.pack("<II", len(blob), 0x004E4942)
        + blob
    )


_MODEL_VECTORS_JS = """(id) => {
    const v = window.threejsViewer;
    const obj = v._objects.get(id);
    if (!obj) return null;
    const groups = [];
    let fallback = null;
    obj.traverse((o) => {
        if (o.userData.vectorEntities) groups.push(o);
        if (o.userData.vectorFallback) fallback = o.visible;
        if (o.isLine || o.isLineSegments) {
            if (!o.userData.vectorFallback && !o.userData.__vectorLine) fallback = fallback ?? o.visible;
        }
    });
    const g = groups[0];
    const st = g && g.userData.vectorEntities;
    return {
        groups: groups.length,
        tracked: groups.every((x) => v._vectors.groups.has(x)),
        underScene: !!g && g.parent === obj.userData.gltfScene,
        records: st ? st.records.map((r) => ({ layer: r.layer ?? null,
            center: r.bound.center.toArray() })) : [],
        skipped: st ? st.skipped : null,
        segments: st ? st.segmentCount : 0,
        fallbackVisible: fallback,
        layers: Object.fromEntries(v._vectors.listLayers().map((l) => [l.name, l])),
        linetypes: Object.fromEntries([...v._vectors.linetypes].map(([k, p]) => [k, p])),
    };
}"""

_PAYLOAD = {
    "version": 1,
    "units": "m",
    "linetypes": [{"name": "FILEDASH", "pattern": [0.1, -0.05]}],
    "layers": [
        {"name": "CUT", "color": {"aci": 1}, "linetype": "FILEDASH", "lineWidth": 3},
        {"name": "MARK", "color": {"aci": 12}},
        {"name": "KEEP", "color": 0x123456},
    ],
    "entitySets": [
        {
            "layer": "CUT",
            "entities": [
                {"type": "circle", "center": [0.5, 0.5, 0], "radius": 0.2},
                {"type": "line", "start": [0, 0, 0], "end": [1, 0, 0], "layer": "MARK"},
            ],
        },
        {
            "entities": [
                {"type": "insert", "block": "DOT", "positions": [0, 0, 0, 1, 0, 0]},
            ]
        },
    ],
    "blocks": {"DOT": [{"type": "circle", "center": [0, 0, 0], "radius": 0.05}]},
}


@pytest.mark.browser
def test_gltf_extras_payload_builds_vectors_and_hides_fallback(
    viewer_client, viewer_page
):
    # A table set from Python wins over the file's entry of the same name.
    viewer_client.set_layer("KEEP", color=0xABCDEF)
    viewer_client.add_model_binary("m", _glb_with_vectors(_PAYLOAD), format="glb")
    settle(viewer_client)
    frames(viewer_page, 3)
    r = viewer_page.evaluate(_MODEL_VECTORS_JS, "m")
    assert r["groups"] == 1 and r["tracked"] and r["underScene"]
    assert r["skipped"] == 0 and r["segments"] > 0
    layers = [x["layer"] for x in r["records"]]
    assert layers == ["CUT", "MARK", None, None]  # set default; own layer; block copies
    assert r["fallbackVisible"] is False
    assert r["layers"]["CUT"]["color"] == 0xFF0000  # ACI 1
    assert r["layers"]["MARK"]["color"] == 0xA50000  # ACI 12: red at 65%
    assert r["layers"]["CUT"]["linetype"] == "FILEDASH"
    assert r["layers"]["KEEP"]["color"] == 0xABCDEF
    assert r["linetypes"]["FILEDASH"] == [0.1, -0.05]

    # Python layer toggles reach glTF vectors too.
    viewer_client.set_layer("CUT", visible=False)
    settle(viewer_client)
    vis = viewer_page.evaluate(
        """() => { const o = window.threejsViewer._objects.get('m'); const out = {};
            o.traverse((x) => { if (x.userData.__vectorLine) out[x.userData.vectorLayer] = x.visible; });
            return out; }"""
    )
    assert vis["CUT"] is False and vis["MARK"] is True

    viewer_client.delete("m")
    settle(viewer_client)
    assert viewer_page.evaluate("() => window.threejsViewer._vectors.groups.size") == 0


@pytest.mark.browser
def test_gltf_vectors_share_y_up_correction_and_instances_are_independent(
    viewer_client, viewer_page
):
    glb = _glb_with_vectors(_PAYLOAD)
    viewer_client.add_model_binary("a", glb, format="glb", y_up=True)
    viewer_client.add_model_binary("b", glb, format="glb", position=[5, 0, 0])
    settle(viewer_client)
    frames(viewer_page, 3)
    world = viewer_page.evaluate(
        """() => { const v = window.threejsViewer; const out = {};
            for (const id of ['a', 'b']) { let g;
                v._objects.get(id).traverse((x) => { if (x.userData.vectorEntities) g = x; });
                g.updateWorldMatrix(true, false);
                out[id] = { group: g.uuid, centre: g.userData.vectorEntities.records[0]
                    .bound.center.clone().applyMatrix4(g.matrixWorld).toArray() }; }
            out.tracked = v._vectors.groups.size; return out; }"""
    )
    assert world["a"]["group"] != world["b"]["group"] and world["tracked"] == 2
    # Y-up file: glTF (0.5, 0.5, 0) becomes viewer (0.5, 0, 0.5).
    assert world["a"]["centre"] == pytest.approx([0.5, 0, 0.5], abs=1e-6)
    assert world["b"]["centre"] == pytest.approx([5.5, 0.5, 0], abs=1e-6)


@pytest.mark.browser
def test_gltf_without_payload_keeps_fallback_and_bad_payload_degrades(
    viewer_client, viewer_page
):
    viewer_client.add_model_binary("plain", _glb_with_vectors(None), format="glb")
    bad = {"entitySets": [{"entities": [{"type": "circle", "radius": -1}, "junk"]}]}
    viewer_client.add_model_binary(
        "bad", _glb_with_vectors(bad, marker=False), format="glb"
    )
    settle(viewer_client)
    plain = viewer_page.evaluate(_MODEL_VECTORS_JS, "plain")
    assert plain["groups"] == 0 and plain["fallbackVisible"] is True
    badr = viewer_page.evaluate(_MODEL_VECTORS_JS, "bad")
    assert badr["groups"] == 1 and badr["records"] == [] and badr["skipped"] == 2
    assert badr["fallbackVisible"] is True  # nothing drawable: keep the fallback
