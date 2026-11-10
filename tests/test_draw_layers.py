"""glTF draw layers: `extras.draw_layer` orders coplanar meshes in one model."""

import json

import numpy as np
import pytest

from conftest import frames, settle


def _glb_with_draw_layers(layers, group_layer=None) -> bytes:
    """Coplanar one-triangle meshes sharing one material, one node per entry of
    ``layers`` (name -> ``extras.draw_layer`` value, or None for no extras).
    With ``group_layer`` the nodes sit under a parent node on that layer."""
    import struct

    tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    blob = tri.tobytes()
    nodes = []
    for name, layer in layers.items():
        node = {"name": name, "mesh": 0}
        if layer is not None:
            node["extras"] = {"draw_layer": layer}
        nodes.append(node)
    roots = list(range(len(nodes)))
    if group_layer is not None:
        nodes.append(
            {"name": "GROUP", "children": roots, "extras": {"draw_layer": group_layer}}
        )
        roots = [len(nodes) - 1]
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": roots}],
        "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
        "materials": [{"name": "shared"}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0, 0, 0],
                "max": [1, 1, 0],
            }
        ],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(blob)}],
        "buffers": [{"byteLength": len(blob)}],
    }
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    return (
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(js), 0x4E4F534A)
        + js
        + struct.pack("<II", len(blob), 0x004E4942)
        + blob
    )


_DRAW_LAYERS_JS = """(id) => {
    const out = {};
    window.threejsViewer._objects.get(id).traverse((o) => {
        if (!o.isMesh) return;
        const m = o.material;
        out[o.name] = { offset: m.polygonOffset, factor: m.polygonOffsetFactor,
            units: m.polygonOffsetUnits, mat: m.uuid, clay: m.isMeshLambertMaterial === true };
    });
    return out;
}"""


@pytest.mark.browser
def test_gltf_draw_layers_offset_coplanar_meshes(viewer_client, viewer_page):
    glb = _glb_with_draw_layers(
        {"FLOOR": 0, "PATH": 1, "BORDERS": 2, "WALLS": None, "BAD": 1.5}
    )
    viewer_client.add_model_binary("plan", glb, format="glb")
    settle(viewer_client)
    r = viewer_page.evaluate(_DRAW_LAYERS_JS, "plan")
    # Layer 0, no extras and an invalid value draw without offset.
    for name in ("FLOOR", "WALLS", "BAD"):
        assert not r[name]["offset"], name
    assert r["FLOOR"]["mat"] == r["WALLS"]["mat"]  # untouched meshes keep sharing
    assert r["PATH"]["offset"] and r["BORDERS"]["offset"]
    assert r["BORDERS"]["factor"] < r["PATH"]["factor"] < 0
    assert r["BORDERS"]["units"] < r["PATH"]["units"] < 0
    # The shared material was cloned per layer, never modified in place.
    assert len({r[n]["mat"] for n in ("FLOOR", "PATH", "BORDERS")}) == 3

    # Clay swaps in one shared material; layered meshes get an offset copy.
    viewer_client.set_display_quality("low")
    settle(viewer_client)
    frames(viewer_page, 2)
    low = viewer_page.evaluate(_DRAW_LAYERS_JS, "plan")
    assert all(low[n]["clay"] for n in ("FLOOR", "PATH", "BORDERS", "WALLS"))
    assert not low["FLOOR"]["offset"]
    assert low["PATH"]["factor"] == r["PATH"]["factor"]
    assert low["BORDERS"]["units"] == r["BORDERS"]["units"]

    viewer_client.set_display_quality("high")
    settle(viewer_client)
    frames(viewer_page, 2)
    assert viewer_page.evaluate(_DRAW_LAYERS_JS, "plan") == r


@pytest.mark.browser
def test_gltf_draw_layer_inherits_from_nearest_ancestor(viewer_client, viewer_page):
    glb = _glb_with_draw_layers({"INHERIT": None, "OWN": 3, "RESET": 0}, group_layer=2)
    viewer_client.add_model_binary("plan", glb, format="glb")
    settle(viewer_client)
    r = viewer_page.evaluate(_DRAW_LAYERS_JS, "plan")
    assert r["INHERIT"]["factor"] == -2
    assert r["OWN"]["factor"] == -3
    assert not r["RESET"]["offset"]  # an explicit 0 overrides the parent's layer


@pytest.mark.browser
def test_gltf_draw_layers_survive_cached_instances(viewer_client, viewer_page):
    glb = _glb_with_draw_layers({"FLOOR": 0, "PATH": 1})
    viewer_client.add_model_binary("a", glb, format="glb")
    viewer_client.add_model_binary("b", glb, format="glb")
    settle(viewer_client)
    a = viewer_page.evaluate(_DRAW_LAYERS_JS, "a")
    b = viewer_page.evaluate(_DRAW_LAYERS_JS, "b")
    assert a["PATH"]["offset"] and b["PATH"]["offset"]
    assert b["PATH"]["units"] == a["PATH"]["units"] < 0
    assert not b["FLOOR"]["offset"]
