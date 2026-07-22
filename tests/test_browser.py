"""Integration tests using Playwright — verify browser-side behavior end-to-end."""

import json
import math
import socket
import struct
import threading
import time
from http.server import HTTPServer

import numpy as np
import pytest

from threejs_viewer import Animation, Frame, ViewerClient
from threejs_viewer.client import _BlobHandler


@pytest.mark.browser
def test_viewer_connects(viewer_client, viewer_page):
    """Viewer opens and WebSocket connects."""
    assert viewer_client._ws is not None


@pytest.mark.browser
def test_add_box_appears_in_scene(viewer_client, viewer_page):
    """Adding a box from Python creates it in the browser scene graph."""
    viewer_client.add_box("mybox")
    time.sleep(0.1)
    result = viewer_client.query_scene()
    assert "mybox" in result["objects"]
    assert result["objects"]["mybox"]["type"] == "Mesh"


@pytest.mark.browser
def test_add_grid_appears_and_is_excluded_from_bounds(viewer_client, viewer_page):
    """add_grid creates a tracked mesh that never inflates scene bounds."""
    viewer_client.add_box("ref")
    viewer_client.add_grid("floor", cell_size=10.0, extent=10000.0)
    time.sleep(0.2)
    objects = viewer_client.query_scene()["objects"]
    assert objects["floor"]["type"] == "Mesh"
    spheres = viewer_page.evaluate(
        "() => { const v = window.threejsViewer;"
        " v._camController.updateSceneBounds();"
        " return { content: v._sceneSphere.radius,"
        "          nearFar: v._nearFarSphere.radius }; }"
    )
    # 10000-unit grid plane must not count toward framing bounds...
    assert spheres["content"] < 100
    # ...but the near/far fit must still reach it (no far-plane clip).
    assert spheres["nearFar"] > 4000
    viewer_client.delete("floor")
    time.sleep(0.1)
    assert "floor" not in viewer_client.query_scene()["objects"]


@pytest.mark.browser
def test_grouping(viewer_client, viewer_page):
    """Parent-child hierarchy works end-to-end."""
    viewer_client.add_group("arm")
    viewer_client.add_box("joint", parent="arm")
    time.sleep(0.1)
    objects = viewer_client.query_scene()["objects"]
    assert objects["arm"]["type"] == "Group"
    assert "joint" in objects["arm"]["children"]
    assert objects["joint"]["parent"] == "arm"


@pytest.mark.browser
def test_delete_object(viewer_client, viewer_page):
    """Deleting an object removes it from the scene."""
    viewer_client.add_sphere("s1")
    time.sleep(0.05)
    viewer_client.delete("s1")
    time.sleep(0.1)
    objects = viewer_client.query_scene()["objects"]
    assert "s1" not in objects


@pytest.mark.browser
def test_visibility(viewer_client, viewer_page):
    """set_visible toggles object visibility."""
    viewer_client.add_box("v1")
    time.sleep(0.05)
    viewer_client.set_visible("v1", False)
    time.sleep(0.1)
    objects = viewer_client.query_scene()["objects"]
    assert objects["v1"]["visible"] is False


@pytest.mark.browser
def test_set_scene_visibility_before_add_is_honoured(viewer_client, viewer_page):
    """set_scene_visibility for an id that doesn't exist yet must apply once the
    object loads. Regression test for the race where a visibility flip arriving
    during a slow GLB fetch was silently dropped, leaving the loaded object
    permanently at its initial `visible` state (PR #47)."""
    viewer_client.set_scene_visibility({"m1": False})
    time.sleep(0.05)
    viewer_client.add_box("m1")
    time.sleep(0.1)
    objects = viewer_client.query_scene()["objects"]
    assert "m1" in objects
    assert objects["m1"]["visible"] is False


@pytest.mark.browser
def test_baseline_visibility_pruned_on_delete(viewer_client, viewer_page):
    """Deleting an object prunes its baseline so a later re-add isn't shadowed
    by stale visibility from a prior set_scene_visibility."""
    viewer_client.add_box("m1")
    viewer_client.set_scene_visibility({"m1": False})
    time.sleep(0.05)
    viewer_client.delete("m1")
    time.sleep(0.05)
    viewer_client.add_box("m1")
    time.sleep(0.1)
    objects = viewer_client.query_scene()["objects"]
    assert objects["m1"]["visible"] is True


def _get_material_color(page, obj_id):
    """Read the first material color (hex) for an object by id, or None."""
    return page.evaluate(
        "(id) => {"
        " const o = window.threejsViewer._objects.get(id);"
        " if (!o) return null;"
        " let c = null;"
        " o.traverse((child) => {"
        "  if (c !== null || !child.material) return;"
        "  const m = Array.isArray(child.material) ? child.material[0] : child.material;"
        "  if (m && m.color) c = m.color.getHex();"
        " });"
        " return c;"
        "}",
        obj_id,
    )


@pytest.mark.browser
def test_set_color_during_binary_load_is_honoured(viewer_client, viewer_page):
    """set_color sent immediately after add_mesh races the binary HTTP fetch.
    Before the inflight-load deferral fix, set_color silently no-opped because
    _objects.get(id) was undefined when the message dispatched. Regression test
    for the add_*_binary race."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    viewer_client.add_mesh("rc", positions, indices)
    viewer_client.set_color("rc", 0xFF0000)  # fire immediately, no sleep
    # Poll until the mesh lands and the color stuck. The deferred replay
    # happens in a microtask after the load resolves, so a couple of polls
    # past first registration is enough.
    color = None
    for _ in range(40):
        time.sleep(0.05)
        color = _get_material_color(viewer_page, "rc")
        if color == 0xFF0000:
            break
    assert color == 0xFF0000, f"expected 0xff0000, got {color!r}"


@pytest.mark.browser
def test_set_visibility_during_binary_load_is_honoured(viewer_client, viewer_page):
    """set_visible sent immediately after add_mesh races the binary HTTP fetch
    the same way set_color does. The general per-id deferred queue should
    apply the visibility flip once the mesh registers."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    viewer_client.add_mesh("vc", positions, indices)
    viewer_client.set_visible("vc", False)
    objects = None
    for _ in range(40):
        time.sleep(0.05)
        objects = viewer_client.query_scene()["objects"]
        if "vc" in objects:
            break
    assert objects is not None and "vc" in objects
    assert objects["vc"]["visible"] is False


@pytest.mark.browser
def test_delete_during_binary_load_drops_queued_ops(viewer_client, viewer_page):
    """A read-side op queued onto an in-flight load whose target gets deleted
    must drop the op (with a warn) instead of applying to a re-add with the
    same id or raising. The mesh should be absent from the scene at the end."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    viewer_client.add_mesh("dc", positions, indices)
    viewer_client.set_color("dc", 0x00FF00)  # queued on inflight
    viewer_client.delete("dc")  # rejects inflight → set_color drops
    time.sleep(0.4)
    objects = viewer_client.query_scene()["objects"]
    assert "dc" not in objects


@pytest.mark.browser
def test_two_queued_set_colors_apply_in_order(viewer_client, viewer_page):
    """Two set_color calls during a single binary load apply in FIFO order;
    the second call wins. Regression for the microtask-FIFO ordering claim —
    the deferred .then() chain must replay queued ops in arrival order."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    viewer_client.add_mesh("fifo", positions, indices)
    viewer_client.set_color("fifo", 0xFF0000)  # red first
    viewer_client.set_color("fifo", 0x0000FF)  # blue second — must win
    color = None
    for _ in range(40):
        time.sleep(0.05)
        color = _get_material_color(viewer_page, "fifo")
        if color == 0x0000FF:
            break
    assert color == 0x0000FF, f"expected 0x0000ff (blue), got {color!r}"


def _get_material_opacity(page, obj_id):
    """Read the first material opacity for an object by id, or None."""
    return page.evaluate(
        "(id) => {"
        " const o = window.threejsViewer._objects.get(id);"
        " if (!o) return null;"
        " let opacity = null;"
        " o.traverse((child) => {"
        "  if (opacity !== null || !child.material) return;"
        "  const m = Array.isArray(child.material) ? child.material[0] : child.material;"
        "  if (m && typeof m.opacity === 'number') opacity = m.opacity;"
        " });"
        " return opacity;"
        "}",
        obj_id,
    )


@pytest.mark.browser
def test_set_opacity_during_binary_load_is_honoured(viewer_client, viewer_page):
    """set_opacity queued onto an in-flight binary load applies once the
    object lands."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    viewer_client.add_mesh("op", positions, indices)
    viewer_client.set_opacity("op", 0.5)
    opacity = None
    for _ in range(40):
        time.sleep(0.05)
        opacity = _get_material_opacity(viewer_page, "op")
        if opacity is not None and abs(opacity - 0.5) < 1e-3:
            break
    assert opacity is not None and abs(opacity - 0.5) < 1e-3, (
        f"expected opacity 0.5, got {opacity!r}"
    )


@pytest.mark.browser
def test_update_transform_during_binary_load_is_honoured(viewer_client, viewer_page):
    """update_transform (set_matrix) queued onto an in-flight binary load
    applies once the object lands."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    viewer_client.add_mesh("tx", positions, indices)
    # 4x4 translation matrix in column-major order: translate (5, 0, 0).
    viewer_client.set_matrix("tx", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1])
    px = None
    for _ in range(40):
        time.sleep(0.05)
        px = viewer_page.evaluate(
            "() => {"
            " const o = window.threejsViewer._objects.get('tx');"
            " return o ? o.position.x : null;"
            "}"
        )
        if px is not None and abs(px - 5.0) < 1e-3:
            break
    assert px is not None and abs(px - 5.0) < 1e-3, f"expected position.x=5, got {px!r}"


@pytest.mark.browser
def test_set_draw_range_during_binary_load_is_honoured(viewer_client, viewer_page):
    """set_draw_range queued onto an in-flight binary load applies once the
    mesh lands."""
    # Two triangles (6 indices) so a 0.5 draw range produces a stable half-count.
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    indices = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
    viewer_client.add_mesh("dr", positions, indices)
    viewer_client.set_draw_range("dr", 0.5)
    dr = None
    for _ in range(40):
        time.sleep(0.05)
        objects = viewer_client.query_scene()["objects"]
        if "dr" in objects:
            dr = objects["dr"]["drawRange"]
            if abs(dr - 0.5) < 1e-3:
                break
    assert dr is not None and abs(dr - 0.5) < 1e-3, (
        f"expected drawRange 0.5, got {dr!r}"
    )


@pytest.mark.browser
def test_add_points_appears_in_scene(viewer_client, viewer_page):
    """add_points creates a THREE.Points cloud in the browser scene graph."""
    pts = np.random.default_rng(0).random((500, 3)).astype(np.float32)
    scalars = pts[:, 2]
    viewer_client.add_points("cloud", pts, colors=scalars, colormap="turbo")
    obj = None
    for _ in range(40):
        time.sleep(0.05)
        objects = viewer_client.query_scene()["objects"]
        if "cloud" in objects:
            obj = objects["cloud"]
            break
    assert obj is not None, "point cloud never landed in the scene"
    assert obj["type"] == "Points"


@pytest.mark.browser
def test_set_draw_range_on_points(viewer_client, viewer_page):
    """set_draw_range reveals a leading fraction of a point cloud."""
    pts = np.random.default_rng(1).random((1000, 3)).astype(np.float32)
    viewer_client.add_points("cloud", pts)
    viewer_client.set_draw_range("cloud", 0.5)
    dr = None
    for _ in range(40):
        time.sleep(0.05)
        objects = viewer_client.query_scene()["objects"]
        if "cloud" in objects:
            dr = objects["cloud"]["drawRange"]
            if abs(dr - 0.5) < 1e-3:
                break
    assert dr is not None and abs(dr - 0.5) < 1e-3, (
        f"expected drawRange 0.5, got {dr!r}"
    )


@pytest.mark.browser
def test_points_time_window_attributes_and_scrub(viewer_client, viewer_page):
    """birth/removal times land as vertex attributes, the patched shader
    compiles cleanly, and set_points_time drives the shared uniform."""
    shader_errors = []
    viewer_page.on(
        "console",
        lambda msg: (
            shader_errors.append(msg.text)
            if "Shader Error" in msg.text or "THREE.WebGLProgram" in msg.text
            else None
        ),
    )
    pts = np.array([[i, 0.0, 0.0] for i in range(4)], dtype=np.float32)
    birth = np.array([0.0, 1.0, 2.0, np.nan])  # NaN = always existed
    removal = np.array([10.0, 10.0, 10.0, 1.5])
    viewer_client.add_points("pc", pts, birth_times=birth, removal_times=removal)
    info = None
    for _ in range(40):
        time.sleep(0.05)
        info = viewer_page.evaluate(
            "() => {"
            " const o = window.threejsViewer._objects.get('pc');"
            " if (!o) return null;"
            " return {"
            "  hasBirth: !!o.geometry.getAttribute('birthTime'),"
            "  hasRemoval: !!o.geometry.getAttribute('removalTime'),"
            "  time: o.userData.timeUniform ? o.userData.timeUniform.value : null,"
            " };"
            "}"
        )
        if info:
            break
    assert info == {"hasBirth": True, "hasRemoval": True, "time": 0}

    viewer_client.set_points_time("pc", 2.5)
    t = None
    for _ in range(40):
        time.sleep(0.05)
        t = viewer_page.evaluate(
            "() => window.threejsViewer._objects.get('pc').userData.timeUniform.value"
        )
        if t == 2.5:
            break
    assert t == 2.5
    # Let a couple of frames render with the patched program before checking
    # for compile errors.
    time.sleep(0.2)
    assert not shader_errors, f"shader errors with time filter: {shader_errors}"


@pytest.mark.browser
def test_point_times_channel_drives_uniform(viewer_client, viewer_page):
    """The point_times binary animation channel scrubs the cloud's time
    uniform from the playhead (lerped between keyframes)."""
    pts = np.zeros((3, 3), dtype=np.float32)
    viewer_client.add_points("pc", pts, removal_times=np.array([1.0, 2.0, 3.0]))
    time.sleep(0.2)

    anim = Animation(loop=False)
    anim.set_frame_times(np.array([0.0, 1.0]))
    anim.set_point_time_data(["pc"], np.array([[0.0], [5.0]], dtype=np.float32))
    viewer_client.load_animation(anim, autoplay=False, initial_time="end")
    t = None
    for _ in range(40):
        time.sleep(0.05)
        t = viewer_page.evaluate(
            "() => {"
            " const o = window.threejsViewer._objects.get('pc');"
            " return o && o.userData.timeUniform ? o.userData.timeUniform.value : null;"
            "}"
        )
        if t == 5.0:
            break
    assert t == 5.0, f"expected playhead at end to scrub uniform to 5.0, got {t!r}"


@pytest.mark.browser
def test_points_lod_streams_nodes_within_budget(viewer_client, viewer_page):
    """add_points(lod=...) creates a streamed octree cloud: the hierarchy
    loads, node payloads stream on demand, the visible set respects the
    point budget, and the scrub-time uniform reaches the shared material."""
    rng = np.random.default_rng(3)
    n = 60_000
    pts = (rng.random((n, 3)) * [8, 3, 1.5]).astype(np.float32)
    birth = pts[:, 0].astype(np.float64)
    viewer_client.add_points(
        "cloud",
        pts,
        colors=pts[:, 2],
        birth_times=birth,
        removal_times=birth + 4.0,
        lod={"node_capacity": 4000, "point_budget": 30_000, "refine_pixels": 2},
    )
    # All births are > 0, so at the default scrub time t=0 every node is
    # time-culled and nothing streams (that per-node culling is itself part
    # of the design). Scrub into the live range to start streaming.
    viewer_client.set_points_time("cloud", 3.5)
    info = None
    for _ in range(100):
        time.sleep(0.1)
        info = viewer_page.evaluate(
            "() => {"
            " const g = window.threejsViewer._objects.get('cloud');"
            " if (!g || !g.userData.pointsLOD) return null;"
            " const lod = g.userData.pointsLOD;"
            " let loaded = 0, visiblePts = 0, visibleNodes = 0;"
            " for (let i = 0; i < lod.nodes.count; i++) {"
            "   const o = lod.objects[i];"
            "   if (!o) continue;"
            "   loaded++;"
            "   if (o.visible) { visibleNodes++; visiblePts += lod.nodes.counts[i]; }"
            " }"
            " return {"
            "  isGroup: g.isGroup === true,"
            "  nodeCount: lod.nodes.count,"
            "  loaded: loaded, visibleNodes: visibleNodes, visiblePts: visiblePts,"
            "  budget: lod.budget,"
            "  time: g.userData.timeUniform ? g.userData.timeUniform.value : null,"
            " };"
            "}"
        )
        # Wait until streaming has materialized more than just the root.
        if info and info["loaded"] >= 2 and info["visibleNodes"] >= 1:
            break
    assert info, "LOD cloud never appeared"
    assert info["isGroup"] and info["nodeCount"] > 8
    assert info["loaded"] >= 2, f"nodes never streamed in: {info}"
    assert 0 < info["visiblePts"] <= info["budget"], (
        f"visible points {info['visiblePts']} exceed budget {info['budget']}"
    )
    assert info["time"] == 3.5  # set_points_time reached the shared uniform

    # Scrub past every removal time: all nodes must time-cull back out.
    viewer_client.set_points_time("cloud", 100.0)
    visible = None
    for _ in range(40):
        time.sleep(0.05)
        visible = viewer_page.evaluate(
            "() => {"
            " const lod = window.threejsViewer._objects.get('cloud').userData.pointsLOD;"
            " let v = 0;"
            " for (const o of lod.objects) if (o && o.visible) v++;"
            " return v;"
            "}"
        )
        if visible == 0:
            break
    assert visible == 0, f"{visible} nodes still visible after all removals"


@pytest.mark.browser
def test_add_swept_tool_appears_in_scene(viewer_client, viewer_page):
    """add_swept_tool lofts an oriented tool-body mesh into the scene."""
    n = 30
    t = np.linspace(0, 1, n)
    positions = np.column_stack([t * 6 - 3, np.sin(t * 6), 0 * t]).astype(np.float32)
    lean = 0.6 * np.sin(t * 6)
    axes = np.column_stack([np.sin(lean), 0 * t, np.cos(lean)]).astype(np.float32)
    profile = np.array([[0, 0.0], [0.5, 0.5], [0.5, 0.4], [4.0, 0.4]], dtype=np.float32)
    viewer_client.add_swept_tool("shank", positions, axes, profile, sections=16)
    obj = None
    for _ in range(40):
        time.sleep(0.05)
        objects = viewer_client.query_scene()["objects"]
        if "shank" in objects:
            obj = objects["shank"]
            break
    assert obj is not None, "swept tool never landed in the scene"
    assert obj["type"] == "Mesh"


@pytest.mark.browser
def test_set_draw_range_on_swept_tool(viewer_client, viewer_page):
    """set_draw_range reveals the swept tool body progressively along the path."""
    n = 30
    t = np.linspace(0, 1, n)
    positions = np.column_stack([t * 6 - 3, 0 * t, 0 * t]).astype(np.float32)
    axes = np.tile([0, 0, 1.0], (n, 1)).astype(np.float32)
    profile = np.array([[0, 0.4], [4.0, 0.4]], dtype=np.float32)
    viewer_client.add_swept_tool("shank", positions, axes, profile)
    viewer_client.set_draw_range("shank", 0.5)
    dr = None
    for _ in range(40):
        time.sleep(0.05)
        objects = viewer_client.query_scene()["objects"]
        if "shank" in objects:
            dr = objects["shank"]["drawRange"]
            if abs(dr - 0.5) < 0.05:
                break
    assert dr is not None and abs(dr - 0.5) < 0.05, (
        f"expected drawRange ~0.5, got {dr!r}"
    )


def _two_triangle_glb() -> bytes:
    """Build a minimal valid GLB in memory: one mesh, one primitive, 4 verts,
    2 indexed triangles (6 indices). No external assets, no materials."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)
    bin_chunk = positions.tobytes() + indices.tobytes()
    bin_chunk += b"\x00" * (-len(bin_chunk) % 4)
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(bin_chunk)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 48, "target": 34962},
            {"buffer": 0, "byteOffset": 48, "byteLength": 12, "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 4,
                "type": "VEC3",
                "min": [0, 0, 0],
                "max": [1, 1, 0],
            },
            {"bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR"},
        ],
    }
    json_chunk = json.dumps(gltf, separators=(",", ":")).encode()
    json_chunk += b" " * (-len(json_chunk) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    return (
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
        + struct.pack("<II", len(bin_chunk), 0x004E4942)
        + bin_chunk
    )


@pytest.mark.browser
def test_set_draw_range_on_glb_model(viewer_client, viewer_page):
    """set_draw_range applies to GLB meshes loaded via add_model_binary: each
    descendant mesh is stamped isMesh/totalIndexCount after load and the model
    group dispatches the fraction per child (issue #104)."""
    viewer_client.add_model_binary("bellows", _two_triangle_glb(), format="glb")
    objects = {}
    for _ in range(60):
        time.sleep(0.05)
        objects = viewer_client.query_scene()["objects"]
        if "bellows" in objects:
            break
    assert "bellows" in objects, "GLB model did not load"

    viewer_client.set_draw_range("bellows", 0.5)
    time.sleep(0.2)
    state = viewer_page.evaluate(
        "() => {"
        " const o = window.threejsViewer._objects.get('bellows');"
        " const meshes = o.userData.drawRangeMeshes;"
        " return {"
        "  isModelGroup: o.userData.isModelGroup === true,"
        "  nMeshes: meshes.length,"
        "  total: meshes[0].userData.totalIndexCount,"
        "  count: meshes[0].geometry.drawRange.count,"
        " };"
        "}"
    )
    assert state["isModelGroup"] is True
    assert state["nMeshes"] == 1
    assert state["total"] == 6
    assert state["count"] == 3  # half of the 6-index buffer

    # query_scene reports the fraction from the stamped children (a Group has
    # no geometry of its own).
    assert (
        abs(viewer_client.query_scene()["objects"]["bellows"]["drawRange"] - 0.5) < 0.05
    )

    # Full reveal (the unload_animation reset path uses the same dispatcher).
    viewer_client.set_draw_range("bellows", 1.0)
    time.sleep(0.2)
    count = viewer_page.evaluate(
        "() => window.threejsViewer._objects.get('bellows')"
        ".userData.drawRangeMeshes[0].geometry.drawRange.count"
    )
    assert count == 6


@pytest.mark.browser
def test_binary_draw_ranges_channel_on_glb_model(viewer_client, viewer_page):
    """The binary `draw_ranges` animation channel (set_draw_range_data) drives
    the draw range of a GLB model group — a DIFFERENT code path
    (makeChannelApply.draw_ranges) from the `set_draw_range` message
    (_setDrawRange), both wired for isModelGroup in issue #104."""
    viewer_client.add_model_binary("bellows", _two_triangle_glb(), format="glb")
    objects = {}
    for _ in range(60):
        time.sleep(0.05)
        objects = viewer_client.query_scene()["objects"]
        if "bellows" in objects:
            break
    assert "bellows" in objects, "GLB model did not load"

    n_frames = 11
    anim = Animation(loop=False)
    anim.set_frame_times(np.linspace(0, 1.0, n_frames, dtype=np.float32))
    # Values ramp 0 -> 1 so t=0.5 -> 0.5.
    ramp = np.linspace(0, 1, n_frames, dtype=np.float32).reshape(n_frames, 1)
    anim.set_draw_range_data(["bellows"], ramp)
    viewer_client.load_animation(anim, autoplay=False)
    loaded = False
    for _ in range(40):
        time.sleep(0.05)
        if viewer_page.evaluate("() => window.threejsViewer._animation != null"):
            loaded = True
            break
    assert loaded, "animation never loaded"

    # Seek to mid-animation; the channel applier must halve the child mesh's
    # 6-index buffer.
    viewer_page.evaluate("() => window.threejsViewer._seekToTime(0.5)")
    count = None
    for _ in range(40):
        time.sleep(0.05)
        count = viewer_page.evaluate(
            "() => window.threejsViewer._objects.get('bellows')"
            ".userData.drawRangeMeshes[0].geometry.drawRange.count"
        )
        if count == 3:
            break
    assert count == 3, f"expected child drawRange.count 3 at t=0.5, got {count!r}"

    # unload restores the full buffer on the stamped child.
    viewer_client.unload_animation()
    time.sleep(0.2)
    count = viewer_page.evaluate(
        "() => window.threejsViewer._objects.get('bellows')"
        ".userData.drawRangeMeshes[0].geometry.drawRange.count"
    )
    assert count == 6


@pytest.mark.browser
def test_clear_scene(viewer_client, viewer_page):
    """clear() removes all objects."""
    viewer_client.add_box("a")
    viewer_client.add_sphere("b")
    time.sleep(0.05)
    viewer_client.clear()
    time.sleep(0.1)
    objects = viewer_client.query_scene()["objects"]
    assert "a" not in objects
    assert "b" not in objects


@pytest.mark.browser
def test_unload_animation_resets_draw_range(viewer_client, viewer_page):
    """unload_animation() resets draw ranges to full."""
    # Use a real mesh so draw_range metadata (userData.isMesh, totalIndexCount) is set
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    viewer_client.add_mesh("m1", positions, indices)
    time.sleep(0.3)  # wait for HTTP fetch of binary mesh data
    anim = Animation(
        frames=[
            Frame(time=0, transforms={}, draw_ranges={"m1": 0.5}),
            Frame(time=1, transforms={}, draw_ranges={"m1": 0.5}),
        ],
        loop=False,
    )
    viewer_client.load_animation(anim)
    # Wait for async HTTP animation load to complete
    for _ in range(20):
        time.sleep(0.1)
        result = viewer_client.query_scene()
        if result["meta"]["animation"]["playing"]:
            break
    assert result["meta"]["animation"]["playing"] is True, "Animation did not start"
    viewer_client.unload_animation()
    time.sleep(0.1)
    result = viewer_client.query_scene()
    assert result["objects"]["m1"]["drawRange"] == 1.0


def _get_animation_time(page):
    return page.evaluate("() => window.threejsViewer._animationTime")


def _is_playing(page):
    return page.evaluate("() => window.threejsViewer._animationPlaying")


def _has_animation(page):
    return page.evaluate("() => window.threejsViewer._animation != null")


def _get_animation_duration(page):
    return page.evaluate(
        "() => window.threejsViewer._animation ? window.threejsViewer._animation.duration : null"
    )


def _wait_for_animation_loaded(page, timeout_s=2.0):
    """Block until the viewer has an animation attached; raise on timeout.

    Works for both autoplay=True and autoplay=False, since it only checks
    for animation presence — not whether it's playing.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _has_animation(page):
            return
        time.sleep(0.05)
    raise AssertionError(f"animation did not load within {timeout_s:.2f}s")


def _wait_for_animation_duration(page, expected, timeout_s=2.0):
    """Block until the loaded animation reports the expected duration."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _get_animation_duration(page) == expected:
            return
        time.sleep(0.05)
    raise AssertionError(
        f"animation swap to duration={expected} did not land within "
        f"{timeout_s:.2f}s (saw duration={_get_animation_duration(page)})"
    )


@pytest.mark.browser
def test_swap_preserves_playhead_and_play_state(viewer_client, viewer_page):
    """Swapping animations preserves playhead time and play state."""
    viewer_client.add_box("sbox")
    time.sleep(0.1)
    anim_a = Animation(
        frames=[Frame(time=0, transforms={}), Frame(time=5, transforms={})],
        loop=False,
    )
    viewer_client.load_animation(anim_a)
    _wait_for_animation_loaded(viewer_page)
    # Pause first so the playhead doesn't drift between seek and swap.
    viewer_client.pause_animation()
    time.sleep(0.1)
    viewer_page.evaluate("() => window.threejsViewer._seekToTime(2.5)")
    assert _is_playing(viewer_page) is False
    assert abs(_get_animation_time(viewer_page) - 2.5) < 1e-6

    anim_b = Animation(
        frames=[Frame(time=0, transforms={}), Frame(time=10, transforms={})],
        loop=False,
    )
    viewer_client.load_animation(anim_b)
    _wait_for_animation_duration(viewer_page, 10)
    assert _is_playing(viewer_page) is False, "paused state not preserved on swap"
    assert abs(_get_animation_time(viewer_page) - 2.5) < 1e-6, (
        "playhead not preserved on swap"
    )

    # Resume, swap again, and verify playing state is preserved too.
    viewer_client.resume_animation()
    time.sleep(0.1)
    assert _is_playing(viewer_page) is True
    viewer_client.load_animation(anim_a)
    _wait_for_animation_duration(viewer_page, 5)
    assert _is_playing(viewer_page) is True, "playing state not preserved on swap"


@pytest.mark.browser
def test_restart_resets_to_zero(viewer_client, viewer_page):
    """load_animation(restart=True) resets playhead to 0 on a swap."""
    viewer_client.add_box("rbox")
    time.sleep(0.1)
    anim = Animation(
        frames=[Frame(time=0, transforms={}), Frame(time=5, transforms={})],
        loop=False,
    )
    viewer_client.load_animation(anim)
    _wait_for_animation_loaded(viewer_page)
    # Pause so the playhead doesn't drift between seek and the restart swap.
    viewer_client.pause_animation()
    time.sleep(0.1)
    viewer_page.evaluate("() => window.threejsViewer._seekToTime(3.0)")
    assert abs(_get_animation_time(viewer_page) - 3.0) < 1e-6

    # autoplay=False keeps the restart deterministic — playhead sits at 0.0
    # instead of advancing from 0 the moment the animation reloads.
    viewer_client.load_animation(anim, restart=True, autoplay=False)
    # Wait for the restart to land (playhead snaps back to 0, still paused).
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if (
            _is_playing(viewer_page) is False
            and _get_animation_time(viewer_page) == 0.0
        ):
            break
        time.sleep(0.05)
    assert _is_playing(viewer_page) is False, (
        "autoplay=False should keep restart paused"
    )
    assert _get_animation_time(viewer_page) == 0.0, "restart did not reset playhead"


@pytest.mark.browser
def test_autoplay_false_loads_paused(viewer_client, viewer_page):
    """load_animation(autoplay=False) loads paused on first-load."""
    viewer_client.add_box("abox")
    time.sleep(0.1)
    anim = Animation(
        frames=[Frame(time=0, transforms={}), Frame(time=1, transforms={})],
        loop=False,
    )
    viewer_client.load_animation(anim, autoplay=False)
    _wait_for_animation_loaded(viewer_page)
    assert _is_playing(viewer_page) is False, "autoplay=False still started playing"


@pytest.mark.browser
def test_initial_time_end_lands_at_duration(viewer_client, viewer_page):
    """load_animation(initial_time='end', autoplay=False) parks playhead at duration."""
    viewer_client.add_box("ebox")
    time.sleep(0.1)
    anim = Animation(
        frames=[Frame(time=0, transforms={}), Frame(time=5, transforms={})],
        loop=False,
    )
    viewer_client.load_animation(anim, autoplay=False, initial_time="end")
    _wait_for_animation_loaded(viewer_page)
    # Playhead should snap to duration immediately, no t=0 flash.
    assert _is_playing(viewer_page) is False
    assert abs(_get_animation_time(viewer_page) - 5.0) < 1e-6, (
        f"expected playhead at 5.0, got {_get_animation_time(viewer_page)}"
    )


@pytest.mark.browser
def test_initial_time_numeric_seek(viewer_client, viewer_page):
    """load_animation(initial_time=2.5) lands at 2.5s on first load."""
    viewer_client.add_box("nbox")
    time.sleep(0.1)
    anim = Animation(
        frames=[Frame(time=0, transforms={}), Frame(time=5, transforms={})],
        loop=False,
    )
    viewer_client.load_animation(anim, autoplay=False, initial_time=2.5)
    _wait_for_animation_loaded(viewer_page)
    assert abs(_get_animation_time(viewer_page) - 2.5) < 1e-6


@pytest.mark.browser
def test_loop_override_false_holds_at_end(viewer_client, viewer_page):
    """load_animation(loop=False) disables looping even when the Animation is loop=True."""
    viewer_client.add_box("lbox")
    time.sleep(0.1)
    # Animation is baked with loop=True — the kwarg must override.
    anim = Animation(
        frames=[Frame(time=0, transforms={}), Frame(time=0.5, transforms={})],
        loop=True,
    )
    viewer_client.load_animation(anim, loop=False, initial_time="end")
    _wait_for_animation_loaded(viewer_page)
    # Playhead starts at duration; with loop override=False it should not wrap.
    # Wait past the duration and verify we're still holding at 0.5 (not at 0).
    time.sleep(0.5)
    t = _get_animation_time(viewer_page)
    assert abs(t - 0.5) < 0.1, (
        f"loop=False override failed: playhead at {t} instead of holding at 0.5"
    )


@pytest.mark.browser
def test_pause_and_resume_animation(viewer_client, viewer_page):
    """pause_animation() / resume_animation() toggle meta.animation.playing."""
    viewer_client.add_box("pbox")
    time.sleep(0.1)
    anim = Animation(
        frames=[Frame(time=0, transforms={}), Frame(time=1, transforms={})],
        loop=True,
    )
    viewer_client.load_animation(anim)
    _wait_for_animation_loaded(viewer_page)
    # Autoplay default is True, so the animation should be playing after load.
    deadline = time.time() + 2.0
    while time.time() < deadline and not _is_playing(viewer_page):
        time.sleep(0.05)
    assert viewer_client.query_scene()["meta"]["animation"]["playing"] is True

    viewer_client.pause_animation()
    time.sleep(0.1)
    assert viewer_client.query_scene()["meta"]["animation"]["playing"] is False

    viewer_client.resume_animation()
    time.sleep(0.1)
    assert viewer_client.query_scene()["meta"]["animation"]["playing"] is True


@pytest.mark.browser
def test_clear_resets_animation_state(viewer_client, viewer_page):
    """clear() resets animation state."""
    viewer_client.add_box("obj1")
    time.sleep(0.1)
    anim = Animation(
        frames=[
            Frame(time=0, transforms={}),
            Frame(time=1, transforms={}),
        ],
        loop=True,
    )
    viewer_client.load_animation(anim)
    # Wait for async HTTP animation load to complete
    for _ in range(20):
        time.sleep(0.1)
        result = viewer_client.query_scene()
        if result["meta"]["animation"]["playing"]:
            break
    assert result["meta"]["animation"]["playing"] is True, "Animation did not start"
    viewer_client.clear()
    time.sleep(0.2)
    result = viewer_client.query_scene()
    assert result["meta"]["animation"]["playing"] is False


@pytest.mark.browser
def test_handle_message_dispatches_without_websocket(viewer_client, viewer_page):
    """`viewer.handleMessage(data)` is the public, WS-decoupled control-message
    entry point (issue #71). Feeding a message straight in — no WebSocket frame —
    must mutate the scene exactly as an `onmessage` would, so an embedder can drive
    the viewer from local/static data with `{ autoConnect: false }`. We call it in
    the browser directly, bypassing the socket, and assert the object appears; a
    delete_object then removes it. Query messages (`list_objects`) must not throw
    when dispatched this way — `_reply()` guards the send."""
    present = viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            v.handleMessage({ type: 'add_group', id: 'hm_group' });
            // A query message must be safe to dispatch directly (routes via _reply).
            v.handleMessage({ type: 'list_objects', requestId: 1 });
            return v._objects.has('hm_group');
        }"""
    )
    assert present is True

    removed = viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            v.handleMessage({ type: 'delete_object', id: 'hm_group' });
            return v._objects.has('hm_group');
        }"""
    )
    assert removed is False


@pytest.mark.browser
def test_show_grid(viewer_client, viewer_page):
    """show_grid() toggles grid visibility."""
    time.sleep(0.1)
    meta = viewer_client.query_scene()["meta"]
    # Grid is hidden by default
    assert meta["grid"]["visible"] is False

    viewer_client.show_grid(visible=True)
    time.sleep(0.1)
    meta = viewer_client.query_scene()["meta"]
    assert meta["grid"]["visible"] is True

    viewer_client.show_grid(visible=False)
    time.sleep(0.1)
    meta = viewer_client.query_scene()["meta"]
    assert meta["grid"]["visible"] is False


# --- Debug display cycles (M / N keys) ---


def _press_key(page, code):
    """Dispatch a keydown event on the viewer container."""
    page.evaluate(
        """(code) => {
            const el = window.threejsViewer.container;
            const evt = new KeyboardEvent('keydown', { code, bubbles: true });
            el.dispatchEvent(evt);
        }""",
        code,
    )


@pytest.mark.browser
def test_m_key_cycles_wireframe_mode(viewer_client, viewer_page):
    """M key cycles wireframe mode 0 → 1 → 2 → 0 across the whole scene."""
    viewer_client.add_box("wbox")
    time.sleep(0.1)
    get_mode = "() => window.threejsViewer._shading.wireframeMode"
    assert viewer_page.evaluate(get_mode) == 0

    expected = [1, 2, 0]
    for want in expected:
        _press_key(viewer_page, "KeyM")
        time.sleep(0.05)
        assert viewer_page.evaluate(get_mode) == want

    # In combined mode (2), the box should have a wireframe overlay child.
    _press_key(viewer_page, "KeyM")  # back to 1
    _press_key(viewer_page, "KeyM")  # to 2
    time.sleep(0.05)
    has_overlay = viewer_page.evaluate(
        """() => {
            const obj = window.threejsViewer._objects.get('wbox');
            const ov = obj.userData.wireframeOverlay;
            return !!(ov && ov.visible);
        }"""
    )
    assert has_overlay


@pytest.mark.browser
def test_n_key_cycles_shading_mode(viewer_client, viewer_page):
    """N key cycles shading debug mode 0 → 1 → 2 → 3 → 0."""
    viewer_client.add_sphere("sdebug")
    time.sleep(0.1)
    get_mode = "() => window.threejsViewer._shading.shadingMode"
    assert viewer_page.evaluate(get_mode) == 0

    for want in [1, 2, 3, 0]:
        _press_key(viewer_page, "KeyN")
        time.sleep(0.05)
        assert viewer_page.evaluate(get_mode) == want


@pytest.mark.browser
def test_m_and_n_compose(viewer_client, viewer_page):
    """M and N modes are independent and compose."""
    viewer_client.add_box("compose_box")
    time.sleep(0.1)
    _press_key(viewer_page, "KeyM")  # wireframe = 1
    _press_key(viewer_page, "KeyN")  # shading = 1
    time.sleep(0.05)
    state = viewer_page.evaluate(
        "() => ({w: window.threejsViewer._shading.wireframeMode, s: window.threejsViewer._shading.shadingMode})"
    )
    assert state == {"w": 1, "s": 1}


# --- ViewerControls ---


@pytest.mark.browser
def test_viewer_controls_installed(viewer_client, viewer_page):
    """ViewerControls is wired up with a writable target Vector3."""
    info = viewer_page.evaluate(
        """() => {
            const c = window.threejsViewer._controls;
            if (!c) return null;
            return {
                hasTarget: c.target && typeof c.target.x === 'number',
                hasUpdate: typeof c.update === 'function',
                mode: c.mode,
            };
        }"""
    )
    assert info is not None, "ViewerControls not installed"
    assert info["hasTarget"]
    assert info["hasUpdate"]
    assert info["mode"] in ("turntable", "free")


@pytest.mark.browser
def test_viewer_controls_target_move_does_not_move_camera(viewer_client, viewer_page):
    """The no-view-shift guarantee: moving target alone leaves the camera pose unchanged."""
    delta = viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            const c = v._controls;
            const cam = v._camera;
            const p0 = cam.position.clone();
            const q0 = cam.quaternion.clone();
            // Move the pivot target arbitrarily.
            c.target.set(5, -3, 2);
            c.update();
            const dp = cam.position.distanceTo(p0);
            const dq = Math.abs(1 - Math.abs(cam.quaternion.dot(q0)));
            return { dp, dq };
        }"""
    )
    assert delta["dp"] < 1e-6, delta
    assert delta["dq"] < 1e-6, delta


@pytest.mark.browser
def test_viewer_controls_r_key_toggles_orbit_mode(viewer_client, viewer_page):
    """R key toggles orbit mode between turntable and free."""
    start = viewer_page.evaluate("() => window.threejsViewer._controls.mode")
    _press_key(viewer_page, "KeyR")
    time.sleep(0.05)
    after = viewer_page.evaluate("() => window.threejsViewer._controls.mode")
    assert after != start
    assert {start, after} == {"turntable", "free"}


# --- Framing honors visibility ---


@pytest.mark.browser
def test_reset_view_skips_invisible_objects(viewer_client, viewer_page):
    """Hidden objects must not pull the framing bbox.

    Setup: a tiny visible box near the origin and a huge hidden box far away.
    If resetView/frameAll honor `.visible`, the orbit target lands on the
    visible box's center, not the midpoint between the two.
    """
    viewer_client.add_box("near", width=0.1, height=0.1, depth=0.1, position=[0, 0, 0])
    viewer_client.add_box("far", width=2, height=2, depth=2, position=[100, 100, 100])
    viewer_client.set_visible("far", False)
    # query_scene round-trips through the WS, which guarantees the queued
    # add/set_visibility messages have been applied before we frame.
    objects = viewer_client.query_scene()["objects"]
    assert objects["far"]["visible"] is False

    # frameAll: target should be at origin (visible box center), not at (~50,50,50).
    target = viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            v.frameAll();
            const t = v._controls.target;
            return { x: t.x, y: t.y, z: t.z };
        }"""
    )
    # Visible box center is the origin; allow a small slack for floating point.
    assert abs(target["x"]) < 1e-3, target
    assert abs(target["y"]) < 1e-3, target
    assert abs(target["z"]) < 1e-3, target

    # resetView: same expectation — orbit target snaps to the visible content.
    target = viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            v.resetView();
            const t = v._controls.target;
            return { x: t.x, y: t.y, z: t.z };
        }"""
    )
    assert abs(target["x"]) < 1e-3, target
    assert abs(target["y"]) < 1e-3, target
    assert abs(target["z"]) < 1e-3, target

    # Re-show the hidden box: framing should now include it.
    viewer_client.set_visible("far", True)
    # query_scene round-trips through the WS to the browser, which guarantees
    # any preceding messages (the set_visibility above) have been processed.
    objects = viewer_client.query_scene()["objects"]
    assert objects["far"]["visible"] is True
    state = viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            const far = v._objects.get('far');
            const near = v._objects.get('near');
            v.frameAll();
            const t = v._controls.target;
            return {
                target: { x: t.x, y: t.y, z: t.z },
                farVisible: far ? far.visible : null,
                nearVisible: near ? near.visible : null,
                farPos: far ? { x: far.position.x, y: far.position.y, z: far.position.z } : null,
            };
        }"""
    )
    assert state["farVisible"] is True, state
    assert state["nearVisible"] is True, state
    # With both boxes visible, the bbox is ~([-0.05, 101], [-0.05, 101], [-0.05, 101])
    # so the center sits well above 40 on every axis.
    assert state["target"]["x"] > 40, state
    assert state["target"]["y"] > 40, state
    assert state["target"]["z"] > 40, state

    # Hide everything: empty-bbox path. resetView must fall through to the
    # origin-and-default-distance fallback without crashing.
    viewer_client.set_visible("near", False)
    viewer_client.set_visible("far", False)
    objects = viewer_client.query_scene()["objects"]
    assert objects["near"]["visible"] is False
    assert objects["far"]["visible"] is False
    target = viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            v.resetView();
            const t = v._controls.target;
            return { x: t.x, y: t.y, z: t.z };
        }"""
    )
    assert abs(target["x"]) < 1e-3, target
    assert abs(target["y"]) < 1e-3, target
    assert abs(target["z"]) < 1e-3, target


# --- update_polyline_colors round-trip ---


def _read_polyline_first_color(page, id_):
    return page.evaluate(
        """(id) => {
            const obj = window.threejsViewer._objects.get(id);
            const start = obj.geometry.attributes.instanceColorStart;
            return { r: start.array[0], g: start.array[1], b: start.array[2] };
        }""",
        id_,
    )


@pytest.mark.browser
def test_update_polyline_colors_swaps_colors(viewer_client, viewer_page):
    """update_polyline_colors replaces the per-vertex colors on an existing polyline."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    rgb_red = np.array([[1, 0, 0]] * 3, dtype=np.float32)
    viewer_client.add_polyline("pl_swap", pts, colors=rgb_red)
    # Polyline create is async (HTTP fetch); poll until the object exists.
    for _ in range(40):
        time.sleep(0.05)
        if viewer_client.query_scene()["objects"].get("pl_swap"):
            break
    else:
        pytest.fail("polyline 'pl_swap' did not appear within 2s")
    before = _read_polyline_first_color(viewer_page, "pl_swap")
    assert abs(before["r"] - 1.0) < 1e-3
    assert before["g"] < 0.01

    rgb_blue = np.array([[0, 0, 1]] * 3, dtype=np.float32)
    viewer_client.update_polyline_colors("pl_swap", rgb_blue)
    # Color update is also async; poll for the swap to land.
    for _ in range(40):
        time.sleep(0.05)
        c = _read_polyline_first_color(viewer_page, "pl_swap")
        if c["b"] > 0.99 and c["r"] < 0.01:
            break
    else:
        pytest.fail(f"color swap on 'pl_swap' did not land within 2s; last={c}")
    after = _read_polyline_first_color(viewer_page, "pl_swap")
    assert after["r"] < 0.01, after
    assert abs(after["b"] - 1.0) < 1e-3, after


@pytest.mark.browser
def test_update_polyline_colors_flips_material_when_no_initial_colors(
    viewer_client, viewer_page
):
    """If a polyline was created without per-vertex colors, the update must
    flip the material into vertex-color mode so the new colors are used."""
    pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    # Use a non-white base color: confirms the white-tint reset on flip.
    # If the base color stayed red, the green vertex colors would render
    # as black (red × green = 0).
    viewer_client.add_polyline("pl_noinit", pts, color=0xFF0000)
    for _ in range(40):
        time.sleep(0.05)
        if viewer_client.query_scene()["objects"].get("pl_noinit"):
            break
    else:
        pytest.fail("polyline 'pl_noinit' did not appear within 2s")
    initial_vertex_colors = viewer_page.evaluate(
        "(id) => window.threejsViewer._objects.get(id).material.vertexColors",
        "pl_noinit",
    )
    assert initial_vertex_colors is False

    rgb = np.array([[0, 1, 0], [0, 1, 0]], dtype=np.float32)
    viewer_client.update_polyline_colors("pl_noinit", rgb)
    for _ in range(40):
        time.sleep(0.05)
        flipped = viewer_page.evaluate(
            "(id) => window.threejsViewer._objects.get(id).material.vertexColors",
            "pl_noinit",
        )
        if flipped:
            break
    else:
        pytest.fail("vertexColors flip on 'pl_noinit' did not land within 2s")
    assert flipped is True
    # Material's base color must be white after the flip — otherwise the
    # vertex green would be tinted/zeroed by the prior 0xFF0000 base.
    base_color = viewer_page.evaluate(
        "(id) => window.threejsViewer._objects.get(id).material.color.getHex()",
        "pl_noinit",
    )
    assert base_color == 0xFFFFFF, hex(base_color)
    color = _read_polyline_first_color(viewer_page, "pl_noinit")
    assert color["r"] < 0.01, color
    assert abs(color["g"] - 1.0) < 1e-3, color


# --- ViewHelper setViewport shim regression ---


@pytest.mark.browser
def test_view_helper_setviewport_shim_no_stack_overflow(viewer_client, viewer_page):
    """Render many frames with the animation toolbar visible (lift > 0).
    The shim must cache the original setViewport once and never re-wrap.

    Regression for a prior bug where the shim re-wrapped the already-wrapped
    setViewport every frame, deepening the call chain by one level per frame
    until the stack blew."""
    result = viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            // Force toolbar visible so the lift > 0 branch runs. The render
            // loop reads the cached CSS-pixel lift (updated by the show/hide
            // paths) rather than offsetHeight; set it directly here.
            v._animControlsEl.classList.add('visible');
            v._animLiftCss = 40;
            // Trigger many render passes synchronously.
            const origAnimate = v._animate.bind(v);
            for (let i = 0; i < 200; i++) {
                origAnimate();
            }
            return {
                cached: !!v._rendererSetViewportOriginal,
                restored: v._renderer.setViewport === v._rendererSetViewportOriginal,
            };
        }"""
    )
    assert result["cached"], "shim never cached the original setViewport"
    assert result["restored"], "setViewport was not restored after _viewHelper.render()"


@pytest.mark.browser
def test_anim_lift_tracks_toolbar_reflow_on_resize(viewer_client, viewer_page):
    """Toolbar height depends on viewport width (timeline-row wraps when
    controls don't fit). The render-shim/hit-test cache + --tjsv-anim-lift
    CSS var must follow the toolbar so the gizmo and Home button stay
    clear of the toolbar after a resize.

    Regression: prior behavior only wrote the cache at load/unload, so
    shrinking the viewport left the cache stale and the Home button
    overlapped the now-taller toolbar."""
    viewer_page.set_viewport_size({"width": 1600, "height": 900})
    viewer_client.add_sphere("s")
    identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    frames = [Frame(time=t / 10, transforms={"s": identity}) for t in range(10)]
    viewer_client.load_animation(Animation(frames=frames))
    viewer_page.wait_for_function(
        "() => window.threejsViewer._animLiftCss > 0", timeout=5000
    )

    def snapshot():
        return viewer_page.evaluate(
            """() => {
                const v = window.threejsViewer;
                const home = v.el.querySelector('.tjsv-view-home');
                const homeRect = home.getBoundingClientRect();
                const tbRect = v._animControlsEl.getBoundingClientRect();
                return {
                    animLiftCss: v._animLiftCss,
                    tbHeight: v._animControlsEl.offsetHeight,
                    cssVar: getComputedStyle(v.el)
                        .getPropertyValue('--tjsv-anim-lift')
                        .trim(),
                    homeBottom: homeRect.bottom,
                    tbTop: tbRect.top,
                };
            }"""
        )

    wide = snapshot()
    assert wide["animLiftCss"] == wide["tbHeight"]
    assert wide["cssVar"] == f"{wide['animLiftCss']}px"

    # Force timeline-row to wrap by narrowing the viewport. The toolbar
    # grows; the ResizeObserver must update the cache + CSS var.
    viewer_page.set_viewport_size({"width": 500, "height": 900})
    viewer_page.wait_for_function(
        f"() => window.threejsViewer._animLiftCss > {wide['animLiftCss']}",
        timeout=2000,
    )
    narrow = snapshot()
    assert narrow["tbHeight"] > wide["tbHeight"], (
        f"toolbar didn't grow on shrink: wide={wide['tbHeight']} "
        f"narrow={narrow['tbHeight']}"
    )
    assert narrow["animLiftCss"] == narrow["tbHeight"], (
        f"cache stale after shrink: {narrow}"
    )
    assert narrow["cssVar"] == f"{narrow['animLiftCss']}px", (
        f"CSS var stale after shrink: {narrow}"
    )
    # Home button sits above the toolbar (1px tolerance for sub-pixel rounding).
    assert narrow["homeBottom"] <= narrow["tbTop"] + 1, (
        f"Home button overlaps toolbar after shrink: {narrow}"
    )

    # Expand back — cache returns to original.
    viewer_page.set_viewport_size({"width": 1600, "height": 900})
    viewer_page.wait_for_function(
        f"() => window.threejsViewer._animLiftCss === {wide['animLiftCss']}",
        timeout=2000,
    )


# --- Lighting panel: URL → renderer wiring + precedence vs localStorage ---


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _start_client(**kwargs):
    """Start a ViewerClient + its HTTP sidecar without waiting for a browser.

    Mirrors the bare-bones setup the ``viewer_client`` fixture does, but
    accepts arbitrary ``ViewerClient`` kwargs — the fixture doesn't, and the
    lighting tests need to drive the constructor.
    """
    port = _free_port()
    client = ViewerClient(port=port, open_browser=False, **kwargs)
    client._http_port = port + 1
    http_server = HTTPServer((client.host, client._http_port), _BlobHandler)
    http_server.blob_store = client._blob_store
    client._http_server = http_server
    threading.Thread(target=http_server.serve_forever, daemon=True).start()
    client._server_thread = threading.Thread(target=client._run_server, daemon=True)
    client._server_thread.start()
    return client


def _wait_for_viewer(page):
    """Block until window.threejsViewer has finished its constructor."""
    page.wait_for_function(
        "() => window.threejsViewer && window.threejsViewer._renderer",
        timeout=10_000,
    )


@pytest.mark.browser
def test_lighting_url_params_beat_localstorage(page):
    """The PR's central claim: URL-pinned lighting values win over localStorage on reload.

    Flow: visit once with no URL params and seed localStorage with rival
    values; then visit again with the four lighting query params pinned and
    assert the renderer/scene/ambient-light state lands on the URL values,
    not the localStorage ones.
    """
    client = _start_client(
        tone_mapping="neutral",
        tone_mapping_exposure=2.3,
        environment_intensity=0.5,
        ambient_intensity=0.7,
    )
    try:
        # First visit: no lighting query params, just ws_port. Seed localStorage
        # with values that disagree with every URL-pinned value above.
        path_uri = client.viewer_path.resolve().as_uri()
        page.goto(f"{path_uri}?ws_port={client.port}")
        _wait_for_viewer(page)
        page.evaluate(
            """() => {
                localStorage.setItem('tjsv.toneMappingExposure', '0.1');
                localStorage.setItem('tjsv.environmentIntensity', '3.9');
                localStorage.setItem('tjsv.ambientIntensity', '2.9');
                localStorage.setItem('tjsv.toneMapping', 'agx');
            }"""
        )

        # Second visit: URL now pins lighting values. Same origin, so the
        # localStorage seeded above is still present — URL must beat it.
        page.goto(client.viewer_url)
        _wait_for_viewer(page)
        state = page.evaluate(
            """() => {
                const v = window.threejsViewer;
                return {
                    exposure: v._renderer.toneMappingExposure,
                    envIntensity: v._scene.environmentIntensity,
                    ambient: v._ambientLight.intensity,
                    toneMapping: v._lightingDefaults.toneMapping,
                };
            }"""
        )
        assert state["exposure"] == pytest.approx(2.3)
        assert state["envIntensity"] == pytest.approx(0.5)
        assert state["ambient"] == pytest.approx(0.7)
        assert state["toneMapping"] == "neutral"
    finally:
        client.disconnect()


@pytest.mark.browser
def test_lighting_panel_edits_persist_in_localstorage(page):
    """Panel slider writes go to localStorage under the ``tjsv.`` namespace and
    are re-applied on reload when no URL param pins the value."""
    client = _start_client()
    try:
        page.goto(f"{client.viewer_path.resolve().as_uri()}?ws_port={client.port}")
        _wait_for_viewer(page)
        # Start from a clean slate so this test is reentrant across runs.
        page.evaluate(
            """() => {
                localStorage.removeItem('tjsv.toneMappingExposure');
                localStorage.removeItem('tjsv.environmentIntensity');
                localStorage.removeItem('tjsv.ambientIntensity');
                localStorage.removeItem('tjsv.toneMapping');
            }"""
        )
        # Simulate a user dragging the exposure slider.
        page.evaluate(
            """() => {
                const slider = window.threejsViewer._lightingExposureSlider;
                slider.value = '0.25';
                slider.dispatchEvent(new Event('input', { bubbles: true }));
            }"""
        )
        ls_value = page.evaluate(
            "() => localStorage.getItem('tjsv.toneMappingExposure')"
        )
        assert ls_value == "0.25"

        # Reload: with no URL param, localStorage should drive the initial value.
        page.reload()
        _wait_for_viewer(page)
        applied = page.evaluate(
            "() => window.threejsViewer._renderer.toneMappingExposure"
        )
        assert applied == pytest.approx(0.25)
    finally:
        client.disconnect()


@pytest.mark.browser
def test_tone_mapping_change_flushes_materials(page):
    """Switching tone-mapping mode must set ``needsUpdate = true`` on every
    material so three.js recompiles shaders against the new tone-mapping
    constant. Without this flush the renderer value changes but already-
    compiled programs keep the old look."""
    client = _start_client()
    try:
        page.goto(f"{client.viewer_path.resolve().as_uri()}?ws_port={client.port}")
        _wait_for_viewer(page)
        # Wait for the WS handshake so we can push a box into the scene.
        assert client._connected_event.wait(timeout=10)
        client.add_box("flushbox")
        time.sleep(0.1)
        # Force the box material's `version` to a known state, then swap modes
        # and confirm three.js bumped it (which is how `needsUpdate = true` is
        # observable — it increments `.version`).
        before = page.evaluate(
            """() => {
                const obj = window.threejsViewer._objects.get('flushbox');
                const mat = Array.isArray(obj.material) ? obj.material[0] : obj.material;
                return mat.version;
            }"""
        )
        page.evaluate(
            """() => {
                const sel = window.threejsViewer._lightingToneMappingSelect;
                sel.value = 'agx';
                sel.dispatchEvent(new Event('change', { bubbles: true }));
            }"""
        )
        after = page.evaluate(
            """() => {
                const obj = window.threejsViewer._objects.get('flushbox');
                const mat = Array.isArray(obj.material) ? obj.material[0] : obj.material;
                return mat.version;
            }"""
        )
        assert after > before, (
            f"material.version did not increment after tone-mapping swap "
            f"(before={before}, after={after}) — materials were not flushed"
        )
        # Renderer constant must have moved away from the default (ACESFilmic).
        initial_tm = page.evaluate(
            "() => window.threejsViewer._lightingDefaults.reset.toneMapping"
        )
        current_tm = page.evaluate(
            "() => window.threejsViewer._lightingToneMappingSelect.value"
        )
        assert initial_tm == "aces"
        assert current_tm == "agx"
    finally:
        client.disconnect()


@pytest.mark.browser
def test_environment_map_toggle_drops_and_restores(page):
    """The lighting panel's Environment map checkbox nulls scene.environment
    (uglier-but-faster) and restores the retained PMREM map when re-checked."""
    client = _start_client()
    try:
        # Pin environment_map=true in the URL so a persisted localStorage
        # `tjsv.environmentMap=false` from another test/run can't make the
        # "starts enabled" assertion flaky.
        page.goto(
            f"{client.viewer_path.resolve().as_uri()}"
            f"?ws_port={client.port}&environment_map=true"
        )
        _wait_for_viewer(page)
        # Wait for the cubemap PMREM env map to finish loading (async images).
        page.wait_for_function(
            "() => window.threejsViewer._envMap != null", timeout=10_000
        )
        assert page.evaluate("() => window.threejsViewer._scene.environment != null")
        # Uncheck -> scene.environment nulled, retained map preserved.
        page.evaluate(
            """() => {
                const cb = window.threejsViewer._lightingEnvMapCheck;
                cb.checked = false;
                cb.dispatchEvent(new Event('change', { bubbles: true }));
            }"""
        )
        assert page.evaluate("() => window.threejsViewer._scene.environment == null")
        assert page.evaluate("() => window.threejsViewer._envMap != null")
        # Re-check -> restored from the retained map.
        page.evaluate(
            """() => {
                const cb = window.threejsViewer._lightingEnvMapCheck;
                cb.checked = true;
                cb.dispatchEvent(new Event('change', { bubbles: true }));
            }"""
        )
        assert page.evaluate("() => window.threejsViewer._scene.environment != null")
    finally:
        client.disconnect()


@pytest.mark.browser
def test_environment_map_url_param_starts_disabled(page):
    """environment_map=false in the URL starts with the env map off."""
    client = _start_client()
    try:
        page.goto(
            f"{client.viewer_path.resolve().as_uri()}"
            f"?ws_port={client.port}&environment_map=false"
        )
        _wait_for_viewer(page)
        page.wait_for_function(
            "() => window.threejsViewer._envMap != null", timeout=10_000
        )
        # Map loaded but not attached; checkbox reflects the off state.
        assert page.evaluate("() => window.threejsViewer._scene.environment == null")
        assert page.evaluate(
            "() => window.threejsViewer._lightingEnvMapCheck.checked === false"
        )
    finally:
        client.disconnect()


@pytest.mark.browser
def test_polyline_pick_roundtrip(viewer_client, viewer_page):
    """Hovering + clicking a polyline in the browser sends a pick back to
    Python with the right arc-length fraction and on-line coordinate."""
    picks = []

    def on_pick(p):
        picks.append(p)
        # Mirror the example: issue a viewer command from inside the callback.
        # This runs on the WebSocket receive thread, so it also checks that a
        # re-entrant send (recv loop → ws.send) doesn't deadlock.
        viewer_client.add_sphere("hit", radius=0.1, position=p["point"])

    viewer_client.on_polyline_pick(on_pick)

    # A straight 3D segment, symmetric about the origin and evenly sampled, so
    # the geometric midpoint (0,0,0) sits at fraction 0.5. The diagonal keeps it
    # from being edge-on under the default 3/4 view.
    direction = np.array([1.0, 0.6, 0.4], dtype=np.float32)
    pts = np.array([t * direction for t in (-2, -1, 0, 1, 2)], dtype=np.float32)
    viewer_client.add_polyline("pickline", pts, color=0xFF8800, line_width=6)

    # Wait until the browser has fetched + created the polyline.
    deadline = time.time() + 5
    while time.time() < deadline:
        if "pickline" in viewer_client.query_scene()["objects"]:
            break
        time.sleep(0.05)
    else:
        pytest.fail("polyline was never created in the browser")

    # Frame the scene so the line is on-screen, then let a frame settle.
    viewer_page.evaluate("() => window.threejsViewer.resetView()")
    time.sleep(0.4)

    # Project the world midpoint (0,0,0) to client pixel coordinates using the
    # live camera matrices (manual mat4*vec4 — THREE isn't a global here).
    cx, cy = viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            const cam = v._camera;
            cam.updateMatrixWorld();
            cam.matrixWorldInverse.copy(cam.matrixWorld).invert();
            const view = cam.matrixWorldInverse.elements;
            const proj = cam.projectionMatrix.elements;
            const apply = (m, x, y, z, w) => [
                m[0]*x + m[4]*y + m[8]*z  + m[12]*w,
                m[1]*x + m[5]*y + m[9]*z  + m[13]*w,
                m[2]*x + m[6]*y + m[10]*z + m[14]*w,
                m[3]*x + m[7]*y + m[11]*z + m[15]*w,
            ];
            const e = apply(view, 0, 0, 0, 1);
            const c = apply(proj, e[0], e[1], e[2], e[3]);
            const ndcx = c[0] / c[3], ndcy = c[1] / c[3];
            const rect = v._renderer.domElement.getBoundingClientRect();
            return [
                rect.left + (ndcx * 0.5 + 0.5) * rect.width,
                rect.top + (-ndcy * 0.5 + 0.5) * rect.height,
            ];
        }"""
    )

    # Hover (shows the marker), then a stationary click (down+up, no drag).
    viewer_page.mouse.move(cx, cy)
    time.sleep(0.05)
    viewer_page.mouse.down()
    viewer_page.mouse.up()
    time.sleep(0.25)

    assert picks, "no polyline_pick was received from the browser"
    pick = picks[-1]
    assert pick["id"] == "pickline"
    assert pick["kind"] == "line", pick["kind"]
    assert 0.4 <= pick["fraction"] <= 0.6, pick["fraction"]
    px, py, pz = pick["point"]
    assert abs(px) < 0.25 and abs(py) < 0.25 and abs(pz) < 0.25, pick["point"]

    # The sphere the callback added from the receive thread must have landed.
    deadline = time.time() + 2
    while time.time() < deadline:
        if "hit" in viewer_client.query_scene()["objects"]:
            break
        time.sleep(0.05)
    else:
        pytest.fail("sphere added from the pick callback never appeared")


@pytest.mark.browser
def test_polyline_pick_disabled_by_default(viewer_client, viewer_page):
    """With picking never enabled, a click on a polyline sends nothing back."""
    picks = []
    # Watch for picks WITHOUT enabling picking in the viewer.
    viewer_client._pick_callbacks.append(lambda p: picks.append(p))

    pts = np.array([[-2, 0, 0], [0, 0, 0], [2, 0, 0]], dtype=np.float32)
    viewer_client.add_polyline("noline", pts, color=0x44AAFF, line_width=6)
    deadline = time.time() + 5
    while time.time() < deadline:
        if "noline" in viewer_client.query_scene()["objects"]:
            break
        time.sleep(0.05)
    viewer_page.evaluate("() => window.threejsViewer.resetView()")
    time.sleep(0.4)
    cx, cy = viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            const rect = v._renderer.domElement.getBoundingClientRect();
            return [rect.left + rect.width / 2, rect.top + rect.height / 2];
        }"""
    )
    viewer_page.mouse.move(cx, cy)
    viewer_page.mouse.down()
    viewer_page.mouse.up()
    time.sleep(0.25)
    assert picks == [], "picking should be inert until enabled"


@pytest.mark.browser
def test_polyline_pick_between_nodes_no_snapping(viewer_client, viewer_page):
    """Picking resolves a continuous point BETWEEN vertices — it must not snap
    to the nearest node. A single 2-point segment has no interior nodes, so any
    interior fraction proves sub-segment interpolation."""
    picks = []
    viewer_client.on_polyline_pick(lambda p: picks.append(p))

    a = np.array([-2.0, -1.2, 0.0])
    b = np.array([2.0, 1.2, 0.0])
    pts = np.array([a, b], dtype=np.float32)  # ONE long segment, no middle node
    viewer_client.add_polyline("seg", pts, color=0xFFAA00, line_width=6)
    deadline = time.time() + 5
    while time.time() < deadline:
        if "seg" in viewer_client.query_scene()["objects"]:
            break
        time.sleep(0.05)
    else:
        pytest.fail("polyline was never created")

    viewer_page.evaluate("() => window.threejsViewer.resetView()")
    time.sleep(0.4)

    # Aim the cursor at the world point 30% of the way along the segment.
    target = (a + 0.30 * (b - a)).tolist()
    cx, cy = viewer_page.evaluate(
        """(target) => {
            const v = window.threejsViewer;
            const cam = v._camera;
            cam.updateMatrixWorld();
            cam.matrixWorldInverse.copy(cam.matrixWorld).invert();
            const view = cam.matrixWorldInverse.elements;
            const proj = cam.projectionMatrix.elements;
            const apply = (m, x, y, z, w) => [
                m[0]*x + m[4]*y + m[8]*z  + m[12]*w,
                m[1]*x + m[5]*y + m[9]*z  + m[13]*w,
                m[2]*x + m[6]*y + m[10]*z + m[14]*w,
                m[3]*x + m[7]*y + m[11]*z + m[15]*w,
            ];
            const e = apply(view, target[0], target[1], target[2], 1);
            const c = apply(proj, e[0], e[1], e[2], e[3]);
            const ndcx = c[0] / c[3], ndcy = c[1] / c[3];
            const rect = v._renderer.domElement.getBoundingClientRect();
            return [
                rect.left + (ndcx * 0.5 + 0.5) * rect.width,
                rect.top + (-ndcy * 0.5 + 0.5) * rect.height,
            ];
        }""",
        target,
    )
    viewer_page.mouse.move(cx, cy)
    viewer_page.mouse.down()
    viewer_page.mouse.up()
    time.sleep(0.25)

    assert picks, "no pick received"
    pick = picks[-1]
    # Interior fraction (not snapped to 0.0 or 1.0), and the on-line point sits
    # at ~30% — i.e. the picker interpolated within the segment.
    assert 0.22 <= pick["fraction"] <= 0.38, pick["fraction"]
    assert pick["segment"] == 0
    px, py, pz = pick["point"]
    assert abs(px - target[0]) < 0.3 and abs(py - target[1]) < 0.3, pick["point"]
    # And it's genuinely between the endpoints, not on either node.
    assert abs(px - a[0]) > 0.3 and abs(px - b[0]) > 0.3, pick["point"]


# Project a world point to client pixel coordinates using the live camera
# matrices (manual mat4*vec4 — THREE isn't a global on the page).
_PROJECT_WORLD_TO_PIXELS = """(target) => {
    const v = window.threejsViewer;
    const cam = v._camera;
    cam.updateMatrixWorld();
    cam.matrixWorldInverse.copy(cam.matrixWorld).invert();
    const view = cam.matrixWorldInverse.elements;
    const proj = cam.projectionMatrix.elements;
    const apply = (m, x, y, z, w) => [
        m[0]*x + m[4]*y + m[8]*z  + m[12]*w,
        m[1]*x + m[5]*y + m[9]*z  + m[13]*w,
        m[2]*x + m[6]*y + m[10]*z + m[14]*w,
        m[3]*x + m[7]*y + m[11]*z + m[15]*w,
    ];
    const e = apply(view, target[0], target[1], target[2], 1);
    const c = apply(proj, e[0], e[1], e[2], e[3]);
    const ndcx = c[0] / c[3], ndcy = c[1] / c[3];
    const rect = v._renderer.domElement.getBoundingClientRect();
    return [
        rect.left + (ndcx * 0.5 + 0.5) * rect.width,
        rect.top + (-ndcy * 0.5 + 0.5) * rect.height,
    ];
}"""


@pytest.mark.browser
def test_parametric_tube_pick(viewer_client, viewer_page):
    """A click on a parametric tube (the bead) reports a pick with
    ``kind == "tube"``, resolved on the tube's full-resolution spine."""
    picks = []
    viewer_client.on_polyline_pick(lambda p: picks.append(p))

    # A straight bead along a diagonal, symmetric about the origin and evenly
    # sampled, so the geometric midpoint (0,0,0) sits at fraction 0.5.
    direction = np.array([1.0, 0.6, 0.4], dtype=np.float32)
    spine = np.array([t * direction for t in (-2, -1, 0, 1, 2)], dtype=np.float32)
    widths = np.full(len(spine), 0.5, dtype=np.float32)
    heights = np.full(len(spine), 0.5, dtype=np.float32)
    viewer_client.add_parametric_tube("bead", spine, widths, heights, color=0x44AAFF)

    deadline = time.time() + 5
    while time.time() < deadline:
        if "bead" in viewer_client.query_scene()["objects"]:
            break
        time.sleep(0.05)
    else:
        pytest.fail("parametric tube was never created in the browser")

    viewer_page.evaluate("() => window.threejsViewer.resetView()")
    time.sleep(0.4)

    # Aim at the bead's midpoint (0,0,0).
    cx, cy = viewer_page.evaluate(_PROJECT_WORLD_TO_PIXELS, [0.0, 0.0, 0.0])
    viewer_page.mouse.move(cx, cy)
    time.sleep(0.05)
    viewer_page.mouse.down()
    viewer_page.mouse.up()
    time.sleep(0.25)

    assert picks, "no pick was received from clicking the bead"
    pick = picks[-1]
    assert pick["id"] == "bead"
    assert pick["kind"] == "tube", pick["kind"]
    assert 0.4 <= pick["fraction"] <= 0.6, pick["fraction"]
    # The resolved point sits on the spine at ~the midpoint.
    px, py, pz = pick["point"]
    assert abs(px) < 0.4 and abs(py) < 0.4 and abs(pz) < 0.4, pick["point"]


@pytest.mark.browser
def test_polyline_pick_js_hook(viewer_client, viewer_page):
    """A client-side JS hook (``viewer.onPolylinePick`` / ``onPolylineHover``)
    receives picks directly in the browser — no Python round-trip — and
    auto-enables picking."""
    pts = np.array([[-2, 0, 0], [0, 0, 0], [2, 0, 0]], dtype=np.float32)
    viewer_client.add_polyline("jsline", pts, color=0x44AAFF, line_width=6)
    deadline = time.time() + 5
    while time.time() < deadline:
        if "jsline" in viewer_client.query_scene()["objects"]:
            break
        time.sleep(0.05)
    else:
        pytest.fail("polyline was never created in the browser")

    # Register hooks purely in the browser; this also enables picking (no
    # enable_polyline_picking() call from Python).
    viewer_page.evaluate(
        """() => {
            window.__jsPicks = [];
            window.__jsHovers = 0;
            window.threejsViewer.onPolylinePick(p => window.__jsPicks.push(p));
            window.threejsViewer.onPolylineHover(p => { if (p) window.__jsHovers++; });
        }"""
    )

    viewer_page.evaluate("() => window.threejsViewer.resetView()")
    time.sleep(0.4)

    cx, cy = viewer_page.evaluate(_PROJECT_WORLD_TO_PIXELS, [0.0, 0.0, 0.0])
    viewer_page.mouse.move(cx, cy)
    time.sleep(0.05)
    viewer_page.mouse.down()
    viewer_page.mouse.up()
    time.sleep(0.2)

    js_picks = viewer_page.evaluate("() => window.__jsPicks")
    js_hovers = viewer_page.evaluate("() => window.__jsHovers")
    assert js_picks, "JS pick hook never fired"
    pick = js_picks[-1]
    assert pick["id"] == "jsline"
    assert pick["kind"] == "line", pick["kind"]
    assert 0.4 <= pick["fraction"] <= 0.6, pick["fraction"]
    # Payload point is a plain {x, y, z} object for JS consumers.
    assert abs(pick["point"]["x"]) < 0.25, pick["point"]
    assert js_hovers > 0, "JS hover hook never fired on pointer move"


@pytest.mark.browser
def test_polyline_pick_pickable_false(viewer_client, viewer_page):
    """A polyline added with ``pickable=False`` is excluded from picking even
    when picking is enabled — a click on it sends nothing back, yet the object
    is still present and rendered (only its hit-testing is opted out)."""
    picks = []
    viewer_client.on_polyline_pick(lambda p: picks.append(p))

    pts = np.array([[-2, 0, 0], [0, 0, 0], [2, 0, 0]], dtype=np.float32)
    viewer_client.add_polyline(
        "optout", pts, color=0x44AAFF, line_width=6, pickable=False
    )
    deadline = time.time() + 5
    while time.time() < deadline:
        if "optout" in viewer_client.query_scene()["objects"]:
            break
        time.sleep(0.05)
    else:
        pytest.fail("polyline was never created in the browser")

    viewer_page.evaluate("() => window.threejsViewer.resetView()")
    time.sleep(0.4)

    cx, cy = viewer_page.evaluate(_PROJECT_WORLD_TO_PIXELS, [0.0, 0.0, 0.0])
    viewer_page.mouse.move(cx, cy)
    time.sleep(0.05)
    viewer_page.mouse.down()
    viewer_page.mouse.up()
    time.sleep(0.25)

    assert picks == [], "pickable=False object must be excluded from picking"
    assert "optout" in viewer_client.query_scene()["objects"]


@pytest.mark.browser
def test_parametric_tube_pickable_false(viewer_client, viewer_page):
    """A parametric tube added with ``pickable=False`` is likewise excluded —
    a click on the bead body sends nothing back."""
    picks = []
    viewer_client.on_polyline_pick(lambda p: picks.append(p))

    direction = np.array([1.0, 0.6, 0.4], dtype=np.float32)
    spine = np.array([t * direction for t in (-2, -1, 0, 1, 2)], dtype=np.float32)
    widths = np.full(len(spine), 0.5, dtype=np.float32)
    heights = np.full(len(spine), 0.5, dtype=np.float32)
    viewer_client.add_parametric_tube(
        "optoutbead", spine, widths, heights, color=0x44AAFF, pickable=False
    )
    deadline = time.time() + 5
    while time.time() < deadline:
        if "optoutbead" in viewer_client.query_scene()["objects"]:
            break
        time.sleep(0.05)
    else:
        pytest.fail("parametric tube was never created in the browser")

    viewer_page.evaluate("() => window.threejsViewer.resetView()")
    time.sleep(0.4)

    cx, cy = viewer_page.evaluate(_PROJECT_WORLD_TO_PIXELS, [0.0, 0.0, 0.0])
    viewer_page.mouse.move(cx, cy)
    time.sleep(0.05)
    viewer_page.mouse.down()
    viewer_page.mouse.up()
    time.sleep(0.25)

    assert picks == [], "pickable=False tube must be excluded from picking"
    assert "optoutbead" in viewer_client.query_scene()["objects"]


def _get_material_fog(page, obj_id):
    """Read the first material's `.fog` flag for an object by id, or None."""
    return page.evaluate(
        "(id) => {"
        " const o = window.threejsViewer._objects.get(id);"
        " if (!o) return null;"
        " let fog = null;"
        " o.traverse((c) => {"
        "  if (fog !== null || !c.material) return;"
        "  const m = Array.isArray(c.material) ? c.material[0] : c.material;"
        "  if (m) fog = m.fog;"
        " });"
        " return fog;"
        "}",
        obj_id,
    )


@pytest.mark.browser
def test_depth_cue_fog_scoped_to_polylines(viewer_client, viewer_page):
    """Distance fog must darken only polylines. `scene.fog` is global and every
    material defaults to `fog:true`, so without scoping the mesh would dim too.
    Assert the mesh material's `.fog` is forced off while fog is active (line
    on), then restored to its original value when fog is turned off."""
    viewer_client.add_box("fogbox")
    pts = np.array([[-2, 0, 0], [0, 1, 0], [2, 0, 0]], dtype=np.float32)
    viewer_client.add_polyline("fogline", pts, color=0x44AAFF, line_width=4)

    # Wait for the (binary-loaded) polyline to register.
    deadline = time.time() + 5
    while time.time() < deadline:
        if "fogline" in viewer_client.query_scene()["objects"]:
            break
        time.sleep(0.05)
    else:
        pytest.fail("polyline was never created in the browser")

    # Mesh materials default to fog enabled.
    assert _get_material_fog(viewer_page, "fogbox") is True

    viewer_client.set_depth_cue(fog=True)
    box_fog = line_fog = None
    for _ in range(40):
        time.sleep(0.05)
        box_fog = _get_material_fog(viewer_page, "fogbox")
        line_fog = _get_material_fog(viewer_page, "fogline")
        if box_fog is False and line_fog is True:
            break
    assert box_fog is False, (
        f"mesh fog should be forced off while fog active, got {box_fog!r}"
    )
    assert line_fog is True, (
        f"polyline fog should be on while fog active, got {line_fog!r}"
    )

    # Turning fog off restores the mesh material to its original fog value.
    viewer_client.set_depth_cue(fog=False)
    box_fog = None
    for _ in range(40):
        time.sleep(0.05)
        box_fog = _get_material_fog(viewer_page, "fogbox")
        if box_fog is True:
            break
    assert box_fog is True, (
        f"mesh fog should be restored after fog off, got {box_fog!r}"
    )


@pytest.mark.browser
def test_depth_cue_edl_depth_is_line_only(viewer_client, viewer_page):
    """Eye-dome lighting must sculpt only polylines. The EDL pass is fed a
    line-only depth texture (polylines are placed on a dedicated camera layer
    rendered alone in a depth pre-pass), with full-scene depth bound separately
    only for the occlusion guard. Assert the polyline carries the EDL layer, the
    mesh does not, and the EDL pass samples the line-only depth target."""
    viewer_client.add_box("edlbox")
    pts = np.array([[-2, 0, 0], [0, 1, 0], [2, 0, 0]], dtype=np.float32)
    viewer_client.add_polyline("edlline", pts, color=0x44AAFF, line_width=4)

    deadline = time.time() + 5
    while time.time() < deadline:
        if "edlline" in viewer_client.query_scene()["objects"]:
            break
        time.sleep(0.05)
    else:
        pytest.fail("polyline was never created in the browser")

    viewer_client.set_depth_cue(edl=True)

    state = None
    for _ in range(40):
        time.sleep(0.05)
        state = viewer_page.evaluate(
            "() => {"
            " const dc = window.threejsViewer._depthCue;"
            " const line = window.threejsViewer._objects.get('edlline');"
            " const box = window.threejsViewer._objects.get('edlbox');"
            " const LINE_BIT = 1 << 1;"  # EDL_LINE_LAYER = 1
            " return {"
            "  edlActive: dc.edlActive,"
            "  hasComposer: !!dc._edlPass,"
            "  lineOnEdlLayer: line ? ((line.layers.mask & LINE_BIT) !== 0) : null,"
            "  boxOnEdlLayer: box ? ((box.layers.mask & LINE_BIT) !== 0) : null,"
            "  tDepthIsLineOnly: (dc._edlPass && dc._lineDepthTarget)"
            "   ? (dc._edlPass.uniforms.tDepth.value === dc._lineDepthTarget.depthTexture) : null,"
            "  tSceneDepthBound: dc._edlPass ? (dc._edlPass.uniforms.tSceneDepth.value !== null) : null,"
            " };"
            "}"
        )
        if state and state["hasComposer"]:
            break
    assert state and state["edlActive"] is True
    assert state["lineOnEdlLayer"] is True, (
        "polyline must be on the EDL line-only layer"
    )
    assert state["boxOnEdlLayer"] is False, (
        "mesh must NOT be on the EDL line-only layer"
    )
    assert state["tDepthIsLineOnly"] is True, (
        "EDL pass must sample the line-only depth target"
    )
    assert state["tSceneDepthBound"] is True, (
        "EDL pass must bind full-scene depth for the occlusion guard"
    )


@pytest.mark.browser
def test_depth_cue_edl_preserves_background(viewer_client, viewer_page):
    """Enabling EDL must not change the background colour. The EffectComposer's
    OutputPass tone-maps everything it renders, which would darken a solid
    background (ACES toe: #222 -> #101). The fix renders the background
    transparent through the composer (NoBlending output pass over an alpha
    canvas) so the untone-mapped canvas CSS background-color shows instead,
    matching the direct render path. Assert the structural guarantees: the GL
    context has alpha, the canvas CSS background is the #222222 clear colour, and
    the composer's final pass replaces pixels (NoBlending) rather than blending
    a tone-mapped background over them."""
    pts = np.array([[-2, 0, 0], [0, 1, 0], [2, 0, 0]], dtype=np.float32)
    viewer_client.add_polyline("bgline", pts, color=0x44AAFF, line_width=4)

    deadline = time.time() + 5
    while time.time() < deadline:
        if "bgline" in viewer_client.query_scene()["objects"]:
            break
        time.sleep(0.05)
    else:
        pytest.fail("polyline was never created in the browser")

    viewer_client.set_depth_cue(edl=True)

    state = None
    for _ in range(40):
        time.sleep(0.05)
        state = viewer_page.evaluate(
            "() => {"
            " const v = window.threejsViewer;"
            " const dc = v._depthCue;"
            " const gl = v._renderer.getContext();"
            " const passes = (dc._composer && dc._composer.passes) || [];"
            " const out = passes[passes.length - 1];"
            " const NO_BLENDING = 0;"  # THREE.NoBlending
            " return {"
            "  hasComposer: !!dc._composer,"
            "  ctxAlpha: gl.getContextAttributes().alpha,"
            "  canvasBg: v._renderer.domElement.style.backgroundColor,"
            "  outNoBlend: out && out.material"
            "   ? (out.material.blending === NO_BLENDING) : null,"
            " };"
            "}"
        )
        if state and state["hasComposer"]:
            break
    assert state and state["hasComposer"], "composer never built after EDL on"
    assert state["ctxAlpha"] is True, (
        "renderer must use an alpha context so the canvas can be transparent"
    )
    assert state["canvasBg"] == "rgb(34, 34, 34)", (
        f"canvas CSS background must be the #222222 clear colour, got {state['canvasBg']!r}"
    )
    assert state["outNoBlend"] is True, (
        "composer output pass must use NoBlending so background pixels are "
        "replaced (transparent) rather than blended as a tone-mapped colour"
    )


@pytest.mark.browser
def test_depth_cue_fog_rescopes_on_shading_toggle(viewer_client, viewer_page):
    """The `M`/`N` shading-debug toggles swap a mesh's material (a shared
    MeshNormalMaterial) or add a wireframe-overlay child mesh — both default to
    `fog:true` and do NOT bump `_objGeneration`. With fog active the per-frame
    `update()` must re-scope on a wireframe/shading mode change, or those newly
    assigned/created materials dim under the global `scene.fog`, breaking the
    polyline-only promise. Assert the swapped debug material and the added
    wireframe overlay both end up fog-disabled while fog is active."""
    viewer_client.add_box("fognbox")
    pts = np.array([[-2, 0, 0], [0, 1, 0], [2, 0, 0]], dtype=np.float32)
    viewer_client.add_polyline("fognline", pts, color=0x44AAFF, line_width=4)

    deadline = time.time() + 5
    while time.time() < deadline:
        if "fognline" in viewer_client.query_scene()["objects"]:
            break
        time.sleep(0.05)
    else:
        pytest.fail("polyline was never created in the browser")

    viewer_client.set_depth_cue(fog=True)
    for _ in range(40):
        time.sleep(0.05)
        if _get_material_fog(viewer_page, "fognbox") is False:
            break

    # N -> shading mode 1 swaps in a shared MeshNormalMaterial (fog:true default).
    cur_mat_fog = (
        "() => {"
        " const o = window.threejsViewer._objects.get('fognbox');"
        " const m = Array.isArray(o.material) ? o.material[0] : o.material;"
        " return m ? m.fog : null;"
        "}"
    )
    _press_key(viewer_page, "KeyN")
    swapped_fog = None
    for _ in range(40):
        time.sleep(0.05)
        swapped_fog = viewer_page.evaluate(cur_mat_fog)
        if swapped_fog is False:
            break
    assert swapped_fog is False, (
        f"swapped shading-debug material must be fog-scoped off, got {swapped_fog!r}"
    )

    # Cycle N back to mode 0 (restore original), then M twice -> combined overlay.
    for _ in range(3):
        _press_key(viewer_page, "KeyN")
    _press_key(viewer_page, "KeyM")
    _press_key(viewer_page, "KeyM")
    overlay_fog = None
    for _ in range(40):
        time.sleep(0.05)
        overlay_fog = viewer_page.evaluate(
            "() => {"
            " const o = window.threejsViewer._objects.get('fognbox');"
            " const ov = o.userData.wireframeOverlay;"
            " return ov && ov.material ? ov.material.fog : null;"
            "}"
        )
        if overlay_fog is False:
            break
    assert overlay_fog is False, (
        f"wireframe overlay material must be fog-scoped off, got {overlay_fog!r}"
    )


# Move/rotate gizmo: top-down camera so a horizontal drag maps to world +X.
_GIZMO_TOPDOWN = """() => {
  const v = window.threejsViewer;
  v._camera.position.set(0,0,8); v._camera.up.set(0,1,0);
  v._controls.target.set(0,0,0); v._camera.lookAt(0,0,0);
  v._controls.update(); v._camera.updateMatrixWorld(true);
}"""

_GIZMO_PROJECT_ORIGIN = """() => {
  const v = window.threejsViewer;
  const w = v._renderer.domElement.clientWidth, h = v._renderer.domElement.clientHeight;
  const ndc = v._camera.position.clone().set(0,0,0).project(v._camera);
  return { x: (ndc.x*0.5+0.5)*w, y: (-ndc.y*0.5+0.5)*h };
}"""

# Browser viewer state lands asynchronously (WS round-trip from the Python client,
# plus a render-loop tick for things like camera-sync). Poll the actual condition
# instead of sleeping a fixed interval, which races under CPU contention.


def _wait_for(page, js_predicate, timeout=5000):
    """Wait until a JS predicate (an arrow-function string returning truthy)
    holds in the page. A throw inside the predicate (e.g. touching viewer state
    that isn't constructed yet) is treated as "not ready" so the poll keeps
    going, rather than failing the wait. Raises on timeout, so it doubles as an
    assertion."""
    guarded = f"() => {{ try {{ return ({js_predicate})(); }} catch (e) {{ return false; }} }}"
    page.wait_for_function(guarded, timeout=timeout)


def _wait_until(predicate, timeout=5.0, interval=0.02):
    """Poll a Python-side predicate until it returns truthy — for state delivered
    on the client's WS receive thread (e.g. move-callback dispatch). Returns True
    if it became truthy within `timeout`, else False (one last check is made at
    the deadline)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


@pytest.mark.browser
def test_move_gizmo_attaches_and_reports(viewer_client, viewer_page):
    """enable_move_gizmo(id) attaches the gizmo; dragging the X arrow moves the
    object in +X and reports the new transform back to on_object_move."""
    moves = []
    viewer_client.on_object_move(moves.append)
    viewer_client.add_box("box")
    _wait_for(viewer_page, "() => window.threejsViewer._objects.has('box')")
    viewer_page.evaluate(_GIZMO_TOPDOWN)
    viewer_client.enable_move_gizmo("box")
    _wait_for(
        viewer_page,
        "() => { const g = window.threejsViewer._transformGizmo;"
        " return g.objectId === 'box' && g.helper.visible; }",
    )

    state = viewer_page.evaluate(
        "() => { const g = window.threejsViewer._transformGizmo;"
        " return { id: g.objectId, vis: g.helper.visible, mode: g.control.mode }; }"
    )
    assert state == {"id": "box", "vis": True, "mode": "translate"}

    # Re-assert the top-down camera right before dragging (so the projection is
    # current), then grab the centre screen-plane handle, which sits exactly at
    # the projected origin — at this camera it translates in world XY, so a
    # rightward drag is +X. Deterministic regardless of viewport size.
    viewer_page.evaluate(_GIZMO_TOPDOWN)
    proj = viewer_page.evaluate(_GIZMO_PROJECT_ORIGIN)
    cx, cy = proj["x"], proj["y"]
    x0 = viewer_page.evaluate(
        "() => window.threejsViewer._objects.get('box').position.x"
    )
    viewer_page.mouse.move(cx, cy)
    viewer_page.mouse.down()
    for i in range(1, 13):
        viewer_page.mouse.move(cx + i * 12, cy)
    viewer_page.mouse.up()
    assert _wait_until(lambda: bool(moves) and moves[-1]["phase"] == "end"), (
        "on_object_move never delivered an 'end' report"
    )
    x1 = viewer_page.evaluate(
        "() => window.threejsViewer._objects.get('box').position.x"
    )

    assert x1 > x0 + 0.1, f"box did not move in +X ({x0} -> {x1})"
    assert moves[-1]["id"] == "box"


@pytest.mark.browser
def test_move_gizmo_mode_switch_and_disable(viewer_client, viewer_page):
    """setGizmoMode swaps to rotate; disable_move_gizmo detaches and hides it."""
    viewer_client.add_box("box")
    _wait_for(viewer_page, "() => window.threejsViewer._objects.has('box')")
    viewer_client.enable_move_gizmo("box", mode="translate")
    _wait_for(
        viewer_page,
        "() => { const g = window.threejsViewer._transformGizmo;"
        " return g.objectId === 'box' && g.helper.visible; }",
    )
    viewer_page.evaluate("() => window.threejsViewer.setGizmoMode('rotate')")
    mode = viewer_page.evaluate(
        "() => window.threejsViewer._transformGizmo.control.mode"
    )
    assert mode == "rotate"

    viewer_client.disable_move_gizmo()
    _wait_for(
        viewer_page,
        "() => { const g = window.threejsViewer._transformGizmo;"
        " return !g.object && !g.helper.visible && !g.enabled; }",
    )
    st = viewer_page.evaluate(
        "() => { const g = window.threejsViewer._transformGizmo;"
        " return { hasObj: !!g.object, vis: g.helper.visible, enabled: g.enabled }; }"
    )
    assert st == {"hasObj": False, "vis": False, "enabled": False}


@pytest.mark.browser
def test_move_gizmo_click_to_select(viewer_client, viewer_page):
    """With click-select on, clicking an object attaches the gizmo to it."""
    viewer_client.add_box("box")
    _wait_for(viewer_page, "() => window.threejsViewer._objects.has('box')")
    viewer_page.evaluate(_GIZMO_TOPDOWN)
    viewer_client.enable_move_gizmo()  # no id → wait for a click
    _wait_for(viewer_page, "() => window.threejsViewer._transformGizmo.enabled")
    assert (
        viewer_page.evaluate("() => window.threejsViewer._transformGizmo.objectId")
        is None
    )

    proj = viewer_page.evaluate(_GIZMO_PROJECT_ORIGIN)
    # The gizmo isn't attached yet (no handles drawn), so a click on the box body
    # near screen-centre selects it. Small offset keeps it well within the box.
    viewer_page.mouse.click(proj["x"] - 15, proj["y"] + 15)
    _wait_for(
        viewer_page,
        "() => window.threejsViewer._transformGizmo.objectId === 'box'",
    )


@pytest.mark.browser
def test_move_gizmo_tracks_camera_switch(viewer_client, viewer_page):
    """The gizmo follows the active camera when the viewer switches persp↔ortho,
    so hit-testing/projection don't break (TransformControls keeps its own camera
    ref). Regression for the construction-time-camera bug."""
    viewer_client.add_box("box")
    _wait_for(viewer_page, "() => window.threejsViewer._objects.has('box')")
    viewer_client.enable_move_gizmo("box")
    _wait_for(
        viewer_page,
        "() => window.threejsViewer._transformGizmo.objectId === 'box'",
    )
    assert viewer_page.evaluate(
        "() => window.threejsViewer._transformGizmo.control.camera.isPerspectiveCamera === true"
    )
    viewer_page.evaluate("() => window.threejsViewer._switchCamera(true)")  # → ortho
    # control.camera is synced in the render-loop update(), a frame or two later.
    _wait_for(
        viewer_page,
        "() => { const v = window.threejsViewer;"
        " return v._transformGizmo.control.camera === v._camera"
        " && v._camera.isOrthographicCamera === true; }",
    )


@pytest.mark.browser
def test_attach_move_gizmo_reaches_untracked_object(viewer_client, viewer_page):
    """attachMoveGizmo attaches the gizmo to a bare Object3D the viewer never
    tracked in _objects (the embedder's sentinel case) and auto-enables the
    controller — enableMoveGizmo({id}) can only reach _objects members."""
    state = viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            const g = v._transformGizmo;
            const wasEnabled = g.enabled;   // never enabled in this test → false
            // Reach the real Object3D class via the scene's prototype chain
            // (THREE is module-scoped, not exposed on window).
            const Object3D = Object.getPrototypeOf(Object.getPrototypeOf(v._scene)).constructor;
            const obj = new Object3D();
            obj.position.set(1, 2, 3);
            v._scene.add(obj);
            v.attachMoveGizmo(obj);
            return {
                wasEnabled,
                enabled: g.enabled,
                vis: g.helper.visible,
                isTarget: g.object === obj,
                objectId: g.objectId,
                tracked: [...v._objects.values()].includes(obj),
            };
        }"""
    )
    assert state == {
        "wasEnabled": False,
        "enabled": True,  # attach auto-activated the controller
        "vis": True,
        "isTarget": True,
        "objectId": None,  # not in _objects → reverse lookup is null
        "tracked": False,
    }


@pytest.mark.browser
def test_move_gizmo_alt_is_momentary(viewer_client, viewer_page):
    """Alt is a momentary rotate override: from a translate base it switches to
    rotate while held and back on release; from a caller-set rotate base an Alt
    tap leaves the base untouched (regression — Alt release used to hard-reset to
    translate, clobbering setGizmoMode('rotate'))."""
    viewer_client.add_box("box")
    _wait_for(viewer_page, "() => window.threejsViewer._objects.has('box')")
    viewer_client.enable_move_gizmo("box")  # base mode = translate
    _wait_for(
        viewer_page,
        "() => { const g = window.threejsViewer._transformGizmo;"
        " return g.enabled && g.objectId === 'box'; }",
    )

    # Dispatch an Alt keydown/keyup (with altKey set) to the gizmo's window
    # listener and read back the effective control mode + the persistent base.
    alt = """(down) => {
        const el = window.threejsViewer.container;
        el.dispatchEvent(new KeyboardEvent(down ? 'keydown' : 'keyup', {
            key: 'Alt', code: 'AltLeft', altKey: down, bubbles: true }));
        const g = window.threejsViewer._transformGizmo;
        return { control: g.control.getMode(), base: g.mode };
    }"""

    # Translate base: Alt down → rotate, Alt up → translate (normal toggle intact).
    assert viewer_page.evaluate(alt, True) == {"control": "rotate", "base": "translate"}
    assert viewer_page.evaluate(alt, False) == {
        "control": "translate",
        "base": "translate",
    }

    # Caller sets a rotate base; an Alt tap must not clobber it back to translate.
    viewer_page.evaluate("() => window.threejsViewer.setGizmoMode('rotate')")
    assert viewer_page.evaluate(alt, True) == {"control": "rotate", "base": "rotate"}
    assert viewer_page.evaluate(alt, False) == {"control": "rotate", "base": "rotate"}


_GIZMO_AXES = (
    "() => { const c = window.threejsViewer._transformGizmo.control;"
    " return { x: c.showX, y: c.showY, z: c.showZ }; }"
)


@pytest.mark.browser
def test_set_gizmo_axes_constrains_and_resets_on_detach(viewer_client, viewer_page):
    """set_gizmo_axes drives TransformControls.showX/Y/Z over the wire; detaching
    the gizmo (disable) restores all axes so the next attach isn't constrained."""
    viewer_client.add_box("box")
    _wait_for(viewer_page, "() => window.threejsViewer._objects.has('box')")
    viewer_client.enable_move_gizmo("box")
    _wait_for(
        viewer_page,
        "() => window.threejsViewer._transformGizmo.objectId === 'box'",
    )

    viewer_client.set_gizmo_axes(x=False, y=False, z=True)
    _wait_for(
        viewer_page,
        "() => { const c = window.threejsViewer._transformGizmo.control;"
        " return c.showX === false && c.showY === false && c.showZ === true; }",
    )
    assert viewer_page.evaluate(_GIZMO_AXES) == {"x": False, "y": False, "z": True}

    viewer_client.disable_move_gizmo()
    _wait_for(
        viewer_page,
        "() => { const c = window.threejsViewer._transformGizmo.control;"
        " return c.showX && c.showY && c.showZ; }",
    )
    assert viewer_page.evaluate(_GIZMO_AXES) == {"x": True, "y": True, "z": True}


# Project the 'box' object's world position to screen pixels (its gizmo's centre
# handle sits there once attached). Like _GIZMO_PROJECT_ORIGIN but for the object.
_GIZMO_PROJECT_BOX = """() => {
  const v = window.threejsViewer;
  const w = v._renderer.domElement.clientWidth, h = v._renderer.domElement.clientHeight;
  const o = v._objects.get('box');
  o.updateMatrixWorld(true);
  const ndc = o.position.clone().setFromMatrixPosition(o.matrixWorld).project(v._camera);
  return { x: (ndc.x*0.5+0.5)*w, y: (-ndc.y*0.5+0.5)*h };
}"""


@pytest.mark.browser
def test_move_gizmo_relative_snap_steps_from_grab(viewer_client, viewer_page):
    """translate_snap_relative quantises the drag delta from the grab-time
    position, not an absolute world grid: a box starting at a non-grid x lands on
    start + k*step (preserving its off-grid offset), proving relative snapping."""
    start_x, step = 0.347, 0.1
    viewer_client.add_box("box")
    _wait_for(viewer_page, "() => window.threejsViewer._objects.has('box')")
    viewer_page.evaluate(_GIZMO_TOPDOWN)
    viewer_client.enable_move_gizmo(
        "box", translate_snap=step, translate_snap_relative=True
    )
    _wait_for(
        viewer_page,
        "() => { const g = window.threejsViewer._transformGizmo;"
        " return g.objectId === 'box' && g.helper.visible; }",
    )
    # Park the box at an off-grid x, then grab its (now off-origin) centre handle.
    viewer_page.evaluate(
        f"() => window.threejsViewer._objects.get('box').position.set({start_x}, 0, 0)"
    )
    viewer_page.evaluate(_GIZMO_TOPDOWN)
    proj = viewer_page.evaluate(_GIZMO_PROJECT_BOX)
    cx, cy = proj["x"], proj["y"]
    viewer_page.mouse.move(cx, cy)
    viewer_page.mouse.down()
    for i in range(1, 13):
        viewer_page.mouse.move(cx + i * 12, cy)
    viewer_page.mouse.up()
    assert _wait_until(
        lambda: viewer_page.evaluate(
            "() => !window.threejsViewer._transformGizmo.control.dragging"
        )
    )
    x1 = viewer_page.evaluate(
        "() => window.threejsViewer._objects.get('box').position.x"
    )
    steps = round((x1 - start_x) / step)
    assert steps >= 1, f"box did not move in +X ({start_x} -> {x1})"
    # Lands exactly on a relative step (offset 0.047 preserved); absolute snapping
    # would instead land on a multiple of 0.1, ~0.047 away from this.
    assert abs(x1 - (start_x + steps * step)) < 1e-6, (
        f"x1={x1} is not start+{steps}*{step}; relative snap not applied"
    )


@pytest.mark.browser
def test_move_gizmo_object_change_hook_runs_before_report(viewer_client, viewer_page):
    """onObjectChange fires per drag-frame before the report is sampled, and a
    mutation it makes is reflected in the onObjectMove payload (ordering: snap →
    change hooks → report). Also asserts positionStart is carried in the report."""
    viewer_client.add_box("box")
    _wait_for(viewer_page, "() => window.threejsViewer._objects.has('box')")
    viewer_page.evaluate(_GIZMO_TOPDOWN)
    viewer_client.enable_move_gizmo("box")
    _wait_for(
        viewer_page,
        "() => { const g = window.threejsViewer._transformGizmo;"
        " return g.objectId === 'box' && g.helper.visible; }",
    )
    # A change hook that forces y=5 each frame, plus a move-report collector.
    viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            window.__chg = { n: 0, lastId: null };
            window.__moves = [];
            v.onObjectChange(p => { window.__chg.n++; window.__chg.lastId = p.id;
                                    p.object3D.position.y = 5; });
            v.onObjectMove(m => { window.__moves.push(m); });
        }"""
    )
    viewer_page.evaluate(_GIZMO_TOPDOWN)
    proj = viewer_page.evaluate(_GIZMO_PROJECT_BOX)
    cx, cy = proj["x"], proj["y"]
    viewer_page.mouse.move(cx, cy)
    viewer_page.mouse.down()
    for i in range(1, 13):
        viewer_page.mouse.move(cx + i * 12, cy)
    viewer_page.mouse.up()
    assert _wait_until(
        lambda: viewer_page.evaluate(
            "() => (window.__moves || []).some(m => m.phase === 'end')"
        )
    )
    state = viewer_page.evaluate(
        """() => {
            const moves = window.__moves.filter(m => m.phase === 'move');
            const end = window.__moves.filter(m => m.phase === 'end').at(-1);
            return {
                n: window.__chg.n,
                lastId: window.__chg.lastId,
                // A 'move' report is sampled inside _onObjectChange, AFTER the hook
                // runs that frame — so it pins the hook-runs-before-report contract
                // (the 'end' report is fired separately and only reads the carried
                // pose). Require at least one and that its y is the hook's mutation.
                moveCount: moves.length,
                moveY: moves.length ? moves.at(-1).position[1] : null,
                endY: end.position[1],
                startLen: (end.positionStart || []).length,
                quatStartLen: (end.quaternionStart || []).length,
            };
        }"""
    )
    assert state["n"] >= 1, "onObjectChange never fired"
    assert state["lastId"] == "box"
    # The hook mutated y before the report sampled it (verified on the move path,
    # which routes through _onObjectChange; end carries the last hooked pose too).
    assert state["moveCount"] >= 1, "no mid-drag 'move' report was sampled"
    assert state["moveY"] == 5
    assert state["endY"] == 5
    assert state["startLen"] == 3 and state["quatStartLen"] == 4


# Locate a pinned (add_gizmo) gizmo handle by name and return its screen-pixel
# centre, so a drag can grab the actual arrow / plane chip (not just the gizmo
# origin). Searches only the *translate* gizmo group (so the huge "infinite axis"
# helper lines and the other-mode handles are out of scope) and picks the
# matching handle whose geometry sits furthest from the gizmo centre — for an
# arrow that's a point on a cone, for a plane chip there's only the one. Reads
# the visible handle; the picker shares its geometry, so the hover hit lands.
_GIZMO_HANDLE_PX = """(args) => {
  const [idx, name] = args;
  const v = window.threejsViewer;
  const g = v._transformGizmo._extra[idx];
  if (!g) return null;
  g.helper.updateMatrixWorld(true);
  const V = v._camera.position.constructor;
  const group = g.control._gizmo.gizmo.translate;
  let best = null, bestLen = -1;
  group.traverse(o => {
    if (o.name !== name || o.tag === 'helper' || !o.geometry
        || !o.material || o.material.visible === false) return;
    o.geometry.computeBoundingBox();
    const c = o.geometry.boundingBox.getCenter(new V());
    const len = c.length();
    if (len > bestLen) { bestLen = len; best = o; }
  });
  if (!best) return null;
  best.geometry.computeBoundingBox();
  const c = best.geometry.boundingBox.getCenter(new V());
  c.applyMatrix4(best.matrixWorld);
  const w = v._renderer.domElement.clientWidth, h = v._renderer.domElement.clientHeight;
  c.project(v._camera);
  return { x: (c.x * 0.5 + 0.5) * w, y: (-c.y * 0.5 + 0.5) * h };
}"""

# Count / read the drag ghost (the translucent clone left at the start pose).
_GHOST_COUNT = (
    "() => { let n = 0; window.threejsViewer._scene.traverse("
    "o => { if (o.userData && o.userData.__gizmoGhost) n++; }); return n; }"
)
_GHOST_X = (
    "() => { let g = null; window.threejsViewer._scene.traverse("
    "o => { if (o.userData && o.userData.__gizmoGhost) g = o; });"
    " return g ? g.position.x : null; }"
)


def _box_pos(page, axis):
    return page.evaluate(
        f"() => window.threejsViewer._objects.get('box').position.{axis}"
    )


def _drag_handle(page, idx, name, dx, dy, steps=12):
    """Grab pinned-gizmo `idx`'s `name` handle and drag it by `steps` increments
    of (dx, dy) screen pixels, then wait for the drag to finish."""
    page.evaluate(_GIZMO_TOPDOWN)
    p = page.evaluate(_GIZMO_HANDLE_PX, [idx, name])
    assert p is not None, f"could not locate gizmo handle {name!r}"
    cx, cy = p["x"], p["y"]
    page.mouse.move(cx, cy)
    page.mouse.down()
    for i in range(1, steps + 1):
        page.mouse.move(cx + i * dx, cy + i * dy)
    page.mouse.up()
    assert _wait_until(
        lambda: page.evaluate(
            f"() => !window.threejsViewer._transformGizmo._extra[{idx}].control.dragging"
        )
    )


@pytest.mark.browser
def test_add_gizmo_multi_dof_and_plane_margin(viewer_client, viewer_page):
    """add_gizmo pins several gizmos at once, each with its own axis constraint
    (1-DOF rail / 2-DOF plane / 3-DOF free), and the plane chips are pushed out
    from the gizmo centre by the margin."""
    for name in ("rail", "tile", "cube"):
        viewer_client.add_box(name)
    _wait_for(
        viewer_page,
        "() => ['rail','tile','cube'].every(n => window.threejsViewer._objects.has(n))",
    )
    viewer_client.add_gizmo("rail", x=False, y=False, z=True)  # 1D
    viewer_client.add_gizmo("tile", x=True, y=True, z=False)  # 2D
    viewer_client.add_gizmo("cube")  # 3D
    _wait_for(
        viewer_page,
        "() => window.threejsViewer._transformGizmo._extra.length === 3",
    )

    axes = viewer_page.evaluate(
        "() => window.threejsViewer._transformGizmo._extra.map(g => ({"
        " id: g.id, x: g.control.showX, y: g.control.showY, z: g.control.showZ,"
        " vis: g.helper.visible }))"
    )
    assert axes == [
        {"id": "rail", "x": False, "y": False, "z": True, "vis": True},
        {"id": "tile", "x": True, "y": True, "z": False, "vis": True},
        {"id": "cube", "x": True, "y": True, "z": True, "vis": True},
    ]

    # The XY plane chip's geometry centroid: stock sits at ~0.21 from the gizmo
    # centre; the margin pushes it past 0.30 (scale alone keeps the centroid put).
    off = viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            const g = v._transformGizmo._extra[1];  // 'tile' XY-plane gizmo
            const V = v._camera.position.constructor;
            let chip = null;
            g.control._gizmo.gizmo.translate.traverse(o => {
                if (chip) return;
                if (o.name === 'XY' && o.geometry && o.material
                    && o.material.visible !== false) chip = o;
            });
            chip.geometry.computeBoundingBox();
            return chip.geometry.boundingBox.getCenter(new V()).length();
        }"""
    )
    assert off > 0.30, f"XY plane chip not pushed out by the margin (len={off})"


@pytest.mark.browser
def test_add_gizmo_space_and_refined_handles(viewer_client, viewer_page):
    """space='local' orients the handles to the object (TransformControls space),
    'world' (default) keeps them world-aligned; and the one-time handle refinement
    strips the bulky rotate handles (E / XYZE) and shades the translate cones."""
    viewer_client.add_box("w")
    viewer_client.add_box("l")
    _wait_for(
        viewer_page,
        "() => ['w','l'].every(n => window.threejsViewer._objects.has(n))",
    )
    viewer_client.add_gizmo("w")  # default → world
    viewer_client.add_gizmo("l", space="local")
    _wait_for(
        viewer_page,
        "() => window.threejsViewer._transformGizmo._extra.length === 2",
    )

    spaces = viewer_page.evaluate(
        "() => window.threejsViewer._transformGizmo._extra.map(g => g.control.space)"
    )
    assert spaces == ["world", "local"]

    refined = viewer_page.evaluate(
        """() => {
            const g = window.threejsViewer._transformGizmo._extra[0];
            const gm = g.control._gizmo;
            const rotNames = grp => grp.children.map(o => o.name);
            // Translate arrows (single-axis, coloured) are swapped to a lit material.
            let litArrows = 0, basicArrows = 0;
            gm.gizmo.translate.children.forEach(o => {
                if (!o.name || o.name.length !== 1) return;  // arrows only
                if (o.material && o.material.isMeshStandardMaterial) litArrows++;
                else if (o.material && o.material.isMeshBasicMaterial) basicArrows++;
            });
            return {
                gizmoRot: rotNames(gm.gizmo.rotate),
                pickerRot: rotNames(gm.picker.rotate),
                helperRot: rotNames(gm.helper.rotate),
                litArrows, basicArrows,
            };
        }"""
    )
    # The outer screen-space ring (E), the gray backdrop circle (XYZE), and the
    # gray AXIS helper line are gone; the three coloured rings remain.
    assert "E" not in refined["gizmoRot"] and "XYZE" not in refined["gizmoRot"]
    assert "E" not in refined["pickerRot"] and "XYZE" not in refined["pickerRot"]
    assert set(refined["gizmoRot"]) == {"X", "Y", "Z"}
    assert "AXIS" not in refined["helperRot"]
    # The cones are lit (shaded) now, not flat MeshBasicMaterial.
    assert refined["litArrows"] >= 3 and refined["basicArrows"] == 0


@pytest.mark.browser
def test_gizmo_arrow_drag_both_directions(viewer_client, viewer_page):
    """A 1-DOF (X-only) pinned gizmo: dragging the X arrow right moves the box in
    +X, dragging it left moves it back in -X (the arrow works from both sides)."""
    viewer_client.add_box("box", position=[0, 0, 0])
    _wait_for(viewer_page, "() => window.threejsViewer._objects.has('box')")
    viewer_page.evaluate(_GIZMO_TOPDOWN)
    viewer_client.add_gizmo("box", x=True, y=False, z=False)
    _wait_for(
        viewer_page,
        "() => { const g = window.threejsViewer._transformGizmo._extra[0];"
        " return g && g.helper.visible && g.control.showX"
        " && !g.control.showY && !g.control.showZ; }",
    )

    x0 = _box_pos(viewer_page, "x")
    _drag_handle(viewer_page, 0, "X", +14, 0)
    x1 = _box_pos(viewer_page, "x")
    assert x1 > x0 + 0.15, f"+X arrow drag did not increase x ({x0} -> {x1})"

    _drag_handle(viewer_page, 0, "X", -14, 0)
    x2 = _box_pos(viewer_page, "x")
    assert x2 < x1 - 0.15, f"-X arrow drag did not decrease x ({x1} -> {x2})"
    # Constrained to X: y stays put throughout.
    assert abs(_box_pos(viewer_page, "y")) < 1e-6


@pytest.mark.browser
def test_gizmo_plane_drag_both_directions(viewer_client, viewer_page):
    """A 2-DOF (XY) pinned gizmo: dragging the XY plane chip moves the box in both
    X and Y at once, and reverses cleanly when dragged the other way."""
    viewer_client.add_box("box", position=[0, 0, 0])
    _wait_for(viewer_page, "() => window.threejsViewer._objects.has('box')")
    viewer_page.evaluate(_GIZMO_TOPDOWN)
    viewer_client.add_gizmo("box", x=True, y=True, z=False)
    _wait_for(
        viewer_page,
        "() => { const g = window.threejsViewer._transformGizmo._extra[0];"
        " return g && g.helper.visible && g.control.showX"
        " && g.control.showY && !g.control.showZ; }",
    )

    x0, y0 = _box_pos(viewer_page, "x"), _box_pos(viewer_page, "y")
    # Top-down: screen right = +X, screen up (dy<0) = +Y.
    _drag_handle(viewer_page, 0, "XY", +11, -11, steps=10)
    x1, y1 = _box_pos(viewer_page, "x"), _box_pos(viewer_page, "y")
    assert x1 > x0 + 0.15 and y1 > y0 + 0.15, (
        f"plane +drag did not move both axes ({x0},{y0} -> {x1},{y1})"
    )

    _drag_handle(viewer_page, 0, "XY", -11, +11, steps=10)
    x2, y2 = _box_pos(viewer_page, "x"), _box_pos(viewer_page, "y")
    assert x2 < x1 - 0.15 and y2 < y1 - 0.15, (
        f"plane -drag did not reverse both axes ({x1},{y1} -> {x2},{y2})"
    )


@pytest.mark.browser
def test_gizmo_drag_ghost_present_until_release(viewer_client, viewer_page):
    """A translucent ghost is dropped at the grab-time pose while dragging and
    removed on release; it stays at the original location as the box moves."""
    viewer_client.add_box("box", position=[0, 0, 0])
    _wait_for(viewer_page, "() => window.threejsViewer._objects.has('box')")
    viewer_page.evaluate(_GIZMO_TOPDOWN)
    viewer_client.add_gizmo("box", x=True, y=False, z=False)
    _wait_for(
        viewer_page,
        "() => { const g = window.threejsViewer._transformGizmo._extra[0];"
        " return g && g.helper.visible; }",
    )

    assert viewer_page.evaluate(_GHOST_COUNT) == 0  # none before a drag

    viewer_page.evaluate(_GIZMO_TOPDOWN)
    p = viewer_page.evaluate(_GIZMO_HANDLE_PX, [0, "X"])
    cx, cy = p["x"], p["y"]
    viewer_page.mouse.move(cx, cy)
    viewer_page.mouse.down()
    viewer_page.mouse.move(cx + 30, cy)
    viewer_page.mouse.move(cx + 60, cy)

    # Mid-drag: exactly one ghost, frozen at the start x≈0 while the box moved off.
    assert viewer_page.evaluate(_GHOST_COUNT) == 1
    assert abs(viewer_page.evaluate(_GHOST_X)) < 1e-3, "ghost drifted from start"
    assert _box_pos(viewer_page, "x") > 0.15, "box did not move during drag"

    viewer_page.mouse.up()
    assert _wait_until(lambda: viewer_page.evaluate(_GHOST_COUNT) == 0), (
        "ghost was not removed on release"
    )


@pytest.mark.browser
def test_gizmo_drag_ghost_survives_circular_userdata(viewer_client, viewer_page):
    """Regression: many objects stash circular / class-instance refs in userData
    (e.g. a tube's userData.parametricTube points back at its mesh). Object3D
    clone() deep-copies userData via JSON.stringify, which would throw on those —
    _spawnGhost blanks userData across the subtree while cloning, so the ghost
    still spawns and the drag isn't broken."""
    viewer_client.add_box("box", position=[0, 0, 0])
    _wait_for(viewer_page, "() => window.threejsViewer._objects.has('box')")
    viewer_page.evaluate(_GIZMO_TOPDOWN)
    viewer_client.add_gizmo("box", x=True, y=False, z=False)
    _wait_for(
        viewer_page,
        "() => { const g = window.threejsViewer._transformGizmo._extra[0];"
        " return g && g.helper.visible; }",
    )
    # Plant a circular ref + a fake class-instance back-ref on the box's userData,
    # exactly the shape that makes JSON.stringify(userData) throw.
    viewer_page.evaluate(
        """() => {
            const o = window.threejsViewer._objects.get('box');
            o.userData.self = o;                       // direct cycle
            o.userData.fake = { mesh: o, big: new Float32Array(8) };
        }"""
    )
    viewer_page.evaluate(_GIZMO_TOPDOWN)
    p = viewer_page.evaluate(_GIZMO_HANDLE_PX, [0, "X"])
    cx, cy = p["x"], p["y"]
    viewer_page.mouse.move(cx, cy)
    viewer_page.mouse.down()
    viewer_page.mouse.move(cx + 30, cy)
    viewer_page.mouse.move(cx + 60, cy)
    # Clone didn't throw → a ghost exists; the drag still moved the box; the
    # source userData is restored intact (the cycle survives the round-trip).
    assert viewer_page.evaluate(_GHOST_COUNT) == 1, "ghost did not spawn"
    assert _box_pos(viewer_page, "x") > 0.15, "drag broke (box did not move)"
    assert viewer_page.evaluate(
        "() => { const o = window.threejsViewer._objects.get('box');"
        " return o.userData.self === o && o.userData.fake.mesh === o; }"
    ), "source userData was not restored after cloning"
    viewer_page.mouse.up()
    assert _wait_until(lambda: viewer_page.evaluate(_GHOST_COUNT) == 0)


@pytest.mark.browser
def test_gizmo_snap_default_inverts_shift(viewer_client, viewer_page):
    """snap_default=True makes snap the resting state: a plain drag lands on the
    grid, and holding Shift releases snap for free movement (the inverse of the
    default free / Shift-to-snap)."""
    viewer_client.add_box("box", position=[0, 0, 0])
    _wait_for(viewer_page, "() => window.threejsViewer._objects.has('box')")
    viewer_page.evaluate(_GIZMO_TOPDOWN)
    viewer_client.enable_move_gizmo(translate_snap=0.5, click_select=False)
    viewer_client.add_gizmo("box", x=True, y=False, z=False, snap_default=True)
    _wait_for(
        viewer_page,
        "() => { const g = window.threejsViewer._transformGizmo._extra[0];"
        " return g && g.helper.visible && g.snapDefault === true; }",
    )

    snap_live = (
        "() => window.threejsViewer._transformGizmo._extra[0].control.translationSnap"
    )

    # Plain drag (no modifier): snap is engaged and the box lands on the 0.5 grid.
    viewer_page.evaluate(_GIZMO_TOPDOWN)
    p = viewer_page.evaluate(_GIZMO_HANDLE_PX, [0, "X"])
    cx, cy = p["x"], p["y"]
    viewer_page.mouse.move(cx, cy)
    viewer_page.mouse.down()
    assert viewer_page.evaluate(snap_live) == 0.5, "snap not engaged by default"
    for i in range(1, 15):
        viewer_page.mouse.move(cx + i * 16, cy)
    viewer_page.mouse.up()
    assert _wait_until(
        lambda: viewer_page.evaluate(
            "() => !window.threejsViewer._transformGizmo._extra[0].control.dragging"
        )
    )
    x = _box_pos(viewer_page, "x")
    assert x > 0.4, f"box did not move ({x})"
    assert abs(round(x / 0.5) * 0.5 - x) < 1e-6, (
        f"default drag not snapped to grid: {x}"
    )

    # Now hold Shift while dragging → snap released → free (snap state is null).
    viewer_page.evaluate(
        "() => window.threejsViewer._objects.get('box').position.set(0, 0, 0)"
    )
    viewer_page.evaluate(_GIZMO_TOPDOWN)
    p = viewer_page.evaluate(_GIZMO_HANDLE_PX, [0, "X"])
    cx, cy = p["x"], p["y"]
    viewer_page.keyboard.down("Shift")
    try:
        viewer_page.mouse.move(cx, cy)
        viewer_page.mouse.down()
        assert viewer_page.evaluate(snap_live) is None, (
            "Shift did not release snap on a snap_default gizmo"
        )
        for i in range(1, 15):
            viewer_page.mouse.move(cx + i * 16, cy)
        viewer_page.mouse.up()
    finally:
        viewer_page.keyboard.up("Shift")


def _read_persp_fov(page):
    """Read the live perspective camera's vertical FOV (degrees), or None."""
    return page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " return v && v._perspCamera ? v._perspCamera.fov : null;"
        "}"
    )


@pytest.mark.browser
def test_fov_defaults_to_40(viewer_client, page):
    """With no `fov` query param the perspective camera uses the 40° default."""
    viewer_path = viewer_client.viewer_path.resolve()
    page.goto(f"file://{viewer_path}?ws_port={viewer_client.port}")
    page.wait_for_function(
        "() => window.threejsViewer && window.threejsViewer._perspCamera"
    )
    assert _read_persp_fov(page) == 40


@pytest.mark.browser
def test_fov_url_param_overrides_default(viewer_client, page):
    """A `fov` query param sets the perspective camera's FOV at construction."""
    viewer_path = viewer_client.viewer_path.resolve()
    page.goto(f"file://{viewer_path}?ws_port={viewer_client.port}&fov=28")
    page.wait_for_function(
        "() => window.threejsViewer && window.threejsViewer._perspCamera"
    )
    assert _read_persp_fov(page) == 28


@pytest.mark.browser
@pytest.mark.parametrize("raw", ["500", "Infinity", "-5"])
def test_fov_url_param_clamped_to_range(viewer_client, page, raw):
    """Out-of-range `fov` params — including ±Infinity — are clamped (not thrown)."""
    viewer_path = viewer_client.viewer_path.resolve()
    page.goto(f"file://{viewer_path}?ws_port={viewer_client.port}&fov={raw}")
    page.wait_for_function(
        "() => window.threejsViewer && window.threejsViewer._perspCamera"
    )
    assert _read_persp_fov(page) == (1 if raw == "-5" else 179)


@pytest.mark.browser
def test_clip_tool_has_refined_rotate_and_slide_gizmos(viewer_client, viewer_page):
    """Enabling the clip tool brings up both gizmos (rotate + normal-slide), and
    disabling it puts them away."""
    viewer_client.add_sphere("s", radius=1.5)
    time.sleep(0.2)
    viewer_client.set_clipping_plane(
        normal=[0.6, 0.2, 0.78], distance=0.0, show_helper=True
    )
    state = None
    for _ in range(40):
        time.sleep(0.05)
        state = viewer_page.evaluate(
            "() => {"
            " const v = window.threejsViewer;"
            " return {"
            "  rotEnabled: v._clipGizmo.enabled,"
            "  rotMode: v._clipGizmo.getMode(),"
            "  rotVisible: v._clipGizmoHelper.visible,"
            "  moveEnabled: v._clipMoveGizmo.enabled,"
            "  moveMode: v._clipMoveGizmo.getMode(),"
            "  moveVisible: v._clipMoveGizmoHelper.visible,"
            "  moveSpace: v._clipMoveGizmo.space,"
            "  moveShowX: v._clipMoveGizmo.showX,"
            "  moveShowZ: v._clipMoveGizmo.showZ,"
            " };"
            "}"
        )
        if state and state["moveEnabled"]:
            break
    assert state is not None
    assert state["rotEnabled"] and state["rotMode"] == "rotate" and state["rotVisible"]
    # The plane-slide gizmo is a local-space, Z-only (normal) translate handle.
    assert (
        state["moveEnabled"]
        and state["moveMode"] == "translate"
        and state["moveVisible"]
    )
    assert state["moveSpace"] == "local"
    assert state["moveShowX"] is False and state["moveShowZ"] is True

    # Disabling the clip tool puts both gizmos away.
    viewer_client.disable_clipping_plane()
    off = None
    for _ in range(40):
        time.sleep(0.05)
        off = viewer_page.evaluate(
            "() => ({rot: window.threejsViewer._clipGizmo.enabled,"
            " move: window.threejsViewer._clipMoveGizmo.enabled})"
        )
        if off and not off["move"]:
            break
    assert off is not None and not off["rot"] and not off["move"]


@pytest.mark.browser
def test_binary_draw_ranges_channel_on_swept_tool_and_points(
    viewer_client, viewer_page
):
    """The binary `draw_ranges` animation channel (set_draw_range_data) drives
    draw range on a swept tool AND a point cloud. This is a DIFFERENT code path
    (makeChannelApply.draw_ranges) from the `set_draw_range` message
    (_setDrawRange); a regression in the channel applier would otherwise pass
    every other test while breaking the example reveals."""
    n = 30
    t = np.linspace(0, 1, n)
    positions = np.column_stack([t * 6 - 3, 0 * t, 0 * t]).astype(np.float32)
    axes = np.tile([0, 0, 1.0], (n, 1)).astype(np.float32)
    profile = np.array([[0, 0.4], [4.0, 0.4]], dtype=np.float32)
    viewer_client.add_swept_tool("shank", positions, axes, profile)
    pts = np.random.default_rng(0).random((400, 3)).astype(np.float32)
    viewer_client.add_points("cloud", pts)
    time.sleep(0.4)  # let the binary HTTP loads land

    n_frames = 11
    anim = Animation(loop=False)
    anim.set_frame_times(np.linspace(0, 1.0, n_frames, dtype=np.float32))
    # One channel covering both ids; values ramp 0 -> 1 so t=0.5 -> ~0.5.
    ramp = np.tile(
        np.linspace(0, 1, n_frames, dtype=np.float32).reshape(n_frames, 1), (1, 2)
    )
    anim.set_draw_range_data(["shank", "cloud"], ramp)
    viewer_client.load_animation(anim, autoplay=False)
    loaded = False
    for _ in range(40):
        time.sleep(0.05)
        if viewer_page.evaluate("() => window.threejsViewer._animation != null"):
            loaded = True
            break
    assert loaded, "animation never loaded"
    # Seek to mid-animation; the binary channel applier must set both draw ranges.
    viewer_page.evaluate("() => window.threejsViewer._seekToTime(0.5)")
    got = None
    for _ in range(40):
        time.sleep(0.05)
        objs = viewer_client.query_scene()["objects"]
        shank = objs.get("shank", {}).get("drawRange")
        cloud = objs.get("cloud", {}).get("drawRange")
        if shank is not None and cloud is not None:
            got = (shank, cloud)
            if abs(shank - 0.5) < 0.1 and abs(cloud - 0.5) < 0.1:
                break
    assert got is not None, "objects never appeared"
    assert abs(got[0] - 0.5) < 0.1, (
        f"swept tool draw range did not advance via channel: {got[0]}"
    )
    assert abs(got[1] - 0.5) < 0.1, (
        f"point cloud draw range did not advance via channel: {got[1]}"
    )


@pytest.mark.browser
def test_flat_black_color_is_honored_not_falsy_substituted(viewer_client, viewer_page):
    """color=0x000000 (no vertex colors) must render black, not the default —
    guards the `data.color ?? default` (vs `||`) falsy-zero fix on points and
    the swept tool."""
    viewer_client.add_points(
        "blackpts", np.zeros((10, 3), dtype=np.float32), color=0x000000
    )
    n = 6
    positions = np.column_stack(
        [np.linspace(0, 3, n), np.zeros(n), np.zeros(n)]
    ).astype(np.float32)
    axes = np.tile([0, 0, 1.0], (n, 1)).astype(np.float32)
    profile = np.array([[0, 0.3], [2.0, 0.3]], dtype=np.float32)
    viewer_client.add_swept_tool("blacktool", positions, axes, profile, color=0x000000)
    got = None
    for _ in range(40):
        time.sleep(0.05)
        got = viewer_page.evaluate(
            "() => {"
            " const o = window.threejsViewer._objects;"
            " const p = o.get('blackpts'), t = o.get('blacktool');"
            " return p && t ? {pts: p.material.color.getHex(), tool: t.material.color.getHex()} : null;"
            "}"
        )
        if got:
            break
    assert got is not None, "objects never landed"
    assert got["pts"] == 0x000000, f"black point cloud rendered {got['pts']:#08x}"
    assert got["tool"] == 0x000000, f"black swept tool rendered {got['tool']:#08x}"


@pytest.mark.browser
def test_camera_set_get_roundtrip_and_orientation(viewer_client, viewer_page):
    """set_camera moves AND re-orients the camera (ViewerControls never calls
    lookAt itself — regression: position/target changed but the camera kept
    facing its old direction); get_camera reads the same pose back."""
    viewer_client.add_box("b")
    time.sleep(0.1)
    viewer_client.set_camera(position=[5.0, -5.0, 4.0], target=[0.5, 0.25, 0.0], fov=45)
    time.sleep(0.3)

    cam = viewer_client.get_camera()
    assert cam["position"] == pytest.approx([5.0, -5.0, 4.0], abs=1e-6)
    assert cam["target"] == pytest.approx([0.5, 0.25, 0.0], abs=1e-6)
    assert cam["fov"] == pytest.approx(45.0)

    # The view direction must point at the target (dot ~ 1), not wherever
    # the camera happened to face before.
    dot = viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " v._camera.updateMatrixWorld(true);"
        " const e = v._camera.matrixWorld.elements;"
        " const dir = [-e[8], -e[9], -e[10]];"
        " const p = v._camera.position, t = v._controls.target;"
        " const d = [t.x - p.x, t.y - p.y, t.z - p.z];"
        " const n = Math.hypot(d[0], d[1], d[2]);"
        " return (dir[0]*d[0] + dir[1]*d[1] + dir[2]*d[2]) / n;"
        "}"
    )
    assert dot > 0.999, f"camera not oriented at target (dot={dot})"


_VIEW_STATE_JS = (
    "() => {"
    " const v = window.threejsViewer;"
    " const p = v._camera.position, t = v._controls.target, u = v._camera.up;"
    " return { pos: [p.x, p.y, p.z], target: [t.x, t.y, t.z],"
    "          up: [u.x, u.y, u.z], tweening: !!v._viewTween };"
    "}"
)


@pytest.mark.browser
def test_set_view_top_reorients_camera(viewer_client, viewer_page):
    """set_view('top') puts the camera straight above the orbit target (+Z)
    with a +Y up vector, preserving the target and the orbit distance —
    reorient only, no framing. 'front' (animate=False) lands on -Y with
    +Z up immediately."""
    viewer_client.add_box("b")
    time.sleep(0.2)
    # Pin a known oblique pose first so distance preservation is checkable.
    viewer_client.set_camera(position=[6.0, -6.0, 3.0], target=[1.0, 2.0, 0.5])
    time.sleep(0.3)
    dist = math.sqrt(5.0**2 + 8.0**2 + 2.5**2)

    viewer_client.set_view("top")
    state = None
    for _ in range(80):  # WS delivery + ~450 ms tween
        time.sleep(0.05)
        state = viewer_page.evaluate(_VIEW_STATE_JS)
        if not state["tweening"] and abs(state["up"][1] - 1.0) < 1e-6:
            break
    assert state is not None and not state["tweening"], "view tween never finished"
    assert state["target"] == pytest.approx([1.0, 2.0, 0.5], abs=1e-6)
    assert state["pos"] == pytest.approx([1.0, 2.0, 0.5 + dist], abs=1e-4)
    assert state["up"] == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)

    viewer_client.set_view("front", animate=False)
    time.sleep(0.3)
    state = viewer_page.evaluate(_VIEW_STATE_JS)
    assert state["pos"] == pytest.approx([1.0, 2.0 - dist, 0.5], abs=1e-4)
    assert state["up"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)


@pytest.mark.browser
def test_set_view_cancels_drag_inertia(viewer_client, viewer_page):
    """Residual damped drag inertia (pending ViewerControls rot/pan deltas
    from a just-finished orbit) must not drift the camera off the preset:
    _animate() bleeds those deltas into the camera every frame, and pan
    inertia even moves the orbit target. setView() cancels them. Deltas are
    injected and setView called in one evaluate so the check is
    deterministic (no damping decay between the two)."""
    viewer_client.add_box("b")
    time.sleep(0.2)
    viewer_client.set_camera(position=[6.0, -6.0, 3.0], target=[1.0, 2.0, 0.5])
    time.sleep(0.3)
    dist = math.sqrt(5.0**2 + 8.0**2 + 2.5**2)

    viewer_page.evaluate(
        "() => {"
        " const c = window.threejsViewer._controls;"
        " c._rotDeltaTheta = 0.8; c._rotDeltaPhi = 0.4;"
        " c._panDeltaX = 3.0; c._panDeltaY = 2.0;"
        " window.threejsViewer.setView('top', { animate: false });"
        "}"
    )
    time.sleep(0.4)  # several frames of controls.update() — would drain inertia
    state = viewer_page.evaluate(_VIEW_STATE_JS)
    assert state["target"] == pytest.approx([1.0, 2.0, 0.5], abs=1e-6)
    assert state["pos"] == pytest.approx([1.0, 2.0, 0.5 + dist], abs=1e-4)
    assert state["up"] == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)
    deltas = viewer_page.evaluate(
        "() => { const c = window.threejsViewer._controls;"
        " return [c._rotDeltaTheta, c._rotDeltaPhi, c._panDeltaX, c._panDeltaY]; }"
    )
    assert deltas == [0, 0, 0, 0]


@pytest.mark.browser
def test_edl_auto_enables_on_points_and_pin_wins(viewer_client, viewer_page):
    """EDL switches on automatically when the first point cloud is added,
    but an explicit set_edl choice (including OFF) pins the state so the
    auto-enable never overrides it; strength/radius reach the shader."""
    pts = np.random.default_rng(0).random((500, 3)).astype(np.float32)
    viewer_client.add_points("pc", pts)
    active = None
    for _ in range(40):
        time.sleep(0.05)
        active = viewer_page.evaluate("() => window.threejsViewer._depthCue.edlActive")
        if active:
            break
    assert active is True, "EDL did not auto-enable on first point cloud"

    # Explicit OFF pins the state: a second cloud must not re-enable it.
    viewer_client.set_edl(False)
    time.sleep(0.2)
    viewer_client.add_points("pc2", pts + 2.0)
    time.sleep(0.4)
    state = viewer_page.evaluate(
        "() => ({active: window.threejsViewer._depthCue.edlActive,"
        "        pinned: window.threejsViewer._depthCue._edlUserSet})"
    )
    assert state == {"active": False, "pinned": True}, (
        f"auto-enable overrode the pinned OFF: {state}"
    )

    # Tuning params reach the live shader pass.
    viewer_client.set_edl(True, strength=77.0, radius=3.5)
    vals = None
    for _ in range(40):
        time.sleep(0.05)
        vals = viewer_page.evaluate(
            "() => {"
            " const dc = window.threejsViewer._depthCue;"
            " if (!dc._edlPass) return null;"
            " return [dc._edlPass.uniforms.edlStrength.value,"
            "         dc._edlPass.uniforms.edlRadius.value];"
            "}"
        )
        if vals == [77.0, 3.5]:
            break
    assert vals == [77.0, 3.5], f"EDL tuning did not reach the shader: {vals}"


@pytest.mark.browser
def test_follow_path_pose_scale_and_track_target(viewer_client, viewer_page):
    """set_follow_path drives the object's exact pose from the path at the
    animation playhead: position lerped, local +z onto the (nlerped) axis,
    and the object's own scale preserved (regression: composing with
    (1,1,1) silently un-scaled the tool). The followed object is also the
    preferred camera-track target."""
    viewer_client.add_box("fp_tool")
    for _ in range(40):
        time.sleep(0.05)
        if viewer_page.evaluate("() => window.threejsViewer._objects.has('fp_tool')"):
            break
    viewer_page.evaluate(
        "() => window.threejsViewer._objects.get('fp_tool').scale.set(2, 2, 2)"
    )
    viewer_client.set_follow_path(
        "fp_tool",
        times=[0.0, 2.0],
        positions=[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        axes=[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    )
    anim = Animation(
        frames=[Frame(time=0, transforms={}), Frame(time=2, transforms={})],
        loop=False,
    )
    viewer_client.load_animation(anim, autoplay=False, initial_time=1.0)
    _wait_for_animation_loaded(viewer_page)

    state = None
    for _ in range(40):
        time.sleep(0.05)
        state = viewer_page.evaluate(
            "() => {"
            " const v = window.threejsViewer;"
            " if (!v._followPaths || !v._followPaths.has('fp_tool')) return null;"
            " const e = v._objects.get('fp_tool').matrix.elements;"
            " return {pos: [e[12], e[13], e[14]],"
            "         zcol: [e[8], e[9], e[10]],"
            "         xlen: Math.hypot(e[0], e[1], e[2])};"
            "}"
        )
        if state is not None:
            break
    assert state is not None, "follow-path track never arrived in the browser"
    # position: halfway along the path at t=1 of [0, 2]
    assert state["pos"] == pytest.approx([2.0, 0.0, 0.0], abs=1e-5)
    # local +z: nlerp of [0,0,1] and [1,0,0] at w=0.5, scaled by 2
    zdir = np.array(state["zcol"]) / np.linalg.norm(state["zcol"])
    assert zdir == pytest.approx([2**-0.5, 0.0, 2**-0.5], abs=1e-5)
    # scale survives the per-tick matrix compose
    assert state["xlen"] == pytest.approx(2.0, abs=1e-5)
    assert np.linalg.norm(state["zcol"]) == pytest.approx(2.0, abs=1e-5)

    # The followed object is the preferred auto camera-track target.
    guess = viewer_page.evaluate("() => window.threejsViewer._guessTrackTarget()")
    assert guess == "fp_tool"


@pytest.mark.browser
def test_follow_path_float64_time_precision(viewer_client, viewer_page):
    """Keys 8 ms apart at t=160,000 s must interpolate, not collapse.
    float32's ulp at that magnitude is ~15.6 ms, so the old f32-packed
    times quantized both keys to the same value (dt=0 -> the tool held at
    the first key); the (K,) float64 time vector keeps them distinct and
    the pose lands halfway."""
    viewer_client.add_box("fp_prec")
    for _ in range(40):
        time.sleep(0.05)
        if viewer_page.evaluate("() => window.threejsViewer._objects.has('fp_prec')"):
            break
    t0, t1 = 160_000.0, 160_000.008
    viewer_client.set_follow_path(
        "fp_prec",
        times=[t0, t1],
        positions=[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        axes=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    )
    anim = Animation(
        frames=[Frame(time=t0, transforms={}), Frame(time=t1, transforms={})],
        loop=False,
    )
    viewer_client.load_animation(anim, autoplay=False, initial_time=(t0 + t1) / 2)
    _wait_for_animation_loaded(viewer_page)

    pos = None
    for _ in range(40):
        time.sleep(0.05)
        pos = viewer_page.evaluate(
            "() => {"
            " const v = window.threejsViewer;"
            " if (!v._followPaths || !v._followPaths.has('fp_prec')) return null;"
            " const e = v._objects.get('fp_prec').matrix.elements;"
            " return [e[12], e[13], e[14]];"
            "}"
        )
        if pos is not None:
            break
    assert pos is not None, "follow-path track never arrived in the browser"
    assert pos == pytest.approx([2.0, 0.0, 0.0], abs=1e-3)


@pytest.mark.browser
def test_follow_path_cleaned_up_on_delete_and_clear(viewer_client, viewer_page):
    """Follow-path tracks must not leak: delete_object drops that id's
    track, clear() empties the map (issue #85)."""
    viewer_client.add_box("fp_a")
    viewer_client.add_box("fp_b")
    time.sleep(0.1)
    path = dict(
        times=[0.0, 1.0],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        axes=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    )
    viewer_client.set_follow_path("fp_a", **path)
    viewer_client.set_follow_path("fp_b", **path)
    for _ in range(40):
        time.sleep(0.05)
        if viewer_page.evaluate("() => window.threejsViewer._followPaths.size") == 2:
            break
    assert viewer_page.evaluate("() => window.threejsViewer._followPaths.size") == 2

    viewer_client.delete("fp_a")
    time.sleep(0.2)
    assert viewer_page.evaluate(
        "() => [...window.threejsViewer._followPaths.keys()]"
    ) == ["fp_b"]

    viewer_client.clear()
    time.sleep(0.2)
    assert viewer_page.evaluate("() => window.threejsViewer._followPaths.size") == 0


@pytest.mark.browser
def test_gizmo_report_carries_effective_mode(viewer_client, viewer_page):
    """Every gizmo report carries the *effective* mode of the drag, read off
    the live control — so an Alt momentary rotate override is observable by
    consumers even though the base mode stays translate (issue #84: without
    the field, embedders branched on the base mode and silently discarded
    Alt rotate-drags)."""
    viewer_client.add_box("box", position=[0, 0, 0])
    _wait_for(viewer_page, "() => window.threejsViewer._objects.has('box')")
    viewer_page.evaluate(_GIZMO_TOPDOWN)
    viewer_client.add_gizmo("box", x=True, y=False, z=False)
    _wait_for(
        viewer_page,
        "() => { const g = window.threejsViewer._transformGizmo._extra[0];"
        " return g && g.helper.visible; }",
    )
    viewer_page.evaluate(
        "() => { window.__modes = [];"
        " window.threejsViewer.onObjectMove(p => window.__modes.push("
        "   [p.mode, p.phase])); }"
    )

    # A plain arrow drag: every report (throttled moves + final end) says
    # translate.
    _drag_handle(viewer_page, 0, "X", +10, 0)
    modes = viewer_page.evaluate("() => window.__modes")
    assert modes, "drag produced no reports"
    assert all(m == "translate" for m, _ in modes), modes
    assert modes[-1][1] == "end"

    # Hold Alt: the live control flips to rotate while the base mode stays
    # translate; a report issued during the override must say rotate.
    viewer_page.keyboard.down("Alt")
    _wait_for(
        viewer_page,
        "() => { const tg = window.threejsViewer._transformGizmo;"
        " const g = tg._extra[0];"
        " return g.control.getMode() === 'rotate' && g.mode === 'translate'; }",
    )
    viewer_page.evaluate(
        "() => { window.__modes = [];"
        " const tg = window.threejsViewer._transformGizmo;"
        " tg._report(tg._extra[0], true); }"
    )
    viewer_page.keyboard.up("Alt")
    modes = viewer_page.evaluate("() => window.__modes")
    assert modes == [["rotate", "end"]], (
        f"Alt override not visible in the report: {modes}"
    )


@pytest.mark.browser
def test_embedder_camera_pose_and_frame_box(viewer_client, viewer_page):
    """getCameraPose/setCameraPose round-trip from JS (both vector forms),
    re-orient the camera at the target, and frameBox fits a world AABB
    (issue #77)."""
    viewer_client.add_box("b")
    _wait_for(viewer_page, "() => window.threejsViewer._objects.has('b')")

    got = viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " v.setCameraPose({position: {x: 5, y: -5, z: 4},"
        "                  target: [0.5, 0.25, 0], fov: 45});"
        " const pose = v.getCameraPose();"
        " v._camera.updateMatrixWorld(true);"
        " const e = v._camera.matrixWorld.elements;"
        " const dir = [-e[8], -e[9], -e[10]];"
        " const p = v._camera.position, t = v._controls.target;"
        " const d = [t.x - p.x, t.y - p.y, t.z - p.z];"
        " const n = Math.hypot(d[0], d[1], d[2]);"
        " const dot = (dir[0]*d[0] + dir[1]*d[1] + dir[2]*d[2]) / n;"
        " return {pose, dot};"
        "}"
    )
    pose = got["pose"]
    assert [pose["position"][k] for k in "xyz"] == pytest.approx([5, -5, 4])
    assert [pose["target"][k] for k in "xyz"] == pytest.approx([0.5, 0.25, 0])
    assert pose["fov"] == pytest.approx(45.0)
    assert got["dot"] > 0.999, f"camera not oriented at target (dot={got['dot']})"

    target = viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " v.frameBox([10, 10, 10], {x: 14, y: 14, z: 14});"
        " const t = v._controls.target;"
        " return [t.x, t.y, t.z];"
        "}"
    )
    assert target == pytest.approx([12.0, 12.0, 12.0], abs=1e-4)

    # invalid input degrades to a warning, never throws
    assert viewer_page.evaluate(
        "() => { window.threejsViewer.frameBox(null, [1,2,3]); return true; }"
    )


@pytest.mark.browser
def test_set_points_lod_options_runtime_tuning(viewer_client, viewer_page):
    """set_points_lod_options re-tunes a streamed cloud's traversal live —
    no re-upload (issue #87): budget/refine_pixels land on the runtime
    state the per-frame traversal reads, and size_boost_max re-derives the
    point size on already-streamed node materials."""
    rng = np.random.default_rng(11)
    n = 40_000
    pts = (rng.random((n, 3)) * [8, 3, 1.5]).astype(np.float32)
    viewer_client.add_points(
        "cloud",
        pts,
        size=2.0,
        lod={"node_capacity": 4000, "point_budget": 30_000, "refine_pixels": 2},
    )
    state = None
    for _ in range(100):
        time.sleep(0.1)
        state = viewer_page.evaluate(
            "() => {"
            " const g = window.threejsViewer._objects.get('cloud');"
            " if (!g || !g.userData.pointsLOD) return null;"
            " const lod = g.userData.pointsLOD;"
            " let loaded = 0;"
            " for (const o of lod.objects) if (o) loaded++;"
            " return {loaded, budget: lod.budget};"
            "}"
        )
        if state and state["loaded"] >= 2:
            break
    assert state and state["loaded"] >= 2, f"nodes never streamed in: {state}"
    assert state["budget"] == 30_000

    viewer_client.set_points_lod_options(
        "cloud", point_budget=10_000, refine_pixels=50, size_boost_max=1.0
    )
    tuned = None
    for _ in range(40):
        time.sleep(0.05)
        tuned = viewer_page.evaluate(
            "() => {"
            " const lod = window.threejsViewer._objects.get('cloud')"
            "   .userData.pointsLOD;"
            " const sizes = [];"
            " for (const o of lod.objects) if (o) sizes.push(o.material.size);"
            " return {budget: lod.budget, refinePixels: lod.refinePixels,"
            "         sizeBoostMax: lod.sizeBoostMax, sizes,"
            "         baseSize: lod.baseSize};"
            "}"
        )
        if tuned and tuned["budget"] == 10_000:
            break
    assert tuned["budget"] == 10_000
    assert tuned["refinePixels"] == 50
    assert tuned["sizeBoostMax"] == 1.0
    # boost capped at 1.0 => every already-loaded node reverts to baseSize
    assert tuned["sizes"], "no loaded node materials to check"
    assert all(s == tuned["baseSize"] for s in tuned["sizes"]), tuned["sizes"]

    # The traversal reads the new budget on the next frames: the visible
    # set shrinks under the tightened budget.
    visible = None
    for _ in range(60):
        time.sleep(0.05)
        visible = viewer_page.evaluate(
            "() => {"
            " const lod = window.threejsViewer._objects.get('cloud')"
            "   .userData.pointsLOD;"
            " let v = 0;"
            " for (let i = 0; i < lod.nodes.count; i++) {"
            "   const o = lod.objects[i];"
            "   if (o && o.visible) v += lod.nodes.counts[i];"
            " }"
            " return v;"
            "}"
        )
        if visible is not None and 0 < visible <= 10_000:
            break
    assert visible is not None and 0 < visible <= 10_000, (
        f"visible points {visible} did not shrink under the new 10k budget"
    )


@pytest.mark.browser
def test_embedder_pick_and_controls_toggle(viewer_client, viewer_page):
    """viewer.pick() raycasts meshes and point clouds from client coords,
    resolves the top-level object id, returns null on empty space; and
    setControlsEnabled toggles orbiting (issue #77)."""
    viewer_client.add_box("part", position=[0, 0, 0])
    pts = np.array([[6.0, 0.0, 0.0], [6.0, 1.0, 0.0], [6.0, -1.0, 0.0]])
    viewer_client.add_points("cloud", pts, size=8.0)
    _wait_for(
        viewer_page,
        "() => window.threejsViewer._objects.has('part')"
        " && window.threejsViewer._objects.has('cloud')",
    )

    # Screen position of a world point -> client coords -> pick.
    pick_at = (
        "(args) => {"
        " const [wx, wy, wz, opts] = args;"
        " const v = window.threejsViewer;"
        " const rect = v._renderer.domElement.getBoundingClientRect();"
        " v._camera.updateMatrixWorld(true);"
        " const nd = v._camera.position.clone().set(wx, wy, wz).project(v._camera);"
        " const cx = rect.left + (nd.x * 0.5 + 0.5) * rect.width;"
        " const cy = rect.top + (-nd.y * 0.5 + 0.5) * rect.height;"
        " const hit = v.pick(cx, cy, opts || {});"
        " return hit && {objectId: hit.objectId, point: hit.point,"
        "                distance: hit.distance};"
        "}"
    )
    viewer_page.evaluate(
        "() => window.threejsViewer.setCameraPose("
        "{position: [0, 0, 9], target: [0, 0, 0], up: [0, 1, 0]})"
    )
    hit = viewer_page.evaluate(pick_at, [0.0, 0.0, 0.0, None])
    assert hit is not None, "pick at box centre missed"
    assert hit["objectId"] == "part"
    # box is 1 unit deep centred at origin, camera on +Z: front face at z=0.5
    assert hit["point"]["z"] == pytest.approx(0.5, abs=1e-3)

    # Empty space -> null (aim well away from both objects).
    assert viewer_page.evaluate(pick_at, [0.0, 3.5, 0.0, None]) is None

    # Point cloud pick with a world-space threshold.
    viewer_page.evaluate(
        "() => window.threejsViewer.setCameraPose("
        "{position: [6, 0, 9], target: [6, 0, 0]})"
    )
    hit = viewer_page.evaluate(pick_at, [6.0, 0.0, 0.0, {"pointsThreshold": 0.5}])
    assert hit is not None and hit["objectId"] == "cloud"

    # ids filter: restricted to the part, the same click misses the cloud.
    assert viewer_page.evaluate(pick_at, [6.0, 0.0, 0.0, {"ids": ["part"]}]) is None

    # Controls toggle.
    assert viewer_page.evaluate(
        "() => { const v = window.threejsViewer;"
        " v.setControlsEnabled(false); const off = v._controls.enabled;"
        " v.setControlsEnabled(true); return {off, on: v._controls.enabled}; }"
    ) == {"off": False, "on": True}


@pytest.mark.browser
def test_embedder_animation_transport(viewer_client, viewer_page):
    """seekAnimationTime / getAnimationState / setAnimationPlaying /
    setAnimationSpeed / onAnimationTime — the public animation transport for
    embedders (issue #74), all against the one shared clock."""
    viewer_client.add_box("tbox")
    time.sleep(0.1)
    anim = Animation(
        frames=[Frame(time=0, transforms={}), Frame(time=4, transforms={})],
        loop=False,
    )
    viewer_client.load_animation(anim, autoplay=False)
    _wait_for_animation_loaded(viewer_page)

    state = viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " window.__ticks = [];"
        " window.__unsub = v.onAnimationTime(s => window.__ticks.push(s.time));"
        " v.seekAnimationTime(1.5);"
        " return v.getAnimationState();"
        "}"
    )
    assert state["time"] == pytest.approx(1.5)
    assert state["duration"] == pytest.approx(4.0)
    assert state["playing"] is False
    assert state["loop"] is False
    # the hook saw the seek
    assert viewer_page.evaluate("() => window.__ticks.slice(-1)[0]") == pytest.approx(
        1.5
    )

    # seek clamps
    assert viewer_page.evaluate(
        "() => { const v = window.threejsViewer;"
        " v.seekAnimationTime(99); return v.getAnimationState().time; }"
    ) == pytest.approx(4.0)

    # play/pause + speed; hook fires during playback ticks
    viewer_page.evaluate(
        "() => { const v = window.threejsViewer;"
        " v.seekAnimationTime(0); window.__ticks = [];"
        " v.setAnimationSpeed(2.0); v.setAnimationPlaying(true); }"
    )
    time.sleep(0.4)
    playing = viewer_page.evaluate("() => window.threejsViewer.getAnimationState()")
    assert playing["playing"] is True
    assert playing["speed"] == pytest.approx(2.0)
    assert playing["time"] > 0.3, "clock did not advance under playback"
    n_ticks = viewer_page.evaluate("() => window.__ticks.length")
    assert n_ticks >= 5, f"hook fired only {n_ticks}× during playback"
    viewer_page.evaluate("() => window.threejsViewer.setAnimationPlaying(false)")
    assert (
        viewer_page.evaluate("() => window.threejsViewer.getAnimationState().playing")
        is False
    )
    # unsubscribe stops the hook
    viewer_page.evaluate(
        "() => { window.__unsub(); window.__ticks = [];"
        " window.threejsViewer.seekAnimationTime(1.0); }"
    )
    assert viewer_page.evaluate("() => window.__ticks.length") == 0
    # invalid speed is rejected without change
    assert viewer_page.evaluate(
        "() => { const v = window.threejsViewer; v.setAnimationSpeed(0);"
        " return v.getAnimationState().speed; }"
    ) == pytest.approx(2.0)


@pytest.mark.browser
def test_embedder_get_object_and_message_replies(viewer_client, viewer_page):
    """getObject(id) hands out the loaded Object3D (issue #75), and
    handleMessage() returns the reply payload for query messages so no-WS
    embedders don't need the socket round-trip."""
    viewer_client.add_box("gbox")
    _wait_for(viewer_page, "() => window.threejsViewer._objects.has('gbox')")

    assert viewer_page.evaluate(
        "() => { const v = window.threejsViewer;"
        " const o = v.getObject('gbox');"
        " return !!o && o === v._objects.get('gbox') && o.isObject3D === true; }"
    )
    assert viewer_page.evaluate("() => window.threejsViewer.getObject('nope')") is None

    reply = viewer_page.evaluate(
        "async () => await window.threejsViewer.handleMessage("
        "{type: 'list_objects', requestId: 7})"
    )
    assert reply["type"] == "list_objects_response"
    assert reply["requestId"] == 7
    assert "gbox" in reply["objects"]
    # non-query messages resolve to null
    assert (
        viewer_page.evaluate(
            "async () => await window.threejsViewer.handleMessage("
            "{type: 'add_group', id: 'g2'})"
        )
        is None
    )


@pytest.mark.browser
def test_embedder_overlays(viewer_client, viewer_page):
    """addOverlay/removeOverlay (issue #76): overlays mount in the scene,
    are excluded from framing bounds unless includeInBounds, survive a
    scene clear, and removeOverlay unmounts without disposing."""
    viewer_client.add_box("obox")
    _wait_for(viewer_page, "() => window.threejsViewer._objects.has('obox')")

    setup = viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " const ov = v.getObject('obox').clone();"
        " ov.position.set(1000, 0, 0);"
        " window.__ov = ov;"
        " const id = v.addOverlay(ov, {id: 'cutter'});"
        " const bounds = v._collectFrameableBounds();"
        " return {id, mounted: ov.parent === v._scene,"
        "         framedMaxX: bounds.max.x};"
        "}"
    )
    assert setup["id"] == "cutter"
    assert setup["mounted"] is True
    # default: excluded from framing — bounds stop at the real box, not 1000
    assert setup["framedMaxX"] < 100, setup

    # opt-in inclusion
    included = viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " v.addOverlay(window.__ov, {id: 'cutter', includeInBounds: true});"
        " return v._collectFrameableBounds().max.x;"
        "}"
    )
    assert included > 999

    # survives a scene clear (embedder owns it; _objects is emptied)
    viewer_client.clear()
    time.sleep(0.2)
    assert viewer_page.evaluate(
        "() => { const v = window.threejsViewer;"
        " return v._objects.size === 0 && window.__ov.parent === v._scene; }"
    )

    # removeOverlay unmounts, does not dispose (geometry attrs intact)
    assert viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " const removed = v.removeOverlay('cutter');"
        " return removed && window.__ov.parent === null"
        "   && !!window.__ov.geometry.attributes.position"
        "   && v.removeOverlay('cutter') === false;"
        "}"
    )

    # id reuse: a replaced (stale) instance must not be able to remove the
    # overlay currently registered under that id (#93 review follow-up)
    assert viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " const a = window.__ov.clone(), b = window.__ov.clone();"
        " v.addOverlay(a, {id: 'reused'});"
        " v.addOverlay(b, {id: 'reused'});"  # replaces a
        " const staleNoop = v.removeOverlay(a) === false;"
        " const bStill = b.parent === v._scene;"
        " const bGone = v.removeOverlay(b) === true && b.parent === null;"
        " return staleNoop && bStill && bGone;"
        "}"
    )


@pytest.mark.browser
def test_status_chip_neutral_default_and_set_status(viewer_client, viewer_page):
    """autoConnect:false defaults the status chip to a neutral 'Local data'
    instead of 'Waiting for Python...' (issue #78); setStatus lets the
    embedder drive text + state."""
    result = viewer_page.evaluate(
        "() => {"
        " const live = window.threejsViewer;"
        " const V = live.constructor;"
        " const div = document.createElement('div');"
        " div.style.cssText ="
        "   'width:300px;height:200px;position:absolute;left:-2000px;top:0';"
        " document.body.appendChild(div);"
        " const v2 = new V(div, {"
        "   htmlTemplate: live._options.htmlTemplate,"
        "   cubemapData: live._options.cubemapData,"
        "   autoConnect: false });"
        " const initial = {dot: v2._statusDot.className,"
        "                  text: v2._statusText.textContent};"
        " v2.setStatus('Static demo', 'connected');"
        " const set = {dot: v2._statusDot.className,"
        "              text: v2._statusText.textContent};"
        " v2.setStatus('Odd', 'bogus-state');"
        " const fallback = v2._statusDot.className;"
        " return {initial, set, fallback};"
        "}"
    )
    assert result["initial"] == {
        "dot": "tjsv-status-dot neutral",
        "text": "Local data",
    }
    assert result["set"] == {
        "dot": "tjsv-status-dot connected",
        "text": "Static demo",
    }
    assert result["fallback"] == "tjsv-status-dot neutral"


@pytest.mark.browser
def test_toolpath_travel_line_lockstep_reveal(viewer_client, viewer_page):
    """add_toolpath(travel="line") mounts one LineSegments child over the
    travel hops and reveals whole edges in lockstep with the beads via the
    group draw-range distribution (issue #88)."""
    from threejs_viewer import Toolpath

    pts = np.zeros((8, 3), dtype=np.float32)
    pts[:, 0] = np.arange(8, dtype=np.float32)
    widths = np.array([0.4, 0.4, 0.0, 0.0, 0.4, 0.4, 0.0, 0.0], dtype=np.float32)
    heights = np.where(widths > 0, 0.2, 0.0).astype(np.float32)
    tp = Toolpath.from_points(pts, bead_width=widths, bead_height=heights)
    viewer_client.add_toolpath("tl", tp, travel="line", travel_color=0xFF8800)
    _wait_for(
        viewer_page,
        "() => { const v = window.threejsViewer;"
        " const g = v._objects.get('tl');"
        " return !!g && g.userData.isToolpathGroup"
        "   && !!g.userData.toolpathTravelId"
        "   && !!v._objects.get('tl_travel'); }",
    )

    info = viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " const line = v._objects.get('tl_travel');"
        " return {type: line.type,"
        "         isSegs: line.userData.isLineSegments === true,"
        "         points: line.userData.totalPointCount,"
        "         parentIsGroup: line.parent === v._objects.get('tl')};"
        "}"
    )
    assert info["type"] == "LineSegments"
    assert info["isSegs"] is True
    assert info["points"] == 10  # 5 travel edges
    assert info["parentIsGroup"] is True

    def travel_count(frac):
        viewer_client.set_draw_range("tl", frac)
        time.sleep(0.15)
        return viewer_page.evaluate(
            "() => window.threejsViewer._objects.get('tl_travel')"
            ".geometry.drawRange.count"
        )

    # end fracs are [2/7, 3/7, 4/7, 6/7, 7/7] (draw_range convention
    # index/(n-1)): whole edges appear as the global fraction passes each
    # edge's end point.
    assert travel_count(0.0) == 0
    assert travel_count(0.20) == 0
    assert travel_count(0.30) == 2  # first hop edge (ends at 2/7 ~ 0.286)
    assert travel_count(0.45) == 4  # second edge (3/7 ~ 0.429)
    assert travel_count(0.60) == 6  # the whole first hop (4/7 ~ 0.571)
    assert travel_count(1.0) == 10  # everything


@pytest.mark.browser
def test_uniform_dt_fast_path_with_offset_start_time(viewer_client, viewer_page):
    """_getFrameAtTime's uniform-dt fast path must be relative to
    frames[0].time: a uniformly spaced timeline starting at t=100000
    previously clamped every lookup to the last frame (issue #96)."""
    viewer_client.add_box("obox")
    time.sleep(0.1)
    n = 200
    times = 100_000.0 + np.arange(n, dtype=np.float64) * 0.01
    transforms = np.zeros((n, 1, 16), dtype=np.float32)
    transforms[:, 0, [0, 5, 10, 15]] = 1.0
    transforms[:, 0, 12] = np.linspace(0.0, 1.0, n)  # x slides 0 -> 1
    anim = Animation(loop=False)
    anim.set_frame_times(times)
    anim.set_transform_data(["obox"], transforms)
    viewer_client.load_animation(anim, autoplay=False)
    _wait_for_animation_loaded(viewer_page)

    state = viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " const mid = v._getFrameAtTime(100_001.005);"
        " v._seekToTime(100_001.0);"
        " const x = v._objects.get('obox').matrix.elements[12];"
        " return {index: mid.index, t: mid.t, x,"
        "         uniformDt: v._animation.uniformDt};"
        "}"
    )
    assert state["uniformDt"] > 0, "test premise: fast path must be active"
    assert state["index"] == 100
    assert state["t"] == pytest.approx(0.5, abs=1e-6)
    # seek to halfway: the box sits mid-slide, not at the end
    assert state["x"] == pytest.approx(0.5, abs=0.02)


@pytest.mark.browser
def test_playback_advances_by_wall_clock_and_caps_stalls(viewer_client, viewer_page):
    """The playhead advances by the RAW wall-clock delta (the issue #97 EMA
    smoothing is gone — the jitter it papered over was float32 quantization,
    fixed at the source), but a single stalled frame (fake 5 s old
    _lastAnimationUpdate, e.g. a backgrounded tab) is capped at
    PLAYBACK_MAX_FRAME_DELTA so it can't teleport the playhead by
    5 s x speed."""
    viewer_client.add_box("pbox")
    time.sleep(0.1)
    anim = Animation(
        frames=[Frame(time=0, transforms={}), Frame(time=10_000, transforms={})],
        loop=False,
    )
    viewer_client.load_animation(anim, autoplay=False)
    _wait_for_animation_loaded(viewer_page)

    viewer_page.evaluate(
        "() => { const v = window.threejsViewer;"
        " v.seekAnimationTime(0); v.setAnimationSpeed(100);"
        " v.setAnimationPlaying(true); }"
    )
    time.sleep(0.5)
    t = viewer_page.evaluate("() => window.threejsViewer.getAnimationState().time")
    # Raw pacing: ~0.5 s of wall time x 100 = ~50 s of timeline (generous
    # bounds — headless rAF cadence is noisy under suite load).
    assert t > 5, f"playhead barely advanced: {t}"

    # Fake a 5 s render stall: raw delta would advance 5 s x 100 = 500 s;
    # the cap must limit the next tick to <= 0.25 s x 100 = 25 s.
    jump = viewer_page.evaluate(
        "() => new Promise(resolve => {"
        " const v = window.threejsViewer;"
        " const before = v._animationTime;"
        " v._lastAnimationUpdate = performance.now() - 5000;"
        " requestAnimationFrame(() => requestAnimationFrame("
        "   () => resolve(v._animationTime - before)));"
        "})"
    )
    assert jump < 30, (
        f"stalled frame advanced the playhead by {jump}s (uncapped would be ~500s)"
    )


@pytest.mark.browser
def test_lod_frontier_lands_on_true_point_under_rdp_collapse(
    viewer_client, viewer_page
):
    """Tube-LOD draw-range remap: RDP collapses collinear runs regardless of
    point spacing, and interpolating across the collapsed span BY INDEX put
    the frontier ~arbitrarily far from the true point (one long segment +
    a dense flatten-tolerance cluster => ~98 mm error). The remap now
    projects the true original-spine position onto the reduced chord."""
    n_dense = 60
    xs = np.concatenate([[0.0, 100.0], 100.0 + 0.1 * np.arange(1, n_dense + 1)])
    n = len(xs)
    spine = np.zeros((n, 3), dtype=np.float32)
    spine[:, 0] = xs
    viewer_client.add_parametric_tube(
        "tube",
        spine,
        np.full(n, 2.0, dtype=np.float32),
        np.full(n, 1.0, dtype=np.float32),
        lod={"threshold": 0},
    )
    _wait_for(
        viewer_page,
        "() => { const o = window.threejsViewer._objects.get('tube');"
        " return !!o && !!o.userData.tubeLOD"
        "   && !!o.userData.tubeLOD.keptIndices; }",
        timeout=20000,
    )
    # frontier at original point index 1 => true position x = 100.0
    viewer_client.set_draw_range("tube", 1.0 / (n - 1))
    time.sleep(0.3)
    x = viewer_page.evaluate(
        "() => {"
        " const o = window.threejsViewer._objects.get('tube');"
        " const md = o.userData.tubeMorphData;"
        " if (!md || md.savedRingIndex == null) return null;"
        " const nCs = o.userData.tubeNCs;"
        " const pos = o.geometry.getAttribute('position').array;"
        " let s = 0;"
        " const rb = md.savedRingIndex * nCs;"
        " for (let j = 0; j < nCs; j++) s += pos[(rb + j) * 3];"
        " return s / nCs;"
        "}"
    )
    assert x is not None, "no morphed frontier ring"
    # buggy index-lerp put this at ~1.74; chord projection puts it at 100
    assert x == pytest.approx(100.0, abs=0.5), f"frontier at x={x}, want ~100"


@pytest.mark.browser
def test_group_frontier_tracks_true_point_index(viewer_client, viewer_page):
    """Travel-split toolpath groups: segmentRanges divided by n instead of
    n-1, skewing the recovered frontier index by `value` points — up to a
    full G-code segment near the end of the path (hundreds of mm on long
    moves: the nozzle-vs-frontier desync). The frontier must land on the
    exact point the draw_range value addresses, in every segment."""
    from threejs_viewer import Toolpath

    # segments with wildly different point spacing + a long trailing move
    pts = np.zeros((10, 3), dtype=np.float32)
    pts[:, 0] = [0, 1, 2, 3, 50, 51, 52, 53, 300, 301]
    widths = np.array(
        [0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5], dtype=np.float32
    )
    heights = np.where(widths > 0, 0.3, 0.0).astype(np.float32)
    tp = Toolpath.from_points(pts, bead_width=widths, bead_height=heights)
    viewer_client.add_toolpath("g", tp)
    _wait_for(
        viewer_page,
        "() => { const g = window.threejsViewer._objects.get('g');"
        " return !!g && g.userData.isToolpathGroup"
        "   && g.children.filter(c => c.userData.isParametricTube).length === 3; }",
    )

    def frontier_x(value):
        viewer_client.set_draw_range("g", value)
        time.sleep(0.2)
        return viewer_page.evaluate(
            "() => {"
            " const g = window.threejsViewer._objects.get('g');"
            " for (const c of g.children) {"
            "   const md = c.userData.tubeMorphData;"
            "   if (!md || md.savedRingIndex == null) continue;"
            "   const nCs = c.userData.tubeNCs;"
            "   const pos = c.geometry.getAttribute('position').array;"
            "   let s = 0;"
            "   const rb = md.savedRingIndex * nCs;"
            "   for (let j = 0; j < nCs; j++) s += pos[(rb + j) * 3];"
            "   return s / nCs;"
            " }"
            " return null;"
            "}"
        )

    n = len(pts)
    # halfway between points 1 and 2 (x = 1.5), early in the path
    x = frontier_x(1.5 / (n - 1))
    assert x == pytest.approx(1.5, abs=0.05), f"early frontier at {x}"
    # halfway into the LAST segment's first edge (points 8-9, x = 300.5):
    # with the /n skew this recovered index ~9.4 -> clamped/wrong position
    x = frontier_x(8.5 / (n - 1))
    assert x == pytest.approx(300.5, abs=0.05), f"late frontier at {x}"


@pytest.mark.browser
def test_resize_noop_guard(viewer_client, viewer_page):
    """resize() skips the GL realloc when the size is unchanged (issue #128):
    embedders call viewer.resize() on every mousemove, and the ResizeObserver
    fires per-event during splitter drags. A genuinely new size still applies,
    including the very first explicit resize."""
    result = viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " let calls = 0;"
        " const orig = v._renderer.setSize.bind(v._renderer);"
        " v._renderer.setSize = (w, h) => { calls++; return orig(w, h); };"
        " v.resize(300, 200);"
        " const afterFirst = calls;"
        " for (let i = 0; i < 50; i++) v.resize(300, 200);"
        " const afterSame = calls;"
        " v.resize(320, 200);"
        " const afterNew = calls;"
        " v.resize(0, 0);"
        " const afterZero = calls;"
        " v._renderer.setSize = orig;"
        " return { afterFirst, afterSame, afterNew, afterZero };"
        "}"
    )
    assert result["afterFirst"] == 1, "first new size must apply"
    assert result["afterSame"] == 1, "repeated same-size resize must be a no-op"
    assert result["afterNew"] == 2, "a genuinely new size must apply"
    assert result["afterZero"] == 2, "zero-size rects stay guarded"


@pytest.mark.browser
def test_gizmo_axis_click_snaps_ortho_and_flips(viewer_client, viewer_page):
    """Gizmo axis-bubble click snaps to an ortho view down that axis (#514);
    re-clicking the same axis flips to the opposite side, a different axis does
    not."""
    viewer_client.add_box("b")
    assert "b" in viewer_client.query_scene()["objects"]  # sync: box is in-scene
    result = viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " v._gizmoAxisClick('front');"
        " const afterFront = { ortho: v._isOrtho, axis: v._gizmoAxisView,"
        "   zoom: v._orthoCamera.zoom };"
        " v._gizmoAxisClick('front');"  # same axis -> flip
        " const afterReclick = v._gizmoAxisView;"
        " v._gizmoAxisClick('top');"  # different axis -> no flip
        " const afterTop = v._gizmoAxisView;"
        " return { afterFront, afterReclick, afterTop };"
        "}"
    )
    assert result["afterFront"]["ortho"] is True, "axis click must switch to ortho"
    assert result["afterFront"]["axis"] == "front"
    assert result["afterFront"]["zoom"] > 0
    assert result["afterReclick"] == "back", "re-clicking the same axis flips it"
    assert result["afterTop"] == "top", "a different axis snaps without flipping"


@pytest.mark.browser
def test_gizmo_leaving_ortho_clears_axis_snap(viewer_client, viewer_page):
    """Switching back to perspective clears the gizmo axis snap so the next
    bubble click is treated as a fresh snap, not a flip (#514)."""
    viewer_client.add_box("b")
    assert "b" in viewer_client.query_scene()["objects"]  # sync: box is in-scene
    axis = viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " v._gizmoAxisClick('front');"
        " v._switchCamera(false);"  # back to perspective
        " const cleared = v._gizmoAxisView;"
        " v._gizmoAxisClick('front');"  # fresh snap, must not flip
        " return { cleared, after: v._gizmoAxisView };"
        "}"
    )
    assert axis["cleared"] is None, "leaving ortho clears the axis snap"
    assert axis["after"] == "front", (
        "a fresh click after re-entering ortho does not flip"
    )


@pytest.mark.browser
def test_gizmo_axis_click_keeps_zoom(viewer_client, viewer_page):
    """A bubble click reorients only — the user's ortho zoom is preserved
    across snaps and flips (a click must not reset the viewing distance)."""
    viewer_client.add_box("b")
    assert "b" in viewer_client.query_scene()["objects"]  # sync: box is in-scene
    result = viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " v._switchCamera(true);"  # ortho, zoom matched to persp framing
        " v._orthoCamera.zoom *= 3.7;"  # user zooms in
        " v._orthoCamera.updateProjectionMatrix();"
        " const before = v._orthoCamera.zoom;"
        " v._gizmoAxisClick('front');"
        " const afterSnap = v._orthoCamera.zoom;"
        " v._gizmoAxisClick('front');"  # flip
        " const afterFlip = v._orthoCamera.zoom;"
        " return { before, afterSnap, afterFlip };"
        "}"
    )
    assert result["afterSnap"] == pytest.approx(result["before"])
    assert result["afterFlip"] == pytest.approx(result["before"])


@pytest.mark.browser
def test_iso_button_returns_to_perspective(viewer_client, viewer_page):
    """The ISO corner button is the way back to 3D: after a bubble click puts
    the viewer in an ortho plan view, clicking ISO switches to a perspective
    isometric (and there is no separate ortho toolbar toggle anymore)."""
    viewer_client.add_box("b")
    assert "b" in viewer_client.query_scene()["objects"]  # sync: box is in-scene
    result = viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " v._gizmoAxisClick('top');"
        " const orthoAfterBubble = v._isOrtho;"
        " v._viewIsoBtn.click();"
        " const orthoAfterIso = v._isOrtho;"
        " const toolbarOrtho = !!document.querySelector('.tjsv-btn-ortho');"
        " return { orthoAfterBubble, orthoAfterIso, toolbarOrtho };"
        "}"
    )
    assert result["orthoAfterBubble"] is True
    assert result["orthoAfterIso"] is False, "ISO must return to perspective"
    assert result["toolbarOrtho"] is False, "ortho toolbar toggle removed"


@pytest.mark.browser
def test_auto_projection_orbit_returns_to_perspective(viewer_client, viewer_page):
    """Auto-projection: ortho entered BY a bubble snap auto-exits back to
    perspective when the user orbits away; a manual `O` ortho never does."""
    viewer_client.add_box("b")
    assert "b" in viewer_client.query_scene()["objects"]  # sync: box is in-scene
    result = viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " const fakeDrag = () => {"
        "   const orig = v._controls.isDragging;"
        "   v._controls.isDragging = () => true;"
        "   v._controls.dispatchEvent({ type: 'change' });"
        "   v._controls.isDragging = orig;"
        " };"
        " v._gizmoAxisClick('top');"  # auto-enters ortho
        " const orthoSnapped = v._isOrtho;"
        " fakeDrag();"  # orbit away -> should return to perspective
        " const orthoAfterOrbit = v._isOrtho;"
        " v._switchCamera(true);"  # manual ortho (O key path)
        " v._gizmoAxisClick('top');"  # snap within manual ortho
        " fakeDrag();"  # orbit away -> manual ortho is respected
        " const manualOrthoKept = v._isOrtho;"
        " return { orthoSnapped, orthoAfterOrbit, manualOrthoKept };"
        "}"
    )
    assert result["orthoSnapped"] is True
    assert result["orthoAfterOrbit"] is False, "auto-entered ortho exits on orbit"
    assert result["manualOrthoKept"] is True, "manual O ortho is never auto-exited"


@pytest.mark.browser
def test_axis_snap_survives_pivot_but_clears_on_orbit(viewer_client, viewer_page):
    """The gizmo axis snap is preserved through a plain click-to-pivot (a
    controls 'change' fired while not dragging) so a re-click still flips, but
    an actual orbit/pan drag ('change' while dragging) clears it (#514)."""
    viewer_client.add_box("b")
    assert "b" in viewer_client.query_scene()["objects"]  # sync: box is in-scene
    result = viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " const c = v._controls;"
        " v._gizmoAxisClick('front');"
        " const snapped = v._gizmoAxisView;"
        # click-to-pivot: controls emit 'change' while NOT dragging.
        " c._state = 0;"
        " c.dispatchEvent({ type: 'change' });"
        " const afterPivot = v._gizmoAxisView;"
        " v._gizmoAxisClick('front');"  # snap preserved -> flip
        " const afterReclick = v._gizmoAxisView;"
        " v._gizmoAxisClick('front');"  # from 'back' -> 'front'
        # orbit drag: controls emit 'change' while dragging (state != NONE).
        " c._state = 1;"
        " c.dispatchEvent({ type: 'change' });"
        " const afterOrbit = v._gizmoAxisView;"
        " c._state = 0;"
        " v._gizmoAxisClick('front');"  # snap cleared -> fresh, no flip
        " const afterFreshClick = v._gizmoAxisView;"
        " return { snapped, afterPivot, afterReclick, afterOrbit,"
        "   afterFreshClick };"
        "}"
    )
    assert result["snapped"] == "front"
    assert result["afterPivot"] == "front", "click-to-pivot preserves the snap"
    assert result["afterReclick"] == "back", "re-click flips while snap preserved"
    assert result["afterOrbit"] is None, "an orbit drag clears the snap"
    assert result["afterFreshClick"] == "front", (
        "a fresh click after an orbit does not flip"
    )


@pytest.mark.browser
def test_orbit_pivot_falls_back_to_bounds_center(viewer_client, viewer_page):
    """A click that hits no component pivots on the scene bounding-box center,
    not the old z=0 floor-plane intersection; the grid is excluded (#520)."""
    # Box centered at (10, 20, 30); a large grid that must not sway the center.
    viewer_client.add_box("b", position=[10.0, 20.0, 30.0])
    viewer_client.add_grid("floor", cell_size=10.0, extent=10000.0)
    assert "b" in viewer_client.query_scene()["objects"]  # sync: box is in-scene
    result = viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " const c = v._controls;"
        " c.target.set(999, 999, 999);"  # somewhere off-model
        " const fb = c._fallbackPivotGetter();"  # what a component-miss triggers
        " return fb ? { x: fb.x, y: fb.y, z: fb.z } : null;"
        "}"
    )
    assert result is not None, "fallback pivot must resolve when the scene has bounds"
    assert abs(result["x"] - 10.0) < 1.0
    assert abs(result["y"] - 20.0) < 1.0
    assert abs(result["z"] - 30.0) < 1.0
