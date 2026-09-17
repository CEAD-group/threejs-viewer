"""Integration tests using Playwright — verify browser-side behavior end-to-end."""

import base64
import json
import math
import socket
import struct
import time

import numpy as np
import pytest

from conftest import frames, settle
from threejs_viewer import Animation, Frame, ViewerClient


@pytest.mark.browser
def test_viewer_connects(viewer_client, viewer_page):
    """Viewer opens and WebSocket connects."""
    assert viewer_client._ws is not None


@pytest.mark.browser
def test_add_box_appears_in_scene(viewer_client, viewer_page):
    """Adding a box from Python creates it in the browser scene graph."""
    viewer_client.add_box("mybox")
    settle(viewer_client)
    result = viewer_client.query_scene()
    assert "mybox" in result["objects"]
    assert result["objects"]["mybox"]["type"] == "Mesh"


@pytest.mark.browser
def test_add_grid_appears_and_is_excluded_from_bounds(viewer_client, viewer_page):
    """add_grid creates a tracked mesh that never inflates scene bounds."""
    viewer_client.add_box("ref")
    viewer_client.add_grid("floor", cell_size=10.0, extent=10000.0)
    settle(viewer_client)
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
    # F/Home framing uses _collectFrameableBounds (an AABB, distinct from
    # the sphere above) — the grid must be excluded there too, or framing
    # over-zooms to fit the whole floor plane.
    frame_extent = viewer_page.evaluate(
        "() => { const b = window.threejsViewer._collectFrameableBounds();"
        " const s = new window.tjsv.THREE.Vector3(); b.getSize(s);"
        " return Math.max(s.x, s.y, s.z); }"
    )
    assert frame_extent < 100
    viewer_client.delete("floor")
    settle(viewer_client)
    assert "floor" not in viewer_client.query_scene()["objects"]


@pytest.mark.browser
def test_grouping(viewer_client, viewer_page):
    """Parent-child hierarchy works end-to-end."""
    viewer_client.add_group("arm")
    viewer_client.add_box("joint", parent="arm")
    settle(viewer_client)
    objects = viewer_client.query_scene()["objects"]
    assert objects["arm"]["type"] == "Group"
    assert "joint" in objects["arm"]["children"]
    assert objects["joint"]["parent"] == "arm"


def _world_position(page, obj_id):
    """World position of a tracked object by id, or None."""
    return page.evaluate(
        "(id) => {"
        " const o = window.threejsViewer._objects.get(id);"
        " if (!o) return null;"
        " o.updateWorldMatrix(true, false);"
        " const e = o.matrixWorld.elements;"
        " return [e[12], e[13], e[14]]; }",
        obj_id,
    )


@pytest.mark.browser
def test_deferred_reparent_child_before_parent(viewer_client, viewer_page):
    """A child added with parent="P" BEFORE P exists renders at the scene root,
    then re-parents under P when P arrives (issue #138)."""
    viewer_client.add_box("orphan", parent="P", position=[1.0, 0.0, 0.0])
    settle(viewer_client)
    # Renders immediately at the scene root (not dropped).
    objects = viewer_client.query_scene()["objects"]
    assert "orphan" in objects
    assert objects["orphan"]["parent"] is None
    assert _world_position(viewer_page, "orphan") == pytest.approx([1.0, 0.0, 0.0])
    # Parent arrives late with its own transform.
    viewer_client.add_group("P", position=[5.0, 0.0, 0.0])
    settle(viewer_client)
    objects = viewer_client.query_scene()["objects"]
    assert objects["orphan"]["parent"] == "P"
    assert "orphan" in objects["P"]["children"]
    # Local transform was authored parent-local, so world = parent + local.
    assert _world_position(viewer_page, "orphan") == pytest.approx([6.0, 0.0, 0.0])


@pytest.mark.browser
def test_deferred_reparent_follows_parent_transforms(viewer_client, viewer_page):
    """After a deferred re-parent, transform updates to the parent move the child."""
    viewer_client.add_box("child", parent="P", position=[1.0, 0.0, 0.0])
    settle(viewer_client)
    viewer_client.add_group("P", position=[5.0, 0.0, 0.0])
    settle(viewer_client)
    viewer_client.batch_update({"P": {"position": [0.0, 10.0, 0.0]}})
    settle(viewer_client)
    assert _world_position(viewer_page, "child") == pytest.approx([1.0, 10.0, 0.0])


@pytest.mark.browser
def test_deferred_reparent_pruned_on_child_delete(viewer_client, viewer_page):
    """Deleting a waiting child before its parent arrives prunes the pending
    entry, so a later add of the parent doesn't touch anything — and an
    unrelated object reusing the child's id is not mis-parented."""
    viewer_client.add_box("ephemeral", parent="P", position=[1.0, 0.0, 0.0])
    settle(viewer_client)
    viewer_client.delete("ephemeral")
    settle(viewer_client)
    # Reuse the id with NO parent — must stay at the scene root.
    viewer_client.add_box("ephemeral", position=[2.0, 0.0, 0.0])
    settle(viewer_client)
    viewer_client.add_group("P", position=[5.0, 0.0, 0.0])
    settle(viewer_client)
    objects = viewer_client.query_scene()["objects"]
    assert objects["ephemeral"]["parent"] is None
    assert "ephemeral" not in objects["P"]["children"]
    assert _world_position(viewer_page, "ephemeral") == pytest.approx([2.0, 0.0, 0.0])


@pytest.mark.browser
def test_delete_object(viewer_client, viewer_page):
    """Deleting an object removes it from the scene."""
    viewer_client.add_sphere("s1")
    settle(viewer_client)
    viewer_client.delete("s1")
    settle(viewer_client)
    objects = viewer_client.query_scene()["objects"]
    assert "s1" not in objects


@pytest.mark.browser
def test_visibility(viewer_client, viewer_page):
    """set_visible toggles object visibility."""
    viewer_client.add_box("v1")
    settle(viewer_client)
    viewer_client.set_visible("v1", False)
    settle(viewer_client)
    objects = viewer_client.query_scene()["objects"]
    assert objects["v1"]["visible"] is False


@pytest.mark.browser
def test_set_scene_visibility_before_add_is_honoured(viewer_client, viewer_page):
    """set_scene_visibility for an id that doesn't exist yet must apply once the
    object loads. Regression test for the race where a visibility flip arriving
    during a slow GLB fetch was silently dropped, leaving the loaded object
    permanently at its initial `visible` state (PR #47)."""
    viewer_client.set_scene_visibility({"m1": False})
    settle(viewer_client)
    viewer_client.add_box("m1")
    settle(viewer_client)
    objects = viewer_client.query_scene()["objects"]
    assert "m1" in objects
    assert objects["m1"]["visible"] is False


@pytest.mark.browser
def test_baseline_visibility_pruned_on_delete(viewer_client, viewer_page):
    """Deleting an object prunes its baseline so a later re-add isn't shadowed
    by stale visibility from a prior set_scene_visibility."""
    viewer_client.add_box("m1")
    viewer_client.set_scene_visibility({"m1": False})
    settle(viewer_client)
    viewer_client.delete("m1")
    settle(viewer_client)
    viewer_client.add_box("m1")
    settle(viewer_client)
    objects = viewer_client.query_scene()["objects"]
    assert objects["m1"]["visible"] is True


def _highlight_state(page, obj_id):
    """Outline-child count + material identity snapshot for an object's meshes."""
    return page.evaluate(
        "(id) => {"
        " const o = window.threejsViewer._objects.get(id);"
        " if (!o) return null;"
        " const out = { outlines: 0, outlineColors: [], outlineSegs: [],"
        "               outlineThresholds: [], outlineStyles: [],"
        "               outlineSharesGeometry: [], meshes: [] };"
        " o.traverse((child) => {"
        "  if (child.userData.__highlightOutline) {"
        "   out.outlines++;"
        "   out.outlineColors.push(child.material.color.getHex());"
        "   out.outlineSegs.push(child.geometry.attributes.position.count / 2);"
        "   out.outlineThresholds.push(child.userData.__highlightThreshold);"
        "   out.outlineStyles.push(child.userData.__highlightStyle);"
        "   out.outlineSharesGeometry.push("
        "     child.geometry === child.parent.geometry);"
        "   return;"
        "  }"
        "  if (!child.isMesh) return;"
        "  const m = Array.isArray(child.material) ? child.material[0] : child.material;"
        "  out.meshes.push({"
        "   uuid: m.uuid, color: m.color ? m.color.getHex() : null,"
        "   opacity: m.opacity, transparent: m.transparent,"
        "   hasEdgeRef: child.userData.__highlightEdge !== undefined,"
        "  });"
        " });"
        " return out;"
        "}",
        obj_id,
    )


def _wait_highlight_state(page, obj_id, predicate):
    """Poll _highlight_state until predicate(state) holds (fixed sleeps flake
    under machine load — messages apply asynchronously); returns last state."""
    state = None
    for _ in range(80):
        state = _highlight_state(page, obj_id)
        if state and predicate(state):
            break
        time.sleep(0.05)
    return state


@pytest.mark.browser
def test_set_highlight_toggle_is_reversible(viewer_client, viewer_page):
    """set_highlight on adds one outline per mesh (traversing group children),
    is idempotent (no stacking on a second enable), and off removes the
    outlines leaving the meshes' own materials byte-identical (issue #147)."""
    viewer_client.add_group("cell")
    viewer_client.add_box("link1", parent="cell", color=0x4488CC)
    viewer_client.add_box("link2", parent="cell", color=0xCC8844)
    before = _wait_highlight_state(viewer_page, "cell", lambda s: len(s["meshes"]) == 2)
    assert before["outlines"] == 0
    assert len(before["meshes"]) == 2

    viewer_client.set_highlight("cell")
    on = _wait_highlight_state(viewer_page, "cell", lambda s: s["outlines"] == 2)
    assert on["outlines"] == 2  # one per descendant mesh
    assert all(m["hasEdgeRef"] for m in on["meshes"])

    # Idempotent: a second enable re-uses the outlines, never stacks.
    viewer_client.set_highlight("cell", color=0xFF00FF)
    again = _wait_highlight_state(
        viewer_page, "cell", lambda s: s["outlineColors"] == [0xFF00FF, 0xFF00FF]
    )
    assert again["outlines"] == 2
    assert again["outlineColors"] == [0xFF00FF, 0xFF00FF]  # re-tinted in place

    viewer_client.set_highlight("cell", enabled=False)
    after = _wait_highlight_state(viewer_page, "cell", lambda s: s["outlines"] == 0)
    assert after["outlines"] == 0
    assert not any(m["hasEdgeRef"] for m in after["meshes"])
    # Original appearance restored exactly: same material instances, same state.
    assert after["meshes"] == before["meshes"]


@pytest.mark.browser
def test_set_highlight_composes_with_set_color_and_opacity(viewer_client, viewer_page):
    """set_color/set_opacity restyle the mesh but leave the outline's own
    selection colour and opacity alone (guarded on __highlightOutline)."""
    viewer_client.add_box("hbox", color=0x4488CC)
    _wait_highlight_state(viewer_page, "hbox", lambda s: len(s["meshes"]) == 1)
    viewer_client.set_highlight("hbox", color=0xFF00FF)
    _wait_highlight_state(viewer_page, "hbox", lambda s: s["outlines"] == 1)
    viewer_client.set_color("hbox", 0x00FF00)
    viewer_client.set_opacity("hbox", 0.5)
    state = _wait_highlight_state(
        viewer_page,
        "hbox",
        lambda s: (
            s["meshes"][0]["color"] == 0x00FF00 and s["meshes"][0]["opacity"] == 0.5
        ),
    )
    assert state["outlines"] == 1
    assert state["outlineColors"] == [0xFF00FF]  # not recoloured by set_color
    assert state["meshes"][0]["color"] == 0x00FF00
    assert state["meshes"][0]["opacity"] == 0.5
    # Turning the mesh translucent re-resolves the style: a transparent mesh
    # renders after the opaque hull, so it cannot prime the depth buffer the
    # hull is culled against, and the silhouette gives way to the edge outline.
    assert state["outlineStyles"] == ["edges"]
    outline_opacity = viewer_page.evaluate(
        "() => {"
        " let op = null;"
        " window.threejsViewer._objects.get('hbox').traverse((c) => {"
        "  if (c.userData.__highlightOutline) op = c.material.opacity;"
        " });"
        " return op;"
        "}"
    )
    # The outline keeps its own styling — 0.95 is the edge outline's baked
    # opacity, not the mesh's 0.5 leaking into the selection indicator.
    assert outline_opacity == 0.95


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
        settle(viewer_client)
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
    settle(viewer_client)
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


@pytest.mark.browser
def test_binary_fetch_404_logs_and_skips(viewer_client, viewer_page):
    """Issue #142: a 404 blob fetch (missing/expired blob) must log a clear
    console.error with the message type, id, and HTTP status, and skip
    cleanly — not hand the error body to the model loader, which threw an
    uncaught `RangeError: Invalid typed array length`."""
    console_msgs = []
    page_errors = []
    viewer_page.on("console", lambda msg: console_msgs.append((msg.type, msg.text)))
    viewer_page.on("pageerror", lambda exc: page_errors.append(str(exc)))

    url = f"http://{viewer_client.host}:{viewer_client._http_port}/no_such_blob"
    viewer_page.evaluate(
        "(url) => window.threejsViewer.handleMessage({"
        " type: 'add_model_binary', id: 'missing_model',"
        " format: 'glb', blob_url: url })",
        url,
    )

    def _found():
        return any(
            t == "error" and "add_model_binary 'missing_model'" in m and "HTTP 404" in m
            for t, m in console_msgs
        )

    # Poll with wait_for_timeout, not time.sleep: the sync Playwright API only
    # dispatches queued page events (console/pageerror) while the main thread
    # is inside a Playwright call.
    for _ in range(100):
        viewer_page.wait_for_timeout(100)
        if _found():
            break
    assert _found(), f"expected a clear 404 console.error, got: {console_msgs!r}"
    assert "missing_model" not in viewer_client.query_scene()["objects"]
    assert page_errors == [], f"uncaught page errors: {page_errors!r}"


_RACE_SETUP_JS = """
(badUrl) => {
    // A valid add_mesh_binary payload for a single triangle: f32 xyz * 3,
    // then u32 indices * 3 (no normals / no vertex colors).
    const buf = new ArrayBuffer(3 * 3 * 4 + 3 * 4);
    new Float32Array(buf, 0, 9).set([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    new Uint32Array(buf, 36, 3).set([0, 1, 2]);
    window.__raceGoodUrl = URL.createObjectURL(new Blob([buf]));
    window.__raceBadUrl = badUrl;
    window.__raceAdd = (id, url) => window.threejsViewer.handleMessage({
        type: 'add_mesh_binary', id, blob_url: url,
        numVertices: 3, numIndices: 3,
    });
}
"""


@pytest.mark.browser
def test_repush_during_blob_fetch_is_quiet(viewer_client, viewer_page):
    """Issue #157: deleting + re-adding an id while its blob fetch is still in
    flight is expected traffic — the producer drops the old blob, so the fetch
    404s. The re-added object must load and the superseded fetch must NOT log a
    console.error (a 404 for a still-live object still does — issue #142, see
    test_binary_fetch_404_logs_and_skips)."""
    console_msgs = []
    page_errors = []
    viewer_page.on("console", lambda msg: console_msgs.append((msg.type, msg.text)))
    viewer_page.on("pageerror", lambda exc: page_errors.append(str(exc)))

    bad_url = f"http://{viewer_client.host}:{viewer_client._http_port}/no_such_blob"
    viewer_page.evaluate(_RACE_SETUP_JS, bad_url)

    def _errors(needle):
        return [m for t, m in console_msgs if t == "error" and needle in m]

    def _logged(needle):
        return any(needle in m for _, m in console_msgs)

    # --- Ordering A: the delete/re-push reaches the viewer while the fetch is
    # still open. The fetch is aborted, so the dead blob is never even fetched
    # to completion.
    viewer_page.evaluate(
        "() => { window.__raceAdd('race_a', window.__raceBadUrl);"
        " window.threejsViewer.handleMessage({ type: 'delete_object', id: 'race_a' });"
        " window.__raceAdd('race_a', window.__raceGoodUrl); }"
    )
    # --- Ordering B: the 404 wins the race and lands *before* the delete/
    # re-push that explains it (HTTP and WebSocket are separate transports).
    # The grace is stretched past production's 500ms so the delete is certain
    # to land inside the window even on a loaded machine, without the test
    # having to sit out a long fixed sleep (below).
    viewer_page.evaluate("() => { window.__supersededFetchGraceMs = 1000; }")
    viewer_page.evaluate("() => window.__raceAdd('race_b', window.__raceBadUrl)")
    viewer_page.wait_for_timeout(500)  # let the 404 land first
    viewer_page.evaluate(
        "() => { window.threejsViewer.handleMessage({ type: 'delete_object', id: 'race_b' });"
        " window.__raceAdd('race_b', window.__raceGoodUrl); }"
    )
    # The re-check after the grace window always says something — either the
    # superseded discard or a loud error — so wait for whichever lands rather
    # than sleeping out the window blind. Finishes as soon as the decision is
    # made, and still fails correctly if it is the wrong one.
    for _ in range(100):
        if _logged("superseded mesh fetch for 'race_b'") or _errors("race_b"):
            break
        viewer_page.wait_for_timeout(50)

    objects = viewer_client.query_scene()["objects"]
    assert objects.get("race_a", {}).get("type") == "Mesh"
    assert objects.get("race_b", {}).get("type") == "Mesh"
    assert _errors("race_a") == [], f"superseded fetch shouted: {console_msgs!r}"
    assert _errors("race_b") == [], f"superseded fetch shouted: {console_msgs!r}"
    # ...and it was actually classified as superseded (not silently skipped).
    assert _logged("Discarding superseded mesh fetch for 'race_a'"), console_msgs
    assert _logged("Discarding superseded mesh fetch for 'race_b'"), console_msgs
    assert page_errors == [], f"uncaught page errors: {page_errors!r}"


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
        settle(viewer_client)
        objects = viewer_client.query_scene()["objects"]
        if "dr" in objects:
            dr = objects["dr"]["drawRange"]
            if abs(dr - 0.5) < 1e-3:
                break
    assert dr is not None and abs(dr - 0.5) < 1e-3, (
        f"expected drawRange 0.5, got {dr!r}"
    )


@pytest.mark.browser
def test_add_mesh_rgba_vertex_colors_browser(viewer_client, viewer_page):
    """add_mesh with (N, 4) colors sets itemSize=4 color attribute and material.transparent=true."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    colors = np.array(
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.5], [0.0, 0.0, 1.0, 0.2]],
        dtype=np.float32,
    )
    viewer_client.add_mesh("rgba_mesh", positions, indices, colors=colors)
    settle(viewer_client)
    res = viewer_page.evaluate(
        "() => {"
        " const o = window.threejsViewer._objects.get('rgba_mesh');"
        " if (!o) return null;"
        " const col = o.geometry.getAttribute('color');"
        " return {"
        "   itemSize: col ? col.itemSize : null,"
        "   transparent: o.material.transparent,"
        "   vertexColors: o.material.vertexColors,"
        "   count: col ? col.count : null,"
        "   colors: col ? Array.from(col.array) : null,"
        " };"
        "}"
    )
    assert res is not None, "rgba_mesh never landed in scene"
    assert res["itemSize"] == 4
    assert res["transparent"] is True
    assert res["vertexColors"] is True
    assert res["count"] == 3
    assert np.allclose(
        res["colors"], [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.2]
    )


@pytest.mark.browser
def test_add_invisible_object_browser(viewer_client, viewer_page):
    """Adding an object with visible=False sets its initial visibility to False in the scene."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    viewer_client.add_mesh("inv_mesh", positions, indices, visible=False)
    viewer_client.add_box("inv_box", visible=False)
    settle(viewer_client)
    res = viewer_page.evaluate(
        "() => {"
        " const m = window.threejsViewer._objects.get('inv_mesh');"
        " const b = window.threejsViewer._objects.get('inv_box');"
        " return {"
        "   meshVisible: m ? m.visible : null,"
        "   boxVisible: b ? b.visible : null,"
        " };"
        "}"
    )
    assert res is not None
    assert res["meshVisible"] is False
    assert res["boxVisible"] is False


@pytest.mark.browser
def test_billboard_faces_camera_browser(viewer_client, viewer_page):
    """A full billboard copies the camera orientation; a Z-locked one stays upright.

    Also covers the parented case: the billboard under a rotated group must
    divide the parent's rotation out rather than tumble with it.
    """
    viewer_client.add_billboard("bb_full", position=[0, 0, 0])
    viewer_client.add_billboard(
        "bb_up", mode="hinge", hinge="+y", hinge_world=[0, 0, 1], position=[3, 0, 0]
    )
    # A group rotated 90 deg about Z (column-major 4x4).
    viewer_client.add_group("bb_group")
    viewer_client.set_matrix(
        "bb_group", [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    )
    viewer_client.add_billboard("bb_child", parent="bb_group")
    settle(viewer_client)
    frames(viewer_page)

    res = viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            const THREE = window.tjsv.THREE;
            const q = new THREE.Quaternion();
            const camQ = v._camera.getWorldQuaternion(new THREE.Quaternion());
            const worldQuat = (id) => {
                const o = v._objects.get(id);
                return o ? o.getWorldQuaternion(q.clone()).toArray() : null;
            };
            const localY = (id) => {
                const o = v._objects.get(id);
                if (!o) return null;
                return new THREE.Vector3(0, 1, 0)
                    .applyQuaternion(o.getWorldQuaternion(q.clone())).toArray();
            };
            return {
                camQuat: camQ.toArray(),
                fullQuat: worldQuat('bb_full'),
                childQuat: worldQuat('bb_child'),
                upY: localY('bb_up'),
            };
        }"""
    )
    assert res["fullQuat"] is not None, "bb_full never landed in scene"
    # A full billboard's world orientation is the camera's, parent or no parent.
    assert np.allclose(res["fullQuat"], res["camQuat"], atol=1e-5)
    assert np.allclose(res["childQuat"], res["camQuat"], atol=1e-5), (
        "billboard under a rotated group tumbled with the parent"
    )
    # A Z-locked billboard's local +Y stays pinned to world +Z.
    assert np.allclose(res["upY"], [0, 0, 1], atol=1e-5)


@pytest.mark.browser
def test_billboard_invalid_axis_degrades_browser(viewer_page):
    """A zero/non-finite axis off handleMessage degrades to the default axis.

    Python validates the axes, but handleMessage is a public embedder surface,
    and a degenerate axis would otherwise reach the aim math as a zero basis on
    every frame.
    """
    viewer_page.evaluate(
        """() => {
            window.threejsViewer.handleMessage(
                {type: 'add_billboard', id: 'bb_zero', width: 1, height: 1,
                 mode: 'hinge', hinge: [0, 0, 0]});
            window.threejsViewer.handleMessage(
                {type: 'add_billboard', id: 'bb_nan', width: 1, height: 1,
                 mode: 'hinge', hinge: [0, 0, null]});
        }"""
    )
    frames(viewer_page)
    res = viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            const state = (id) => {
                const o = v._objects.get(id);
                if (!o) return null;
                return {
                    hinge: o.userData.__billboardOpts.hinge.toArray(),
                    finite: o.quaternion.toArray().every(Number.isFinite)
                        && o.matrixWorld.elements.every(Number.isFinite),
                };
            };
            return {zero: state('bb_zero'), nan: state('bb_nan')};
        }"""
    )
    for name in ("zero", "nan"):
        assert res[name] is not None, f"bb_{name} never landed in scene"
        # Degenerate hinge falls back to the documented default (+y).
        assert res[name]["hinge"] == pytest.approx([0.0, 1.0, 0.0])
        assert res[name]["finite"] is True, (
            "invalid axis produced a non-finite transform"
        )


_BB_PROBE_JS = """(id) => {
    const v = window.threejsViewer;
    const THREE = window.tjsv.THREE;
    const o = v._objects.get(id);
    if (!o) return null;
    o.updateWorldMatrix(true, false);
    const q = o.getWorldQuaternion(new THREE.Quaternion());
    const camPos = v._camera.getWorldPosition(new THREE.Vector3());
    const objPos = o.getWorldPosition(new THREE.Vector3());
    const opts = o.userData.__billboardOpts;
    const face = opts ? opts.face.clone() : new THREE.Vector3(0, 0, 1);
    return {
        quat: q.toArray(),
        pos: objPos.toArray(),
        // World direction of the object's local `face` axis.
        faceWorld: face.applyQuaternion(q).toArray(),
        // Unit vector from the object toward the camera.
        toCamera: camPos.sub(objPos).normalize().toArray(),
        finite: o.matrixWorld.elements.every(Number.isFinite),
    };
}"""


def _spin_animation(object_id, axis="y", turns=1.0, n=61, duration=6.0):
    """A transforms channel spinning one object about its own local axis.

    Binary channel on purpose: that is the path that pins `matrixAutoUpdate`
    off and writes `obj.matrix` directly.
    """
    from threejs_viewer import Animation

    times = np.linspace(0.0, duration, n)
    data = np.zeros((n, 1, 16), dtype=np.float32)
    for i, t in enumerate(times):
        a = 2 * np.pi * turns * t / duration
        c, s = np.cos(a), np.sin(a)
        m = np.eye(4)
        if axis == "y":
            m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
        else:
            m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
        data[i, 0] = m.T.flatten()
    anim = Animation(frames=[])
    anim.set_frame_times(times)
    anim.set_transform_data([object_id], data)
    return anim


@pytest.mark.browser
def test_billboard_aim_does_not_drift_browser(viewer_client, viewer_page):
    """`aim` composes on a retained base, never on its own previous output.

    The regression test for the core hazard: if the viewer read the pose back
    as R_own it would compose R_aim into itself every frame and the object
    would spin away on its own.
    """
    viewer_client.add_box("drifter", position=[0, 0, 0])
    viewer_client.set_billboard("drifter", mode="aim", face="+y")
    settle(viewer_client)
    frames(viewer_page, 2)
    early = viewer_page.evaluate(_BB_PROBE_JS, "drifter")
    frames(viewer_page, 30)
    late = viewer_page.evaluate(_BB_PROBE_JS, "drifter")

    assert early is not None and late is not None
    assert np.allclose(early["quat"], late["quat"], atol=1e-6), (
        f"pose drifted over 30 still frames: {early['quat']} -> {late['quat']}"
    )
    # And it is actually aimed: local +y points at the camera.
    assert np.allclose(late["faceWorld"], late["toCamera"], atol=1e-5)


@pytest.mark.browser
def test_billboard_aim_tracks_moving_camera_browser(viewer_client, viewer_page):
    """The aim follows the camera and stays a pure function of it."""
    viewer_client.add_box("tracker", position=[0, 0, 0])
    viewer_client.set_billboard("tracker", mode="aim", face="+y")
    settle(viewer_client)
    frames(viewer_page)

    for position in ([10, 0, 0], [0, 12, 3], [-6, -6, 8]):
        viewer_client.set_camera(position=position, target=[0, 0, 0])
        settle(viewer_client)
        frames(viewer_page)
        res = viewer_page.evaluate(_BB_PROBE_JS, "tracker")
        assert res["finite"] is True
        assert np.allclose(res["faceWorld"], res["toCamera"], atol=1e-5), (
            f"camera at {position}: face {res['faceWorld']} != toCamera {res['toCamera']}"
        )


@pytest.mark.browser
def test_billboard_wheel_keeps_spinning_browser(viewer_client, viewer_page):
    """The headline case: an animated wheel spins while its axle tracks the camera.

    Covers the `matrixAutoUpdate === false` write path — an object driven by a
    binary `transforms` channel. Before this change the viewer wrote
    `obj.quaternion`, which that path ignores, so an animated billboard did not
    billboard at all.
    """
    viewer_client.add_group("wheel")
    # An off-axis marker makes the spin about the axle observable.
    viewer_client.add_box("wheel_marker", width=0.2, position=[1, 0, 0], parent="wheel")
    viewer_client.load_animation(_spin_animation("wheel"), loop=True)
    viewer_client.set_billboard("wheel", mode="aim", face="+y")
    settle(viewer_client)
    frames(viewer_page, 2)

    first = viewer_page.evaluate(_BB_PROBE_JS, "wheel")
    marker_first = viewer_page.evaluate(_BB_PROBE_JS, "wheel_marker")
    time.sleep(0.4)  # let the animation clock advance (wall-clock playback)
    frames(viewer_page, 2)
    second = viewer_page.evaluate(_BB_PROBE_JS, "wheel")
    marker_second = viewer_page.evaluate(_BB_PROBE_JS, "wheel_marker")

    # The axle points at the camera at both instants...
    for probe in (first, second):
        assert probe["finite"] is True
        assert np.allclose(probe["faceWorld"], probe["toCamera"], atol=1e-5), (
            "the wheel's axle is not aimed at the camera"
        )
    # ...while the wheel keeps turning about it (the marker moved).
    assert not np.allclose(marker_first["pos"], marker_second["pos"], atol=1e-3), (
        "the wheel stopped spinning — the billboard overwrote its own rotation"
    )


@pytest.mark.browser
def test_billboard_stable_while_animation_paused_browser(viewer_client, viewer_page):
    """Paused animation + orbit is what makes a read-back implementation blow up.

    `_applyFrame` only runs while playing, so nothing refreshes `obj.matrix`
    between frames; a viewer that recovered R_own from the object would
    integrate a fresh aim every frame into runaway rotation.
    """
    viewer_client.add_group("paused_wheel")
    viewer_client.add_box(
        "paused_marker", width=0.2, position=[1, 0, 0], parent="paused_wheel"
    )
    viewer_client.load_animation(_spin_animation("paused_wheel"), autoplay=False)
    viewer_client.set_billboard("paused_wheel", mode="aim", face="+y")
    viewer_client.set_camera(position=[8, 0, 0], target=[0, 0, 0])
    settle(viewer_client)
    frames(viewer_page, 2)
    early = viewer_page.evaluate(_BB_PROBE_JS, "paused_wheel")
    frames(viewer_page, 40)
    late = viewer_page.evaluate(_BB_PROBE_JS, "paused_wheel")

    assert np.allclose(early["quat"], late["quat"], atol=1e-6), (
        f"paused billboard drifted: {early['quat']} -> {late['quat']}"
    )
    assert np.allclose(late["faceWorld"], late["toCamera"], atol=1e-5)


@pytest.mark.browser
def test_billboard_pivot_holds_anchor_point_browser(viewer_client, viewer_page):
    """A pivot keeps that local point fixed while the object swings about it."""
    viewer_client.add_box("post", width=0.4, height=2.0, depth=0.4, position=[0, 0, 1])
    # Pivot at the foot of the box, in its own local coordinates.
    viewer_client.set_billboard("post", mode="aim", face="+z", pivot=[0, -1.0, 0])
    settle(viewer_client)
    frames(viewer_page)

    anchor_js = """() => {
        const v = window.threejsViewer;
        const THREE = window.tjsv.THREE;
        const o = v._objects.get('post');
        o.updateWorldMatrix(true, false);
        return new THREE.Vector3(0, -1, 0).applyMatrix4(o.matrixWorld).toArray();
    }"""
    anchors = []
    for position in ([9, 0, 2], [0, 9, 2], [-7, -7, 5]):
        viewer_client.set_camera(position=position, target=[0, 0, 1])
        settle(viewer_client)
        frames(viewer_page)
        anchors.append(viewer_page.evaluate(anchor_js))

    for later in anchors[1:]:
        assert np.allclose(anchors[0], later, atol=1e-5), (
            f"pivot point moved while re-aiming: {anchors[0]} -> {later}"
        )


@pytest.mark.browser
def test_billboard_disable_restores_pose_browser(viewer_client, viewer_page):
    """enabled=False restores the producer's pose exactly."""
    viewer_client.add_box("revert", position=[1, 2, 3], rotation=[0.3, 0.4, 0.5])
    settle(viewer_client)
    frames(viewer_page)
    before = viewer_page.evaluate(_BB_PROBE_JS, "revert")

    viewer_client.set_billboard("revert", mode="aim", face="+y", pivot=[0, 0.5, 0])
    settle(viewer_client)
    frames(viewer_page, 3)
    during = viewer_page.evaluate(_BB_PROBE_JS, "revert")
    assert not np.allclose(before["quat"], during["quat"], atol=1e-4), (
        "the billboard never changed the pose, so restoring it proves nothing"
    )

    viewer_client.set_billboard("revert", enabled=False)
    settle(viewer_client)
    frames(viewer_page, 3)
    after = viewer_page.evaluate(_BB_PROBE_JS, "revert")
    assert np.allclose(before["quat"], after["quat"], atol=1e-6)
    assert np.allclose(before["pos"], after["pos"], atol=1e-6)


@pytest.mark.browser
def test_billboard_nested_has_no_parent_lag_browser(viewer_client, viewer_page):
    """A billboard under a billboard resolves in one frame, not two.

    The child is registered FIRST on purpose: with plain insertion order it
    would read its parent's previous-frame world quaternion and lag behind
    whenever the camera moves.
    """
    viewer_client.add_group("outer")
    viewer_client.add_billboard("inner", parent="outer", position=[0, 0, 1])
    # Registered after the child, so insertion order is child-then-parent.
    viewer_client.set_billboard("outer", mode="camera")
    settle(viewer_client)
    frames(viewer_page)

    viewer_client.set_camera(position=[7, -5, 4], target=[0, 0, 0])
    settle(viewer_client)
    frames(viewer_page, 1)  # exactly one frame: a lag would still be visible

    res = viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            const THREE = window.tjsv.THREE;
            const o = v._objects.get('inner');
            o.updateWorldMatrix(true, false);
            return {
                child: o.getWorldQuaternion(new THREE.Quaternion()).toArray(),
                cam: v._camera.getWorldQuaternion(new THREE.Quaternion()).toArray(),
            };
        }"""
    )
    assert np.allclose(res["child"], res["cam"], atol=1e-5), (
        "nested billboard lagged its billboard parent by a frame"
    )


@pytest.mark.browser
def test_billboard_hinge_holds_axis_browser(viewer_client, viewer_page):
    """A hinge billboard spins about its axis only, keeping it on hinge_world."""
    viewer_client.add_box("hinged", position=[0, 0, 0])
    viewer_client.set_billboard(
        "hinged", mode="hinge", hinge="+y", hinge_world=[0, 0, 1], face="+z"
    )
    settle(viewer_client)

    for position in ([9, 1, 2], [-4, 8, 6]):
        viewer_client.set_camera(position=position, target=[0, 0, 0])
        settle(viewer_client)
        frames(viewer_page)
        res = viewer_page.evaluate(
            """() => {
                const v = window.threejsViewer;
                const THREE = window.tjsv.THREE;
                const o = v._objects.get('hinged');
                o.updateWorldMatrix(true, false);
                const q = o.getWorldQuaternion(new THREE.Quaternion());
                return {
                    hingeWorld: new THREE.Vector3(0, 1, 0).applyQuaternion(q).toArray(),
                    faceWorld: new THREE.Vector3(0, 0, 1).applyQuaternion(q).toArray(),
                    finite: o.matrixWorld.elements.every(Number.isFinite),
                };
            }"""
        )
        assert res["finite"] is True
        # The local hinge stays pinned to the requested world direction...
        assert np.allclose(res["hingeWorld"], [0, 0, 1], atol=1e-5)
        # ...and the face turns toward the camera in the plane perpendicular
        # to it (compare only the in-plane part, which is all 1 DOF can fix).
        face = np.array(res["faceWorld"])[:2]
        to_cam = np.array(position, dtype=float)[:2]
        face /= np.linalg.norm(face)
        to_cam /= np.linalg.norm(to_cam)
        assert np.allclose(face, to_cam, atol=1e-4)


@pytest.mark.browser
def test_billboard_pinned_hinge_absorbs_own_rotation_browser(
    viewer_client, viewer_page
):
    """`hinge` + `hinge_world` leaves the own rotation with nowhere to go.

    Aligning the hinge onto `hinge_world` and then solving the spin to aim
    `face` fixes every degree of freedom, so the pose is a function of
    (hinge_world, face, camera) alone. Two objects with wildly different own
    rotations must land on the same pose — the property the demo's "still"
    donuts are showing.
    """
    for name, rotation in (("pinned_a", [0, 0, 0]), ("pinned_b", [0.9, 0.4, 1.3])):
        viewer_client.add_box(name, position=[0, 0, 0], rotation=rotation)
        viewer_client.set_billboard(
            name, mode="hinge", hinge="+y", hinge_world=[0, 0, 1], face="+z"
        )
    viewer_client.set_camera(position=[6, -5, 3], target=[0, 0, 0])
    settle(viewer_client)
    frames(viewer_page, 2)

    a = viewer_page.evaluate(_BB_PROBE_JS, "pinned_a")
    b = viewer_page.evaluate(_BB_PROBE_JS, "pinned_b")
    assert a is not None and b is not None
    assert np.allclose(a["quat"], b["quat"], atol=1e-5), (
        f"pinned hinge should not depend on the object's own rotation: "
        f"{a['quat']} vs {b['quat']}"
    )


@pytest.mark.browser
def test_billboard_degenerate_hinge_holds_pose_browser(viewer_client, viewer_page):
    """face ∥ hinge cannot be aimed by a spin: hold a finite pose, don't NaN."""
    viewer_client.add_box("degenerate", position=[0, 0, 0])
    viewer_client.set_billboard("degenerate", mode="hinge", hinge="+z", face="+z")
    settle(viewer_client)
    frames(viewer_page, 3)
    res = viewer_page.evaluate(_BB_PROBE_JS, "degenerate")
    assert res["finite"] is True
    assert all(np.isfinite(res["quat"]))


@pytest.mark.browser
def test_add_points_appears_in_scene(viewer_client, viewer_page):
    """add_points creates a THREE.Points cloud in the browser scene graph."""
    pts = np.random.default_rng(0).random((500, 3)).astype(np.float32)
    scalars = pts[:, 2]
    viewer_client.add_points("cloud", pts, colors=scalars, colormap="turbo")
    obj = None
    for _ in range(40):
        settle(viewer_client)
        objects = viewer_client.query_scene()["objects"]
        if "cloud" in objects:
            obj = objects["cloud"]
            break
    assert obj is not None, "point cloud never landed in the scene"
    assert obj["type"] == "Points"


_POINTS_STATE_JS = """
(id) => {
    const o = window.threejsViewer._objects.get(id);
    if (!o) return null;
    const p = o.geometry.getAttribute('position');
    const n = o.userData.totalPointCount;
    return {
        count: n,
        capacity: p.count,
        drawn: o.geometry.drawRange.count,
        colorCapacity: o.geometry.getAttribute('color')?.count ?? 0,
        last: Array.from(p.array.slice((n - 1) * 3, n * 3)),
        radius: o.geometry.boundingSphere ? o.geometry.boundingSphere.radius : null,
    };
}
"""


@pytest.mark.browser
def test_append_points_grows_the_cloud_in_place(viewer_client, viewer_page):
    """append_points grows an existing THREE.Points without re-uploading it:
    the logical count grows, the buffer keeps its geometric spare capacity,
    the full-cloud draw range follows, and the last chunk lands at the tail."""
    rng = np.random.default_rng(2)
    pts = rng.random((100, 3)).astype(np.float32)
    viewer_client.add_points("cloud", pts, colors=pts[:, 2], colormap="turbo")
    for _ in range(40):
        settle(viewer_client)
        if "cloud" in viewer_client.query_scene()["objects"]:
            break
    chunk = None
    for _ in range(3):
        chunk = rng.random((50, 3)).astype(np.float32)
        viewer_client.append_points("cloud", chunk, colors=chunk[:, 2])

    state = None
    for _ in range(40):
        time.sleep(0.05)
        state = viewer_page.evaluate(_POINTS_STATE_JS, "cloud")
        if state and state["count"] == 250:
            break
    assert state is not None and state["count"] == 250, f"got {state!r}"
    # One geometric growth to the 1024 floor covered all three appends —
    # the existing points were copied once, not per append.
    assert state["capacity"] == 1024
    assert state["colorCapacity"] == 1024
    assert state["drawn"] == 250
    # Buffer order is send order: the last append sits at the tail.
    assert state["last"] == pytest.approx(chunk[-1].tolist(), abs=1e-6)
    # Bounds were expanded (not left stale) so culling/framing see the growth.
    assert state["radius"] is not None and state["radius"] > 0


@pytest.mark.browser
def test_append_points_bounds_stay_incremental_across_a_capacity_growth(
    viewer_client, viewer_page
):
    """Growing capacity moves the data into a fresh BufferGeometry. If that
    drops the bounds, the next expandPointsBounds() re-seeds its box with an
    O(count) scan of the whole cloud — so the one append that happens to
    trigger a doubling costs a full re-scan, a latency spike on exactly the
    multi-million-point stream this path exists for.

    Only observable by how much work it does, so count Box3.expandByPoint:
    one call per *new* point when the bounds carried over, one per existing
    point on top of that when they did not.
    """
    rng = np.random.default_rng(7)
    pts = rng.random((100, 3)).astype(np.float32)
    viewer_client.add_points("cloud", pts)
    for _ in range(40):
        settle(viewer_client)
        if "cloud" in viewer_client.query_scene()["objects"]:
            break
    # First append: grows 100 -> the 1024 floor, and legitimately seeds the
    # box (there is no prior boundingBox to carry) — not what is measured.
    viewer_client.append_points("cloud", rng.random((50, 3)).astype(np.float32))
    for _ in range(40):
        time.sleep(0.05)
        state = viewer_page.evaluate(_POINTS_STATE_JS, "cloud")
        if state and state["count"] == 150:
            break

    # Count expandByPoint from here. Box3 is not exposed globally; reach its
    # prototype through the box the viewer already built.
    viewer_page.evaluate(
        "() => {"
        " const g = window.threejsViewer._objects.get('cloud').geometry;"
        " const proto = Object.getPrototypeOf(g.boundingBox);"
        " if (!proto.__origExpand) proto.__origExpand = proto.expandByPoint;"
        " window.__expandCalls = 0;"
        " proto.expandByPoint = function (p) {"
        "  window.__expandCalls++; return proto.__origExpand.call(this, p);"
        " };"
        "}"
    )
    # 150 + 900 = 1050 > 1024, so this append triggers the second growth.
    viewer_client.append_points("cloud", rng.random((900, 3)).astype(np.float32))
    state = None
    for _ in range(60):
        time.sleep(0.05)
        state = viewer_page.evaluate(_POINTS_STATE_JS, "cloud")
        if state and state["count"] == 1050:
            break
    assert state is not None and state["count"] == 1050, f"got {state!r}"
    assert state["capacity"] == 2048, "this append must have grown the buffer"

    calls = viewer_page.evaluate(
        "() => {"
        " const g = window.threejsViewer._objects.get('cloud').geometry;"
        " const proto = Object.getPrototypeOf(g.boundingBox);"
        " if (proto.__origExpand) proto.expandByPoint = proto.__origExpand;"
        " return window.__expandCalls;"
        "}"
    )
    # 900 (plus a handful from scene-bounds bookkeeping, which expands by a
    # box's 8 corners) with the bounds carried across; 150 + 900 without.
    assert 900 <= calls < 1000, (
        f"bounds were re-seeded across the growth: {calls} expandByPoint calls "
        f"for a 900-point append onto a 150-point cloud"
    )
    assert state["radius"] is not None and state["radius"] > 0


@pytest.mark.browser
def test_append_points_to_unknown_id_warns(viewer_client, viewer_page):
    """An append whose target the viewer does not have warns loudly instead
    of silently dropping stream data (the Python client raises before this
    for ids it never created, so drive handleMessage directly)."""
    warnings = []
    viewer_page.on(
        "console",
        lambda msg: warnings.append(msg.text) if msg.type == "warning" else None,
    )
    viewer_page.evaluate(
        "() => window.threejsViewer.handleMessage("
        "{type: 'append_points_binary', id: 'ghost', numPoints: 1,"
        " hasVertexColors: false, blob_url: 'http://127.0.0.1:1/nope'})"
    )
    for _ in range(40):
        time.sleep(0.05)
        if any("append_points: 'ghost'" in w for w in warnings):
            break
    assert any("append_points: 'ghost'" in w for w in warnings), warnings
    assert "ghost" not in viewer_client.query_scene()["objects"]


@pytest.mark.browser
def test_set_draw_range_on_points(viewer_client, viewer_page):
    """set_draw_range reveals a leading fraction of a point cloud."""
    pts = np.random.default_rng(1).random((1000, 3)).astype(np.float32)
    viewer_client.add_points("cloud", pts)
    viewer_client.set_draw_range("cloud", 0.5)
    dr = None
    for _ in range(40):
        settle(viewer_client)
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
        settle(viewer_client)
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
        settle(viewer_client)
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
        settle(viewer_client)
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
        settle(viewer_client)
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


def _scale_animated_glb() -> bytes:
    """The `_two_triangle_glb` mesh plus one embedded animation clip: a
    2-second LINEAR scale track on the (named) mesh node, keyframes
    scale=0.059 at t=0 and scale=0.857 at t=2 (mirroring the compressed ->
    extended bellows clips from issue #135). The node itself carries NO
    scale property, so its authored bind pose is scale 1.0 — distinct from
    every keyframe, which lets tests tell "bind pose held" apart from
    "bound to t=0"."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)
    times = np.array([0.0, 2.0], dtype=np.float32)
    scales = np.array([[0.059] * 3, [0.857] * 3], dtype=np.float32)
    bin_chunk = positions.tobytes() + indices.tobytes()
    bin_chunk += b"\x00" * (-len(bin_chunk) % 4)  # 4-align the float views
    times_off = len(bin_chunk)
    bin_chunk += times.tobytes()
    scales_off = len(bin_chunk)
    bin_chunk += scales.tobytes()
    bin_chunk += b"\x00" * (-len(bin_chunk) % 4)
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "anode"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "animations": [
            {
                "samplers": [{"input": 2, "output": 3, "interpolation": "LINEAR"}],
                "channels": [{"sampler": 0, "target": {"node": 0, "path": "scale"}}],
            }
        ],
        "buffers": [{"byteLength": len(bin_chunk)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 48, "target": 34962},
            {"buffer": 0, "byteOffset": 48, "byteLength": 12, "target": 34963},
            {"buffer": 0, "byteOffset": times_off, "byteLength": 8},
            {"buffer": 0, "byteOffset": scales_off, "byteLength": 24},
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
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": 2,
                "type": "SCALAR",
                "min": [0.0],
                "max": [2.0],
            },
            {"bufferView": 3, "componentType": 5126, "count": 2, "type": "VEC3"},
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


def _load_animated_glb(viewer_client, obj_id="anim"):
    viewer_client.add_model_binary(obj_id, _scale_animated_glb(), format="glb")
    objects = {}
    for _ in range(60):
        settle(viewer_client)
        objects = viewer_client.query_scene()["objects"]
        if obj_id in objects:
            return objects
    raise AssertionError("animated GLB did not load")


def _node_scale(viewer_page, obj_id="anim"):
    return viewer_page.evaluate(
        f"() => window.threejsViewer._objects.get('{obj_id}')"
        ".getObjectByName('anode').scale.x"
    )


def _wait_for_scale(viewer_page, expected, obj_id="anim", tol=1e-3):
    scale = None
    for _ in range(40):
        time.sleep(0.05)
        scale = _node_scale(viewer_page, obj_id)
        if abs(scale - expected) < tol:
            return scale
    raise AssertionError(f"expected node scale ~{expected}, got {scale!r}")


@pytest.mark.browser
def test_animated_glb_holds_bind_pose_until_driven(viewer_client, viewer_page):
    """An animated GLB must NOT deform on load (issue #135): the mixer's
    actions are created but not played, so the authored bind pose (scale 1.0,
    no keyframe has that value) holds until the first clip-drive message.
    set_clip_progress then maps 0.0 -> the t=0 keyframe and 1.0 -> the end
    keyframe (not wrapped back to 0 by LoopRepeat)."""
    objects = _load_animated_glb(viewer_client)

    # Bind pose held: 1.0, NOT the t=0 keyframe (0.059) the old eager
    # play()+setTime(0) statically bound.
    assert abs(_node_scale(viewer_page) - 1.0) < 1e-6

    # query_scene reports the clip durations for discovery (additive field).
    assert objects["anim"].get("clipDurations") == [2.0]

    viewer_client.set_clip_progress("anim", 0.0)
    _wait_for_scale(viewer_page, 0.059)
    viewer_client.set_clip_progress("anim", 1.0)
    _wait_for_scale(viewer_page, 0.857)
    # Out-of-range input clamps to [0, 1].
    viewer_client.set_clip_progress("anim", -3.0)
    _wait_for_scale(viewer_page, 0.059)


@pytest.mark.browser
def test_set_clip_progress_maps_to_clip_duration(viewer_client, viewer_page):
    """set_clip_progress(0.5) seeks to 0.5 * the clip's own duration (2 s ->
    t=1 s -> mid pose), i.e. the same pose as set_clip_time(id, 1.0) — and a
    driven model behaves exactly like the pre-#135 eager bind thereafter."""
    _load_animated_glb(viewer_client)

    viewer_client.set_clip_progress("anim", 0.5)
    _wait_for_scale(viewer_page, (0.059 + 0.857) / 2)  # t=1.0 of a 2 s clip

    # Absolute-seconds seek still works identically once driven.
    viewer_client.set_clip_time("anim", 0.0)
    _wait_for_scale(viewer_page, 0.059)
    viewer_client.set_clip_time("anim", 1.0)
    _wait_for_scale(viewer_page, (0.059 + 0.857) / 2)


@pytest.mark.browser
def test_set_clip_time_clamps_at_clip_end(viewer_client, viewer_page):
    """set_clip_time(duration) lands on the END pose (issue #143): the
    LoopRepeat actions used to wrap `time == duration` modulo the clip back
    to the t=0 pose, snapping a fully-driven drag chain to fully-compressed.
    Absolute seeks now clamp per clip to [0, duration): past-the-end holds
    the end pose, negative holds the start pose."""
    _load_animated_glb(viewer_client)

    viewer_client.set_clip_time("anim", 2.0)  # exactly the clip duration
    _wait_for_scale(viewer_page, 0.857)  # end pose, NOT wrapped to 0.059
    viewer_client.set_clip_time("anim", 5.0)  # past the end: still end pose
    _wait_for_scale(viewer_page, 0.857)
    viewer_client.set_clip_time("anim", -1.0)  # before the start: start pose
    _wait_for_scale(viewer_page, 0.059)


@pytest.mark.browser
def test_binary_clip_times_channel_drives_deferred_mixer(viewer_client, viewer_page):
    """The binary clip_times animation channel is a first drive too: applying
    it to a freshly-loaded (bind-pose-held) model activates the deferred
    actions and seeks the clip, exactly as before issue #135."""
    _load_animated_glb(viewer_client)
    assert abs(_node_scale(viewer_page) - 1.0) < 1e-6  # still bind pose

    anim = Animation(loop=False)
    anim.set_frame_times(np.array([0.0, 2.0], dtype=np.float32))
    # Clip time ramps 0 -> 2 s with the playhead.
    anim.set_clip_time_data(["anim"], np.array([[0.0], [2.0]]))
    viewer_client.load_animation(anim, autoplay=False)
    loaded = False
    for _ in range(40):
        time.sleep(0.05)
        if viewer_page.evaluate("() => window.threejsViewer._animation != null"):
            loaded = True
            break
    assert loaded, "animation never loaded"

    viewer_page.evaluate("() => window.threejsViewer._seekToTime(1.0)")
    _wait_for_scale(viewer_page, (0.059 + 0.857) / 2)  # clip time 1.0 of 2 s

    # Sweeping the channel to exactly the clip duration lands on the end
    # pose instead of wrapping to t=0 (issue #143).
    viewer_page.evaluate("() => window.threejsViewer._seekToTime(2.0)")
    _wait_for_scale(viewer_page, 0.857)


@pytest.mark.browser
def test_clear_scene(viewer_client, viewer_page):
    """clear() removes all objects."""
    viewer_client.add_box("a")
    viewer_client.add_sphere("b")
    settle(viewer_client)
    viewer_client.clear()
    settle(viewer_client)
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
        settle(viewer_client)
        result = viewer_client.query_scene()
        if result["meta"]["animation"]["playing"]:
            break
    assert result["meta"]["animation"]["playing"] is True, "Animation did not start"
    viewer_client.unload_animation()
    settle(viewer_client)
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
def test_nonfinite_matrix_holds_last_good_pose(viewer_client, viewer_page):
    """A NaN keyframe must not make the object vanish (issue #162).

    A single non-finite element poisons matrixWorld for the whole subtree and
    three.js drops the object outright. The guard skips the bad matrix, so the
    object stays visible at its last good pose.
    """
    viewer_client.add_box("nanbox")
    settle(viewer_client)
    eye = np.eye(4, dtype=np.float32).flatten(order="F")
    data = np.tile(eye, (3, 1, 1)).astype(np.float32)  # (3 frames, 1 obj, 16)
    data[1, 0, 12] = 5.0  # x = 5 at t=1 — the last good pose
    data[2, 0, :] = np.nan  # poisoned keyframe at t=2
    anim = Animation(loop=False)
    anim.set_frame_times(np.array([0.0, 1.0, 2.0]))
    anim.set_transform_data(["nanbox"], data)
    viewer_client.load_animation(anim, autoplay=False, initial_time=1.0)
    _wait_for_animation_loaded(viewer_page)
    frames(viewer_page)
    assert _world_position(viewer_page, "nanbox") == pytest.approx(
        [5.0, 0.0, 0.0], abs=1e-4
    ), "setup: object should sit at x=5 on the last good keyframe"

    # Between the good and the NaN keyframe the lerp result is NaN...
    viewer_page.evaluate("() => window.threejsViewer._seekToTime(1.5)")
    frames(viewer_page)
    pos = _world_position(viewer_page, "nanbox")
    assert all(math.isfinite(c) for c in pos), f"NaN leaked into matrixWorld: {pos}"
    assert pos == pytest.approx([5.0, 0.0, 0.0], abs=1e-4)

    # ...and landing exactly on the NaN keyframe holds it too.
    viewer_page.evaluate("() => window.threejsViewer._seekToTime(2.0)")
    frames(viewer_page)
    pos = _world_position(viewer_page, "nanbox")
    assert all(math.isfinite(c) for c in pos), f"NaN leaked into matrixWorld: {pos}"
    assert pos == pytest.approx([5.0, 0.0, 0.0], abs=1e-4)

    # The object is still in the scene and visible (not dropped by three.js).
    state = viewer_page.evaluate(
        "() => { const o = window.threejsViewer._objects.get('nanbox');"
        " return { visible: o.visible,"
        "          finite: o.matrix.elements.every(Number.isFinite),"
        "          warned: !!o.userData.__warnedNonFiniteMatrix }; }"
    )
    assert state["visible"] is True
    assert state["finite"] is True
    assert state["warned"] is True, "expected the one-time non-finite warning"


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
        settle(viewer_client)
    assert viewer_client.query_scene()["meta"]["animation"]["playing"] is True

    viewer_client.pause_animation()
    settle(viewer_client)
    assert viewer_client.query_scene()["meta"]["animation"]["playing"] is False

    viewer_client.resume_animation()
    settle(viewer_client)
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
        settle(viewer_client)
        result = viewer_client.query_scene()
        if result["meta"]["animation"]["playing"]:
            break
    assert result["meta"]["animation"]["playing"] is True, "Animation did not start"
    viewer_client.clear()
    settle(viewer_client)
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
def test_load_progress_and_assets_loaded_hooks(viewer_client, viewer_page):
    """Issue #163: `viewer.onLoadProgress(cb)` / `viewer.onAssetsLoaded(cb)` /
    `viewer.getLoadState()` are the public replacement for polling the private
    `_pendingFetches` counter. Progress fires on every fetch start and end with
    the running batch counters; assets-loaded fires when the batch drains AND
    the producer has sent `mark_assets_complete` — and, crucially, it must fire
    on the direct `handleMessage()` path with no WebSocket, which the old
    WS-OPEN gate on `_maybeNotifyAssetsLoaded` blocked. Both return an
    unsubscribe fn. Two 404 blob fetches drive the counters (the fetch is
    balanced in a finally block regardless of the HTTP status)."""
    url = f"http://{viewer_client.host}:{viewer_client._http_port}/no_such_blob"
    started = viewer_page.evaluate(
        """(url) => {
            const v = window.threejsViewer;
            window.__progress = [];
            window.__complete = [];
            window.__offProgress = v.onLoadProgress((s) => window.__progress.push(s));
            window.__offComplete = v.onAssetsLoaded((s) => window.__complete.push(s));
            // Detach the socket: the embedder case is a page with no WS at all.
            window.__ws = v._ws;
            v._ws = null;
            v.handleMessage({ type: 'add_model_binary', id: 'lp_a', format: 'glb', blob_url: url });
            v.handleMessage({ type: 'add_model_binary', id: 'lp_b', format: 'glb', blob_url: url });
            return v.getLoadState();
        }""",
        url,
    )
    # Two fetches registered synchronously: the batch total counts what has
    # been *seen*, and nothing has landed yet.
    assert started == {"loaded": 0, "total": 2, "pending": 2, "assetsComplete": False}

    for _ in range(100):
        viewer_page.wait_for_timeout(100)
        if (
            viewer_page.evaluate("() => window.threejsViewer.getLoadState().pending")
            == 0
        ):
            break

    result = viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            const drained = v.getLoadState();
            const completeBeforeMark = window.__complete.length;
            // The completion marker, fed straight in with no socket open.
            v.handleMessage({ type: 'mark_assets_complete' });
            const afterMark = window.__complete.slice();
            window.__offProgress();
            window.__offComplete();
            v.handleMessage({ type: 'mark_assets_complete' });
            const progressCount = window.__progress.length;
            v.handleMessage({ type: 'add_model_binary', id: 'lp_c', format: 'glb',
                              blob_url: 'http://127.0.0.1:1/nope' });
            return {
                drained,
                completeBeforeMark,
                afterMark,
                progressCount,
                progressAfterUnsub: window.__progress.length,
                completeAfterUnsub: window.__complete.length,
                progress: window.__progress,
            };
        }"""
    )
    assert result["drained"] == {
        "loaded": 2,
        "total": 2,
        "pending": 0,
        "assetsComplete": False,
    }
    # Four progress events: start, start, end, end.
    assert result["progressCount"] == 4
    assert [
        (p["loaded"], p["total"], p["pending"]) for p in result["progress"][:4]
    ] == [
        (0, 1, 1),
        (0, 2, 2),
        (1, 2, 1),
        (2, 2, 0),
    ]
    # No completion until the producer says the stream is done.
    assert result["completeBeforeMark"] == 0
    assert len(result["afterMark"]) == 1
    assert result["afterMark"][0]["assetsComplete"] is True
    # Unsubscribe stops both hooks.
    assert result["completeAfterUnsub"] == 1
    assert result["progressAfterUnsub"] == 4

    viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            v._ws = window.__ws;
            v._assetsComplete = false;
            v.handleMessage({ type: 'delete_object', id: 'lp_a' });
            v.handleMessage({ type: 'delete_object', id: 'lp_b' });
            v.handleMessage({ type: 'delete_object', id: 'lp_c' });
        }"""
    )


@pytest.mark.browser
def test_wait_for_assets_replies_to_every_marker(viewer_client, viewer_page):
    """`assets_loaded` is a reply, not a one-shot event, and must not be
    deduplicated on the "nothing changed since last time" reasoning.

    wait_for_assets() clears its event, sends mark_assets_complete and blocks
    until the viewer answers. A script that stages its scene in two passes
    calls it twice with no new assets in between — suppressing the second
    reply would hang it forever (up to the timeout).
    """
    viewer_client.add_box("wfa", width=1.0, height=1.0, depth=1.0)
    viewer_client.wait_for_assets(timeout=15, disconnect=False)
    # Second pass, deliberately with no new assets: still must be answered.
    viewer_client.wait_for_assets(timeout=15, disconnect=False)
    # And once more after actual new work, the ordinary case.
    viewer_client.add_box("wfa2", width=1.0, height=1.0, depth=1.0)
    viewer_client.wait_for_assets(timeout=15, disconnect=False)
    assert "wfa2" in viewer_client.query_scene()["objects"]
    viewer_client.delete("wfa")
    viewer_client.delete("wfa2")


@pytest.mark.browser
def test_on_unknown_message_hook(viewer_client, viewer_page):
    """`viewer.onUnknownMessage(cb)` (issue #145) is the sanctioned hook for
    application-defined message types: it fires from handleMessage's default
    branch with the parsed message (so it also covers the direct
    `handleMessage()` embedder path), suppresses the "Unknown message type"
    console.warn while registered, returns an unsubscribe function, and never
    fires for known message types."""
    result = viewer_page.evaluate(
        """async () => {
            const v = window.threejsViewer;
            const seen = [];
            const warns = [];
            const origWarn = console.warn;
            console.warn = (...args) => { warns.push(args.join(' ')); };
            try {
                const off = v.onUnknownMessage((data) => seen.push(data));
                // Unknown type: hook fires with the parsed payload, warn suppressed.
                await v.handleMessage({ type: 'totally_custom', payload: 42 });
                // Known type: must NOT fire the hook.
                await v.handleMessage({ type: 'add_group', id: 'unk_group' });
                const seenAfterKnown = seen.length;
                // Unsubscribe: hook stops firing, warn comes back.
                off();
                await v.handleMessage({ type: 'totally_custom_2' });
                return {
                    seen,
                    seenAfterKnown,
                    finalSeen: seen.length,
                    warns: warns.filter((w) => w.includes('Unknown message type')),
                };
            } finally {
                console.warn = origWarn;
                await v.handleMessage({ type: 'delete_object', id: 'unk_group' });
            }
        }"""
    )
    assert result["seen"] == [{"type": "totally_custom", "payload": 42}]
    assert result["seenAfterKnown"] == 1  # known types never fire the hook
    assert result["finalSeen"] == 1  # unsubscribe stops delivery
    # Warn suppressed while registered; restored after unsubscribe.
    assert result["warns"] == ["Unknown message type: 'totally_custom_2'"]


@pytest.mark.browser
def test_show_grid(viewer_client, viewer_page):
    """show_grid() toggles grid visibility."""
    settle(viewer_client)
    meta = viewer_client.query_scene()["meta"]
    # Grid is hidden by default
    assert meta["grid"]["visible"] is False

    viewer_client.show_grid(visible=True)
    settle(viewer_client)
    meta = viewer_client.query_scene()["meta"]
    assert meta["grid"]["visible"] is True

    viewer_client.show_grid(visible=False)
    settle(viewer_client)
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
    settle(viewer_client)
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
    frames(viewer_page)
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
    settle(viewer_client)
    get_mode = "() => window.threejsViewer._shading.shadingMode"
    assert viewer_page.evaluate(get_mode) == 0

    for want in [1, 2, 3, 0]:
        _press_key(viewer_page, "KeyN")
        frames(viewer_page)
        assert viewer_page.evaluate(get_mode) == want


@pytest.mark.browser
def test_m_and_n_compose(viewer_client, viewer_page):
    """M and N modes are independent and compose."""
    viewer_client.add_box("compose_box")
    settle(viewer_client)
    _press_key(viewer_page, "KeyM")  # wireframe = 1
    _press_key(viewer_page, "KeyN")  # shading = 1
    frames(viewer_page)
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
    frames(viewer_page)
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


# --- Framing excludes TransformControls gizmo helpers (issue #144) ---


@pytest.mark.browser
def test_frame_all_excludes_move_gizmo(viewer_client, viewer_page):
    """An attached move gizmo must not inflate frameAll bounds.

    The TransformControls helper carries a ~±50k drag plane whose *material*
    is invisible but whose object is visible, so the frameable-bounds
    traversal used to union it and fly the camera hundreds of km out.
    Covers both the primary interactive gizmo and a pinned add_gizmo extra
    (both tag their helper root with userData.isGizmoHelper).
    """
    viewer_client.add_box("box", width=1, height=1, depth=1, position=[0, 0, 0])
    viewer_client.enable_move_gizmo("box")
    viewer_client.add_gizmo("box")  # pinned extra gizmo leaks the same way
    _wait_for(
        viewer_page,
        "() => { const g = window.threejsViewer._transformGizmo;"
        " return g.enabled && g.objectId === 'box' && g._extra.length === 1; }",
    )
    state = viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            v.frameAll();
            const bbox = v._collectFrameableBounds();
            const size = bbox.max.clone().sub(bbox.min);
            return {
                dist: v._camera.position.distanceTo(v._controls.target),
                size: { x: size.x, y: size.y, z: size.z },
                helperVisible: v._transformGizmo.helper.visible,
                primaryTagged: v._transformGizmo.helper.userData.isGizmoHelper === true,
                extraTagged: v._transformGizmo._extra[0].helper.userData.isGizmoHelper === true,
            };
        }"""
    )
    assert state["helperVisible"] is True, state
    assert state["primaryTagged"] is True, state
    assert state["extraTagged"] is True, state
    # The frameable bbox must span the 1-unit box, not the ~±50k gizmo plane.
    for axis in ("x", "y", "z"):
        assert 0.5 < state["size"][axis] < 2, state
    # Camera stays a few units out, not ~250 km.
    assert state["dist"] < 20, state


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
        settle(viewer_client)
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
        settle(viewer_client)
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


# --- Orbit / P / Home button stack (issue #190) ---


@pytest.mark.browser
def test_view_buttons_stack_left_of_gimbal(viewer_client, viewer_page):
    """Orbit mode, P and Home sit in one vertical column left of the axis-bubble
    cluster, top to bottom, all the same size. Home used to be centred in
    the 128x128 ViewHelper square, in the middle of the six axis bubbles.
    The column may overlap the square's outer margin (the bubbles orbit its
    centre), so the guard is the square's left quarter, not its edge."""
    viewer_page.set_viewport_size({"width": 1000, "height": 700})
    frames(viewer_page)
    r = viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            const rect = (sel) => {
                const b = v.el.querySelector(sel).getBoundingClientRect();
                return {left: b.left, right: b.right, top: b.top,
                        bottom: b.bottom, width: b.width, height: b.height};
            };
            const dom = v._renderer.domElement.getBoundingClientRect();
            const dim = v._gizmoDim;
            return {
                orbit: rect('.tjsv-view-orbit'),
                proj: rect('.tjsv-view-proj'),
                home: rect('.tjsv-view-home'),
                gimbal: {left: dom.right - dim, top: dom.bottom - dim,
                         right: dom.right, bottom: dom.bottom},
            };
        }"""
    )
    orbit, proj, home, gimbal = r["orbit"], r["proj"], r["home"], r["gimbal"]
    for name, b in (("orbit", orbit), ("proj", proj), ("home", home)):
        assert abs(b["width"] - 28) < 1 and abs(b["height"] - 28) < 1, (name, b)
        clear_of_bubbles = gimbal["left"] + (gimbal["right"] - gimbal["left"]) / 4
        assert b["right"] <= clear_of_bubbles, f"{name} reaches the bubbles: {r}"
        assert b["bottom"] <= gimbal["bottom"] + 1, f"{name} below the gimbal: {r}"
    # One column: same left edge, ordered orbit above P above Home, no overlap.
    assert (
        abs(orbit["left"] - proj["left"]) < 1 and abs(proj["left"] - home["left"]) < 1
    ), r
    assert orbit["bottom"] <= proj["top"] + 1, r
    assert proj["bottom"] <= home["top"] + 1, r
    # Vertically centred on the gimbal square, level with the bubble cluster.
    column_mid = (orbit["top"] + home["bottom"]) / 2
    gimbal_mid = (gimbal["top"] + gimbal["bottom"]) / 2
    assert abs(column_mid - gimbal_mid) <= 2, (column_mid, gimbal_mid, r)


@pytest.mark.browser
def test_orbit_button_toggles_mode(viewer_client, viewer_page):
    """The orbit-mode button at the top of the stack flips turntable <-> free
    like the R key, and always shows the current mode (data-mode, `.free`
    accent, tooltip, and which glyph is displayed). The R key drives the same
    indicator, so a keyboard flip updates the button too."""
    frames(viewer_page)

    def snap():
        return viewer_page.evaluate(
            """() => {
                const v = window.threejsViewer;
                const b = v.el.querySelector('.tjsv-view-orbit');
                const shown = (sel) =>
                    getComputedStyle(b.querySelector(sel)).display !== 'none';
                return {
                    mode: v._orbitMode,
                    data: b.dataset.mode,
                    free: b.classList.contains('free'),
                    title: b.title,
                    turntableGlyph: shown('.tjsv-orbit-glyph-turntable'),
                    freeGlyph: shown('.tjsv-orbit-glyph-free'),
                };
            }"""
        )

    start = snap()
    assert start["mode"] == start["data"]
    assert start["free"] == (start["mode"] == "free")
    assert start["turntableGlyph"] != start["freeGlyph"]

    viewer_page.click(".tjsv-view-orbit")
    after_click = snap()
    assert after_click["mode"] != start["mode"]
    assert after_click["data"] == after_click["mode"]
    assert after_click["free"] == (after_click["mode"] == "free")
    assert after_click["turntableGlyph"] == (after_click["mode"] == "turntable")
    assert after_click["freeGlyph"] == (after_click["mode"] == "free")
    assert after_click["title"] != start["title"]
    expected_word = "Free" if after_click["mode"] == "free" else "Turntable"
    assert after_click["title"].startswith(f"Orbit: {expected_word}")

    _press_key(viewer_page, "KeyR")
    after_key = snap()
    assert after_key["mode"] == start["mode"]
    assert after_key["data"] == start["data"]
    assert after_key["free"] == start["free"]
    assert after_key["title"] == start["title"]


@pytest.mark.browser
def test_view_gimbal_arms_and_bubbles_restyled(viewer_client, viewer_page):
    """_configureViewHelper stretches the three arm meshes by _gizmoArmScale,
    moves the six bubble sprites out to the arm tips and shrinks them to
    _gizmoBaseScale. The click hit-test raycasts the live sprites, so a
    synthetic pointer at a bubble's projected screen position must still
    resolve that bubble after the restyle."""
    viewer_page.set_viewport_size({"width": 1000, "height": 700})
    frames(viewer_page)
    r = viewer_page.evaluate(
        """() => {
            const v = window.threejsViewer;
            const h = v._viewHelper;
            const arms = h.children.filter(
                (c) => c.isMesh && !(c.userData && c.userData.type));
            const sprites = h.userData.interactiveSprites;
            // Project the posX bubble the way ViewHelper.render does: the
            // helper's quaternion is the inverse camera rotation, and the
            // stock ortho camera spans -2..2 over the dim x dim viewport.
            const s = sprites.find((o) => o.userData.type === 'posX');
            const p = s.position.clone().applyQuaternion(h.quaternion);
            const dom = v._renderer.domElement.getBoundingClientRect();
            const dim = v._gizmoDim;
            const lift = v._gizmoLiftCss();
            const clientX = dom.right - dim + ((p.x + 2) / 4) * dim;
            const clientY = dom.bottom - dim - lift + ((2 - p.y) / 4) * dim;
            const hit = v._gizmoHitTest({clientX, clientY});
            return {
                armScales: arms.map((a) => [a.scale.x, a.scale.y, a.scale.z]),
                spriteDist: sprites.map((o) => o.position.length()),
                spriteScale: sprites.map((o) => o.scale.x),
                armScale: v._gizmoArmScale,
                baseScale: v._gizmoBaseScale,
                insideRect: hit.insideRect,
                hitType: hit.hit ? hit.hit.userData.type : null,
                projected: [p.x, p.y],
            };
        }"""
    )
    assert r["armScale"] == pytest.approx(1.0)
    assert r["baseScale"] == pytest.approx(1.12)
    assert len(r["armScales"]) == 3, r
    for sx, sy, sz in r["armScales"]:
        assert (sx, sy, sz) == pytest.approx((1.0, 1.0, 1.0)), r
    assert len(r["spriteDist"]) == 6, r
    assert r["spriteDist"] == pytest.approx([1.0] * 6), r
    assert r["spriteScale"] == pytest.approx([1.12] * 6), r
    # The bubble stays inside the helper's +-2 ortho frustum.
    assert max(abs(c) for c in r["projected"]) + 0.56 < 2, r
    assert r["insideRect"] and r["hitType"] == "posX", r


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
    client._start_servers(http_port=0)
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
            " const smaa = passes[passes.length - 1];"
            " const out = passes[passes.length - 2];"
            " const NO_BLENDING = 0;"  # THREE.NoBlending
            " return {"
            "  hasComposer: !!dc._composer,"
            "  ctxAlpha: gl.getContextAttributes().alpha,"
            "  canvasBg: v._renderer.domElement.style.backgroundColor,"
            "  outNoBlend: out && out.material"
            "   ? (out.material.blending === NO_BLENDING) : null,"
            "  smaaLast: !!(smaa && smaa === dc._smaaPass),"
            "  parityStable: dc._composer"
            "   ? (dc._composer.readBuffer === dc._composer.renderTarget2"
            "      && dc._composer.renderTarget2.depthTexture === dc._depthTexture) : null,"
            "  smaaNoBlend: smaa && smaa._materialBlend"
            "   ? (smaa._materialBlend.blending === NO_BLENDING) : null,"
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
    assert state["smaaLast"] is True, (
        "SMAA anti-aliasing pass must be the final composer pass (after OutputPass, "
        "so its luma edge detection sees display-referred colour)"
    )
    assert state["parityStable"] is True, (
        "composer read buffer must be renderTarget2 (the one carrying the EDL "
        "depth texture) between frames; an odd swap count per frame would flip "
        "it every other frame and render blank"
    )
    assert state["smaaNoBlend"] is True, (
        "SMAA blend quad must use NoBlending (writes to screen) so transparent "
        "background pixels are replaced, not blended over the prior frame"
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
def test_move_gizmo_palette_matches_view_helper(viewer_client, viewer_page):
    """The gizmo handles use three's ViewHelper axis colours (issue #191), so
    the X / Y / Z arrows and the corner-gimbal bubbles agree on what each
    axis looks like. Read off the lit arrow materials after a rendered
    frame, so the per-frame restyle has already run."""
    viewer_client.add_box("box")
    settle(viewer_client)  # WS barrier: the box is registered before the attach
    viewer_client.enable_move_gizmo("box")
    _wait_for(
        viewer_page,
        "() => window.threejsViewer._transformGizmo.objectId === 'box'",
    )
    frames(viewer_page)
    r = viewer_page.evaluate(
        """() => {
            const arrows = window.threejsViewer._transformGizmo.control
                ._gizmo.gizmo.translate.children;
            const hex = (name) => {
                const o = arrows.find((c) => c.name === name && c.userData.__litArrow);
                return o ? o.material.color.getHex() : null;
            };
            return { x: hex('X'), y: hex('Y'), z: hex('Z') };
        }"""
    )
    # three r183 ViewHelper.js axis colours.
    assert (r["x"], r["y"], r["z"]) == (0xFF4466, 0x88FF44, 0x4488FF), r


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


# A 3/4 view (the pose where the stock fat arrow pickers bulged in front of the
# flat XY chip and stole its hover — a top-down view faces the chip head-on and
# would not exhibit the shadowing).
_GIZMO_THREEQUARTER = """() => {
  const v = window.threejsViewer;
  v._camera.position.set(2.6, -2.6, 1.9); v._camera.up.set(0, 0, 1);
  v._controls.target.set(0, 0, 0); v._camera.lookAt(0, 0, 0);
  v._controls.update(); v._camera.updateMatrixWorld(true);
}"""

# Project the visible XY chip's bbox to screen pixels, dispatch a pointermove
# grid over it, and tally which handle wins each cell (control.axis).
_GIZMO_CHIP_SCAN = """() => {
  const v = window.threejsViewer;
  const g = v._transformGizmo._primary;
  const control = g.control;
  g.helper.updateMatrixWorld(true);
  const canvas = v._renderer.domElement;
  const rect = canvas.getBoundingClientRect();
  const V = v._camera.position.constructor;
  let box = null;
  control._gizmo.gizmo.translate.traverse(o => {
    if (o.name !== 'XY' || !o.geometry || !o.material || box) return;
    o.geometry.computeBoundingBox();
    const bb = o.geometry.boundingBox;
    let minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
    for (const x of [bb.min.x, bb.max.x])
      for (const y of [bb.min.y, bb.max.y])
        for (const z of [bb.min.z, bb.max.z]) {
          const p = new V(x, y, z).applyMatrix4(o.matrixWorld).project(v._camera);
          const sx = rect.left + (p.x * 0.5 + 0.5) * rect.width;
          const sy = rect.top + (-p.y * 0.5 + 0.5) * rect.height;
          minX = Math.min(minX, sx); maxX = Math.max(maxX, sx);
          minY = Math.min(minY, sy); maxY = Math.max(maxY, sy);
        }
    box = {minX, minY, maxX, maxY};
  });
  if (!box) return null;
  const counts = {};
  let total = 0;
  for (let y = box.minY; y <= box.maxY; y += 3) {
    for (let x = box.minX; x <= box.maxX; x += 3) {
      canvas.dispatchEvent(new PointerEvent('pointermove',
        {clientX: x, clientY: y, pointerType: 'mouse', bubbles: true}));
      total++;
      const a = control.axis || 'none';
      counts[a] = (counts[a] || 0) + 1;
    }
  }
  return {counts, total};
}"""

# Probe every visible mesh of one arrow (shaft + end cones): dispatch a
# pointermove at each mesh's projected centre and collect what hover resolves.
_GIZMO_ARROW_PROBE = """(name) => {
  const v = window.threejsViewer;
  const g = v._transformGizmo._primary;
  const control = g.control;
  g.helper.updateMatrixWorld(true);
  const canvas = v._renderer.domElement;
  const rect = canvas.getBoundingClientRect();
  const V = v._camera.position.constructor;
  const hits = [];
  control._gizmo.gizmo.translate.traverse(o => {
    if (o.name !== name || !o.geometry) return;
    o.geometry.computeBoundingBox();
    const c = o.geometry.boundingBox.getCenter(new V());
    c.applyMatrix4(o.matrixWorld).project(v._camera);
    const sx = rect.left + (c.x * 0.5 + 0.5) * rect.width;
    const sy = rect.top + (-c.y * 0.5 + 0.5) * rect.height;
    canvas.dispatchEvent(new PointerEvent('pointermove',
      {clientX: sx, clientY: sy, pointerType: 'mouse', bubbles: true}));
    hits.push(control.axis || 'none');
  });
  return hits;
}"""


@pytest.mark.browser
def test_gizmo_plane_chip_hover_beats_arrow_pickers(viewer_client, viewer_page):
    """Regression for the stock fat arrow pickers shadowing the plane chips:
    hover resolves closest-intersection-wins, and the unslimmed arrow pickers
    (radius 0.2 cones hugging each axis) sat in front of the flat XY chip from
    any 3/4 view, so most of the chip's visible parallelogram grabbed the X
    arrow instead of the chip. With the pickers slimmed
    (GIZMO_ARROW_PICKER_SLIM / GIZMO_CENTER_PICKER_SCALE) the chip must win the
    bulk of its own footprint while every arrow stays hittable."""
    viewer_client.add_box("box", position=[0, 0, 0])
    _wait_for(viewer_page, "() => window.threejsViewer._objects.has('box')")
    viewer_client.enable_move_gizmo(id="box", click_select=False)
    _wait_for(
        viewer_page,
        "() => { const g = window.threejsViewer._transformGizmo._primary;"
        " return g && g.object && g.helper.visible; }",
    )
    viewer_page.evaluate(_GIZMO_THREEQUARTER)

    res = viewer_page.evaluate(_GIZMO_CHIP_SCAN)
    assert res is not None, "XY chip handle not found"
    counts, total = res["counts"], res["total"]
    xy = counts.get("XY", 0)
    arrows = counts.get("X", 0) + counts.get("Y", 0) + counts.get("Z", 0)
    # Pre-fix this pose gave the chip and the X arrow roughly equal shares of
    # the chip's bbox; post-fix the chip dominates by an order of magnitude.
    assert xy > 3 * max(1, arrows), (
        f"arrow pickers shadow the XY chip inside its own footprint: {counts}"
    )
    assert xy >= 0.3 * total, f"chip hover coverage too low: {counts} of {total}"

    # Slimming must not make the arrows unhittable: each axis still resolves at
    # (at least one of) its shaft/cone centres.
    for name in ("X", "Y", "Z"):
        hits = viewer_page.evaluate(_GIZMO_ARROW_PROBE, name)
        assert name in hits, f"arrow {name} no longer hittable anywhere: {hits}"


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
        settle(viewer_client)
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


_ORBIT_STATE_JS = (
    "() => {"
    " const v = window.threejsViewer;"
    " const c = v._controls;"
    " const f = v._camera.getWorldDirection(v._camera.position.clone());"
    " const u = v._camera.up;"
    " return { fwd: [f.x, f.y, f.z], up: [u.x, u.y, u.z], mode: c.mode,"
    "          ortho: v._isOrtho, dragging: c.isDragging(),"
    "          tweening: !!v._viewTween,"
    "          pending: Math.abs(c._rotDeltaTheta) + Math.abs(c._rotDeltaPhi) };"
    "}"
)
_COS_POLE_EPS = math.cos(math.radians(5.0))


def _drag_canvas(page, dx, dy, steps=20):
    """Real left-button pointer drag across the viewer canvas: press at the
    canvas centre, move by `steps` increments of (dx, dy) screen pixels,
    release, then wait for the damped orbit inertia to drain so the camera
    pose is final."""
    cx, cy = page.evaluate(
        "() => { const r = window.threejsViewer._renderer.domElement"
        ".getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; }"
    )
    page.mouse.move(cx, cy)
    page.mouse.down()
    for i in range(1, steps + 1):
        page.mouse.move(cx + i * dx, cy + i * dy)
    page.mouse.up()

    def drained():
        s = page.evaluate(_ORBIT_STATE_JS)
        return not s["dragging"] and not s["tweening"] and s["pending"] == 0

    assert _wait_until(drained), "orbit inertia never drained"


def _wait_view_tween_done(page):
    assert _wait_until(
        lambda: not page.evaluate("() => !!window.threejsViewer._viewTween")
    )


@pytest.mark.browser
@pytest.mark.parametrize("view", ["top", "bottom"])
def test_turntable_drag_leaves_pole_after_axis_view(viewer_client, viewer_page, view):
    """Issue #202: after a top/bottom view snap the forward vector sits exactly
    on the pole, and every mouse-sized pitch step used to be refused because it
    landed inside the 5 degree pole cone, so a vertical left-drag only yawed.
    A pitch that moves away from the pole is now always applied, and once the
    orbit leaves the cone the turntable re-levels camera.up to world +Z (the
    view preset set it to +Y, which every later lookAt would read as a roll)."""
    viewer_client.add_box("b")
    settle(viewer_client)
    viewer_page.evaluate(
        "(view) => { const v = window.threejsViewer; v._controls.setMode('turntable');"
        " v.setView(view, { animate: false }); }",
        view,
    )
    frames(viewer_page)
    before = viewer_page.evaluate(_ORBIT_STATE_JS)
    assert abs(before["fwd"][2]) == pytest.approx(1.0, abs=1e-6)
    assert before["up"] == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)

    _drag_canvas(viewer_page, 0, -10)  # 200 px vertical drag
    after = viewer_page.evaluate(_ORBIT_STATE_JS)
    assert abs(after["fwd"][2]) < _COS_POLE_EPS - 0.01, (
        f"forward stayed on the pole after a vertical drag: {after}"
    )
    assert after["up"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6), (
        f"turntable did not re-level camera.up after leaving the pole: {after}"
    )


@pytest.mark.browser
def test_turntable_drag_leaves_pole_after_gizmo_top_click(viewer_client, viewer_page):
    """The same escape through the gimbal bubble path: `_gizmoAxisClick('top')`
    auto-enters ortho, and a real vertical drag both pitches the view off the
    pole and returns to perspective (auto-projection)."""
    viewer_client.add_box("b")
    settle(viewer_client)
    viewer_page.evaluate(
        "() => { const v = window.threejsViewer; v._controls.setMode('turntable');"
        " v._gizmoAxisClick('top'); }"
    )
    _wait_view_tween_done(viewer_page)
    before = viewer_page.evaluate(_ORBIT_STATE_JS)
    assert before["ortho"] is True
    assert abs(before["fwd"][2]) == pytest.approx(1.0, abs=1e-6)

    _drag_canvas(viewer_page, 0, 10)
    after = viewer_page.evaluate(_ORBIT_STATE_JS)
    assert after["ortho"] is False, (
        "orbiting away from the snap should return to perspective"
    )
    assert abs(after["fwd"][2]) < _COS_POLE_EPS - 0.01, after
    assert after["up"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6), after


@pytest.mark.browser
def test_turntable_drag_toward_pole_never_flips(viewer_client, viewer_page):
    """The pole clamp is preserved: from an oblique view a hard vertical drag
    in either direction lands the forward vector at the cone boundary at most
    (|fwd.z| <= cos(5 deg)) and the camera never flips through the pole: its
    own up axis keeps a positive world-Z component, so the view is never
    upside down."""
    viewer_client.add_box("b")
    settle(viewer_client)
    for direction in (-1, 1):
        viewer_page.evaluate(
            "() => { const v = window.threejsViewer; v._controls.setMode('turntable');"
            " v.setCameraPose({ position: [6, -6, 3], target: [0, 0, 0], up: [0, 0, 1] }); }"
        )
        frames(viewer_page)
        # 40 x 15 px = 600 px, far more than the ~150 px needed to reach the pole.
        _drag_canvas(viewer_page, 0, 15 * direction, steps=40)
        after = viewer_page.evaluate(_ORBIT_STATE_JS)
        assert abs(after["fwd"][2]) <= _COS_POLE_EPS + 1e-6, (direction, after)
        assert after["up"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
        cam_up_z = viewer_page.evaluate(
            "() => { const c = window.threejsViewer._camera;"
            " return c.up.clone().set(0, 1, 0).applyQuaternion(c.quaternion).z; }"
        )
        assert cam_up_z > 0.05, (direction, after, cam_up_z)


@pytest.mark.browser
def test_camera_switch_hands_over_up_for_relevel(viewer_client, viewer_page):
    """`_switchCamera` copies `up` along with position and quaternion, so the
    turntable re-level lands on whichever camera is active next. A perspective
    `setView('top')` leaves +Y on the persp camera; the gimbal top click enters
    ortho, and the drag auto-returns to perspective, which must then carry the
    re-levelled +Z rather than the stale +Y."""
    viewer_client.add_box("b")
    settle(viewer_client)
    viewer_page.evaluate(
        "() => { const v = window.threejsViewer; v._controls.setMode('turntable');"
        " v.setView('top', { animate: false }); v._gizmoAxisClick('top'); }"
    )
    _wait_view_tween_done(viewer_page)
    before = viewer_page.evaluate(_ORBIT_STATE_JS)
    assert before["ortho"] is True
    persp_up = viewer_page.evaluate(
        "() => { const u = window.threejsViewer._perspCamera.up; return [u.x, u.y, u.z]; }"
    )
    assert persp_up == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)

    _drag_canvas(viewer_page, 0, 10)
    after = viewer_page.evaluate(_ORBIT_STATE_JS)
    assert after["ortho"] is False
    assert abs(after["fwd"][2]) < _COS_POLE_EPS - 0.01, after
    assert after["up"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6), after


@pytest.mark.browser
def test_free_mode_drag_keeps_camera_up(viewer_client, viewer_page):
    """Free mode is untouched by the turntable re-level: after a top snap and a
    vertical drag, camera.up keeps the preset's +Y instead of being reset to
    world +Z. Free mode yaws about camera-up and the horizon may roll; that is
    the intended behaviour, not a regression."""
    viewer_client.add_box("b")
    settle(viewer_client)
    viewer_page.evaluate(
        "() => { const v = window.threejsViewer; v._controls.setMode('free');"
        " v.setView('top', { animate: false }); }"
    )
    frames(viewer_page)
    _drag_canvas(viewer_page, 0, -10)
    after = viewer_page.evaluate(_ORBIT_STATE_JS)
    assert after["mode"] == "free"
    assert abs(after["fwd"][2]) < _COS_POLE_EPS - 0.01, after
    assert after["up"] == pytest.approx([0.0, 1.0, 0.0], abs=1e-6), (
        f"free mode must not force camera.up to world +Z: {after}"
    )


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
    settle(viewer_client)
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
def test_toolbar_hidden_by_default_and_client_toggle(viewer_client, viewer_page):
    """The top-left menu button is hidden on a bare open; the Python client
    shows it with set_toolbar_visible and hides it again (closing the menu)."""
    page = viewer_page
    assert page.evaluate("() => window.threejsViewer._toolbarEl.hidden") is True
    assert page.evaluate("() => window.threejsViewer.isToolbarVisible()") is False

    viewer_client.set_toolbar_visible(True)
    settle(viewer_client)
    assert page.evaluate("() => window.threejsViewer._toolbarEl.hidden") is False
    assert page.locator(".tjsv-btn-menu").is_visible()

    page.locator(".tjsv-btn-menu").click()
    assert page.evaluate("() => window.threejsViewer._menuOpen") is True
    viewer_client.set_toolbar_visible(False)
    settle(viewer_client)
    assert page.evaluate("() => window.threejsViewer._toolbarEl.hidden") is True
    assert page.evaluate("() => window.threejsViewer._menuOpen") is False


@pytest.mark.browser
def test_toolbar_menu_lists_options_with_shortcuts(viewer_client, viewer_page):
    """The menu lists every advanced option with its shortcut key; an item
    click runs the same action as the key, and the state labels follow."""
    page = viewer_page
    viewer_client.set_toolbar_visible(True)
    settle(viewer_client)
    page.locator(".tjsv-btn-menu").click()
    items = page.evaluate(
        "() => [...document.querySelectorAll('[data-menu=viewer] .tjsv-menu-action')]"
        "  .filter(b => !b.hidden)"
        "  .map(b => [b.querySelector('.tjsv-menu-label').textContent,"
        "             b.querySelector('kbd').textContent])"
    )
    assert items == [
        ["Clipping plane", "C"],
        ["Lighting", "E"],
        ["Orbit mode", "R"],
        ["Projection", "O"],
        ["Wireframe", "M"],
        ["Shading debug", "N"],
        ["Distance fog", "D"],
        ["Eye-dome lighting", "\u21e7D"],
        ["Frame all", "F"],
    ]
    # Camera tracking only shows once an animation with a track target exists.
    assert (
        page.evaluate("() => document.querySelector('[data-item=track]').hidden")
        is True
    )

    page.locator("[data-menu=viewer] [data-item=wireframe]").click()
    state = page.evaluate(
        "() => ({mode: window.threejsViewer._shading.wireframeMode,"
        "        active: document.querySelector('[data-item=wireframe]')"
        "                  .classList.contains('active'),"
        "        label: document.querySelector('[data-item=wireframe] .tjsv-menu-state')"
        "                  .textContent,"
        "        open: window.threejsViewer._menuOpen})"
    )
    assert state == {"mode": 1, "active": True, "label": "wire", "open": True}

    # A pointerdown outside the toolbar closes the menu.
    page.mouse.click(400, 300)
    assert page.evaluate("() => window.threejsViewer._menuOpen") is False


@pytest.mark.browser
def test_toolbar_option_and_url_param(viewer_client, viewer_page):
    """`toolbar: true` as a constructor option shows the menu button on a
    fresh viewer, mirroring the `toolbar=true` URL param the Python kwarg sends."""
    result = viewer_page.evaluate(
        "() => {"
        " const live = window.threejsViewer;"
        " const V = live.constructor;"
        " const mk = (opts) => {"
        "   const div = document.createElement('div');"
        "   div.style.cssText ="
        "     'width:300px;height:200px;position:absolute;left:-2000px;top:0';"
        "   document.body.appendChild(div);"
        "   const v = new V(div, Object.assign({"
        "     htmlTemplate: live._options.htmlTemplate,"
        "     cubemapData: live._options.cubemapData,"
        "     autoConnect: false }, opts));"
        "   return v._toolbarEl.hidden;"
        " };"
        " return {plain: mk({}), opt: mk({toolbar: true}), off: mk({toolbar: 'false'})};"
        "}"
    )
    assert result == {"plain": True, "opt": False, "off": True}


def _press_viewer_key(page, key, code, shift=False):
    """Dispatch a keydown on the viewer container (its keyboard handler is
    scoped there, so a page-level keyboard.press needs focus it may not have)."""
    page.evaluate(
        "([key, code, shift]) => window.threejsViewer.container.dispatchEvent("
        "  new KeyboardEvent('keydown', {key, code, shiftKey: shift, bubbles: true}))",
        [key, code, shift],
    )


@pytest.mark.browser
def test_add_menu_dropdown_items_and_callbacks(viewer_client, viewer_page):
    """addMenu builds a dropdown next to the built-in menu; each item type
    renders, fires its callback, and setItem patches from outside."""
    page = viewer_page
    result = page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " window.__log = [];"
        " const m = v.addMenu({id: 'demo', label: 'Demo', items: ["
        "   {type: 'label', label: 'Scene'},"
        "   {id: 'go', label: 'Go', shortcut: 'G', bindKey: true,"
        "    onClick: () => window.__log.push('go')},"
        "   {type: 'toggle', id: 'spin', label: 'Spin', checked: false,"
        "    onChange: (on) => window.__log.push('spin:' + on)},"
        "   {type: 'select', id: 'size', label: 'Size',"
        "    options: [{value: 's', label: 'S'}, {value: 'm', label: 'M'}], value: 'm',"
        "    onChange: (val) => window.__log.push('size:' + val)},"
        "   {type: 'segmented', id: 'side', options: ['left', 'right'], value: 'right',"
        "    onChange: (val) => window.__log.push('side:' + val)},"
        "   {type: 'divider'},"
        "   {type: 'custom', id: 'legend', render: (el) => { el.textContent = 'legend'; }},"
        " ]});"
        " v.onMenuAction(a => window.__log.push('action:' + a.item + ':' + a.type + ':' + a.value));"
        " const root = m.el;"
        " return {inBar: root.parentElement.classList.contains('tjsv-rail'),"
        "         btnText: root.querySelector('.tjsv-menu-btn').textContent.trim(),"
        "         hidden: !root.classList.contains('open'),"
        "         kbd: root.querySelector('[data-item=go] kbd').textContent,"
        "         custom: root.querySelector('[data-item=legend]').textContent,"
        "         segActive: root.querySelector('[data-item=side] .tjsv-seg-btn.active').dataset.value};"
        "}"
    )
    assert result == {
        "inBar": True,
        "btnText": "Demo",
        "hidden": True,
        "kbd": "G",
        "custom": "legend",
        "segActive": "right",
    }

    page.locator("[data-menu=demo] .tjsv-menu-btn").click()
    assert page.evaluate("() => window.threejsViewer.getMenu('demo').isOpen()") is True
    # Opening one dropdown closes the other.
    viewer_client.set_toolbar_visible(True)
    settle(viewer_client)
    page.locator(".tjsv-btn-menu").click()
    assert page.evaluate("() => window.threejsViewer.getMenu('demo').isOpen()") is False
    page.locator("[data-menu=demo] .tjsv-menu-btn").click()

    page.locator("[data-menu=demo] [data-item=go]").click()
    page.locator("[data-menu=demo] [data-item=spin]").click()
    page.locator("[data-menu=demo] [data-item=size] select").select_option("s")
    page.locator(
        "[data-menu=demo] [data-item=side] .tjsv-seg-btn[data-value=left]"
    ).click()
    # A bound shortcut runs the item without the menu.
    page.keyboard.press("Escape")
    _press_viewer_key(page, "g", "KeyG")
    log = page.evaluate("() => window.__log")
    assert log == [
        "go",
        "action:go:button:undefined",
        "spin:true",
        "action:spin:toggle:true",
        "size:s",
        "action:size:select:s",
        "side:left",
        "action:side:segmented:left",
        "go",
        "action:go:button:undefined",
    ]

    state = page.evaluate(
        "() => {"
        " const m = window.threejsViewer.getMenu('demo');"
        " m.setItem('spin', {checked: false, state: 'idle'});"
        " m.setItem('size', {options: [{value: 'xl', label: 'XL'}], value: 'xl'});"
        " m.setItem('go', {disabled: true, label: 'Gone'});"
        " const r = m.el;"
        " return {spinOn: r.querySelector('[data-item=spin]').classList.contains('active'),"
        "         spinState: r.querySelector('[data-item=spin] .tjsv-menu-state').textContent,"
        "         size: r.querySelector('[data-item=size] select').value,"
        "         goDisabled: r.querySelector('[data-item=go]').disabled,"
        "         goLabel: r.querySelector('[data-item=go] .tjsv-menu-label').textContent,"
        "         value: m.getValue('spin')};"
        "}"
    )
    assert state == {
        "spinOn": False,
        "spinState": "idle",
        "size": "xl",
        "goDisabled": True,
        "goLabel": "Gone",
        "value": False,
    }
    assert page.evaluate("() => window.threejsViewer.removeMenu('demo')") is True
    assert page.evaluate("() => document.querySelector('[data-menu=demo]')") is None


@pytest.mark.browser
def test_add_menu_eyes_persist_and_follow_late_objects(viewer_client, viewer_page):
    """An eye item hides every object it owns, keeps hiding objects that are
    added later, persists under storageKey, and hidingEyeFor names it."""
    page = viewer_page
    viewer_client.add_box("box_a", 1, 1, 1)
    settle(viewer_client)
    page.evaluate(
        "() => {"
        " try { localStorage.removeItem('tjsv-test.eyes'); } catch (e) {}"
        " window.threejsViewer.addMenu({id: 'layers', label: 'Layers', storageKey: 'tjsv-test.eyes',"
        "   items: [{type: 'eye', id: 'boxes', label: 'Boxes', prefix: 'box_'}]});"
        "}"
    )
    assert (
        page.evaluate("() => window.threejsViewer.getObject('box_a').visible") is True
    )
    page.locator("[data-menu=layers] .tjsv-menu-btn").click()
    page.locator("[data-menu=layers] [data-item=boxes]").click()
    assert (
        page.evaluate("() => window.threejsViewer.getObject('box_a').visible") is False
    )
    assert page.evaluate("() => window.threejsViewer.hidingEyeFor('box_a')") == "Boxes"
    assert page.evaluate("() => window.threejsViewer.hidingEyeFor('other')") is None

    viewer_client.add_box("box_b", 1, 1, 1, position=[2, 0, 0])
    settle(viewer_client)
    assert (
        page.evaluate("() => window.threejsViewer.getObject('box_b').visible") is False
    )
    assert page.evaluate(
        "() => JSON.parse(localStorage.getItem('tjsv-test.eyes'))"
    ) == {"boxes": False}
    # A fresh mount under the same storageKey starts from the stored choice.
    stored = page.evaluate(
        "() => {"
        " const m = window.threejsViewer.addMenu({id: 'layers', label: 'Layers',"
        "   storageKey: 'tjsv-test.eyes', items: [{type: 'eye', id: 'boxes', prefix: 'box_'}]});"
        " return [m.getValue('boxes'), m.el.querySelector('[data-item=boxes]').classList.contains('off')];"
        "}"
    )
    assert stored == [False, True]


@pytest.mark.browser
def test_add_menu_eye_match_apply_and_veto(viewer_client, viewer_page):
    """An eye may own objects through `match` and show/hide through `apply`;
    an `apply` returning false vetoes the flip and the row reverts."""
    page = viewer_page
    viewer_client.add_box("ws_1_base", 1, 1, 1)
    viewer_client.add_box("ws_1_reach", 1, 1, 1)
    settle(viewer_client)
    result = page.evaluate(
        "() => {"
        " const v = window.threejsViewer; window.__applied = [];"
        " const m = v.addMenu({id: 'e', label: 'E', items: ["
        "   {type: 'eye', id: 'base', label: 'Base', match: id => /_base$/.test(id)},"
        "   {type: 'eye', id: 'tcp', label: 'TCP', ids: [], apply: on => { window.__applied.push(on); }},"
        "   {type: 'eye', id: 'locked', label: 'Locked', ids: [], apply: () => false,"
        "    onChange: () => window.__applied.push('never')},"
        " ]});"
        " m.setItem('base', {checked: false});"
        " m.setItem('tcp', {checked: false});"
        " const r = m.el;"
        " r.querySelector('.tjsv-menu-btn').click();"
        " r.querySelector('[data-item=locked]').click();"
        " return {base: v.getObject('ws_1_base').visible, reach: v.getObject('ws_1_reach').visible,"
        "         hiding: v.hidingEyeFor('ws_1_base'), applied: window.__applied,"
        "         locked: m.getValue('locked'),"
        "         lockedRow: r.querySelector('[data-item=locked]').classList.contains('active')};"
        "}"
    )
    assert result == {
        "base": False,
        "reach": True,
        "hiding": "Base",
        "applied": [True, False],
        "locked": True,
        "lockedRow": True,
    }


@pytest.mark.browser
def test_add_menu_rail_stacks_and_panel_mode(viewer_client, viewer_page):
    """Every menu is a tab on the right-edge rail, stacked top to bottom; a
    `panel` menu starts open and survives an outside click, a dropdown does not."""
    page = viewer_page
    result = page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " const p = v.addMenu({id: 'legend', label: 'Colour', mode: 'panel', bodyWidth: '220px',"
        "   items: [{type: 'select', id: 'mode', label: 'By', options: ['a', 'b']}]});"
        " const d = v.addMenu({id: 'gz', label: 'Gizmo',"
        "   items: [{type: 'toggle', id: 'move', label: 'Move'}, {type: 'toggle', id: 'rot', label: 'Rotate'}]});"
        " const top = el => parseFloat(getComputedStyle(el).top);"
        " return {host: p.el.parentElement.className, tab: p.el.querySelector('.tjsv-menu-btn-label').textContent,"
        "         panelOpen: p.isOpen() && p.el.classList.contains('open'),"
        "         width: getComputedStyle(p.el.querySelector('.tjsv-menu')).width,"
        "         stacked: top(d.el) > top(p.el) + p.el.querySelector('.tjsv-menu-btn').getBoundingClientRect().height,"
        "         items: d.el.querySelectorAll('.tjsv-menu-item').length};"
        "}"
    )
    assert result == {
        "host": "tjsv-rail tjsv-toolbar",
        "tab": "Colour",
        "panelOpen": True,
        "width": "220px",
        "stacked": True,
        "items": 2,
    }
    page.locator("[data-menu=gz] .tjsv-menu-btn").click()
    assert page.evaluate("() => window.threejsViewer.getMenu('gz').isOpen()") is True
    page.locator("[data-menu=gz] [data-item=move]").click()
    assert (
        page.evaluate("() => window.threejsViewer.getMenu('gz').getValue('move')")
        is True
    )
    # Opening one tab folds the others (one open at a time), an outside click
    # folds a dropdown, and a panel re-opened by its tab survives that click.
    assert (
        page.evaluate("() => window.threejsViewer.getMenu('legend').isOpen()") is False
    )
    page.mouse.click(400, 300)
    assert page.evaluate("() => window.threejsViewer.getMenu('gz').isOpen()") is False
    page.locator("[data-menu=legend] .tjsv-menu-btn").click()
    page.mouse.click(400, 300)
    assert (
        page.evaluate("() => window.threejsViewer.getMenu('legend').isOpen()") is True
    )


@pytest.mark.browser
def test_add_menu_reserved_shortcut_is_not_bound(viewer_client, viewer_page):
    """A client item may display a viewer key but never binds it."""
    page = viewer_page
    warned = []
    page.on("console", lambda m: warned.append(m.text) if m.type == "warning" else None)
    page.evaluate(
        "() => { window.__hits = 0; window.threejsViewer.addMenu({id: 'k', label: 'K', items: ["
        "  {id: 'x', label: 'X', shortcut: 'M', bindKey: true, onClick: () => window.__hits++}]}); }"
    )
    _press_viewer_key(page, "m", "KeyM")
    frames(page)
    assert page.evaluate("() => window.__hits") == 0
    assert page.evaluate("() => window.threejsViewer._shading.wireframeMode") == 1
    assert any("is a viewer key" in w for w in warned)


@pytest.mark.browser
def test_python_add_menu_round_trip(viewer_client, viewer_page):
    """add_menu from Python renders in the viewer; a click comes back as a
    menu_action to on_menu_action; update_menu_item patches live."""
    import threading

    page = viewer_page
    got = []
    ev = threading.Event()

    def cb(action):
        got.append(action)
        ev.set()

    viewer_client.on_menu_action(cb)
    viewer_client.add_menu(
        "py",
        label="Py",
        items=[
            {"id": "hello", "label": "Hello", "shortcut": "H", "bind_key": True},
            {"type": "toggle", "id": "flag", "label": "Flag", "checked": True},
        ],
    )
    settle(viewer_client)
    page.locator("[data-menu=py] .tjsv-menu-btn").click()
    page.locator("[data-menu=py] [data-item=flag]").click()
    assert ev.wait(5), "no menu_action reached Python"
    assert got == [{"menu": "py", "item": "flag", "type": "toggle", "value": False}]

    viewer_client.update_menu_item("py", "hello", state="ready", disabled=True)
    settle(viewer_client)
    assert page.evaluate(
        "() => [document.querySelector('[data-item=hello] .tjsv-menu-state').textContent,"
        "       document.querySelector('[data-item=hello]').disabled]"
    ) == ["ready", True]
    viewer_client.remove_menu("py")
    settle(viewer_client)
    assert page.evaluate("() => document.querySelector('[data-menu=py]')") is None


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
def test_iso_snap_is_true_isometric(viewer_client, viewer_page):
    """The iso snap is a true isometric: orthographic projection down the
    (1,-1,1) direction, under the same auto-projection rule as the axis
    bubbles (orbiting away returns to perspective). It has no button since the
    orbit-mode toggle took its slot; `_snapOrthoAxisView('iso')` stays the
    programmatic path. The old ortho toolbar toggle stays removed."""
    viewer_client.add_box("b")
    assert "b" in viewer_client.query_scene()["objects"]  # sync: box is in-scene
    result = viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " v._snapOrthoAxisView('iso');"
        " const afterIso = { ortho: v._isOrtho, snap: v._gizmoAxisView };"
        " const orig = v._controls.isOrbiting;"
        " v._controls.isOrbiting = () => true;"
        " v._controls.dispatchEvent({ type: 'change' });"
        " v._controls.isOrbiting = orig;"
        " const orthoAfterOrbit = v._isOrtho;"
        " const toolbarOrtho = !!document.querySelector('.tjsv-btn-ortho');"
        " return { afterIso, orthoAfterOrbit, toolbarOrtho };"
        "}"
    )
    assert result["afterIso"] == {"ortho": True, "snap": "iso"}, (
        "iso must snap into an orthographic isometric"
    )
    assert result["orthoAfterOrbit"] is False, "orbiting away returns to perspective"
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
        "   const orig = v._controls.isOrbiting;"
        "   v._controls.isOrbiting = () => true;"
        "   v._controls.dispatchEvent({ type: 'change' });"
        "   v._controls.isOrbiting = orig;"
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
def test_projection_button_indicates_and_toggles(viewer_client, viewer_page):
    """The gimbal-corner P/O button shows the CURRENT projection (P/O label +
    .ortho accent), toggles it on click, and tracks auto-projection switches."""
    viewer_client.add_box("b")
    assert "b" in viewer_client.query_scene()["objects"]  # sync: box is in-scene
    result = viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " const btn = v._viewProjBtn;"
        " const state = () => ({ label: btn.textContent,"
        "   accent: btn.classList.contains('ortho'), ortho: v._isOrtho });"
        " const initial = state();"
        " btn.click();"  # manual toggle -> ortho
        " const manualOrtho = state();"
        " btn.click();"  # manual toggle back -> perspective
        " const manualPersp = state();"
        " v._gizmoAxisClick('top');"  # auto-projection also updates the button
        " const autoOrtho = state();"
        " return { initial, manualOrtho, manualPersp, autoOrtho };"
        "}"
    )
    assert result["initial"] == {"label": "P", "accent": False, "ortho": False}
    assert result["manualOrtho"] == {"label": "O", "accent": True, "ortho": True}
    assert result["manualPersp"] == {"label": "P", "accent": False, "ortho": False}
    assert result["autoOrtho"] == {"label": "O", "accent": True, "ortho": True}


@pytest.mark.browser
def test_axis_snap_survives_pivot_but_clears_on_orbit(viewer_client, viewer_page):
    """The gizmo axis snap is preserved through a plain click-to-pivot (a
    controls 'change' fired while not dragging) so a re-click still flips, but
    an actual orbit drag ('change' while orbiting) clears it (#514)."""
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
        # orbit drag: controls emit 'change' while in the ROTATE state with the
        # pointerdown mode snapshot taken.
        " c._state = 1; c._dragMode = c.mode;"
        " c.dispatchEvent({ type: 'change' });"
        " const afterOrbit = v._gizmoAxisView;"
        " c._state = 0; c._dragMode = null;"
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


_CANVAS_CENTER_JS = (
    "() => { const r = window.threejsViewer._renderer.domElement"
    ".getBoundingClientRect();"
    " return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; }"
)

_PROJECTION_STATE_JS = (
    "() => { const v = window.threejsViewer;"
    " return { ortho: v._isOrtho, snap: v._gizmoAxisView,"
    "   auto: v._orthoAutoEntered }; }"
)


def _drag_canvas_button(page, button, steps=8, step_px=10):
    """Press ``button`` at the canvas centre and drag it rightwards."""
    c = page.evaluate(_CANVAS_CENTER_JS)
    page.mouse.move(c["x"], c["y"])
    page.mouse.down(button=button)
    for i in range(1, steps + 1):
        page.mouse.move(c["x"] + i * step_px, c["y"])
    page.mouse.up(button=button)


@pytest.mark.browser
def test_pan_after_axis_snap_keeps_auto_ortho(viewer_client, viewer_page):
    """A right-button pan drag after a bubble snap from perspective keeps the
    orthographic projection and the axis snap, so the view stays axis-aligned
    and a re-click of the same bubble still flips (#193). A wheel zoom keeps
    both as well."""
    viewer_client.add_box("b")
    assert "b" in viewer_client.query_scene()["objects"]  # sync: box is in-scene
    viewer_page.evaluate("() => window.threejsViewer._snapOrthoAxisView('front')")
    frames(viewer_page)
    before = viewer_page.evaluate(_PROJECTION_STATE_JS)
    assert before == {"ortho": True, "snap": "front", "auto": True}

    _drag_canvas_button(viewer_page, "right")
    frames(viewer_page)
    after_pan = viewer_page.evaluate(_PROJECTION_STATE_JS)
    assert after_pan == {"ortho": True, "snap": "front", "auto": True}, (
        "a pan must keep the auto-entered ortho and the axis snap"
    )

    c = viewer_page.evaluate(_CANVAS_CENTER_JS)
    viewer_page.mouse.move(c["x"], c["y"])
    viewer_page.mouse.wheel(0, 200)
    frames(viewer_page)
    after_wheel = viewer_page.evaluate(_PROJECTION_STATE_JS)
    assert after_wheel == {"ortho": True, "snap": "front", "auto": True}, (
        "a wheel zoom must keep the auto-entered ortho and the axis snap"
    )

    flipped = viewer_page.evaluate(
        "() => { const v = window.threejsViewer; v._gizmoAxisClick('front');"
        " return v._gizmoAxisView; }"
    )
    assert flipped == "back", "the snap survived the pan, so a re-click flips"


@pytest.mark.browser
def test_orbit_after_axis_snap_returns_to_perspective(viewer_client, viewer_page):
    """A left-button orbit drag after a bubble snap from perspective returns
    to perspective and clears the axis snap (#193)."""
    viewer_client.add_box("b")
    assert "b" in viewer_client.query_scene()["objects"]  # sync: box is in-scene
    viewer_page.evaluate("() => window.threejsViewer._snapOrthoAxisView('front')")
    frames(viewer_page)
    assert viewer_page.evaluate(_PROJECTION_STATE_JS) == {
        "ortho": True,
        "snap": "front",
        "auto": True,
    }

    _drag_canvas_button(viewer_page, "left")
    frames(viewer_page)
    after_orbit = viewer_page.evaluate(_PROJECTION_STATE_JS)
    assert after_orbit == {"ortho": False, "snap": None, "auto": False}, (
        "an orbit must return to perspective and clear the axis snap"
    )


@pytest.mark.browser
def test_manual_ortho_survives_orbit_after_axis_snap(viewer_client, viewer_page):
    """A manual `O` ortho is never auto-exited: a bubble snap taken while
    already in manual ortho followed by an orbit drag stays orthographic (the
    projection the user had before the snap), though the snap itself clears."""
    viewer_client.add_box("b")
    assert "b" in viewer_client.query_scene()["objects"]  # sync: box is in-scene
    # The viewer binds its shortcuts on the container, so dispatch there.
    viewer_page.evaluate(
        "() => window.threejsViewer.container.dispatchEvent(new KeyboardEvent("
        "'keydown', { key: 'o', code: 'KeyO', bubbles: true }))"
    )
    frames(viewer_page)
    assert viewer_page.evaluate(_PROJECTION_STATE_JS) == {
        "ortho": True,
        "snap": None,
        "auto": False,
    }
    viewer_page.evaluate("() => window.threejsViewer._snapOrthoAxisView('front')")
    frames(viewer_page)
    assert viewer_page.evaluate(_PROJECTION_STATE_JS) == {
        "ortho": True,
        "snap": "front",
        "auto": False,
    }

    _drag_canvas_button(viewer_page, "left")
    frames(viewer_page)
    after_orbit = viewer_page.evaluate(_PROJECTION_STATE_JS)
    assert after_orbit == {"ortho": True, "snap": None, "auto": False}, (
        "a manual ortho stays ortho through an orbit; only the snap clears"
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


# A KHR_draco_mesh_compression-encoded unit quad: 4 verts, 2 triangles, POSITION
# only, EDGEBREAKER, 14-bit quantization — 75 bytes, the smallest useful Draco
# payload. Checked in as base64 rather than as a binary fixture file because the
# test builds the GLB container around it in Python (mirroring _two_triangle_glb,
# so the two fixtures differ only in the compression), and because encoding it at
# test time would need a Draco *encoder* the project does not ship. Regenerate
# with three's examples/jsm/libs/draco/draco_encoder.js:
#   const mesh = new m.Mesh(), mb = new m.MeshBuilder();
#   mb.AddFacesToMesh(mesh, 2, new Uint32Array([0,1,2, 0,2,3]));
#   mb.AddFloatAttributeToMesh(mesh, m.POSITION, 4, 3, positions);  // -> attr id 0
#   encoder.SetEncodingMethod(m.MESH_EDGEBREAKER_ENCODING);
#   encoder.SetAttributeQuantization(m.POSITION, 14);
#   encoder.EncodeMeshToDracoBuffer(mesh, out);
_DRACO_QUAD_B64 = (
    "RFJBQ08CAgEBAAAABAIAAgAAAR//AREB/wAAAQAJAwAAAgEBAQADAwEwARADACiCmAAAAAAA"
    "/z8AAAAAAAAAAAAAAAAAAAAAgD8O"
)


def _draco_quad_glb() -> bytes:
    """The `_two_triangle_glb` quad, but with its geometry Draco-compressed.

    Per the KHR_draco_mesh_compression spec the accessors keep count/type/min/max
    but carry no bufferView — the decoder supplies the data — and the primitive
    points at the compressed bufferView plus the per-attribute unique ids.
    """
    bin_chunk = base64.b64decode(_DRACO_QUAD_B64)
    draco_len = len(bin_chunk)
    bin_chunk += b"\x00" * (-len(bin_chunk) % 4)
    gltf = {
        "asset": {"version": "2.0"},
        "extensionsUsed": ["KHR_draco_mesh_compression"],
        "extensionsRequired": ["KHR_draco_mesh_compression"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "extensions": {
                            "KHR_draco_mesh_compression": {
                                "bufferView": 0,
                                "attributes": {"POSITION": 0},
                            }
                        },
                    }
                ]
            }
        ],
        "buffers": [{"byteLength": len(bin_chunk)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": draco_len}],
        "accessors": [
            {
                "componentType": 5126,
                "count": 4,
                "type": "VEC3",
                "min": [0, 0, 0],
                "max": [1, 1, 0],
            },
            {"componentType": 5125, "count": 6, "type": "SCALAR"},
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
def test_draco_compressed_glb_loads(viewer_client, viewer_page):
    """A KHR_draco_mesh_compression GLB decodes and renders (issue #167).

    The page is loaded from file:// with no network, so this also proves the
    decoder ships inside viewer.html: DRACOLoader never fetches a CDN, it gets
    the wasm + emscripten wrapper from the inlined gzip+base64 payload.
    """
    viewer_client.add_model_binary("quad", _draco_quad_glb(), format="glb")
    objects = {}
    for _ in range(100):
        settle(viewer_client)
        objects = viewer_client.query_scene()["objects"]
        if "quad" in objects:
            break
    assert "quad" in objects, "Draco GLB did not load"

    state = viewer_page.evaluate(
        "() => {"
        " const root = window.threejsViewer._objects.get('quad');"
        " let mesh = null;"
        " root.traverse(o => { if (o.isMesh && !mesh) mesh = o; });"
        " if (!mesh) return null;"
        " const g = mesh.geometry;"
        " g.computeBoundingBox();"
        " const bb = g.boundingBox;"
        " return {"
        "  verts: g.attributes.position.count,"
        "  indices: g.index ? g.index.count : 0,"
        "  min: bb.min.toArray(), max: bb.max.toArray(),"
        "  visible: mesh.visible,"
        "  decoderPath: window.threejsViewer._dracoLoader.decoderPath,"
        "  dracoWorkers: window.threejsViewer._dracoLoader.workerPool.length,"
        " };"
        "}"
    )
    assert state is not None, "no mesh under the loaded Draco model"
    # The decoder really ran (a worker was spun up) and it was never given a
    # URL prefix to fetch from — the bytes came from the inlined payload.
    assert state["dracoWorkers"] >= 1
    assert state["decoderPath"] == ""
    # Decoded geometry: the 4-vertex / 2-triangle unit quad in the XY plane.
    assert state["verts"] == 4
    assert state["indices"] == 6
    assert state["visible"] is True
    for got, want in zip(state["min"], [0.0, 0.0, 0.0]):
        assert abs(got - want) < 1e-3, state["min"]
    for got, want in zip(state["max"], [1.0, 1.0, 0.0]):
        assert abs(got - want) < 1e-3, state["max"]

    # The draw-range stamp (issue #104) runs on Draco models like any other.
    assert state["indices"] == viewer_page.evaluate(
        "() => window.threejsViewer._objects.get('quad')"
        ".userData.drawRangeMeshes[0].userData.totalIndexCount"
    )


@pytest.mark.browser
def test_uncompressed_glb_still_loads_with_draco_wired(viewer_client, viewer_page):
    """Attaching a DRACOLoader must not change the uncompressed path: the
    extension is per-primitive and GLTFLoader falls back on its own, so a plain
    GLB loads without the decoder ever being touched (issue #167). The worker
    count asserts that laziness — DRACOLoader spins one up only on a compressed
    primitive, so an all-uncompressed scene never inflates the wasm."""
    viewer_client.add_model_binary("plain", _two_triangle_glb(), format="glb")
    objects = {}
    for _ in range(60):
        settle(viewer_client)
        objects = viewer_client.query_scene()["objects"]
        if "plain" in objects:
            break
    assert "plain" in objects, "uncompressed GLB did not load"
    state = viewer_page.evaluate(
        "() => {"
        " const root = window.threejsViewer._objects.get('plain');"
        " let mesh = null;"
        " root.traverse(o => { if (o.isMesh && !mesh) mesh = o; });"
        " return { verts: mesh.geometry.attributes.position.count,"
        "          indices: mesh.geometry.index.count,"
        "          dracoWorkers: window.threejsViewer._dracoLoader.workerPool.length };"
        "}"
    )
    assert state["verts"] == 4
    assert state["indices"] == 6
    assert state["dracoWorkers"] == 0


@pytest.mark.browser
def test_orbit_pivot_ignores_invisible_objects(viewer_client, viewer_page):
    """A hidden object must not drag the fallback orbit pivot (#166).

    The near/far content sphere deliberately keeps invisible objects (they may
    be shown later and must not get clipped), so the pivot reads the framing
    bounds instead.
    """
    viewer_client.add_box("visible_box", position=[10.0, 0.0, 0.0])
    viewer_client.add_box("hidden_box", position=[1000.0, 0.0, 0.0])
    viewer_client.set_visible("hidden_box", False)
    assert "hidden_box" in viewer_client.query_scene()["objects"]  # sync

    result = viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " const fb = v._controls._fallbackPivotGetter();"
        " v._camController.updateSceneBounds();"
        " return { pivot: fb ? fb.x : null,"
        "          nearFarRadius: v._nearFarSphere.radius };"
        "}"
    )
    # Pivot sits on the visible box only...
    assert result["pivot"] is not None
    assert abs(result["pivot"] - 10.0) < 1.0
    # ...while the near/far fit still reaches the hidden one, so making it
    # visible later can't leave it clipped.
    assert result["nearFarRadius"] > 400


@pytest.mark.browser
def test_set_highlight_default_style_is_silhouette(viewer_client, viewer_page):
    """The default highlight is an inverted-hull contour that shares the
    mesh's geometry — legible on a smooth body, where a feature-edge outline
    has no crease to draw at all (issues #158/#165)."""
    viewer_client.add_sphere("ball", radius=1.0)
    _wait_highlight_state(viewer_page, "ball", lambda s: len(s["meshes"]) == 1)
    viewer_client.set_highlight("ball")
    state = _wait_highlight_state(viewer_page, "ball", lambda s: s["outlines"] == 1)
    assert state["outlineStyles"] == ["silhouette"]
    assert state["outlineSharesGeometry"] == [True]  # no duplicate buffers
    hull = viewer_page.evaluate(
        "() => {"
        " let h = null;"
        " window.threejsViewer._objects.get('ball').traverse((c) => {"
        "  if (c.userData.__highlightOutline) h = c;"
        " });"
        " const res = window.threejsViewer._highlightResolution.value;"
        " return h ? { side: h.material.side, depthTest: h.material.depthTest,"
        "              depthWrite: h.material.depthWrite,"
        "              width: h.material.userData.__highlightWidth.value,"
        "              resX: res.x } : null;"
        "}"
    )
    assert hull["side"] == 1  # THREE.BackSide
    # Depth-tested on purpose: a depthTest:false hull would paint its
    # backfaces over the object body.
    assert hull["depthTest"] is True
    assert hull["depthWrite"] is False
    assert hull["width"] == 3
    assert hull["resX"] > 0  # viewport uniform wired for screen-constant width


@pytest.mark.browser
def test_set_highlight_silhouette_is_visible_on_a_smooth_body(
    viewer_client, viewer_page
):
    """End-to-end: the silhouette actually renders around a sphere, which is
    the case the feature-edge outline cannot express (issue #165)."""
    viewer_client.add_sphere("ball", radius=1.0, color=0x1133AA)
    _wait_highlight_state(viewer_page, "ball", lambda s: len(s["meshes"]) == 1)
    viewer_client.frame_object("ball")

    def orange_pixels():
        # Render and read the drawing buffer in one JS turn: the canvas has no
        # preserveDrawingBuffer, so a later drawImage() would come back blank.
        return viewer_page.evaluate(
            "() => {"
            " const v = window.threejsViewer;"
            " v._renderer.render(v._scene, v._camera);"
            " const gl = v._renderer.getContext();"
            " const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;"
            " const px = new Uint8Array(w * h * 4);"
            " gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);"
            " let n = 0;"
            " for (let i = 0; i < px.length; i += 4) {"
            "  if (px[i] > 150 && px[i + 1] > 80 && px[i + 1] < 200 && px[i + 2] < 90) n++;"
            " }"
            " return n;"
            "}"
        )

    for _ in range(60):
        if orange_pixels() == 0:
            break
        time.sleep(0.05)
    assert orange_pixels() == 0, "no selection orange before highlighting"

    viewer_client.set_highlight("ball", width_px=6)
    _wait_highlight_state(viewer_page, "ball", lambda s: s["outlines"] == 1)
    lit = 0
    for _ in range(60):
        lit = orange_pixels()
        if lit > 200:
            break
        time.sleep(0.05)
    assert lit > 200, f"silhouette must draw a visible rim, got {lit} px"

    viewer_client.set_highlight("ball", enabled=False)
    for _ in range(60):
        if orange_pixels() == 0:
            break
        time.sleep(0.05)
    assert orange_pixels() == 0, "disable must leave no trace of the outline"


@pytest.mark.browser
def test_set_highlight_silhouette_falls_back_on_a_translucent_mesh(
    viewer_client, viewer_page
):
    """A silhouette hull needs the mesh in the depth buffer to cull its
    backfaces. set_opacity() below 1 clears depthWrite, so the hull would draw
    in full and paint a solid shell of the selection colour over the object —
    fall back to the feature-edge outline there instead."""
    viewer_client.add_box(
        "tbox", width=2.0, height=2.0, depth=2.0, color=0x1133AA, opacity=0.4
    )
    _wait_highlight_state(viewer_page, "tbox", lambda s: len(s["meshes"]) == 1)
    viewer_client.set_highlight("tbox")
    state = _wait_highlight_state(viewer_page, "tbox", lambda s: s["outlines"] == 1)
    assert state["outlineStyles"] == ["edges"], (
        "a non-depth-writing mesh must not get a silhouette hull"
    )
    # And the object is genuinely the translucent case that triggers it.
    assert state["meshes"][0]["transparent"] is True


@pytest.mark.browser
def test_set_highlight_does_not_paint_a_shell_over_a_translucent_mesh(
    viewer_client, viewer_page
):
    """The visible consequence of the fallback: the highlight stays a thin
    outline instead of covering the object's whole projected footprint.

    A box is used rather than a sphere because it has creases for the edge
    outline to draw, so this measures 'outline, not shell' rather than
    'nothing at all'.
    """
    viewer_client.add_box(
        "tbox", width=2.0, height=2.0, depth=2.0, color=0x1133AA, opacity=0.4
    )
    _wait_highlight_state(viewer_page, "tbox", lambda s: len(s["meshes"]) == 1)
    viewer_client.frame_object("tbox")
    viewer_client.set_highlight("tbox", width_px=6)
    _wait_highlight_state(viewer_page, "tbox", lambda s: s["outlines"] == 1)

    def coverage():
        # Orange highlight pixels vs. all non-background pixels, in one JS turn
        # (no preserveDrawingBuffer, so the read must follow the render).
        return viewer_page.evaluate(
            "() => {"
            " const v = window.threejsViewer;"
            " v._renderer.render(v._scene, v._camera);"
            " const gl = v._renderer.getContext();"
            " const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;"
            " const px = new Uint8Array(w * h * 4);"
            " gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);"
            " let orange = 0, covered = 0;"
            " for (let i = 0; i < px.length; i += 4) {"
            "  const r = px[i], g = px[i + 1], b = px[i + 2];"
            "  if (Math.abs(r - 34) < 12 && Math.abs(g - 34) < 12"
            "      && Math.abs(b - 34) < 12) continue;"  # background #222222
            "  covered++;"
            "  if (r > 150 && g > 80 && g < 200 && b < 90) orange++;"
            " }"
            " return { orange, covered };"
            "}"
        )

    seen = {"orange": 0, "covered": 0}
    for _ in range(60):
        seen = coverage()
        if seen["covered"] > 500 and seen["orange"] > 0:
            break
        time.sleep(0.05)
    assert seen["covered"] > 500, "box must be framed and visible"
    assert seen["orange"] > 0, "highlight must draw something"
    # A silhouette hull on a non-depth-writing mesh covers essentially the
    # whole footprint; an outline is a thin fraction of it.
    assert seen["orange"] < 0.35 * seen["covered"], (
        f"highlight painted a shell, not an outline: "
        f"{seen['orange']}/{seen['covered']} px"
    )


@pytest.mark.browser
def test_set_highlight_style_re_resolves_when_opacity_changes(
    viewer_client, viewer_page
):
    """The style is re-resolved when set_opacity flips depthWrite under an
    already-highlighted mesh, so the order of the two calls does not matter."""
    viewer_client.add_box("obox", width=2.0, height=2.0, depth=2.0, color=0x1133AA)
    _wait_highlight_state(viewer_page, "obox", lambda s: len(s["meshes"]) == 1)
    viewer_client.set_highlight("obox")
    state = _wait_highlight_state(viewer_page, "obox", lambda s: s["outlines"] == 1)
    assert state["outlineStyles"] == ["silhouette"], "opaque mesh gets the hull"

    # Highlight first, opacity second: the hull must give way to edges.
    viewer_client.set_opacity("obox", 0.4)
    state = _wait_highlight_state(
        viewer_page, "obox", lambda s: s["outlineStyles"] == ["edges"]
    )
    assert state["outlineStyles"] == ["edges"]
    assert state["outlines"] == 1, "re-resolving must not stack a second outline"

    # Back to opaque: the hull comes back.
    viewer_client.set_opacity("obox", 1.0)
    state = _wait_highlight_state(
        viewer_page, "obox", lambda s: s["outlineStyles"] == ["silhouette"]
    )
    assert state["outlineStyles"] == ["silhouette"]
    assert state["outlines"] == 1

    # Still fully reversible after the style round-trip.
    viewer_client.set_highlight("obox", enabled=False)
    state = _wait_highlight_state(viewer_page, "obox", lambda s: s["outlines"] == 0)
    assert state["outlines"] == 0
    assert state["meshes"][0]["hasEdgeRef"] is False


@pytest.mark.browser
def test_set_highlight_edges_threshold_angle(viewer_client, viewer_page):
    """The outline's edge-detection threshold drops tessellation edges by
    default and is tunable per call, rebuilding in place (issue #158).

    A sphere is the worst case from the issue: at THREE's own 1 degree
    default every triangulation edge survives and the highlight reads as an
    X-ray wireframe.
    """
    viewer_client.add_sphere("ball", radius=1.0)
    _wait_highlight_state(viewer_page, "ball", lambda s: len(s["meshes"]) == 1)

    viewer_client.set_highlight("ball", style="edges")
    default = _wait_highlight_state(viewer_page, "ball", lambda s: s["outlines"] == 1)
    assert default["outlineThresholds"] == [30]

    # Same object at the old 1-degree behaviour: every tessellation edge.
    viewer_client.set_highlight("ball", style="edges", threshold_deg=1.0)
    fine = _wait_highlight_state(
        viewer_page, "ball", lambda s: s["outlineThresholds"] == [1]
    )
    assert fine["outlines"] == 1  # rebuilt in place, never stacked
    # A smooth-shaded sphere has no crease above 30 degrees at all, so the
    # default outline is empty where the 1-degree one is the full wire grid.
    assert default["outlineSegs"][0] == 0
    assert fine["outlineSegs"][0] > 500

    # A box's 90-degree creases survive the default threshold.
    viewer_client.add_box("cube")
    viewer_client.set_highlight("cube", style="edges")
    cube = _wait_highlight_state(viewer_page, "cube", lambda s: s["outlines"] == 1)
    assert cube["outlineSegs"][0] == 12  # exactly the 12 cube edges


@pytest.mark.browser
def test_set_highlight_threshold_survives_toggle_off(viewer_client, viewer_page):
    """Disposing and re-enabling after a custom threshold leaves no stale
    geometry and restores the default."""
    viewer_client.add_box("tbox")
    _wait_highlight_state(viewer_page, "tbox", lambda s: len(s["meshes"]) == 1)
    viewer_client.set_highlight("tbox", style="edges", threshold_deg=120.0)
    coarse = _wait_highlight_state(
        viewer_page, "tbox", lambda s: s["outlineThresholds"] == [120]
    )
    assert coarse["outlineSegs"][0] == 0  # no crease that sharp on a box
    viewer_client.set_highlight("tbox", enabled=False)
    _wait_highlight_state(viewer_page, "tbox", lambda s: s["outlines"] == 0)
    viewer_client.set_highlight("tbox", style="edges")
    back = _wait_highlight_state(viewer_page, "tbox", lambda s: s["outlines"] == 1)
    assert back["outlineThresholds"] == [30]
    assert back["outlineSegs"][0] == 12


@pytest.mark.browser
def test_set_highlight_style_switch_rebuilds_in_place(viewer_client, viewer_page):
    """Switching style disposes the old outline and builds the new one — still
    exactly one outline per mesh, and off leaves nothing behind."""
    viewer_client.add_box("sbox")
    _wait_highlight_state(viewer_page, "sbox", lambda s: len(s["meshes"]) == 1)
    viewer_client.set_highlight("sbox")
    hull = _wait_highlight_state(
        viewer_page, "sbox", lambda s: s["outlineStyles"] == ["silhouette"]
    )
    assert hull["outlines"] == 1
    viewer_client.set_highlight("sbox", style="edges")
    edges = _wait_highlight_state(
        viewer_page, "sbox", lambda s: s["outlineStyles"] == ["edges"]
    )
    assert edges["outlines"] == 1  # replaced, not stacked
    assert edges["outlineSegs"][0] == 12
    assert edges["outlineSharesGeometry"] == [False]  # its own EdgesGeometry
    viewer_client.set_highlight("sbox", style="silhouette")
    back = _wait_highlight_state(
        viewer_page, "sbox", lambda s: s["outlineStyles"] == ["silhouette"]
    )
    assert back["outlines"] == 1
    viewer_client.set_highlight("sbox", enabled=False)
    off = _wait_highlight_state(viewer_page, "sbox", lambda s: s["outlines"] == 0)
    assert off["outlines"] == 0
    # The hull shares the mesh geometry — disabling must not have disposed it.
    still_drawable = viewer_page.evaluate(
        "() => { const g = window.threejsViewer._objects.get('sbox').geometry;"
        " return !!(g && g.attributes.position && g.attributes.position.count > 0); }"
    )
    assert still_drawable


_WORLD_POS_JS = """(id) => {
    const o = window.threejsViewer.getObject(id);
    if (!o) return null;
    o.updateWorldMatrix(true, false);
    const e = o.matrixWorld.elements;
    return { local: o.position.toArray(), world: [e[12], e[13], e[14]] };
}"""


@pytest.mark.browser
@pytest.mark.parametrize(
    "kwargs",
    [{}, {"fat": False}, {"segments": True}],
    ids=["fat", "native", "segments"],
)
def test_add_polyline_applies_transform(viewer_client, viewer_page, kwargs):
    """add_polyline_binary honours data.transform on all three line variants
    (issue #194): the polyline lands at the requested pose, not the origin."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]], dtype=np.float32)
    viewer_client.add_polyline("pl", pts, position=[5, 6, 7], **kwargs)
    settle(viewer_client)
    got = viewer_page.evaluate(_WORLD_POS_JS, "pl")
    assert got is not None
    assert got["local"] == pytest.approx([5, 6, 7])
    assert got["world"] == pytest.approx([5, 6, 7])


@pytest.mark.browser
def test_add_polyline_applies_matrix_under_parent(viewer_client, viewer_page):
    """A matrix transform composes under a transformed parent group."""
    viewer_client.add_group("g", position=[10, 0, 0])
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    mat = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1]
    viewer_client.add_polyline("pl", pts, parent="g", matrix=mat)
    settle(viewer_client)
    got = viewer_page.evaluate(_WORLD_POS_JS, "pl")
    assert got["local"] == pytest.approx([1, 2, 3])
    assert got["world"] == pytest.approx([11, 2, 3])


@pytest.mark.browser
def test_add_points_applies_transform(viewer_client, viewer_page):
    """add_points_binary honours data.transform (same omission as #194)."""
    pts = np.random.default_rng(1).random((100, 3)).astype(np.float32)
    viewer_client.add_points("pc", pts, position=[3, 4, 5])
    settle(viewer_client)
    got = viewer_page.evaluate(_WORLD_POS_JS, "pc")
    assert got is not None
    assert got["world"] == pytest.approx([3, 4, 5])


@pytest.mark.browser
def test_add_points_lod_applies_transform(viewer_client, viewer_page):
    """add_points_lod honours data.transform on the octree group."""
    pts = np.random.default_rng(2).random((5000, 3)).astype(np.float32)
    viewer_client.add_points(
        "cloud", pts, lod={"node_capacity": 1000}, position=[3, 4, 5]
    )
    got = None
    for _ in range(100):
        time.sleep(0.05)
        got = viewer_page.evaluate(_WORLD_POS_JS, "cloud")
        if got is not None:
            break
    assert got is not None
    assert got["world"] == pytest.approx([3, 4, 5])


@pytest.mark.browser
def test_points_lod_nonuniform_scale_uses_max_component(viewer_client, viewer_page):
    """A LOD cloud scaled [1, 1, 4] must refine exactly like one scaled
    [4, 4, 4]: the node-size estimate bounds the radius by the largest scale
    component, so the stretched axis never stops refinement early. Under
    ortho the estimate ignores camera position, so the wanted sets compare
    exactly."""
    pts = np.random.default_rng(5).random((20_000, 3)).astype(np.float32)
    lod = {"node_capacity": 1000, "point_budget": 1_000_000, "refine_pixels": 100}
    for cid, scale in (("s111", [1, 1, 1]), ("s114", [1, 1, 4]), ("s444", [4, 4, 4])):
        viewer_client.add_points(cid, pts, lod=lod, scale=scale)
    viewer_page.evaluate("() => window.threejsViewer._camController.switch(true)")
    wanted = None
    for _ in range(100):
        time.sleep(0.05)
        wanted = viewer_page.evaluate(
            "() => {"
            " const out = {};"
            " for (const id of ['s111', 's114', 's444']) {"
            "   const g = window.threejsViewer._objects.get(id);"
            "   if (!g || !g.userData.pointsLOD) return null;"
            "   out[id] = g.userData.pointsLOD.wanted.reduce((a, b) => a + b, 0);"
            " }"
            " return out;"
            "}"
        )
        if wanted and wanted["s444"] > 1 and wanted["s114"] == wanted["s444"]:
            break
    assert wanted, "LOD clouds never appeared"
    assert wanted["s444"] > 1, f"scaled cloud never refined past the root: {wanted}"
    assert wanted["s114"] == wanted["s444"], wanted
    assert wanted["s111"] < wanted["s444"], wanted


_ZOOM_PROJECT_JS = """([x, y, z]) => {
    const v = window.threejsViewer;
    const THREE = window.tjsv.THREE;
    const rect = v._renderer.domElement.getBoundingClientRect();
    const cam = v._camera;
    cam.updateMatrixWorld(true);
    const ndc = new THREE.Vector3(x, y, z).project(cam);
    return {
        px: rect.left + ((ndc.x + 1) / 2) * rect.width,
        py: rect.top + ((1 - ndc.y) / 2) * rect.height,
        dist: cam.position.distanceTo(v._controls.target),
        zoom: cam.zoom,
        ortho: !!v._isOrtho,
        target: v._controls.target.toArray(),
    };
}"""


def _zoom_drift_at_cursor(page, world_point, delta_y, n_events):
    """Put the mouse on `world_point`'s projection, wheel n times, and return
    the pixel drift of that world point plus the before/after camera state.

    Chromium rounds a synthetic wheel event's clientX/clientY to whole pixels
    (a move to 791.39 arrives as 791), so the anchor sits up to 0.5 px off the
    projected point and the drift after a 1.85x zoom lands around 0.5 px. The
    1 px tolerance covers that; the exact-NDC test below shows the math itself
    is exact to 1e-13 px.
    """
    before = page.evaluate(_ZOOM_PROJECT_JS, world_point)
    page.mouse.move(before["px"], before["py"])
    for _ in range(n_events):
        page.mouse.wheel(0, delta_y)
    frames(page, 3)
    after = page.evaluate(_ZOOM_PROJECT_JS, world_point)
    drift = math.hypot(after["px"] - before["px"], after["py"] - before["py"])
    return drift, before, after


@pytest.mark.browser
def test_wheel_zoom_anchors_on_cursor_perspective_and_ortho(viewer_client, viewer_page):
    """Issue #192: wheel zoom keeps the world point under the cursor at the
    same screen pixel, in both projections, instead of converging on the orbit
    target. The camera still moves (distance / zoom change), and the target
    shifts with the camera so the ViewHelper centre (same Vector3) follows."""
    viewer_client.add_box(
        "b", width=0.5, height=0.5, depth=0.5, position=[1.5, 0.5, 0.0]
    )
    viewer_client.set_camera(position=[6, -6, 5], target=[0, 0, 0], up=[0, 0, 1])
    settle(viewer_client)
    frames(viewer_page, 2)
    world_point = [1.5, 0.5, 0.0]

    # Perspective: zoom in, then zoom out, cursor parked on the box.
    drift_in, before, after = _zoom_drift_at_cursor(viewer_page, world_point, -300, 4)
    assert after["ortho"] is False
    assert after["dist"] < before["dist"] * 0.8, "zoom in shortened the dolly"
    assert drift_in < 1.0, f"perspective zoom-in drift {drift_in:.3f}px"
    assert after["target"] != before["target"], "target rides along with the camera"
    drift_out, before, after = _zoom_drift_at_cursor(viewer_page, world_point, 300, 4)
    assert after["dist"] > before["dist"] * 1.2
    assert drift_out < 1.0, f"perspective zoom-out drift {drift_out:.3f}px"
    # The screen centre (the old anchor) is not fixed anymore when the cursor
    # is off-centre: the target projection should have moved.
    print(f"perspective drift in={drift_in:.4f}px out={drift_out:.4f}px")

    # Orthographic (manual O key path), same check on cam.zoom.
    viewer_page.evaluate("() => window.threejsViewer._switchCamera(true)")
    frames(viewer_page, 2)
    drift_o_in, before, after = _zoom_drift_at_cursor(viewer_page, world_point, -300, 4)
    assert after["ortho"] is True
    assert after["zoom"] > before["zoom"] * 1.2, "ortho zoom increased"
    assert drift_o_in < 1.0, f"ortho zoom-in drift {drift_o_in:.3f}px"
    drift_o_out, before, after = _zoom_drift_at_cursor(viewer_page, world_point, 300, 4)
    assert after["zoom"] < before["zoom"] * 0.8
    assert drift_o_out < 1.0, f"ortho zoom-out drift {drift_o_out:.3f}px"
    print(f"ortho drift in={drift_o_in:.4f}px out={drift_o_out:.4f}px")

    # After the long zoom the ViewHelper centre is still the controls' target
    # (same Vector3, mutated in place), so click-to-pivot and the gimbal agree.
    same = viewer_page.evaluate(
        "() => window.threejsViewer._viewHelper.center === window.threejsViewer._controls.target"
    )
    assert same is True


@pytest.mark.browser
def test_wheel_zoom_anchor_math_is_exact(viewer_client, viewer_page):
    """Drive `_applyZoom` with the float NDC of a world point directly, so the
    wheel event's integer pixel rounding is out of the picture: 12 steps in and
    12 back out leave the point's projection where it was to sub-1e-6 px."""
    viewer_client.add_box(
        "b", width=0.5, height=0.5, depth=0.5, position=[1.5, 0.5, 0.0]
    )
    viewer_client.set_camera(position=[6, -6, 5], target=[0, 0, 0], up=[0, 0, 1])
    settle(viewer_client)
    frames(viewer_page, 2)
    js = (
        "([x, y, z]) => {"
        " const v = window.threejsViewer; const THREE = window.tjsv.THREE;"
        " const cam = v._camera; cam.updateMatrixWorld(true);"
        " const rect = v._renderer.domElement.getBoundingClientRect();"
        " const n0 = new THREE.Vector3(x, y, z).project(cam);"
        " const d0 = cam.position.distanceTo(v._controls.target), z0 = cam.zoom;"
        " const out = [];"
        " for (const s of [1 / 0.95, 0.95]) {"
        "   for (let i = 0; i < 12; i++) v._controls._applyZoom(s, n0.x, n0.y);"
        "   cam.updateMatrixWorld(true);"
        "   const n1 = new THREE.Vector3(x, y, z).project(cam);"
        "   out.push(Math.hypot((n1.x - n0.x) / 2 * rect.width, (n1.y - n0.y) / 2 * rect.height));"
        " }"
        " return { drift: out, d0, d1: cam.position.distanceTo(v._controls.target),"
        "          z0, z1: cam.zoom };"
        "}"
    )
    persp = viewer_page.evaluate(js, [1.5, 0.5, 0.0])
    assert max(persp["drift"]) < 1e-6, persp
    assert abs(persp["d1"] - persp["d0"]) < 1e-9, "in then out restores the distance"
    viewer_page.evaluate("() => window.threejsViewer._switchCamera(true)")
    frames(viewer_page, 2)
    ortho = viewer_page.evaluate(js, [1.5, 0.5, 0.0])
    assert max(ortho["drift"]) < 1e-6, ortho
    assert abs(ortho["z1"] - ortho["z0"]) < 1e-9 * ortho["z0"], (
        "in then out restores the zoom"
    )


@pytest.mark.browser
def test_wheel_zoom_with_cursor_on_target_is_pure_dolly(viewer_client, viewer_page):
    """With the cursor exactly on the orbit target the zoom degenerates to the
    old about-target dolly: the target does not move at all."""
    viewer_client.add_box("b", width=0.5, height=0.5, depth=0.5, position=[0, 0, 0])
    viewer_client.set_camera(position=[6, -6, 5], target=[0, 0, 0], up=[0, 0, 1])
    settle(viewer_client)
    frames(viewer_page, 2)
    drift, before, after = _zoom_drift_at_cursor(viewer_page, [0, 0, 0], -300, 4)
    assert drift < 1.0
    assert after["dist"] < before["dist"] * 0.8
    assert max(abs(a - b) for a, b in zip(after["target"], before["target"])) < 1e-6
    # Programmatic zoom without a cursor keeps the about-target behaviour.
    moved = viewer_page.evaluate(
        "() => {"
        " const v = window.threejsViewer;"
        " const t0 = v._controls.target.toArray();"
        " v._controls._applyZoom(1.5);"
        " const t1 = v._controls.target.toArray();"
        " return t0.some((c, i) => Math.abs(c - t1[i]) > 1e-9);"
        "}"
    )
    assert moved is False


# Object click (issue #178) and dblclick framing switch (issue #177).


def _object_click_setup(viewer_client, viewer_page):
    """Box at the origin under a top-down camera; returns its screen centre."""
    viewer_client.add_box("box")
    _wait_for(viewer_page, "() => window.threejsViewer._objects.has('box')")
    viewer_page.evaluate(_GIZMO_TOPDOWN)
    frames(viewer_page, 2)
    return viewer_page.evaluate(_GIZMO_PROJECT_ORIGIN)


@pytest.mark.browser
def test_object_click_reports_id_to_python_and_js_hook(viewer_client, viewer_page):
    """A stationary click on an object reaches Python as object_clicked with
    the top-level id, the hit point and the button, and fires the JS hook."""
    proj = _object_click_setup(viewer_client, viewer_page)
    # Nothing is enabled yet, so the pointerup path skips the raycast.
    assert (
        viewer_page.evaluate("() => window.threejsViewer._objectClickEnabled") is False
    )
    clicks = []
    viewer_client.on_object_click(clicks.append)
    _wait_for(viewer_page, "() => window.threejsViewer._objectClickEnabled === true")
    viewer_page.evaluate(
        "() => { window.__clicks = [];"
        " window.threejsViewer.onObjectClick((p) => window.__clicks.push("
        "   {id: p.id, button: p.button, hasObj: !!p.object3D})); }"
    )

    viewer_page.mouse.click(proj["x"], proj["y"])
    assert _wait_until(lambda: len(clicks) >= 1), "no object_clicked reached Python"
    c = clicks[-1]
    assert c["id"] == "box"
    assert c["button"] == 0
    assert c["modifiers"] == {
        "shift": False,
        "ctrl": False,
        "alt": False,
        "meta": False,
    }
    # Top-down camera: the ray hits the box's +Z face (unit box, top at z=0.5).
    assert abs(c["point"][0]) < 0.1 and abs(c["point"][1]) < 0.1, c["point"]
    assert abs(c["point"][2] - 0.5) < 1e-3, c["point"]

    # Right-click reports button 2; shift-click carries the modifier.
    viewer_page.mouse.click(proj["x"], proj["y"], button="right")
    assert _wait_until(lambda: len(clicks) >= 2)
    assert clicks[-1]["id"] == "box" and clicks[-1]["button"] == 2
    viewer_page.keyboard.down("Shift")
    viewer_page.mouse.click(proj["x"], proj["y"])
    viewer_page.keyboard.up("Shift")
    assert _wait_until(lambda: len(clicks) >= 3)
    assert clicks[-1]["modifiers"]["shift"] is True

    js = viewer_page.evaluate("() => window.__clicks")
    assert [j["id"] for j in js] == ["box", "box", "box"]
    assert [j["button"] for j in js] == [0, 2, 0]
    assert all(j["hasObj"] for j in js)


@pytest.mark.browser
def test_object_click_drag_is_not_a_click(viewer_client, viewer_page):
    """A 20 px drag between pointerdown and pointerup (an orbit) reports
    nothing; a stationary click afterwards still does."""
    proj = _object_click_setup(viewer_client, viewer_page)
    clicks = []
    viewer_client.on_object_click(clicks.append)
    _wait_for(viewer_page, "() => window.threejsViewer._objectClickEnabled === true")

    cx, cy = proj["x"], proj["y"]
    viewer_page.mouse.move(cx, cy)
    viewer_page.mouse.down()
    for i in range(1, 5):
        viewer_page.mouse.move(cx + i * 5, cy)
    viewer_page.mouse.up()
    # The drag orbited the camera; give the WS a moment to prove silence.
    assert not _wait_until(lambda: bool(clicks), timeout=0.4)

    # Positive control on the same setup: reset the camera the drag moved,
    # re-project the box, and a stationary click reports it.
    viewer_page.evaluate(_GIZMO_TOPDOWN)
    frames(viewer_page, 2)
    proj = viewer_page.evaluate(_GIZMO_PROJECT_ORIGIN)
    viewer_page.mouse.click(proj["x"], proj["y"])
    assert _wait_until(lambda: bool(clicks))
    assert clicks[-1]["id"] == "box"


@pytest.mark.browser
def test_object_click_empty_space_reports_null(viewer_client, viewer_page):
    """A click that hits nothing reports id None and point None, so a
    consumer can deselect on it."""
    proj = _object_click_setup(viewer_client, viewer_page)
    clicks = []
    viewer_client.on_object_click(clicks.append)
    _wait_for(viewer_page, "() => window.threejsViewer._objectClickEnabled === true")
    w = viewer_page.evaluate(
        "() => window.threejsViewer._renderer.domElement.clientWidth"
    )
    # Well clear of the unit box (a few hundred px at this camera distance)
    # and away from the bottom-right view gimbal.
    viewer_page.mouse.click(proj["x"] - 0.3 * w, proj["y"])
    assert _wait_until(lambda: bool(clicks))
    assert clicks[-1] == {
        "id": None,
        "point": None,
        "button": 0,
        "modifiers": {"shift": False, "ctrl": False, "alt": False, "meta": False},
    }


@pytest.mark.browser
def test_object_click_not_on_gizmo_handle(viewer_client, viewer_page):
    """With a move gizmo attached, a press on one of its handles is the
    gizmo's gesture and never reports as an object click; click-select still
    works alongside the object-click event without double effects."""
    proj = _object_click_setup(viewer_client, viewer_page)
    clicks = []
    viewer_client.on_object_click(clicks.append)
    viewer_client.enable_move_gizmo()
    _wait_for(viewer_page, "() => window.threejsViewer._transformGizmo.enabled")
    _wait_for(viewer_page, "() => window.threejsViewer._objectClickEnabled === true")

    # Click-select attaches the gizmo, and the same click reports to Python.
    viewer_page.mouse.click(proj["x"] - 15, proj["y"] + 15)
    _wait_for(
        viewer_page, "() => window.threejsViewer._transformGizmo.objectId === 'box'"
    )
    assert _wait_until(lambda: len(clicks) == 1)
    assert clicks[0]["id"] == "box"

    # Hover the gizmo centre until TransformControls reports an axis, then
    # click there: the press lands on a handle, so no object click fires.
    viewer_page.mouse.move(proj["x"], proj["y"])
    frames(viewer_page, 2)
    _wait_for(
        viewer_page,
        "() => window.threejsViewer._transformGizmo.control.axis != null",
    )
    viewer_page.mouse.down()
    viewer_page.mouse.up()
    assert not _wait_until(lambda: len(clicks) > 1, timeout=0.4)


@pytest.mark.browser
def test_dblclick_frame_option_and_setter(viewer_client, viewer_page):
    """dblclickFrame:false leaves the camera untouched on a double-click
    (issue #177); the default still frames, and the runtime setter flips it."""
    viewer_client.add_box("box", position=[3.0, 0.0, 0.0])
    _wait_for(viewer_page, "() => window.threejsViewer._objects.has('box')")
    viewer_page.evaluate(_GIZMO_TOPDOWN)
    frames(viewer_page, 2)
    proj = viewer_page.evaluate(
        "() => { const v = window.threejsViewer;"
        " const w = v._renderer.domElement.clientWidth,"
        "       h = v._renderer.domElement.clientHeight;"
        " const ndc = v._camera.position.clone().set(3,0,0).project(v._camera);"
        " return { x: (ndc.x*0.5+0.5)*w, y: (-ndc.y*0.5+0.5)*h }; }"
    )
    cam = "() => window.threejsViewer._camera.position.toArray()"

    viewer_page.evaluate("() => window.threejsViewer.setDblclickFrame(false)")
    before = viewer_page.evaluate(cam)
    viewer_page.mouse.dblclick(proj["x"], proj["y"])
    frames(viewer_page, 5)
    assert viewer_page.evaluate(cam) == before

    viewer_page.evaluate("() => window.threejsViewer.setDblclickFrame(true)")
    viewer_page.mouse.dblclick(proj["x"], proj["y"])
    _wait_for(
        viewer_page,
        "() => { const p = window.threejsViewer._camera.position.toArray();"
        f" return p.some((c, i) => Math.abs(c - {json.dumps(before)}[i]) > 1e-3); }}",
    )

    # The constructor option lands on the instance without a runtime call.
    flags = viewer_page.evaluate(
        "() => {"
        " const live = window.threejsViewer;"
        " const V = live.constructor;"
        " const mk = (opts) => {"
        "   const div = document.createElement('div');"
        "   div.style.cssText ="
        "     'width:300px;height:200px;position:absolute;left:-2000px;top:0';"
        "   document.body.appendChild(div);"
        "   return new V(div, { htmlTemplate: live._options.htmlTemplate,"
        "     cubemapData: live._options.cubemapData, autoConnect: false, ...opts });"
        " };"
        " return [mk({})._dblclickFrame, mk({dblclickFrame: false})._dblclickFrame];"
        "}"
    )
    assert flags == [True, False]


@pytest.mark.browser
def test_object_click_skips_hidden_ancestor(viewer_client, viewer_page):
    """A tracked child under a group hidden with set_visible is unrendered,
    so a click on it reports null; showing the group again reports the child.
    The dblclick framing shares the hit test, so it must not frame it either."""
    viewer_client.add_group("grp")
    viewer_client.add_box("child", parent="grp")
    _wait_for(viewer_page, "() => window.threejsViewer._objects.has('child')")
    viewer_client.set_visible("grp", False)
    settle(viewer_client)
    viewer_page.evaluate(_GIZMO_TOPDOWN)
    frames(viewer_page, 2)
    proj = viewer_page.evaluate(_GIZMO_PROJECT_ORIGIN)
    clicks = []
    viewer_client.on_object_click(clicks.append)
    _wait_for(viewer_page, "() => window.threejsViewer._objectClickEnabled === true")

    # The child itself is still .visible === true; only its parent is hidden.
    assert viewer_page.evaluate(
        "() => window.threejsViewer._objects.get('child').visible"
    )
    viewer_page.mouse.click(proj["x"], proj["y"])
    assert _wait_until(lambda: bool(clicks))
    assert clicks[-1]["id"] is None
    assert (
        viewer_page.evaluate(
            f"() => window.threejsViewer._hitTrackedObject({proj['x']}, {proj['y']})"
        )
        is None
    )

    viewer_client.set_visible("grp", True)
    settle(viewer_client)
    viewer_page.mouse.click(proj["x"], proj["y"])
    assert _wait_until(lambda: len(clicks) >= 2)
    assert clicks[-1]["id"] == "child"


@pytest.mark.browser
def test_object_click_pointercancel_drops_press(viewer_client, viewer_page):
    """A pointercancel between press and release (a cancelled touch or pen)
    clears the pending press, so the later pointerup is not a click."""
    proj = _object_click_setup(viewer_client, viewer_page)
    clicks = []
    viewer_client.on_object_click(clicks.append)
    _wait_for(viewer_page, "() => window.threejsViewer._objectClickEnabled === true")
    viewer_page.mouse.move(proj["x"], proj["y"])
    viewer_page.mouse.down()
    assert viewer_page.evaluate("() => window.threejsViewer._objectClickDown !== null")
    viewer_page.evaluate(
        "() => window.dispatchEvent(new PointerEvent('pointercancel', {pointerId: 1}))"
    )
    assert viewer_page.evaluate("() => window.threejsViewer._objectClickDown === null")
    viewer_page.mouse.up()
    assert not _wait_until(lambda: bool(clicks), timeout=0.4)


@pytest.mark.browser
def test_destroy_removes_object_click_listeners(viewer_client, viewer_page):
    """destroy() removes the stored canvas pointerdown and window
    pointerup/pointercancel handlers, so a destroyed instance never raycasts."""
    removed = viewer_page.evaluate(
        "() => {"
        " const live = window.threejsViewer;"
        " const V = live.constructor;"
        " const div = document.createElement('div');"
        " div.style.cssText ="
        "   'width:300px;height:200px;position:absolute;left:-2000px;top:0';"
        " document.body.appendChild(div);"
        " const v2 = new V(div, { htmlTemplate: live._options.htmlTemplate,"
        "   cubemapData: live._options.cubemapData, autoConnect: false });"
        " const canvas = v2._renderer.domElement;"
        " const seen = [];"
        " const origWin = window.removeEventListener.bind(window);"
        " const origCanvas = canvas.removeEventListener.bind(canvas);"
        " window.removeEventListener = (t, fn, ...r) => {"
        "   if (fn === v2._onObjectClickUp || fn === v2._onObjectClickCancel) seen.push('window:' + t);"
        "   return origWin(t, fn, ...r); };"
        " canvas.removeEventListener = (t, fn, ...r) => {"
        "   if (fn === v2._onObjectClickDown) seen.push('canvas:' + t);"
        "   return origCanvas(t, fn, ...r); };"
        " try { v2.destroy(); } finally { window.removeEventListener = origWin; }"
        " return seen.sort();"
        "}"
    )
    assert removed == ["canvas:pointerdown", "window:pointercancel", "window:pointerup"]


# --- ws_host: WebSocket and sidecar on one non-default hostname (issue #187) ---


@pytest.mark.browser
def test_ws_host_param_routes_websocket_and_blobs_to_one_host(page):
    """``ViewerClient(host="127.0.0.1")`` must connect the WebSocket to that
    host (via ``ws_host``) and fetch blobs from it, not from localhost."""
    client = _start_client(host="127.0.0.1")
    try:
        page.goto(client.viewer_url, timeout=90_000)
        assert client._connected_event.wait(timeout=60)
        assert page.evaluate("() => window.threejsViewer._wsUrl") == (
            f"ws://127.0.0.1:{client.port}"
        )
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.uint32)
        client.add_mesh("wh", positions, indices)
        settle(client)
        assert "wh" in client.query_scene()["objects"]
    finally:
        client.disconnect()


# --- Firefox smoke test: sidecar fetch from a file:// page (issue #187) ---


@pytest.mark.browser
def test_firefox_file_page_loads_binary_asset(viewer_client, playwright):
    """One binary asset must land under Firefox from a file:// viewer page.

    Firefox treats a blob URL on a different host than the page's WebSocket
    host as a cross-origin request and refuses it (#185 advertised
    127.0.0.1 while the page used localhost). Chromium allows that, so the
    Chromium-only suite never saw it; this test keeps the one-hostname
    contract honest in the browser that enforces it.
    """
    from playwright.sync_api import Error as PlaywrightError

    try:
        # Headless Firefox on a GPU-less Linux runner refuses to create a WebGL
        # context (the viewer constructor then throws before connect() runs);
        # allow software rendering so it has a chance.
        browser = playwright.firefox.launch(
            firefox_user_prefs={
                "webgl.force-enabled": True,
                "webgl.forbid-software": False,
                "gfx.webrender.software": True,
            }
        )
    except PlaywrightError as exc:
        pytest.skip(f"Firefox not installed for Playwright: {exc}")
    try:
        page = browser.new_page()
        has_webgl2 = page.evaluate(
            "() => !!document.createElement('canvas').getContext('webgl2')"
        )
        if not has_webgl2:
            pytest.skip("Playwright Firefox cannot create a WebGL2 context here")
        # Firefox has no devtools in the CI log; keep its console for the
        # failure message so a non-connecting page explains itself.
        log = []
        page.on("console", lambda m: log.append(f"console[{m.type}]: {m.text}"))
        page.on("pageerror", lambda e: log.append(f"pageerror: {e}"))
        page.on(
            "requestfailed", lambda r: log.append(f"requestfailed: {r.url} {r.failure}")
        )
        viewer_path = viewer_client.viewer_path.resolve()
        page.goto(
            f"{viewer_path.as_uri()}?ws_port={viewer_client.port}", timeout=90_000
        )
        assert viewer_client._connected_event.wait(timeout=120), (
            "Firefox did not connect to the WebSocket server; page log:\n"
            + "\n".join(log[-40:])
        )
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.uint32)
        viewer_client.add_mesh("ff_mesh", positions, indices)
        settle(viewer_client)
        objects = viewer_client.query_scene()["objects"]
        assert "ff_mesh" in objects, f"mesh did not load in Firefox: {sorted(objects)}"
        assert objects["ff_mesh"]["type"] == "Mesh"
    finally:
        browser.close()


def _material_flags(page, obj_id):
    """Read the first material's wireframe/transparent/depthWrite for an object."""
    return page.evaluate(
        "(id) => {"
        "  const obj = window.threejsViewer._objects.get(id);"
        "  if (!obj) return null;"
        "  const m = Array.isArray(obj.material) ? obj.material[0] : obj.material;"
        "  if (!m) return null;"
        "  return {wireframe: m.wireframe, transparent: m.transparent,"
        "          depthWrite: m.depthWrite, opacity: m.opacity,"
        "          polygonOffset: m.polygonOffset,"
        "          polygonOffsetFactor: m.polygonOffsetFactor};"
        "}",
        obj_id,
    )


@pytest.mark.browser
def test_primitive_wireframe_param_is_honoured(viewer_client, viewer_page):
    """add_object's params.wireframe reaches the material (issue #207)."""
    viewer_client.add_box("wf", color=0x00CCFF, opacity=0.25, wireframe=True)
    viewer_client.add_box("solid", color=0x00CCFF, opacity=0.25)
    settle(viewer_client)
    assert _material_flags(viewer_page, "wf")["wireframe"] is True
    assert _material_flags(viewer_page, "solid")["wireframe"] is False


@pytest.mark.browser
def test_primitive_polygon_offset_param_is_honoured(viewer_client, viewer_page):
    """add_object's params.polygonOffset reaches the material (issue #207)."""
    viewer_client.add_box("po", polygon_offset=-1.0, polygon_offset_units=-2.0)
    settle(viewer_client)
    flags = _material_flags(viewer_page, "po")
    assert flags["polygonOffset"] is True
    assert flags["polygonOffsetFactor"] == -1.0


@pytest.mark.browser
def test_translucent_primitive_depth_write_matches_set_opacity(
    viewer_client, viewer_page
):
    """A primitive added translucent sorts like one turned translucent via
    set_color, and a round trip back to the original opacity restores the
    original state (issue #207)."""
    viewer_client.add_box("a", color=0x00CCFF, opacity=0.25)
    settle(viewer_client)
    assert _material_flags(viewer_page, "a")["depthWrite"] is False

    viewer_client.set_color("a", 0xFF2020, opacity=0.45)
    settle(viewer_client)
    assert _material_flags(viewer_page, "a")["depthWrite"] is False

    viewer_client.set_color("a", 0x00CCFF, opacity=0.25)
    settle(viewer_client)
    after = _material_flags(viewer_page, "a")
    assert after["depthWrite"] is False and abs(after["opacity"] - 0.25) < 1e-6

    viewer_client.set_opacity("a", 1.0)
    settle(viewer_client)
    assert _material_flags(viewer_page, "a")["depthWrite"] is True


@pytest.mark.browser
def test_explicit_depth_write_survives_set_opacity(viewer_client, viewer_page):
    """An explicit params.depthWrite outranks applyOpacity's opacity rule
    (issue #207)."""
    viewer_client.add_box("keep", opacity=0.25, depth_write=True)
    settle(viewer_client)
    assert _material_flags(viewer_page, "keep")["depthWrite"] is True
    viewer_client.set_opacity("keep", 0.5)
    settle(viewer_client)
    assert _material_flags(viewer_page, "keep")["depthWrite"] is True


def _material_side(page, obj_id):
    return page.evaluate(
        "(id) => {"
        "  const obj = window.threejsViewer._objects.get(id);"
        "  const m = Array.isArray(obj.material) ? obj.material[0] : obj.material;"
        "  return m.side;"
        "}",
        obj_id,
    )


@pytest.mark.browser
def test_translucent_primitive_is_double_sided(viewer_client, viewer_page):
    """A translucent body draws its far walls too, so the shape reads as a
    solid instead of a flat silhouette; an opaque one stays front-only."""
    viewer_client.add_box("clear", color=0x00CCFF, opacity=0.3)
    viewer_client.add_box("solid", color=0x00CCFF)
    viewer_client.add_box("cage", color=0x00CCFF, opacity=0.3, wireframe=True)
    viewer_client.add_box("forced", color=0x00CCFF, opacity=0.3, side="front")
    settle(viewer_client)
    # three's THREE.FrontSide / THREE.DoubleSide; the module is not a page global.
    front_side, double_side = 0, 2
    assert _material_side(viewer_page, "clear") == double_side
    assert _material_side(viewer_page, "solid") == front_side
    # A cage has no interior to reveal, so it keeps the cheaper front-only draw.
    assert _material_side(viewer_page, "cage") == front_side
    assert _material_side(viewer_page, "forced") == front_side
