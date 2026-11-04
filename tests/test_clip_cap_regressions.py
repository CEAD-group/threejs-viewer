"""Adversarial regressions from the Codex (OpenAI) review of PR #321."""

import numpy as np
import pytest

from conftest import frames, settle
from test_clip_caps import (
    _Glb,
    _PIXEL_AT,
    _box_arrays,
    _caps,
    _model_url,
    _png,
    _shown_rgb,
    _textured_box_glb,
    _use_worker,
    _wait_caps,
    _wait_object,
    _wait_settled,
)

pytestmark = pytest.mark.browser


def _clip(client, page, worker):
    _use_worker(page, worker)
    client.set_clipping_caps(True)
    client.set_clipping_plane(normal=[0, 0, 1], distance=0.123, show_helper=False)
    settle(client)
    _wait_caps(page, "m", 1)


@pytest.mark.parametrize("worker", [False, True])
def test_cap_rebuilds_when_position_attribute_is_replaced(
    viewer_client, viewer_page, worker
):
    pos, _, idx = _box_arrays()
    viewer_client.add_mesh("m", pos, idx)
    _clip(viewer_client, viewer_page, worker)
    assert _caps(viewer_page, "m")[0]["area"] == pytest.approx(4)
    viewer_page.evaluate("""() => {
        const g = window.threejsViewer.getObject('m').geometry;
        const p = g.getAttribute('position').clone();
        for (let i = 0; i < p.count; i++) p.setX(i, p.getX(i) * 2);
        g.setAttribute('position', p);
    }""")
    frames(viewer_page)
    _wait_settled(viewer_page)
    assert _caps(viewer_page, "m")[0]["area"] == pytest.approx(8)


@pytest.mark.parametrize("worker", [False, True])
def test_cap_samples_selected_texture_uv_channel(viewer_client, viewer_page, worker):
    g = _Glb()
    mesh = g.box_mesh()
    pos, _, _ = _box_arrays()
    prim = g.gltf["meshes"][mesh]["primitives"][0]
    for channel, x in [(0, 0.25), (1, 0.75)]:
        uv = np.tile(np.array([x, 0.5], dtype=np.float32), (len(pos), 1))
        prim["attributes"][f"TEXCOORD_{channel}"] = g.accessor(
            g.view(uv.tobytes()), 5126, len(pos), "VEC2"
        )
    png = _png(16, 16, (32, 32, 32), (224, 208, 192))
    g.gltf["images"] = [{"bufferView": g.view(png), "mimeType": "image/png"}]
    g.gltf["textures"] = [{"source": 0}]
    g.gltf["materials"] = [
        {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0, "texCoord": 1}}}
    ]
    prim["material"] = 0
    viewer_client.add_model_binary("m", g.bytes([g.node(mesh=mesh)]), "glb")
    _clip(viewer_client, viewer_page, worker)
    frames(viewer_page)
    assert _shown_rgb(viewer_page, "m") == pytest.approx([224, 208, 192], abs=2)


@pytest.mark.parametrize("vertex_colors", [False, True])
def test_cap_texture_rebake_reuses_color_buffer(
    viewer_client, viewer_page, vertex_colors
):
    viewer_client.add_model_binary(
        "m", _textured_box_glb(_png(8, 8, (32, 64, 96))), "glb"
    )
    settle(viewer_client)
    _wait_object(viewer_page, "m")
    if vertex_colors:
        viewer_page.evaluate("""() => window.threejsViewer.getObject('m').traverse(n => {
            if (!n.isMesh) return;
            const color = n.geometry.getAttribute('position').clone();
            color.array.fill(0.5);
            n.geometry.setAttribute('color', color);
            n.material.vertexColors = true;
            n.material.needsUpdate = true;
        })""")
    _clip(viewer_client, viewer_page, False)
    frames(viewer_page)
    shown = _shown_rgb(viewer_page, "m")
    linear = viewer_page.evaluate("""() => {
        let result;
        window.threejsViewer.getObject('m').traverse(n => {
            if (n.userData.__clipCap) result = Array.from(n.geometry.getAttribute('color').array.slice(0, 3));
        });
        return result;
    }""")
    expected = [((v / 255 + 0.055) / 1.055) ** 2.4 for v in (32, 64, 96)]
    assert linear == pytest.approx(
        np.array(expected) * (0.5 if vertex_colors else 1), rel=1e-5
    )
    viewer_page.evaluate("""() => {
        const root = window.threejsViewer.getObject('m');
        root.traverse(n => {
            if (n.userData.__clipCap) {
                window.reviewCap = n;
                window.reviewCapColor = n.geometry.getAttribute('color');
                window.reviewBake = n.userData.__regionBakeAt;
            }
            else if (n.isMesh) n.material.map.offset.x = 0.1;
        });
    }""")
    # Rebaking is intentionally throttled to 250 ms. Wait for its state change.
    viewer_page.wait_for_function(
        "() => window.reviewCap.userData.__regionBakeAt > window.reviewBake"
    )
    assert viewer_page.evaluate("""() => {
        let same = false;
        window.threejsViewer.getObject('m').traverse(n => {
            if (n.userData.__clipCap) same = n.geometry.getAttribute('color') === window.reviewCapColor;
        });
        return same;
    }""")

    assert _shown_rgb(viewer_page, "m") == shown
    if vertex_colors:
        restored = viewer_page.evaluate("""() => {
            window.threejsViewer._clipCaps._bakeRegionColors(window.reviewCap, [{}]);
            return Array.from(window.reviewCap.geometry.getAttribute('color').array.slice(0, 3));
        }""")
        assert restored == pytest.approx([0.5] * 3)


def test_clones_with_different_maps_reuse_worker_sources(viewer_client, viewer_page):
    url = _model_url(
        viewer_client, "/review_clone.glb", _textured_box_glb(_png(8, 8, (32, 64, 96)))
    )
    viewer_client.add_model("m", url, format="glb")
    viewer_client.add_model("plain", url, format="glb", position=[3, 0, 0])
    settle(viewer_client)
    _wait_object(viewer_page, "m")
    _wait_object(viewer_page, "plain")
    viewer_page.evaluate("""() => window.threejsViewer.getObject('plain').traverse(n => {
        if (n.isMesh) { n.material.map = null; n.material.needsUpdate = true; }
    })""")
    _clip(viewer_client, viewer_page, True)
    _wait_caps(viewer_page, "plain", 1)
    sent = viewer_page.evaluate("() => window.threejsViewer._clipCaps.sourcesSent")
    for distance in [0.25, 0.5, 0.75]:
        viewer_client.set_clipping_plane(
            normal=[0, 0, 1], distance=distance, show_helper=False
        )
        settle(viewer_client)
        frames(viewer_page)
        _wait_settled(viewer_page)
    assert (
        viewer_page.evaluate("() => window.threejsViewer._clipCaps.sourcesSent") == sent
    )


@pytest.mark.parametrize("worker", [False, True])
def test_touching_sheet_does_not_disable_solid_cap(viewer_client, viewer_page, worker):
    pos, _, idx = _box_arrays()
    # The sheet shares exactly one corner with the box and crosses the cut.
    sheet = np.array([[1, 1, 1], [3, 1, 1], [3, 2, -1], [1, 2, -1]], dtype=np.float32)
    viewer_client.add_mesh(
        "m",
        np.concatenate([pos, sheet]),
        np.concatenate([idx, np.array([0, 1, 2, 0, 2, 3]) + len(pos)]).astype(
            np.uint32
        ),
    )
    _use_worker(viewer_page, worker)
    viewer_client.set_clipping_caps(True)
    viewer_client.set_clipping_plane(
        normal=[0, 0, 1], distance=0.123, show_helper=False
    )
    settle(viewer_client)
    frames(viewer_page)
    _wait_settled(viewer_page)
    caps = _caps(viewer_page, "m")
    assert len(caps) == 1
    assert caps[0]["area"] == pytest.approx(4)


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize("offset, expected", [([1, 0.5, 0], 6.5), ([1, 0, 0], 6.0)])
@pytest.mark.parametrize("reversed_winding", [False, True])
def test_overlapping_boxes_have_union_cap_area(
    viewer_client, viewer_page, worker, offset, expected, reversed_winding
):
    pos, _, idx = _box_arrays()
    if reversed_winding:
        idx = idx.reshape(-1, 3)[:, ::-1].ravel()
    other = pos + np.array(offset, dtype=np.float32)
    viewer_client.add_mesh(
        "m", np.concatenate([pos, other]), np.concatenate([idx, idx + len(pos)])
    )
    _clip(viewer_client, viewer_page, worker)
    # Account for the shared rectangle exactly, including collinear boundaries.
    cap = _caps(viewer_page, "m")[0]
    assert cap["area"] == pytest.approx(expected)
    assert cap["dist"][0] < 1e-5


@pytest.mark.parametrize("worker", [False, True])
def test_overlapping_cap_preserves_holes_and_vertex_colors(
    viewer_client, viewer_page, worker
):
    outer, _, oi = _box_arrays(2)
    inner, _, ii = _box_arrays(1)
    plug, _, pi = _box_arrays(1)
    plug += np.array([1, 0.25, 0], dtype=np.float32)
    pos = np.concatenate([outer, inner, plug])
    idx = np.concatenate(
        [
            oi,
            ii.reshape(-1, 3)[:, ::-1].ravel() + len(outer),
            pi + len(outer) + len(inner),
        ]
    )
    # A linear gradient has the same value whichever source triangle contains a
    # newly created intersection vertex. The uncovered hole must remain empty.
    colors = np.zeros_like(pos)
    colors[:, 0] = (pos[:, 0] + 2) / 4
    colors[:, 1] = 0.5
    viewer_client.add_mesh("m", pos, idx, colors=colors)
    _clip(viewer_client, viewer_page, worker)
    cap = _caps(viewer_page, "m")[0]
    assert cap["area"] == pytest.approx(12 + 1.75)
    error = viewer_page.evaluate("""() => {
        const cap = window.threejsViewer.getObject('m').userData.__clipCapState.caps[0];
        const p = cap.geometry.getAttribute('position'), c = cap.geometry.getAttribute('color');
        let e = 0;
        for (let i = 0; i < p.count; i++) e = Math.max(e, Math.abs(c.getX(i) - (p.getX(i) + 2) / 4));
        return e;
    }""")
    # add_mesh quantizes input RGB to bytes.
    assert error <= 1 / 255 + 1e-6


@pytest.mark.parametrize("worker", [False, True])
def test_cap_rebuilds_when_group_assignments_change(viewer_client, viewer_page, worker):
    viewer_client.add_model_binary(
        "m", _textured_box_glb(_png(8, 8, (32, 64, 96))), "glb"
    )
    settle(viewer_client)
    _wait_object(viewer_page, "m")
    viewer_page.evaluate("""() => window.threejsViewer.getObject('m').traverse(n => {
        if (!n.isMesh) return;
        const other = n.material.clone();
        other.map = n.material.map.clone();
        const canvas = document.createElement('canvas'); canvas.width = canvas.height = 8;
        const ctx = canvas.getContext('2d'); ctx.fillStyle = '#d0a040'; ctx.fillRect(0, 0, 8, 8);
        other.map.source = new n.material.map.source.constructor(canvas); other.map.needsUpdate = true;
        n.material = [n.material, other];
        n.geometry.clearGroups(); n.geometry.addGroup(0, n.geometry.index.count, 0);
    })""")
    _clip(viewer_client, viewer_page, worker)
    frames(viewer_page)
    assert _shown_rgb(viewer_page, "m") == pytest.approx([32, 64, 96], abs=2)
    viewer_page.evaluate("""() => window.threejsViewer.getObject('m').traverse(n => {
        if (n.isMesh && !n.userData.__clipCap) n.geometry.groups[0].materialIndex = 1;
    })""")
    frames(viewer_page)
    _wait_settled(viewer_page)
    frames(viewer_page)
    assert _shown_rgb(viewer_page, "m") == pytest.approx([208, 160, 64], abs=2)


@pytest.mark.parametrize("worker", [False, True])
def test_cap_wireframe_cycle_rebuild_and_disposal(viewer_client, viewer_page, worker):
    pos, _, idx = _box_arrays()
    viewer_client.add_mesh("m", pos, idx)
    _clip(viewer_client, viewer_page, worker)
    # Real M-key events: wire-only must keep caps, solid+wire must include them.
    for mode in (1, 2):
        viewer_page.evaluate("""() => window.threejsViewer.container.dispatchEvent(
            new KeyboardEvent('keydown', {key:'m',code:'KeyM',bubbles:true}))""")
        frames(viewer_page)
        _wait_settled(viewer_page)
        assert viewer_page.evaluate(
            """mode => {
            const caps = window.threejsViewer.getObject('m').userData.__clipCapState?.caps;
            return caps?.length === 1 && caps.every(c => c.visible &&
                (mode === 1 ? c.material.wireframe : !c.material.wireframe &&
                c.userData.wireframeOverlay?.visible &&
                c.userData.wireframeOverlay.geometry === c.geometry &&
                c.userData.wireframeOverlay.material.clippingPlanes.length === 0));
        }""",
            mode,
        )
    viewer_page.evaluate("""() => {
        const cap = window.threejsViewer.getObject('m').userData.__clipCapState.caps[0];
        window.oldCapGeometry = cap.geometry;
        window.capOverlayMaterial = cap.userData.wireframeOverlay.material;
        window.overlayDisposed = 0;
        window.capOverlayMaterial.addEventListener('dispose', () => window.overlayDisposed++);
    }""")
    viewer_client.set_clipping_slab(normal=[0, 0, 1], center=0.1, thickness=0.5)
    settle(viewer_client)
    frames(viewer_page)
    _wait_settled(viewer_page)
    assert viewer_page.evaluate("""() => {
        const caps = window.threejsViewer.getObject('m').userData.__clipCapState.caps;
        return caps.length === 2 && caps.every(c => c.geometry !== window.oldCapGeometry &&
            c.userData.wireframeOverlay?.visible && c.userData.wireframeOverlay.geometry === c.geometry &&
            c.userData.wireframeOverlay.material.clippingPlanes.length === 1);
    }""")
    viewer_client.set_clipping_caps(False)
    settle(viewer_client)
    frames(viewer_page)
    assert viewer_page.evaluate("window.overlayDisposed") == 1


@pytest.mark.parametrize("worker", [False, True])
def test_cap_rebuilds_when_interleaved_data_is_replaced(
    viewer_client, viewer_page, worker
):
    g = _Glb()
    mesh = g.box_mesh("interleaved")
    viewer_client.add_model_binary("m", g.bytes([g.node(mesh=mesh)]), "glb")
    _clip(viewer_client, viewer_page, worker)
    assert _caps(viewer_page, "m")[0]["area"] == pytest.approx(4)
    viewer_page.evaluate("""() => window.threejsViewer.getObject('m').traverse(n => {
        if (!n.isMesh || n.userData.__clipCap) return;
        const p = n.geometry.getAttribute('position');
        const previous = p.data;
        p.data = new previous.constructor(previous.array.slice(), previous.stride);
        for (let i=0; i<p.count; i++) p.setX(i,p.getX(i)*2);
    })""")
    frames(viewer_page)
    _wait_settled(viewer_page)
    assert _caps(viewer_page, "m")[0]["area"] == pytest.approx(8)


@pytest.mark.parametrize("worker", [False, True])
def test_coplanar_caps_have_stable_winner(viewer_client, viewer_page, worker):
    pos, _, idx = _box_arrays()
    viewer_client.add_mesh("first", pos, idx, color=0xFF0000)
    angle = 0.37
    rotated = pos.copy()
    rotated[:, 0] = pos[:, 0] * np.cos(angle) - pos[:, 1] * np.sin(angle)
    rotated[:, 1] = pos[:, 0] * np.sin(angle) + pos[:, 1] * np.cos(angle)
    viewer_client.add_mesh("last", rotated, idx, color=0x0000FF)
    viewer_client.set_sun(enabled=False)
    _use_worker(viewer_page, worker)
    viewer_client.set_clipping_caps(True)
    viewer_client.set_clipping_slab(
        normal=[0, 0, 1], center=0, thickness=0.4, show_helper=False
    )
    settle(viewer_client)
    frames(viewer_page)
    _wait_settled(viewer_page)
    for x in (-2, -0.01, 0.01, 2):
        viewer_client.set_camera(position=[x, -3, 8], target=[0, 0, 0], up=[0, 1, 0])
        settle(viewer_client)
        frames(viewer_page, 3)
        for px in (-0.4, 0, 0.4):
            for py in (-0.4, 0, 0.4):
                rgb = viewer_page.evaluate(_PIXEL_AT, [px, py, 0.2])
                assert rgb[2] > 80 and rgb[2] > 2 * rgb[0], (x, px, py, rgb)
