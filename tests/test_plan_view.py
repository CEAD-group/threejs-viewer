"""Browser tests for the plan-view additions: projection, navigation, camera events,
depth order, add completion and tube picking."""

import math

import numpy as np
import pytest

from conftest import frames, settle

pytestmark = pytest.mark.browser

_POSE_JS = "() => window.threejsViewer.getCameraPose()"


def _vec(p):
    return [p["x"], p["y"], p["z"]]


def test_set_projection_message_switches_and_reports(viewer_client, viewer_page):
    """set_projection switches the camera and keeps the view; getCameraPose,
    get_camera and the feature probe report it."""
    viewer_client.add_box("b")
    viewer_client.set_camera(position=[4, -4, 3], target=[0, 0, 0])
    settle(viewer_client)
    pose = viewer_page.evaluate(_POSE_JS)
    assert pose["projection"] == "perspective"
    assert viewer_page.evaluate(
        "() => window.threejsViewer.hasFeature('set_projection')"
    )
    assert not viewer_page.evaluate(
        "() => window.threejsViewer.hasFeature('no_such_thing')"
    )

    viewer_client.set_projection("orthographic")
    settle(viewer_client)
    ortho = viewer_page.evaluate(_POSE_JS)
    assert ortho["projection"] == "orthographic" and ortho["fov"] is None
    assert _vec(ortho["target"]) == pytest.approx([0, 0, 0], abs=1e-9)
    assert viewer_client.get_camera()["projection"] == "orthographic"
    # The projection button follows a message switch.
    assert (
        viewer_page.evaluate(
            "() => document.querySelector('.tjsv-view-proj').textContent"
        )
        == "O"
    )

    viewer_client.set_projection("perspective")
    settle(viewer_client)
    assert viewer_page.evaluate(_POSE_JS)["projection"] == "perspective"
    # Unknown names warn and change nothing.
    assert (
        viewer_page.evaluate("() => window.threejsViewer.setProjection('iso')") is False
    )
    assert viewer_page.evaluate(_POSE_JS)["projection"] == "perspective"


def test_set_camera_projection_lands_with_the_pose(viewer_client, viewer_page):
    """A pose with `projection` switches before the pose applies, so the given
    zoom and orientation hold on the new camera in the same frame."""
    viewer_client.add_box("b")
    settle(viewer_client)
    viewer_page.evaluate(
        "() => window.threejsViewer.setCameraPose({position: [0, 0, 50], target: [0, 0, 0],"
        " up: [0, 1, 0], zoom: 3, projection: 'orthographic'})"
    )
    pose = viewer_page.evaluate(_POSE_JS)
    assert pose["projection"] == "orthographic"
    assert pose["zoom"] == pytest.approx(3)
    assert _vec(pose["position"]) == pytest.approx([0, 0, 50], abs=1e-9)
    assert _vec(pose["up"]) == pytest.approx([0, 1, 0], abs=1e-9)
    viewer_client.set_camera(
        position=[5, -5, 5], target=[0, 0, 0], up=[0, 0, 1], projection="perspective"
    )
    settle(viewer_client)
    pose = viewer_page.evaluate(_POSE_JS)
    assert pose["projection"] == "perspective"
    assert _vec(pose["position"]) == pytest.approx([5, -5, 5], abs=1e-9)


_PROJECT_JS = """([x, y, z]) => {
    const v = window.threejsViewer;
    const r = v.getCanvas().getBoundingClientRect();
    v._camera.updateMatrixWorld(true);
    const p = new v._camera.position.constructor(x, y, z).project(v._camera);
    return [r.left + (p.x + 1) / 2 * r.width, r.top + (1 - p.y) / 2 * r.height];
}"""


def _plan_view(viewer_client, viewer_page):
    """A polyline seen from above in orthographic, left drag panning, rotation off."""
    pts = np.array([[-2, 0, 0], [0, 0, 0], [2, 0, 0]], dtype=np.float32)
    viewer_client.add_polyline("path", pts, color=0x44AAFF, line_width=6)
    viewer_client.set_camera(
        position=[0, 0, 10],
        target=[0, 0, 0],
        up=[0, 1, 0],
        zoom=2,
        projection="orthographic",
    )
    viewer_client.set_navigation(rotate=False, left_button="pan")
    settle(viewer_client)
    frames(viewer_page)


def _dir(pose):
    d = [pose["position"][k] - pose["target"][k] for k in "xyz"]
    n = math.sqrt(sum(x * x for x in d))
    return [x / n for x in d]


def test_navigation_left_drag_pans_without_rotating(viewer_client, viewer_page):
    _plan_view(viewer_client, viewer_page)
    nav = viewer_page.evaluate("() => window.threejsViewer.getNavigation()")
    assert nav == {"rotate": False, "pan": True, "zoom": True, "leftButton": "pan"}
    before = viewer_page.evaluate(_POSE_JS)
    box = viewer_page.locator("canvas").first.bounding_box()
    x, y = box["x"] + box["width"] * 0.3, box["y"] + box["height"] * 0.7
    viewer_page.mouse.move(x, y)
    viewer_page.mouse.down()
    viewer_page.mouse.move(x + 120, y - 60, steps=10)
    viewer_page.mouse.up()
    frames(viewer_page, 30)
    after = viewer_page.evaluate(_POSE_JS)
    assert _dir(after) == pytest.approx(_dir(before), abs=1e-9)
    assert _vec(after["up"]) == pytest.approx(_vec(before["up"]), abs=1e-9)
    moved = [after["target"][k] - before["target"][k] for k in "xyz"]
    assert abs(moved[0]) > 0.1 and abs(moved[2]) < 1e-9, moved


def test_navigation_click_picks_without_moving_the_pivot(viewer_client, viewer_page):
    """A plain click on a line picks it while the left button pans, and does not
    move the orbit target (click-to-pivot is off with rotation)."""
    _plan_view(viewer_client, viewer_page)
    viewer_page.evaluate(
        "() => { window.__picks = [];"
        " window.threejsViewer.onPolylinePick(p => window.__picks.push(p)); }"
    )
    before = viewer_page.evaluate(_POSE_JS)
    cx, cy = viewer_page.evaluate(_PROJECT_JS, [1.0, 0.0, 0.0])
    viewer_page.mouse.move(cx, cy)
    frames(viewer_page)
    viewer_page.mouse.down()
    viewer_page.mouse.up()
    frames(viewer_page, 3)
    picks = viewer_page.evaluate("() => window.__picks")
    assert picks and picks[-1]["id"] == "path"
    assert picks[-1]["point"]["x"] == pytest.approx(1.0, abs=0.05)
    after = viewer_page.evaluate(_POSE_JS)
    assert _vec(after["target"]) == pytest.approx(_vec(before["target"]), abs=1e-12)
    assert _vec(after["position"]) == pytest.approx(_vec(before["position"]), abs=1e-12)


def test_navigation_without_rotation_keeps_the_view(viewer_client, viewer_page):
    """With rotation off the projection key, the gimbal's axis bubbles and the
    projection button keep the view; F frames without reorienting; set_projection
    still switches."""
    _plan_view(viewer_client, viewer_page)
    before = viewer_page.evaluate(_POSE_JS)
    viewer_page.locator("canvas").first.focus()
    viewer_page.keyboard.press("o")
    viewer_page.evaluate("() => window.threejsViewer._gizmoAxisClick('front')")
    viewer_page.evaluate("() => document.querySelector('.tjsv-view-proj').click()")
    frames(viewer_page, 30)
    after = viewer_page.evaluate(_POSE_JS)
    assert after["projection"] == "orthographic"
    assert _dir(after) == pytest.approx(_dir(before), abs=1e-9)
    viewer_page.keyboard.press("f")
    frames(viewer_page, 30)
    framed = viewer_page.evaluate(_POSE_JS)
    assert _dir(framed) == pytest.approx(_dir(before), abs=1e-9)
    assert _vec(framed["up"]) == pytest.approx([0, 1, 0], abs=1e-9)
    viewer_client.set_projection("perspective")
    settle(viewer_client)
    assert viewer_page.evaluate(_POSE_JS)["projection"] == "perspective"


def test_navigation_zoom_off_ignores_the_wheel(viewer_client, viewer_page):
    _plan_view(viewer_client, viewer_page)
    viewer_client.set_navigation(zoom=False)
    settle(viewer_client)
    before = viewer_page.evaluate(_POSE_JS)
    box = viewer_page.locator("canvas").first.bounding_box()
    viewer_page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    viewer_page.mouse.wheel(0, -400)
    frames(viewer_page, 5)
    assert viewer_page.evaluate(_POSE_JS)["zoom"] == pytest.approx(before["zoom"])
    # Defaults restore orbiting.
    viewer_client.set_navigation(rotate=True, zoom=True, left_button="rotate")
    settle(viewer_client)
    assert viewer_page.evaluate("() => window.threejsViewer.getNavigation()") == {
        "rotate": True,
        "pan": True,
        "zoom": True,
        "leftButton": "rotate",
    }


def test_camera_change_fires_once_per_changed_frame(viewer_client, viewer_page):
    """onCameraChange reports the current view after subscribing, stays quiet while
    nothing moves, reports set_camera and zoom with world units per pixel, and
    stops after unsubscribing."""
    viewer_client.add_box("b")
    viewer_client.set_camera(
        position=[0, 0, 10],
        target=[0, 0, 0],
        up=[0, 1, 0],
        zoom=2,
        projection="orthographic",
    )
    settle(viewer_client)
    viewer_page.evaluate(
        "() => { window.__cam = [];"
        " window.__unsub = window.threejsViewer.onCameraChange(c => window.__cam.push(c)); }"
    )
    frames(viewer_page, 3)
    events = viewer_page.evaluate("() => window.__cam")
    assert len(events) == 1
    first = events[0]
    assert first["projection"] == "orthographic"
    expected = viewer_page.evaluate(
        "() => { const v = window.threejsViewer, c = v._camera;"
        " return (c.top - c.bottom) / c.zoom / v.getCanvas().clientHeight; }"
    )
    assert first["worldPerPixel"] == pytest.approx(expected, rel=1e-12)
    frames(viewer_page, 10)
    assert viewer_page.evaluate("() => window.__cam.length") == 1

    viewer_client.set_camera(zoom=4)
    settle(viewer_client)
    frames(viewer_page, 3)
    events = viewer_page.evaluate("() => window.__cam")
    assert len(events) == 2
    assert events[1]["zoom"] == pytest.approx(4)
    assert events[1]["worldPerPixel"] == pytest.approx(
        first["worldPerPixel"] / 2, rel=1e-9
    )

    # A wheel zoom from the controls is reported too.
    box = viewer_page.locator("canvas").first.bounding_box()
    viewer_page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    viewer_page.mouse.wheel(0, -200)
    frames(viewer_page, 3)
    assert viewer_page.evaluate("() => window.__cam.length") >= 3

    viewer_page.evaluate("() => window.__unsub()")
    n = viewer_page.evaluate("() => window.__cam.length")
    viewer_client.set_camera(zoom=1)
    settle(viewer_client)
    frames(viewer_page, 3)
    assert viewer_page.evaluate("() => window.__cam.length") == n


_BLOB_LINE_JS = (
    """(points) => URL.createObjectURL(new Blob([new Float32Array(points)]))"""
)


def test_when_object_ready_resolves_after_the_blob_add(viewer_client, viewer_page):
    """whenObjectReady resolves after a blob-backed add has registered its object,
    false for a failed or deleted load and for an unknown id, true at once for an
    existing object, and follows a newer add for the same id."""
    url = viewer_page.evaluate(_BLOB_LINE_JS, [0, 0, 0, 1, 0, 0, 1, 1, 0])
    result = viewer_page.evaluate(
        """async (url) => {
            const v = window.threejsViewer;
            v.handleMessage({type: 'add_polyline_binary', id: 'l', blob_url: url, numPoints: 3});
            const before = !!v.getObject('l');
            const ready = await v.whenObjectReady('l');
            return {before, ready, after: !!v.getObject('l')};
        }""",
        url,
    )
    assert result == {"before": False, "ready": True, "after": True}

    flags = viewer_page.evaluate(
        """async (url) => {
            const v = window.threejsViewer;
            v.handleMessage({type: 'add_polyline_binary', id: 'bad', blob_url: url + 'x', numPoints: 3});
            const bad = await v.whenObjectReady('bad');
            v.handleMessage({type: 'add_polyline_binary', id: 'gone', blob_url: url, numPoints: 3});
            const pending = v.whenObjectReady('gone');
            v.handleMessage({type: 'delete_object', id: 'gone'});
            const gone = await pending;
            v.handleMessage({type: 'add_polyline_binary', id: 'twice', blob_url: url, numPoints: 3});
            const first = v.whenObjectReady('twice');
            v.handleMessage({type: 'add_polyline_binary', id: 'twice', blob_url: url, numPoints: 2});
            const twice = await first;
            const n = v.getObject('twice').userData.pickPoints.length / 3;
            return {bad, gone, twice, n, unknown: await v.whenObjectReady('nope'),
                    existing: await v.whenObjectReady('l')};
        }""",
        url,
    )
    assert flags == {
        "bad": False,
        "gone": False,
        "twice": True,
        "n": 2,
        "unknown": False,
        "existing": True,
    }


_PIXELS_JS = """(pts) => { const v = window.threejsViewer;
    v._camera.updateMatrixWorld(true);
    v._renderer.render(v._scene, v._camera);
    const gl = v._renderer.getContext();
    const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
    const px = new Uint8Array(4);
    return pts.map(([x, y, z]) => {
        const p = v._camera.position.clone().set(x, y, z).project(v._camera);
        gl.readPixels(Math.round((p.x + 1) / 2 * w), Math.round((p.y + 1) / 2 * h),
                      1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
        return [px[0], px[1], px[2]]; }); }"""


def _top_view(viewer_client):
    viewer_client.set_camera(
        position=[0, 0, 5],
        target=[0, 0, 0],
        up=[0, 1, 0],
        zoom=4,
        projection="orthographic",
    )


def test_depth_offset_orders_coplanar_lines(viewer_client, viewer_page):
    """At equal depth the later line wins; a depthOffset toward the camera makes the
    earlier one win instead, for fat and native lines alike, without moving it."""
    for k, fat in enumerate((True, False)):
        y = 0.5 * k
        pts = np.array([[-1, y, 0], [1, y, 0]], dtype=np.float32)
        # Pair at x < 0: no offset. Pair at x > 0: the first line is offset.
        left = pts * [0.5, 1, 1] - [0.6, 0, 0]
        right = pts * [0.5, 1, 1] + [0.6, 0, 0]
        for side, ps, offset in (("l", left, None), ("r", right, 0.01)):
            viewer_client.add_polyline(
                f"red{side}{k}",
                ps,
                color=0xFF0000,
                fat=fat,
                line_width=8,
                depth_offset=offset,
            )
            settle(viewer_client)
            viewer_client.add_polyline(
                f"blue{side}{k}", ps, color=0x0000FF, fat=fat, line_width=8
            )
            settle(viewer_client)
    _top_view(viewer_client)
    settle(viewer_client)
    frames(viewer_page, 3)
    px = viewer_page.evaluate(
        _PIXELS_JS, [[-0.6, 0, 0], [0.6, 0, 0], [-0.6, 0.5, 0], [0.6, 0.5, 0]]
    )
    for l_px, r_px in ((px[0], px[1]), (px[2], px[3])):
        assert l_px[2] > l_px[0], f"no offset: the later (blue) line wins, got {px}"
        assert r_px[0] > r_px[2], f"offset: the earlier (red) line wins, got {px}"
    assert viewer_page.evaluate(
        "() => window.threejsViewer.getObject('redr0').material.userData.depthOffset"
    ) == pytest.approx(0.01)


def test_render_order_and_tube_depth_offset(viewer_client, viewer_page):
    """renderOrder lands on the object; a tube pushed away by depthOffset lets the
    centreline inside it show."""
    spine = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.float32)
    w = np.full(3, 0.3, dtype=np.float32)
    viewer_client.add_parametric_tube(
        "tube", spine, w, w, color=0xFF0000, depth_offset=-1.0, render_order=3
    )
    viewer_client.add_polyline(
        "centre", spine, color=0x0000FF, line_width=6, render_order=-2
    )
    _top_view(viewer_client)
    settle(viewer_client)
    frames(viewer_page, 3)
    orders = viewer_page.evaluate(
        "() => ['tube', 'centre'].map((id) => window.threejsViewer.getObject(id).renderOrder)"
    )
    assert orders == [3, -2]
    (px,) = viewer_page.evaluate(_PIXELS_JS, [[0.5, 0, 0]])
    assert px[2] > px[0], px


def test_orthographic_near_far_fit_the_scene(viewer_client, viewer_page):
    """The orthographic camera fits near and far around the content along the view
    instead of the fixed 0.1 to 1000 m."""
    viewer_client.add_box("b")
    viewer_client.set_camera(
        position=[0, 0, 20], target=[0, 0, 0], up=[0, 1, 0], projection="orthographic"
    )
    settle(viewer_client)
    frames(viewer_page, 3)
    near, far = viewer_page.evaluate(
        "() => [window.threejsViewer._camera.near, window.threejsViewer._camera.far]"
    )
    r = math.sqrt(3) / 2
    assert 20 - 1.2 * r < near < 20 - r, near
    assert 20 + r < far < 20 + 1.2 * r, far
