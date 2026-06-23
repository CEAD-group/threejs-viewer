"""Integration tests using Playwright — verify browser-side behavior end-to-end."""

import socket
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
