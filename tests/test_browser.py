"""Integration tests using Playwright — verify browser-side behavior end-to-end."""

import time

import numpy as np
import pytest

from threejs_viewer import Animation, Frame


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
            // Force toolbar visible so the lift > 0 branch runs.
            v._animControlsEl.classList.add('visible');
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
