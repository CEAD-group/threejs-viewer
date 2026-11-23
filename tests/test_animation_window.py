"""Browser tests for the embedder playback window (setAnimationWindow)."""

import pytest

from conftest import frames, settle
from threejs_viewer import Animation, Frame

pytestmark = pytest.mark.browser

_STATE = """() => {
    const v = window.threejsViewer, el = v.el;
    const s = v.getAnimationState();
    return {
        time: s.time, playing: s.playing, window: s.window,
        thumb: parseFloat(el.querySelector('.tjsv-timeline-thumb').style.left),
        current: el.querySelector('.tjsv-current-time').textContent,
        total: el.querySelector('.tjsv-total-time').textContent,
        markers: [...el.querySelectorAll('.timeline-marker')].map(
            (m) => [Number(m.dataset.time), parseFloat(m.style.left)]),
    };
}"""


def _load(client, page, loop=False):
    """A 100 s animation with a frame every second and markers at 10, 30 and 70 s."""
    client.add_box("window_box")
    anim = Animation(
        frames=[Frame(time=i, transforms={}) for i in range(101)], loop=loop
    )
    for t in (10, 30, 70):
        anim.add_marker(t, f"m{t}")
    client.load_animation(anim, autoplay=False)
    settle(client)
    page.wait_for_function("() => !!window.threejsViewer.getAnimationState()")


def _window(page, start, end):
    page.evaluate(
        "([a, b]) => window.threejsViewer.setAnimationWindow(a, b)", [start, end]
    )


def _seek(page, t):
    page.evaluate("(t) => window.threejsViewer.seekAnimationTime(t)", t)


def _state(page):
    return page.evaluate(_STATE)


def _key(page, key, code):
    page.evaluate(
        "([key, code]) => window.threejsViewer.container.dispatchEvent("
        "  new KeyboardEvent('keydown', {key, code, bubbles: true}))",
        [key, code],
    )


def test_feature_is_advertised(viewer_page):
    assert viewer_page.evaluate(
        "() => window.threejsViewer.hasFeature('animation_window')"
    )


def test_bar_readout_and_markers_span_the_window(viewer_client, viewer_page):
    _load(viewer_client, viewer_page)
    _seek(viewer_page, 30)
    _window(viewer_page, 20, 40)
    s = _state(viewer_page)
    assert s["time"] == pytest.approx(30), "setting a window does not move the playhead"
    assert s["window"] == {"start": 20, "end": 40}
    assert s["thumb"] == pytest.approx(50)
    assert (s["current"], s["total"]) == ("10.00", "20.00")
    assert s["markers"] == [[30, pytest.approx(50)]], (
        "markers outside the window are hidden"
    )


def test_scrubbing_maps_inside_the_window(viewer_client, viewer_page):
    _load(viewer_client, viewer_page)
    _window(viewer_page, 20, 40)
    box = viewer_page.locator(".tjsv-timeline-container").bounding_box()
    y = box["y"] + box["height"] / 2
    viewer_page.mouse.move(box["x"] + box["width"] * 0.25, y)
    viewer_page.mouse.down()
    assert _state(viewer_page)["time"] == pytest.approx(25, abs=0.2)
    viewer_page.mouse.move(box["x"] + box["width"] + 200, y)
    assert _state(viewer_page)["time"] == pytest.approx(40)
    viewer_page.mouse.move(box["x"] - 200, y)
    assert _state(viewer_page)["time"] == pytest.approx(20)
    viewer_page.mouse.up()


def test_play_stops_at_the_window_end(viewer_client, viewer_page):
    _load(viewer_client, viewer_page)
    _window(viewer_page, 20, 40)
    _seek(viewer_page, 39)
    viewer_page.evaluate(
        "() => { const v = window.threejsViewer;"
        " v.setAnimationSpeed(20); v.setAnimationPlaying(true); }"
    )
    viewer_page.wait_for_function(
        "() => !window.threejsViewer.getAnimationState().playing"
    )
    assert _state(viewer_page)["time"] == pytest.approx(40)


def test_play_loops_to_the_window_start(viewer_client, viewer_page):
    _load(viewer_client, viewer_page, loop=True)
    _window(viewer_page, 20, 40)
    _seek(viewer_page, 39.5)
    viewer_page.evaluate(
        "() => { const v = window.threejsViewer; window.__times = [];"
        " window.__off = v.onAnimationTime((s) => window.__times.push(s.time));"
        " v.setAnimationSpeed(5); v.setAnimationPlaying(true); }"
    )
    viewer_page.wait_for_function("() => window.__times.some((t) => t < 30)")
    times = viewer_page.evaluate(
        "() => { window.__off(); window.threejsViewer.setAnimationPlaying(false);"
        " return window.__times; }"
    )
    assert all(20 <= t <= 40 for t in times), times
    assert _state(viewer_page)["window"] == {"start": 20, "end": 40}


@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("outside", [5, 60, 100])
def test_play_from_outside_starts_at_the_window_start(
    viewer_client, viewer_page, loop, outside
):
    _load(viewer_client, viewer_page, loop=loop)
    _seek(viewer_page, outside)
    _window(viewer_page, 20, 40)
    viewer_page.evaluate(
        "() => { const v = window.threejsViewer; window.__times = [];"
        " window.__off = v.onAnimationTime((s) => window.__times.push(s.time));"
        " v.setAnimationPlaying(true); }"
    )
    viewer_page.wait_for_function(
        "() => window.__times.filter((t) => t > 20).length > 2"
    )
    times = viewer_page.evaluate(
        "() => { window.__off(); window.threejsViewer.setAnimationPlaying(false);"
        " return window.__times; }"
    )
    played = [t for t in times if t != outside]
    assert played[0] == pytest.approx(20), times
    assert all(20 <= t < 22 for t in played), times


@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("target", [5, 60])
def test_seek_outside_while_playing_plays_on_from_there(
    viewer_client, viewer_page, loop, target
):
    _load(viewer_client, viewer_page, loop=loop)
    _window(viewer_page, 20, 40)
    _seek(viewer_page, 25)
    viewer_page.evaluate(
        "(target) => { const v = window.threejsViewer; window.__times = [];"
        " v.setAnimationPlaying(true);"
        " window.__off = v.onAnimationTime((s) => window.__times.push(s.time));"
        " v.seekAnimationTime(target); }",
        target,
    )
    viewer_page.wait_for_function("() => window.__times.length > 8")
    times = viewer_page.evaluate(
        "() => { window.__off(); window.threejsViewer.setAnimationPlaying(false);"
        " return window.__times; }"
    )
    after = times[times.index(target) :]
    assert all(target <= t < target + 2 for t in after), times
    assert after == sorted(after), times
    assert _state(viewer_page)["window"] == {"start": 20, "end": 40}


def test_aria_value_stays_on_the_bar(viewer_client, viewer_page):
    _load(viewer_client, viewer_page)
    _window(viewer_page, 20, 40)
    tl = viewer_page.locator(".tjsv-timeline-container")
    for t, now in ((60, "20.00"), (5, "0.00"), (30, "10.00")):
        _seek(viewer_page, t)
        assert tl.get_attribute("aria-valuenow") == now
        assert tl.get_attribute("aria-valuemax") == "20.00"


def test_steps_home_and_end_stay_inside_the_window(viewer_client, viewer_page):
    _load(viewer_client, viewer_page)
    _window(viewer_page, 20.5, 40.5)
    _key(viewer_page, "Home", "Home")
    assert _state(viewer_page)["time"] == pytest.approx(20.5)
    _key(viewer_page, "ArrowLeft", "ArrowLeft")
    assert _state(viewer_page)["time"] == pytest.approx(20.5)
    _key(viewer_page, "ArrowRight", "ArrowRight")
    assert _state(viewer_page)["time"] == pytest.approx(21)
    _key(viewer_page, "End", "End")
    assert _state(viewer_page)["time"] == pytest.approx(40.5)
    _key(viewer_page, "ArrowRight", "ArrowRight")
    assert _state(viewer_page)["time"] == pytest.approx(40.5)
    viewer_page.locator(".tjsv-btn-start").click()
    assert _state(viewer_page)["time"] == pytest.approx(20.5)
    viewer_page.locator(".tjsv-btn-prev-frame").click()
    assert _state(viewer_page)["time"] == pytest.approx(20.5)
    viewer_page.locator(".tjsv-btn-end").click()
    assert _state(viewer_page)["time"] == pytest.approx(40.5)


def test_api_seek_outside_keeps_the_window(viewer_client, viewer_page):
    _load(viewer_client, viewer_page)
    _window(viewer_page, 20, 40)
    _seek(viewer_page, 60)
    s = _state(viewer_page)
    assert s["time"] == pytest.approx(60)
    assert s["window"] == {"start": 20, "end": 40}
    assert s["thumb"] == pytest.approx(100), "the thumb pins to the nearer end"
    assert s["current"] == "40.00"
    _seek(viewer_page, 5)
    s = _state(viewer_page)
    assert s["thumb"] == pytest.approx(0)
    assert s["current"] == "-15.00"


def test_null_clears_and_invalid_is_rejected(viewer_client, viewer_page):
    _load(viewer_client, viewer_page)
    _seek(viewer_page, 30)
    _window(viewer_page, 20, 40)
    _window(viewer_page, 40, 20)
    assert _state(viewer_page)["window"] == {"start": 20, "end": 40}
    viewer_page.evaluate("() => window.threejsViewer.setAnimationWindow(null)")
    s = _state(viewer_page)
    assert s["window"] is None
    assert s["thumb"] == pytest.approx(30)
    assert (s["current"], s["total"]) == ("30.00", "100.00")
    assert [t for t, _ in s["markers"]] == [10, 30, 70]


def test_loading_an_animation_clears_the_window(viewer_client, viewer_page):
    _load(viewer_client, viewer_page)
    _seek(viewer_page, 30)
    _window(viewer_page, 20, 40)
    anim = Animation(
        frames=[Frame(time=0, transforms={}), Frame(time=100, transforms={})]
    )
    viewer_client.load_animation(anim, autoplay=False)
    settle(viewer_client)
    frames(viewer_page, 1)
    s = _state(viewer_page)
    assert s["window"] is None
    assert s["time"] == pytest.approx(30), "a swap keeps the playhead"
    assert s["total"] == "100.00"
