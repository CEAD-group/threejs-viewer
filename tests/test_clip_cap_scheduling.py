"""When clip caps are built: uncut and off-screen owners, playing tubes, moved owners, frame budget."""

import numpy as np
import pytest

from conftest import frames, settle
from test_clip_caps import (
    _box_arrays,
    _caps,
    _line,
    _union_area,
    _use_worker,
    _wait_caps,
    _wait_object,
    _wait_settled,
)

pytestmark = pytest.mark.browser

_STATS = """() => { const c = window.threejsViewer._clipCaps;
    return { builds: c.builds, requests: c.requests, lastFrameBuilds: c.lastFrameBuilds }; }"""

_VISIBLE_CAPS = """(id) => { let k = 0; window.threejsViewer.getObject(id).traverse(
    (c) => { if (c.userData.__clipCap && c.visible) k++; }); return k; }"""


def _stats(page):
    return page.evaluate(_STATS)


def test_uncut_owner_builds_no_cap(viewer_client, viewer_page):
    """A box wholly on one side of the plane is never sliced; once the plane
    passes through it, it gets its cap."""
    page = viewer_page
    viewer_client.add_box("b", width=2, height=2, depth=2)
    viewer_client.set_clipping_caps(True)
    for distance in (5, -5):
        viewer_client.set_clipping_plane(
            normal=[0, 0, 1], distance=distance, show_helper=False
        )
        settle(viewer_client)
        frames(page, 3)
        _wait_settled(page)
        assert _stats(page)["builds"] == 0
        assert page.evaluate(_VISIBLE_CAPS, "b") == 0
    viewer_client.set_clipping_plane(normal=[0, 0, 1], distance=0.25, show_helper=False)
    settle(viewer_client)
    _wait_caps(page, "b", 1)
    assert _stats(page)["builds"] == 1


def test_off_screen_owner_builds_its_cap_once_in_view(viewer_client, viewer_page):
    """A cut owner outside the view frustum waits; panning to it builds its cap."""
    page = viewer_page
    viewer_client.add_box("near", width=2, height=2, depth=2)
    viewer_client.add_box("far", width=2, height=2, depth=2, position=[200, 0, 0])
    viewer_client.set_camera(position=[0, -8, 0], target=[0, 0, 0])
    viewer_client.set_clipping_caps(True)
    viewer_client.set_clipping_plane(normal=[0, 0, 1], distance=0.25, show_helper=False)
    settle(viewer_client)
    _wait_caps(page, "near", 1)
    frames(page, 3)
    assert page.evaluate(_VISIBLE_CAPS, "far") == 0
    assert _stats(page)["builds"] == 1
    viewer_client.set_camera(position=[200, -8, 0], target=[200, 0, 0])
    settle(viewer_client)
    _wait_caps(page, "far", 1)
    assert _stats(page)["builds"] == 2


@pytest.mark.parametrize("worker", [False, True])
def test_playing_tube_redraws_its_cap_without_rebuilding(
    viewer_client, viewer_page, worker
):
    """A draw-range change only shortens the tube cap's own draw range: no
    rebuild or worker request, and the cap still covers just the drawn part."""
    page = viewer_page
    _use_worker(page, worker)
    spine = _line((0, 0), (40, 0), (40, 40), (80, 40), (80, 0))
    n = len(spine)
    viewer_client.add_parametric_tube(
        "t", spine, np.full(n, 8.0), np.full(n, 3.0), lod=False
    )
    viewer_client.set_clipping_caps(True)
    viewer_client.set_clipping_plane(normal=[0, 0, 1], distance=0.3, show_helper=False)
    settle(viewer_client)
    _wait_object(page, "t")
    _wait_caps(page, "t", 1)
    full = _union_area(page, "t", grid=160)
    before = _stats(page)
    for value in (0.2, 0.4, 0.5, 0.8, 0.5):
        viewer_client.set_draw_range("t", value)
        settle(viewer_client)
        frames(page, 2)
    _wait_settled(page)
    after = _stats(page)
    assert after["builds"] == before["builds"]
    assert after["requests"] == before["requests"]
    half = _union_area(page, "t", grid=160)
    assert 0.35 * full < half < 0.65 * full


def test_frame_budget_spreads_main_thread_builds(viewer_client, viewer_page):
    """With no time left in a frame only one cap is built per frame, and every
    cap is still built in the end."""
    page = viewer_page
    page.evaluate("() => { window.__clipCapFrameBudgetMs = 0; }")
    ids = [f"b{i}" for i in range(12)]
    for i, id in enumerate(ids):
        viewer_client.add_box(
            id, width=1, height=1, depth=1, position=[i * 1.5 - 8, 0, 0]
        )
    viewer_client.set_camera(position=[0, -30, 0], target=[0, 0, 0])
    viewer_client.set_clipping_caps(True)
    viewer_client.set_clipping_plane(normal=[0, 0, 1], distance=0.1, show_helper=False)
    settle(viewer_client)
    seen = []
    for _ in range(30):
        frames(page, 1)
        seen.append(_stats(page)["lastFrameBuilds"])
    assert max(seen) == 1
    _wait_settled(page)
    assert all(page.evaluate(_VISIBLE_CAPS, id) == 1 for id in ids)
    assert _stats(page)["builds"] == len(ids)


_ALL_CAPS = """(id) => { let k = 0; window.threejsViewer.getObject(id).traverse(
    (c) => { if (c.userData.__clipCap) k++; }); return k; }"""

# Moves the owners every frame and records how many of their caps were shown.
_MOVE_AND_COUNT = """async ([ids, frames, amp]) => {
    const v = window.threejsViewer;
    const objs = ids.map((id) => v.getObject(id));
    const base = objs.map((o) => o.position.z);
    const shown = [];
    for (let f = 0; f < frames; f++) {
        await new Promise((r) => requestAnimationFrame(r));
        let k = 0;
        for (const o of objs) o.traverse((c) => { if (c.userData.__clipCap && c.visible) k++; });
        shown.push(k);
        objs.forEach((o, i) => { o.position.z = base[i] + amp * Math.sin(f * 0.15); });
    }
    return shown;
}"""


def test_owner_back_at_its_capped_pose_shows_its_cap(viewer_client, viewer_page):
    """Owners moved and moved back before their rebuild come back to the pose
    their caps were built for: every cap is shown again."""
    page = viewer_page
    page.evaluate("() => { window.__clipCapFrameBudgetMs = 0; }")
    viewer_client.add_box("a", width=2, height=2, depth=2, position=[-2, 0, 0])
    viewer_client.add_box("b", width=2, height=2, depth=2, position=[2, 0, 0])
    viewer_client.set_camera(position=[0, -12, 0], target=[0, 0, 0])
    viewer_client.set_clipping_caps(True)
    viewer_client.set_clipping_plane(normal=[0, 0, 1], distance=0.1, show_helper=False)
    settle(viewer_client)
    _wait_caps(page, "a", 1)
    _wait_caps(page, "b", 1)
    page.evaluate(
        "() => { const v = window.threejsViewer, c = v._clipCaps;"
        " const objs = ['a', 'b'].map((id) => v.getObject(id));"
        " for (const o of objs) o.position.z = 0.3; c.update();"
        " for (const o of objs) o.position.z = 0; c.update(); }"
    )
    frames(page, 3)
    _wait_settled(page)
    assert [page.evaluate(_VISIBLE_CAPS, id) for id in ("a", "b")] == [1, 1]


@pytest.mark.parametrize("worker", [False, True])
def test_moving_owner_keeps_showing_a_cap(viewer_client, viewer_page, worker):
    """An owner moving across the plane every frame shows a cap in nearly every
    frame while its rebuilds catch up, on the worker and on a starved main thread."""
    page = viewer_page
    _use_worker(page, worker)
    if not worker:
        page.evaluate("() => { window.__clipCapFrameBudgetMs = 0; }")
    pos, _, idx = _box_arrays()
    ids = ["m"] if worker else [f"m{i}" for i in range(8)]
    for i, id in enumerate(ids):
        viewer_client.add_mesh(id, pos, idx)
        viewer_client.set_matrix(
            id, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, i * 2.5 - 9, 0, 0, 1]
        )
    viewer_client.set_camera(position=[0, -30, 0], target=[0, 0, 0])
    viewer_client.set_clipping_caps(True)
    viewer_client.set_clipping_plane(normal=[0, 0, 1], distance=0.1, show_helper=False)
    settle(viewer_client)
    for id in ids:
        _wait_caps(page, id, 1)
    shown = page.evaluate(_MOVE_AND_COUNT, [ids, 90, 0.5])
    assert sum(shown) >= 0.95 * 90 * len(ids), shown


def test_cap_too_old_for_its_owner_is_hidden(viewer_client, viewer_page):
    """A cap whose rebuild does not land within CLIP_CAP_STALE_MS is hidden;
    the new one appears on the plane once the result lands."""
    page = viewer_page
    _use_worker(page, True)
    pos, _, idx = _box_arrays()
    viewer_client.add_mesh("m", pos, idx)
    viewer_client.set_clipping_caps(True)
    viewer_client.set_clipping_plane(normal=[0, 0, 1], distance=0.1, show_helper=False)
    settle(viewer_client)
    _wait_caps(page, "m", 1)
    page.evaluate(
        "() => { const c = window.threejsViewer._clipCaps; window.__held = [];"
        " window.__onResult = c.onWorkerResult.bind(c);"
        " c.onWorkerResult = (m) => window.__held.push(m); }"
    )
    viewer_client.set_matrix("m", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0.3, 1])
    settle(viewer_client)
    page.wait_for_function("() => window.__held.length > 0", timeout=10_000)
    frames(page, 2)
    assert page.evaluate(_VISIBLE_CAPS, "m") == 1
    page.wait_for_function(
        "(id) => { let k = 0; window.threejsViewer.getObject(id).traverse("
        "(c) => { if (c.userData.__clipCap && c.visible) k++; }); return k === 0; }",
        arg="m",
        timeout=5_000,
    )
    page.evaluate(
        "() => { const c = window.threejsViewer._clipCaps; c.onWorkerResult = window.__onResult;"
        " for (const m of window.__held) c.onWorkerResult(m); }"
    )
    _wait_caps(page, "m", 1)
    (cap,) = _caps(page, "m")
    assert cap["dist"][0] < 1e-4
    assert cap["area"] == pytest.approx(4)


def test_hidden_caps_are_disposed_when_clipping_goes_off(viewer_client, viewer_page):
    """A cap hidden because the plane left its owner is removed with clipping."""
    page = viewer_page
    viewer_client.add_box("b", width=2, height=2, depth=2)
    viewer_client.set_clipping_caps(True)
    viewer_client.set_clipping_plane(normal=[0, 0, 1], distance=0.25, show_helper=False)
    settle(viewer_client)
    _wait_caps(page, "b", 1)
    viewer_client.set_clipping_plane(normal=[0, 0, 1], distance=5, show_helper=False)
    settle(viewer_client)
    frames(page, 3)
    assert page.evaluate(_VISIBLE_CAPS, "b") == 0
    viewer_client.disable_clipping_plane()
    settle(viewer_client)
    frames(page, 3)
    assert page.evaluate(_ALL_CAPS, "b") == 0


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize("points", [2, 11])
def test_revealed_tube_cap_reaches_the_frontier(
    viewer_client, viewer_page, points, worker
):
    """Mid reveal the tube cap runs up to the frontier ring, also inside the
    frontier's ring pair on a sparse spine."""
    page = viewer_page
    _use_worker(page, worker)
    spine = np.zeros((points, 3), dtype=np.float32)
    spine[:, 0] = np.linspace(0, 100, points)
    viewer_client.add_parametric_tube(
        "t", spine, np.full(points, 8.0), np.full(points, 3.0), lod=False
    )
    viewer_client.set_camera(position=[50, -150, 40], target=[50, 0, 0])
    viewer_client.set_clipping_caps(True)
    viewer_client.set_clipping_plane(normal=[0, 0, 1], distance=0.3, show_helper=False)
    settle(viewer_client)
    _wait_object(page, "t")
    _wait_caps(page, "t", 1)
    full = _union_area(page, "t", grid=200)
    for value in (0.55, 0.3):
        viewer_client.set_draw_range("t", value)
        settle(viewer_client)
        frames(page, 2)
        _wait_settled(page)
        assert _union_area(page, "t", grid=200) == pytest.approx(value * full, rel=0.05)


def test_orthographic_view_builds_only_caps_in_view(viewer_client, viewer_page):
    """The frustum skip follows the orthographic camera's box."""
    page = viewer_page
    viewer_client.add_box("near", width=2, height=2, depth=2)
    viewer_client.add_box("far", width=2, height=2, depth=2, position=[200, 0, 0])
    viewer_client.set_camera(
        position=[0, -8, 0], target=[0, 0, 0], projection="orthographic"
    )
    viewer_client.set_clipping_caps(True)
    viewer_client.set_clipping_plane(normal=[0, 0, 1], distance=0.25, show_helper=False)
    settle(viewer_client)
    _wait_caps(page, "near", 1)
    frames(page, 3)
    assert page.evaluate(_VISIBLE_CAPS, "far") == 0
    viewer_client.set_camera(
        position=[200, -8, 0], target=[200, 0, 0], projection="orthographic"
    )
    settle(viewer_client)
    _wait_caps(page, "far", 1)
    assert _stats(page)["builds"] == 2
