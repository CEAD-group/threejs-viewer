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


def test_moved_owner_hides_its_cap_until_the_new_one_lands(viewer_client, viewer_page):
    """While the worker slices for a moved owner, the old cap is not shown;
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
    frames(page, 3)
    assert page.evaluate(_VISIBLE_CAPS, "m") == 0
    page.evaluate(
        "() => { const c = window.threejsViewer._clipCaps; c.onWorkerResult = window.__onResult;"
        " for (const m of window.__held) c.onWorkerResult(m); }"
    )
    _wait_caps(page, "m", 1)
    (cap,) = _caps(page, "m")
    assert cap["dist"][0] < 1e-4
    assert cap["area"] == pytest.approx(4)


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
