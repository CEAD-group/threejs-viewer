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
    tree = viewer_client.query_scene()
    assert "mybox" in tree
    assert tree["mybox"]["type"] == "Mesh"


@pytest.mark.browser
def test_grouping(viewer_client, viewer_page):
    """Parent-child hierarchy works end-to-end."""
    viewer_client.add_group("arm")
    viewer_client.add_box("joint", parent="arm")
    time.sleep(0.1)
    tree = viewer_client.query_scene()
    assert tree["arm"]["type"] == "Group"
    assert "joint" in tree["arm"]["children"]
    assert tree["joint"]["parent"] == "arm"


@pytest.mark.browser
def test_delete_object(viewer_client, viewer_page):
    """Deleting an object removes it from the scene."""
    viewer_client.add_sphere("s1")
    time.sleep(0.05)
    viewer_client.delete("s1")
    time.sleep(0.1)
    tree = viewer_client.query_scene()
    assert "s1" not in tree


@pytest.mark.browser
def test_visibility(viewer_client, viewer_page):
    """set_visible toggles object visibility."""
    viewer_client.add_box("v1")
    time.sleep(0.05)
    viewer_client.set_visible("v1", False)
    time.sleep(0.1)
    tree = viewer_client.query_scene()
    assert tree["v1"]["visible"] is False


@pytest.mark.browser
def test_clear_scene(viewer_client, viewer_page):
    """clear() removes all objects."""
    viewer_client.add_box("a")
    viewer_client.add_sphere("b")
    time.sleep(0.05)
    viewer_client.clear()
    time.sleep(0.1)
    tree = viewer_client.query_scene()
    # Only metadata keys should remain (no user objects)
    assert "a" not in tree
    assert "b" not in tree


@pytest.mark.browser
def test_stop_animation_resets_draw_range(viewer_client, viewer_page):
    """stop_animation() resets draw ranges to full."""
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
        tree = viewer_client.query_scene()
        if tree["_animation"]["playing"]:
            break
    assert tree["_animation"]["playing"] is True, "Animation did not start"
    viewer_client.stop_animation()
    time.sleep(0.1)
    tree = viewer_client.query_scene()
    assert tree["m1"]["drawRange"] == 1.0


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
        tree = viewer_client.query_scene()
        if tree["_animation"]["playing"]:
            break
    assert tree["_animation"]["playing"] is True, "Animation did not start"
    viewer_client.clear()
    time.sleep(0.2)
    tree = viewer_client.query_scene()
    assert tree["_animation"]["playing"] is False


@pytest.mark.browser
def test_show_grid(viewer_client, viewer_page):
    """show_grid() toggles grid visibility."""
    time.sleep(0.1)
    tree = viewer_client.query_scene()
    # Grid is hidden by default
    assert tree["_grid"]["visible"] is False

    viewer_client.show_grid(visible=True)
    time.sleep(0.1)
    tree = viewer_client.query_scene()
    assert tree["_grid"]["visible"] is True

    viewer_client.show_grid(visible=False)
    time.sleep(0.1)
    tree = viewer_client.query_scene()
    assert tree["_grid"]["visible"] is False
