"""Browser tests for the plan-view additions: projection, navigation, camera events,
depth order, add completion and tube picking."""

import pytest

from conftest import settle

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
