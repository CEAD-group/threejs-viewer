"""Real toolpath regressions for collapse height and reflected tube winding."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from test_parametric_tube import (
    _decode_repro_bead,
    _wait_for_collapse,
    _wait_for_object,
)

pytestmark = pytest.mark.browser
DATA = Path(__file__).parent / "data"
SOURCE = DATA.parent.parent / "src/threejs_viewer/viewer/viewer.js"


def _spine(issue):
    return np.asarray(
        json.loads((DATA / f"tube_issue_{issue}.json").read_text()), dtype=np.float32
    )


@pytest.fixture(scope="module")
def worker_code():
    """Resolve the actual worker template, including its shared function bodies."""
    source = SOURCE.read_text()
    code = re.search(r"const _LOD_WORKER_CODE = `(.*?)\n`;", source, re.S)[1]

    def inject(match):
        expression = match[1]
        if expression.endswith(".toString()"):
            name = expression.split(".")[0]
            return re.search(rf"^function {name}\(.*?^}}", source, re.M | re.S)[0]
        return re.search(rf"^const {expression} = (.*?);", source, re.M)[1]

    return re.sub(r"\$\{([^}]+)\}", inject, code)


def _worker_geometry(page, code, spine, widths, heights, up, collapse=False):
    return page.evaluate(
        """({code, spine, widths, heights, up, collapse}) => {
            const run = new Function('self', 'spine', 'widths', 'heights', 'up', 'collapse',
                code + `
                const geo = buildGeometry(spine, widths, heights, up, null, null, null);
                const original = Array.from(geo.positions);
                if (collapse) collapseTubeStrandFolds(geo.positions, spine, widths, heights,
                    geo.localFrames, widths.length, 6);
                return {positions: Array.from(geo.positions), original,
                    normals: Array.from(geo.normals), indices: Array.from(geo.indices),
                    frames: Array.from(geo.localFrames)};`);
            return run({}, new Float32Array(spine.flat()), new Float32Array(widths),
                new Float32Array(heights), up, collapse);
        }""",
        dict(
            code=code,
            spine=spine.tolist(),
            widths=widths.tolist(),
            heights=heights.tolist(),
            up=up,
            collapse=collapse,
        ),
    )


def _assert_outward(geometry):
    positions = np.asarray(geometry["positions"]).reshape(-1, 3)
    normals = np.asarray(geometry["normals"]).reshape(-1, 3)
    indices = np.asarray(geometry["indices"]).reshape(-1, 3)
    a, b, c = positions[indices].transpose(1, 0, 2)
    faces = np.cross(b - a, c - a)
    outward = normals[indices].sum(axis=1)
    dots = np.einsum("ij,ij->i", faces, outward)
    scale = np.linalg.norm(faces, axis=1) * np.linalg.norm(outward, axis=1)
    assert np.isfinite(positions).all()
    assert np.all(dots >= -1e-6 * np.maximum(scale, 1e-10)), (
        dots.min(),
        np.sum(dots < -1e-6),
    )
    assert np.count_nonzero(dots > 1e-6) > 30


_READ_GEOMETRY = """(id) => {
    const obj = window.threejsViewer._objects.get(id), g = obj.geometry;
    return {positions: Array.from(g.attributes.position.array),
        normals: Array.from(g.attributes.normal.array),
        indices: Array.from(g.index.array.slice(0, Math.min(g.index.count, g.drawRange.count)))};
}"""


@pytest.mark.parametrize("rotated", [False, True])
def test_sub_bead_jog_keeps_height_and_does_not_snap(
    viewer_client, viewer_page, worker_code, rotated
):
    spine = _spine(277)
    up = [0, 0, 1]
    if rotated:
        spine = spine[:, [2, 0, 1]].copy()
        up = [1, 0, 0]
    widths, heights = np.full(len(spine), 11.83), np.full(len(spine), 2.0)
    worker = _worker_geometry(
        viewer_page, worker_code, spine, widths, heights, up, True
    )
    np.testing.assert_array_equal(worker["positions"], worker["original"])
    viewer_client.add_parametric_tube(
        "jog", spine, widths, heights, up_vector=up, lod=False, strand_collapse=True
    )
    _wait_for_object(viewer_page, "jog")
    _wait_for_collapse(viewer_page, "jog")
    result = viewer_page.evaluate("""() => {
        const o = window.threejsViewer._objects.get('jog');
        return {p: Array.from(o.geometry.attributes.position.array),
            original: Array.from(o.userData.uncollapsedPositions)};
    }""")
    np.testing.assert_array_equal(result["p"], result["original"])


@pytest.mark.parametrize("rotated", [False, True])
def test_real_corner_still_collapses_without_height_loss(
    viewer_page, worker_code, rotated
):
    spine, widths, heights = _decode_repro_bead()
    up = [0, 0, 1]
    if rotated:
        spine = spine[:, [2, 0, 1]].copy()
        up = [1, 0, 0]
    geo = _worker_geometry(viewer_page, worker_code, spine, widths, heights, up, True)
    delta = np.asarray(geo["positions"]).reshape(-1, 3) - np.asarray(
        geo["original"]
    ).reshape(-1, 3)
    assert np.linalg.norm(delta, axis=1).max() > 0.01
    ring_up = np.asarray(geo["frames"]).reshape(-1, 6)[:, 3:]
    ring_delta = delta[: len(spine) * 6].reshape(-1, 6, 3)
    np.testing.assert_allclose(
        np.einsum("ijk,ik->ij", ring_delta, ring_up), 0, atol=2e-6
    )


@pytest.mark.parametrize("kind", ["reported", "retrace", "double_hairpin", "vertical"])
def test_hairpin_winding_main_worker_and_frontier(
    viewer_client, viewer_page, worker_code, kind
):
    if kind == "reported":
        spine = _spine(278)
    elif kind == "retrace":
        spine = np.array(
            [[0, 0, 0], [10, 0, 0], [20, 0, 0], [19, 0, 0], [9, 0, 0], [-1, 0, 0]],
            dtype=np.float32,
        )
    elif kind == "double_hairpin":
        spine = np.array(
            [[0, 0, 0], [10, 0, 0], [20, 0, 0], [19, 0, 0], [30, 0, 0], [40, 0, 0]],
            dtype=np.float32,
        )
    else:
        spine = np.array(
            [
                [0, 0, 0],
                [10, 0, 0],
                [10, 0, 1],
                [10, 0, 2],
                [10, 0, 3],
                [20, 0, 3],
                [30, 0, 3],
            ],
            dtype=np.float32,
        )
    widths, heights = np.full(len(spine), 9.46), np.full(len(spine), 2.5)
    _assert_outward(
        _worker_geometry(viewer_page, worker_code, spine, widths, heights, [0, 0, 1])
    )
    viewer_client.add_parametric_tube(
        "hairpin", spine, widths, heights, lod=False, strand_collapse=False
    )
    _wait_for_object(viewer_page, "hairpin")
    initial = viewer_page.evaluate(_READ_GEOMETRY, "hairpin")
    _assert_outward(initial)
    # Direct controller calls keep the geometry probe synchronous with the edit.
    for value in [0.2, 0.51, 0.515, 0.75, 0.5, 0.49, 0.0, 0.8, 1.0]:
        viewer_page.evaluate(
            """value => window.threejsViewer._objects.get('hairpin').userData.parametricTube.setDrawRange(value)""",
            value,
        )
        if value:
            _assert_outward(viewer_page.evaluate(_READ_GEOMETRY, "hairpin"))
    restored = viewer_page.evaluate(_READ_GEOMETRY, "hairpin")
    np.testing.assert_array_equal(restored["indices"], initial["indices"])
    np.testing.assert_allclose(restored["positions"], initial["positions"], atol=1e-5)
