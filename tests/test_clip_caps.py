"""Browser tests for clip caps on clipped solids (issue #259)."""

import math

import numpy as np
import pytest

from conftest import frames, settle

# Per visible cap under `id`: world-space area, the largest distance of a cap
# vertex to each active clip plane, the cap's own clipping-plane count and
# colour, plus the number of boundary vertices left open (0 for closed loops).
_CAP_STATS = """(id) => {
    const v = window.threejsViewer;
    const obj = v.getObject(id);
    if (!obj) return null;
    obj.updateWorldMatrix(true, true);
    const caps = [];
    obj.traverse((n) => { if (n.userData.__clipCap) caps.push(n); });
    const planes = v._clipPlanes.map((p) => [p.normal.x, p.normal.y, p.normal.z, p.constant]);
    return caps.filter((c) => c.visible).map((cap) => {
        cap.updateWorldMatrix(true, false);
        const e = cap.matrixWorld.elements;
        const a = cap.geometry.getAttribute('position').array;
        const W = [];
        for (let i = 0; i < a.length; i += 3) {
            const x = a[i], y = a[i + 1], z = a[i + 2];
            W.push([e[0] * x + e[4] * y + e[8] * z + e[12],
                    e[1] * x + e[5] * y + e[9] * z + e[13],
                    e[2] * x + e[6] * y + e[10] * z + e[14]]);
        }
        let area = 0;
        const edges = new Map();
        const key = (p) => p.map((c) => Math.round(c * 1e5)).join(',');
        for (let t = 0; t < W.length; t += 3) {
            const [p, q, r] = [W[t], W[t + 1], W[t + 2]];
            const u = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
            const w = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
            const c = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
            area += Math.hypot(c[0], c[1], c[2]) / 2;
            for (const [s0, s1] of [[p, q], [q, r], [r, p]]) {
                const k0 = key(s0), k1 = key(s1);
                if (k0 === k1) continue;
                const k = k0 < k1 ? k0 + '|' + k1 : k1 + '|' + k0;
                edges.set(k, (edges.get(k) || 0) + 1);
            }
        }
        // Boundary edges (odd use count) must form closed loops: every
        // boundary vertex then has an even boundary degree.
        const deg = new Map();
        for (const [k, n] of edges) {
            if (n % 2 === 0) continue;
            for (const vk of k.split('|')) deg.set(vk, (deg.get(vk) || 0) + 1);
        }
        let open = 0;
        for (const d of deg.values()) if (d % 2) open++;
        const dist = planes.map(([nx, ny, nz, c]) =>
            Math.max(...W.map((p) => Math.abs(nx * p[0] + ny * p[1] + nz * p[2] + c))));
        const m = cap.material;
        return { area, dist, nTri: W.length / 3, open,
                 planes: (m.clippingPlanes || []).length,
                 color: m.color.getHex(), vertexColors: !!m.vertexColors };
    });
}"""


def _caps(page, id):
    return page.evaluate(_CAP_STATS, id)


def _wait_caps(page, id, n):
    page.wait_for_function(
        "([id, n]) => { const o = window.threejsViewer.getObject(id); if (!o) return false;"
        " let k = 0; o.traverse((c) => { if (c.userData.__clipCap && c.visible) k++; });"
        " return k === n; }",
        arg=[id, n],
        timeout=10_000,
    )


def _key(page, key):
    """The viewer binds its shortcuts on the container, so dispatch there."""
    page.evaluate(
        "(k) => window.threejsViewer.container.dispatchEvent(new KeyboardEvent("
        "'keydown', { key: k, code: 'Key' + k.toUpperCase(), bubbles: true }))",
        key,
    )


def _cap_count(page, id):
    return page.evaluate(
        "(id) => { let k = 0; window.threejsViewer.getObject(id).traverse("
        "(c) => { if (c.userData.__clipCap) k++; }); return k; }",
        id,
    )


@pytest.mark.browser
def test_caps_off_by_default(viewer_client, viewer_page):
    viewer_client.add_box("b", width=2, height=2, depth=2)
    viewer_client.set_clipping_plane(normal=[1, 0, 0], distance=0.3, show_helper=False)
    settle(viewer_client)
    frames(viewer_page, 3)
    assert viewer_page.evaluate("() => window.threejsViewer.getClipCaps()") is False
    assert _cap_count(viewer_page, "b") == 0


@pytest.mark.browser
@pytest.mark.parametrize(
    ("kind", "kwargs", "normal", "distance", "expected"),
    [
        ("box", {"width": 2, "height": 2, "depth": 2}, [1, 0, 0], 0.3, 4.0),
        ("sphere", {"radius": 1.0}, [0, 0, 1], 0.5, math.pi * 0.75),
        # CylinderGeometry runs along local Y: an X cut is a 1 x 2 rectangle.
        (
            "cylinder",
            {"radius_top": 0.5, "radius_bottom": 0.5, "height": 2},
            [1, 0, 0],
            0.0,
            2.0,
        ),
        (
            "cylinder",
            {"radius_top": 0.5, "radius_bottom": 0.5, "height": 2},
            [0, 1, 0],
            0.4,
            math.pi * 0.25,
        ),
        ("capsule", {"radius": 0.25, "length": 0.5}, [0, -1, 0], 0.1, math.pi * 0.0625),
    ],
)
def test_primitive_cap_lies_on_plane_with_analytic_area(
    viewer_client, viewer_page, kind, kwargs, normal, distance, expected
):
    getattr(viewer_client, f"add_{kind}")("p", color=0xFF0000, **kwargs)
    viewer_client.set_clipping_caps(True)
    viewer_client.set_clipping_plane(
        normal=normal, distance=distance, show_helper=False
    )
    settle(viewer_client)
    _wait_caps(viewer_page, "p", 1)
    (cap,) = _caps(viewer_page, "p")
    assert cap["dist"][0] < 1e-5
    # Polygonal sphere/cylinder sections sit slightly inside the analytic circle.
    assert cap["area"] == pytest.approx(expected, rel=0.03)
    assert cap["open"] == 0
    assert cap["planes"] == 0
    assert cap["color"] == 0xFF0000


@pytest.mark.browser
def test_slab_caps_both_planes_and_each_is_clipped_by_the_other(
    viewer_client, viewer_page
):
    viewer_client.add_box("b", width=2, height=2, depth=2)
    viewer_client.set_clipping_caps(True)
    viewer_client.set_clipping_slab(
        normal=[0, 0, 1], center=0.2, thickness=0.6, show_helper=False
    )
    settle(viewer_client)
    _wait_caps(viewer_page, "b", 2)
    caps = _caps(viewer_page, "b")
    assert [c["planes"] for c in caps] == [1, 1]
    assert caps[0]["dist"][0] < 1e-5 and caps[1]["dist"][1] < 1e-5
    assert all(c["area"] == pytest.approx(4.0, rel=1e-4) for c in caps)


@pytest.mark.browser
def test_cap_follows_plane_moves_and_transforms(viewer_client, viewer_page):
    viewer_client.add_sphere("s", radius=1.0)
    viewer_client.set_clipping_caps(True)
    viewer_client.set_clipping_plane(normal=[0, 0, 1], distance=0.0, show_helper=False)
    settle(viewer_client)
    _wait_caps(viewer_page, "s", 1)
    full = _caps(viewer_page, "s")[0]["area"]
    viewer_client.set_clipping_plane(normal=[0, 0, 1], distance=0.8, show_helper=False)
    settle(viewer_client)
    frames(viewer_page, 2)
    moved = _caps(viewer_page, "s")[0]
    assert moved["dist"][0] < 1e-5
    assert moved["area"] == pytest.approx(math.pi * (1 - 0.64), rel=0.05)
    assert moved["area"] < full
    # Moving the object re-cuts it at the same world plane.
    viewer_client.set_matrix("s", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0.8, 1])
    settle(viewer_client)
    frames(viewer_page, 2)
    centred = _caps(viewer_page, "s")[0]
    assert centred["dist"][0] < 1e-5
    assert centred["area"] == pytest.approx(full, rel=1e-3)


@pytest.mark.browser
def test_cap_overrides_translucency_and_switch_off(viewer_client, viewer_page):
    """A per-object clip_cap wins over the switch; translucent solids stay open;
    turning the switch or clipping off removes the caps."""
    viewer_client.add_box("on", clip_cap=True)
    viewer_client.add_box("off", clip_cap=False, position=[2, 0, 0])
    viewer_client.add_box("glass", opacity=0.5, position=[4, 0, 0])
    viewer_client.add_box("plain", position=[6, 0, 0])
    viewer_client.set_clipping_plane(normal=[0, 1, 0], distance=0.1, show_helper=False)
    settle(viewer_client)
    _wait_caps(viewer_page, "on", 1)
    assert _cap_count(viewer_page, "plain") == 0
    viewer_client.set_clipping_caps(True)
    settle(viewer_client)
    _wait_caps(viewer_page, "plain", 1)
    frames(viewer_page, 2)
    assert _cap_count(viewer_page, "off") == 0
    assert _cap_count(viewer_page, "glass") == 0
    # A cap follows set_color, and set_opacity below 1 drops it.
    viewer_client.set_color("plain", 0x00FF00)
    settle(viewer_client)
    frames(viewer_page, 2)
    assert _caps(viewer_page, "plain")[0]["color"] == 0x00FF00
    viewer_client.set_opacity("plain", 0.5)
    settle(viewer_client)
    _wait_caps(viewer_page, "plain", 0)
    viewer_client.disable_clipping_plane()
    settle(viewer_client)
    _wait_caps(viewer_page, "on", 0)
    assert _cap_count(viewer_page, "on") == 0


@pytest.mark.browser
def test_cap_is_skipped_by_highlight_and_debug_cycles(viewer_client, viewer_page):
    viewer_client.add_box("b", color=0x3366FF)
    viewer_client.set_clipping_caps(True)
    viewer_client.set_clipping_plane(normal=[1, 0, 0], distance=0.0, show_helper=False)
    settle(viewer_client)
    _wait_caps(viewer_page, "b", 1)
    viewer_client.set_highlight("b", True)
    settle(viewer_client)
    _key(viewer_page, "n")
    frames(viewer_page, 2)
    out = viewer_page.evaluate(
        "() => { const v = window.threejsViewer; let cap = null;"
        " v.getObject('b').traverse((c) => { if (c.userData.__clipCap) cap = c; });"
        " return {outlined: !!cap.userData.__highlightEdge,"
        "  swapped: cap.userData.originalMaterial !== undefined,"
        "  std: cap.material.isMeshStandardMaterial,"
        "  ownerSwapped: v.getObject('b').userData.originalMaterial !== undefined}; }"
    )
    assert out == {
        "outlined": False,
        "swapped": False,
        "std": True,
        "ownerSwapped": True,
    }


# Read the pixel at a world point after a fresh render.
_PIXEL_AT = """([x, y, z]) => {
    const v = window.threejsViewer;
    v._renderer.render(v._scene, v._camera);
    const gl = v._renderer.getContext();
    const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
    const p = v._camera.position.clone().set(x, y, z).project(v._camera);
    const px = new Uint8Array(4);
    gl.readPixels(Math.round((p.x + 1) / 2 * w), Math.round((p.y + 1) / 2 * h),
                  1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    return [px[0], px[1], px[2]];
}"""


def _is_red(rgb):
    return rgb[0] > 80 and rgb[0] > 2 * rgb[1] and rgb[0] > 2 * rgb[2]


@pytest.mark.browser
def test_slab_cut_reads_cap_colour_in_both_projections(viewer_client, viewer_page):
    """Looking straight down a thin slab through a box shows the background
    without caps (the walls are edge-on) and the box colour with them."""
    page = viewer_page
    viewer_client.set_sun(enabled=False)
    viewer_client.add_box("b", width=2, height=2, depth=2, color=0xFF0000)
    viewer_client.set_clipping_slab(
        normal=[0, 0, 1], center=0.0, thickness=0.4, show_helper=False
    )
    viewer_client.set_camera(position=[0.01, 0.01, 8], target=[0, 0, 0], up=[0, 1, 0])
    settle(viewer_client)
    frames(page, 3)
    assert not _is_red(page.evaluate(_PIXEL_AT, [0, 0, 0.2]))
    viewer_client.set_clipping_caps(True)
    settle(viewer_client)
    _wait_caps(page, "b", 2)
    assert _is_red(page.evaluate(_PIXEL_AT, [0, 0, 0.2]))
    _key(page, "o")
    frames(page, 3)
    assert (
        page.evaluate("() => window.threejsViewer._camera.isOrthographicCamera") is True
    )
    assert _is_red(page.evaluate(_PIXEL_AT, [0, 0, 0.2]))


# Covered area of the union of every visible cap's triangles on clip plane
# `plane`, by sampling a grid in the plane: overlapping coplanar fans (swept
# cells, round joins) would double count in a plain triangle-area sum.
_CAP_UNION_AREA = """([id, plane, grid]) => {
    const v = window.threejsViewer;
    const obj = v.getObject(id);
    obj.updateWorldMatrix(true, true);
    const P = v._clipPlanes[plane];
    const n = [P.normal.x, P.normal.y, P.normal.z];
    const a = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    let u = [a[1] * n[2] - a[2] * n[1], a[2] * n[0] - a[0] * n[2], a[0] * n[1] - a[1] * n[0]];
    const ul = Math.hypot(...u); u = u.map((c) => c / ul);
    const w = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]];
    const tris = [];
    obj.traverse((cap) => {
        if (!cap.userData.__clipCap || !cap.visible) return;
        cap.updateWorldMatrix(true, false);
        const e = cap.matrixWorld.elements;
        const arr = cap.geometry.getAttribute('position').array;
        const pts = [];
        let onPlane = true;
        for (let i = 0; i < arr.length; i += 3) {
            const x = arr[i], y = arr[i + 1], z = arr[i + 2];
            const p = [e[0] * x + e[4] * y + e[8] * z + e[12], e[1] * x + e[5] * y + e[9] * z + e[13],
                       e[2] * x + e[6] * y + e[10] * z + e[14]];
            if (Math.abs(n[0] * p[0] + n[1] * p[1] + n[2] * p[2] + P.constant) > 1e-4) onPlane = false;
            pts.push([p[0] * u[0] + p[1] * u[1] + p[2] * u[2], p[0] * w[0] + p[1] * w[1] + p[2] * w[2]]);
        }
        if (!onPlane) return;
        for (let i = 0; i < pts.length; i += 3) tris.push([pts[i], pts[i + 1], pts[i + 2]]);
    });
    if (!tris.length) return 0;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const t of tris) for (const [x, y] of t) {
        x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
    const dx = (x1 - x0) / grid, dy = (y1 - y0) / grid;
    const inside = (t, x, y) => {
        const [[ax, ay], [bx, by], [cx, cy]] = t;
        const d1 = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
        const d2 = (cx - bx) * (y - by) - (cy - by) * (x - bx);
        const d3 = (ax - cx) * (y - cy) - (ay - cy) * (x - cx);
        return (d1 >= 0 && d2 >= 0 && d3 >= 0) || (d1 <= 0 && d2 <= 0 && d3 <= 0);
    };
    let hit = 0;
    for (let i = 0; i < grid; i++) for (let j = 0; j < grid; j++) {
        const x = x0 + (i + 0.5) * dx, y = y0 + (j + 0.5) * dy;
        if (tris.some((t) => inside(t, x, y))) hit++;
    }
    return hit * dx * dy;
}"""


def _union_area(page, id, plane=0, grid=160):
    return page.evaluate(_CAP_UNION_AREA, [id, plane, grid])


def _straight_tool(client, id, **kwargs):
    """A cylinder (r=0.3, h=1, axis +Z) swept along X from -2 to 2."""
    n = 9
    positions = np.zeros((n, 3), dtype=np.float32)
    positions[:, 0] = np.linspace(-2, 2, n)
    axes = np.tile(np.array([0, 0, 1], dtype=np.float32), (n, 1))
    profile = np.array([[0, 0.3], [1, 0.3]], dtype=np.float32)
    client.add_swept_tool(id, positions, axes, profile, sections=16, **kwargs)


# 16-gon of circumradius 0.3.
_DISC16 = 8 * math.sin(2 * math.pi / 16) * 0.09


@pytest.mark.browser
def test_swept_tool_caps_lie_on_plane_and_cover_the_section(viewer_client, viewer_page):
    page = viewer_page
    colors = np.full(9, 0x3366CC, dtype=np.uint32)
    _straight_tool(viewer_client, "tool", colors=colors)
    viewer_client.set_clipping_caps(True)
    # Across the path: the swept body's section is the 0.6 x 1 wall.
    viewer_client.set_clipping_plane(normal=[1, 0, 0], distance=0.25, show_helper=False)
    settle(viewer_client)
    _wait_caps(page, "tool", 1)
    (cap,) = _caps(page, "tool")
    assert cap["dist"][0] < 1e-5
    assert cap["vertexColors"] is True
    assert _union_area(page, "tool") == pytest.approx(0.6, rel=0.03)
    # Along the path at mid height: a stadium, 4 long plus the two end discs.
    viewer_client.set_clipping_plane(
        normal=[0, 0, -1], distance=-0.5, show_helper=False
    )
    settle(viewer_client)
    frames(page, 2)
    assert _caps(page, "tool")[0]["dist"][0] < 1e-5
    assert _union_area(page, "tool") == pytest.approx(4 * 0.6 + _DISC16, rel=0.03)


@pytest.mark.browser
def test_swept_tool_cap_follows_draw_range_and_delete(viewer_client, viewer_page):
    page = viewer_page
    _straight_tool(viewer_client, "tool")
    viewer_client.set_clipping_caps(True)
    viewer_client.set_clipping_plane(
        normal=[0, 0, -1], distance=-0.5, show_helper=False
    )
    settle(viewer_client)
    _wait_caps(page, "tool", 1)
    full = _union_area(page, "tool")
    viewer_client.set_draw_range("tool", 0.5)
    settle(viewer_client)
    frames(page, 2)
    half = _union_area(page, "tool")
    # Four of eight station pairs: 2 long plus the end discs.
    assert half == pytest.approx(2 * 0.6 + _DISC16, rel=0.04)
    assert half < full
    # A plane cut beyond the frontier leaves no cap.
    viewer_client.set_clipping_plane(normal=[1, 0, 0], distance=1.5, show_helper=False)
    settle(viewer_client)
    _wait_caps(page, "tool", 0)
    viewer_client.delete("tool")
    settle(viewer_client)
    frames(page, 2)
    assert page.evaluate("() => window.threejsViewer.getObject('tool')") is None


@pytest.mark.browser
def test_swept_tool_clip_cap_false_stays_open(viewer_client, viewer_page):
    _straight_tool(viewer_client, "tool", clip_cap=False)
    viewer_client.set_clipping_caps(True)
    viewer_client.set_clipping_plane(normal=[1, 0, 0], distance=0.25, show_helper=False)
    settle(viewer_client)
    frames(viewer_page, 3)
    assert _cap_count(viewer_page, "tool") == 0
