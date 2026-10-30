"""Kinematic chains (issue #292): set_kinematic_chain, set_joints and the joints channel.

tests/data/kinematic_chain/reference_poses.json pins threejs_viewer.kinematics
.compose (world matrix per link at sample poses, row-major, top three rows) on a
chain with every joint type; seeded random chains then hold the viewer's port to
that reference.
"""

import functools
import json
import math
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from websockets.sync.client import connect as ws_connect

from conftest import frames, settle
from threejs_viewer import Animation, ViewerClient, kinematics

REFERENCE = Path(__file__).parent / "data" / "kinematic_chain" / "reference_poses.json"
VIEWER_DIR = Path(__file__).parent.parent / "src" / "threejs_viewer" / "viewer"


def _unit(rng):
    v = rng.normal(size=3)
    return (v / np.linalg.norm(v)).tolist()


def _random_chain(seed, n_links=9):
    """A chain mixing fixed, revolute, prismatic, solved and aim joints, with
    branching parents, coupled terms and random rigid origins."""
    rng = np.random.default_rng(seed)
    axes = [
        {"name": f"A{i}", "kind": "rotary", "limits": [None, None]} for i in range(4)
    ]
    solved = ["S0", "S1"]
    links = []
    for i in range(n_links):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        origin = np.eye(4)
        origin[:3, :3] = q * np.sign(np.linalg.det(q))
        origin[:3, 3] = rng.uniform(-0.5, 0.5, size=3)
        parent = (
            None if i == 0 or rng.random() < 0.15 else links[rng.integers(i)]["name"]
        )
        kind = ["revolute", "prismatic", "solved", "aim", None][i % 5]
        joint = None
        if kind in ("revolute", "prismatic"):
            names = rng.choice(
                [a["name"] for a in axes], size=rng.integers(1, 3), replace=False
            )
            joint = {
                "type": kind,
                "axis": _unit(rng),
                "value": {
                    "offset": float(rng.uniform(-10, 10)),
                    "terms": {str(n): float(rng.uniform(-2, 2)) for n in names},
                },
            }
            if kind == "revolute":
                joint["point"] = rng.uniform(-0.2, 0.2, size=3).tolist()
        elif kind == "solved":
            joint = {
                "type": "solved",
                "axis": _unit(rng),
                "point": rng.uniform(-0.2, 0.2, size=3).tolist(),
                "index": int(rng.integers(len(solved))),
            }
        elif kind == "aim":
            joint = {
                "type": "aim",
                "axis": _unit(rng),
                "point": rng.uniform(-0.2, 0.2, size=3).tolist(),
                "body": _unit(rng),
                "target": {
                    "link": None
                    if rng.random() < 0.3
                    else links[rng.integers(i)]["name"],
                    "point": rng.uniform(-1, 1, size=3).tolist(),
                },
            }
        links.append(
            {
                "name": f"L{i}",
                "parent": parent,
                "origin": origin.tolist(),
                "joint": joint,
            }
        )
    desc = {
        "format": kinematics.FORMAT,
        "version": 1,
        "units": {"length": "m", "angle": "deg"},
        "axes": axes,
        "solved": solved,
        "links": links,
    }
    poses = [
        {
            "axes": rng.uniform(-180, 180, size=len(axes)).tolist(),
            "solved": rng.uniform(-180, 180, size=len(solved)).tolist(),
        }
        for _ in range(8)
    ]
    for pose in poses:
        world = kinematics.compose(desc, pose["axes"], pose["solved"])
        pose["world"] = {n: m.reshape(-1)[:12].tolist() for n, m in world.items()}
    return {"description": desc, "poses": poses}


def _chains():
    """The pinned reference chain plus seeded random ones, with expected poses."""
    return {"reference": json.loads(REFERENCE.read_text())} | {
        f"random{seed}": _random_chain(seed) for seed in range(4)
    }


def _arm(length=2.0):
    """A turn about +Z at the origin carrying a tip `length` out along +X."""
    eye = np.eye(4).tolist()
    tip = np.eye(4)
    tip[0, 3] = length
    return {
        "format": kinematics.FORMAT,
        "version": 1,
        "axes": [{"name": "J1", "kind": "rotary", "limits": [None, None]}],
        "solved": [],
        "links": [
            {
                "name": "J1",
                "parent": None,
                "origin": eye,
                "joint": {
                    "type": "revolute",
                    "axis": [0.0, 0.0, 1.0],
                    "point": [0.0, 0.0, 0.0],
                    "value": {"offset": 0.0, "terms": {"J1": 1.0}},
                },
            },
            {"name": "tip", "parent": "J1", "origin": tip.tolist(), "joint": None},
        ],
    }


class _StubWS:
    def __init__(self):
        self.sent = []

    def send(self, data):
        self.sent.append(json.loads(data))


def test_set_joint_data_packs_one_padded_row_per_chain():
    """A rail plus a 6-axis arm is 7 values per keyframe; a narrower chain in
    the same channel is zero-padded to the widest row."""
    anim = Animation()
    anim.set_frame_times([0.0, 1.0, 2.0])
    cell = np.arange(21, dtype=float).reshape(3, 7)
    anim.set_joint_data({"cell": cell})
    (ch,) = anim._channels
    assert (ch.name, ch.ids, ch.stride, ch.dtype) == ("joints", ["cell"], 7, "float32")
    assert np.asarray(ch.data, dtype=np.float32).nbytes == 3 * 7 * 4

    anim.set_joint_data(
        {"cell": cell[:, :4], "positioner": [[1.0], [2.0], [3.0]]},
        solved={"cell": [[9.0], [8.0], [7.0]]},
        dtype="float64",
        interpolation="hold",
    )
    (ch,) = anim._channels
    assert ch.ids == ["cell", "positioner"] and ch.stride == 5
    rows = np.asarray(ch.data).reshape(3, 2, 5)
    np.testing.assert_array_equal(rows[1, 0], [7, 8, 9, 10, 8])
    np.testing.assert_array_equal(rows[1, 1], [2, 0, 0, 0, 0])
    assert ch.interpolation == "hold"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"values": {}}, "at least one"),
        ({"values": {"a": [1.0, 2.0]}}, "n_frames, n_axes"),
        ({"values": {"a": [[1.0], [2.0]], "b": [[1.0]]}}, "same number of frames"),
        ({"values": {"a": [[float("nan")]]}}, "finite"),
        ({"values": {"a": [[1.0]]}, "solved": {"b": [[1.0]]}}, "without values"),
        ({"values": {"a": [[1.0]]}, "solved": {"a": [[1.0], [2.0]]}}, "n_solved"),
        ({"values": {"a": [[1.0]]}, "dtype": "uint8"}, "dtype"),
    ],
)
def test_set_joint_data_validates(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Animation().set_joint_data(**kwargs)


def test_kinematic_chain_records_validates_and_clear_forgets():
    client = ViewerClient()
    client._ws = _StubWS()

    class _Desc:
        def to_dict(self):
            return _arm()

    client.set_kinematic_chain("arm", _Desc(), object_ids={"J1": "J1", "tip": "tip"})
    spec = client._kinematic_chains["arm"]
    assert spec["object_ids"] == {"J1": "J1", "tip": "tip"}
    assert client._ws.sent[-1] == spec

    client.set_joints("arm", [30])
    assert client._joint_poses["arm"] == {
        "type": "set_joints",
        "id": "arm",
        "values": [30.0],
        "solved": [],
    }
    with pytest.raises(ValueError, match="expected 1 values"):
        client.set_joints("arm", [1, 2])
    with pytest.raises(ValueError, match="solved"):
        client.set_joints("arm", [1], solved=[2])
    with pytest.raises(ValueError, match="finite"):
        client.set_joints("arm", [float("inf")])
    with pytest.raises(ValueError, match="no kinematic chain"):
        client.set_joints("other", [1])

    # A resend that changes the axis count drops the pose that no longer fits.
    two_axes = _arm()
    two_axes["axes"].append({"name": "J2", "kind": "rotary", "limits": [None, None]})
    client.set_kinematic_chain("arm", two_axes, object_ids={"tip": "tool"})
    assert "arm" not in client._joint_poses
    assert client._kinematic_chains["arm"]["object_ids"] == {"tip": "tool"}

    bad = _arm()
    bad["links"].reverse()
    with pytest.raises(ValueError, match="before its parent"):
        client.set_kinematic_chain("bad", bad, object_ids={})
    with pytest.raises(ValueError, match="version 1"):
        client.set_kinematic_chain("bad", {**_arm(), "version": 2}, object_ids={})
    with pytest.raises(ValueError, match="object_ids must map"):
        client.set_kinematic_chain("bad", _arm(), object_ids=None)
    with pytest.raises(ValueError, match=r"not links of the description: \['tool'\]"):
        client.set_kinematic_chain("bad", _arm(), object_ids={"tool": "tool"})
    with pytest.raises(ValueError, match=r"not links.*'J9'"):
        client.set_kinematic_chain(
            "bad", _arm(), object_ids={}, object_scale={"J9": 2.0}
        )

    client.clear()
    assert client._kinematic_chains == {} and client._joint_poses == {}


def test_reconnect_replays_chains_before_their_poses():
    client = ViewerClient(port=0, open_browser=False)
    client._start_servers(http_port=0)
    try:
        client.set_kinematic_chain("arm", _arm(), object_ids={"tip": "tip"})
        client.set_joints("arm", [45])
        seen = []
        with ws_connect(f"ws://127.0.0.1:{client.port}", open_timeout=5) as ws:
            while "set_joints" not in seen:
                seen.append(json.loads(ws.recv(timeout=5))["type"])
        assert seen.index("set_kinematic_chain") < seen.index("set_joints")
    finally:
        client.disconnect()


def test_reference_compose_matches_pinned_poses():
    """compose reproduces the pinned poses, so a change to the reference shows up
    here before it can drift the viewer port along with it."""
    data = json.loads(REFERENCE.read_text())
    for pose in data["poses"]:
        world = kinematics.compose(data["description"], pose["axes"], pose["solved"])
        for name, expected in pose["world"].items():
            assert np.abs(world[name].reshape(-1)[:12] - expected).max() < 1e-12, name


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda d: d.pop("format"), "expected format"),
        (lambda d: d.update(format="other.description"), "expected format"),
        (lambda d: d.update(version=2), "version 1"),
        (lambda d: d.update(units={"length": "mm", "angle": "deg"}), "units"),
        (lambda d: d["links"].reverse(), "before its parent"),
        (lambda d: d["links"][0]["joint"].update(type="screw"), "unknown joint type"),
        (
            lambda d: d["links"][0]["joint"]["value"]["terms"].update(J7=1.0),
            "unknown axes",
        ),
        (lambda d: d["links"][1].update(origin=[[1, 0, 0]]), "finite 4x4"),
        (lambda d: d["links"].append(dict(d["links"][1])), "unique"),
    ],
)
def test_validate_refuses_what_breaks_the_format(change, match):
    desc = _arm()
    change(desc)
    with pytest.raises(ValueError, match=match):
        kinematics.validate(desc)


def _source_page(page, tmp_path):
    """Serve the viewer source dir and import viewer.js as a module into the page."""
    (tmp_path / "index.html").write_text(
        """<!doctype html><html><head>
<script type="importmap">{"imports": {
  "three": "https://unpkg.com/three@0.183.2/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@0.183.2/examples/jsm/"}}</script>
</head><body><script type="module">
window.kc = await import('/viewer/viewer.js');
</script></body></html>"""
    )

    class Handler(SimpleHTTPRequestHandler):
        def translate_path(self, path):
            path = path.split("?")[0]
            if path.startswith("/viewer/"):
                return str(VIEWER_DIR / path[len("/viewer/") :])
            return str(tmp_path / path.lstrip("/"))

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    page.goto(f"http://127.0.0.1:{server.server_address[1]}/index.html")
    page.wait_for_function("() => window.kc != null", timeout=20_000)
    return server


@pytest.mark.browser
def test_viewer_fk_matches_the_python_reference_in_float64(page, tmp_path):
    """The exported pure FK reproduces kinematics.compose to 1e-9 on the pinned
    chain and on seeded random chains covering every joint type, and refuses a
    description in another format."""
    chains = _chains()
    server = _source_page(page, tmp_path)
    try:
        result = page.evaluate(
            """(chains) => {
                const m = window.kc, report = {};
                for (const [key, chain] of Object.entries(chains)) {
                    const c = m.compileKinematicChain(chain.description);
                    const out = new Float64Array(16 * c.names.length);
                    let worst = 0, count = 0;
                    for (const pose of chain.poses) {
                        m.composeKinematicChainInto(c, pose.axes, pose.solved, out);
                        c.names.forEach((n, i) => {
                            for (let k = 0; k < 12; k++) {
                                worst = Math.max(worst, Math.abs(out[16 * i + k] - pose.world[n][k]));
                                count++;
                            }
                        });
                    }
                    const byName = m.composeKinematicChain(
                        chain.description, chain.poses[0].axes, chain.poses[0].solved);
                    report[key] = {worst, count, mapSize: byName.size};
                }
                let refused = null;
                const other = {...chains.reference.description, format: 'other.description'};
                try { m.compileKinematicChain(other); } catch (e) { refused = e.message; }
                return {report, refused};
            }""",
            chains,
        )
    finally:
        server.shutdown()
    assert set(result["report"]) == set(chains)
    for key, r in result["report"].items():
        assert r["count"] > 0 and r["mapSize"] > 0, key
        assert r["worst"] < 1e-9, (key, r["worst"])
    assert "expected format threejs_viewer.kinematic_chain" in result["refused"]


def _object_world_rowmajor(page, ids):
    return page.evaluate(
        """(ids) => Object.fromEntries(ids.map((id) => {
            const o = window.threejsViewer._objects.get(id);
            o.updateWorldMatrix(true, false);
            const e = o.matrixWorld.elements;
            return [id, [0, 1, 2, 3].flatMap((r) => [0, 1, 2, 3].map((c) => e[4 * c + r]))];
        }))""",
        ids,
    )


@pytest.mark.browser
def test_set_joints_poses_objects_like_the_reference(viewer_client, viewer_page):
    """Every linked object lands on the reference world matrix within 1e-5
    (float32), whether it sits at the scene root or is parented under its chain
    parent's object (the viewer then writes the parent-relative matrix)."""
    data = {"chains": _chains()}
    for key, chain in data["chains"].items():
        links = chain["description"]["links"]
        ids = {link["name"]: f"{key}:{link['name']}" for link in links}
        for i, link in enumerate(links):
            # Every other link nested under its parent's object, the rest at the root.
            parent = ids[link["parent"]] if link["parent"] and i % 2 else None
            viewer_client.add_group(ids[link["name"]], parent=parent)
        viewer_client.set_kinematic_chain(key, chain["description"], object_ids=ids)
    for pose_index in range(min(len(c["poses"]) for c in data["chains"].values())):
        for key, chain in data["chains"].items():
            pose = chain["poses"][pose_index]
            viewer_client.set_joints(key, pose["axes"], pose["solved"])
        settle(viewer_client)
        for key, chain in data["chains"].items():
            pose = chain["poses"][pose_index]
            names = [link["name"] for link in chain["description"]["links"]]
            got = _object_world_rowmajor(viewer_page, [f"{key}:{n}" for n in names])
            for n in names:
                err = np.abs(
                    np.array(got[f"{key}:{n}"][:12]) - pose["world"][n][:12]
                ).max()
                assert err < 1e-5, (key, n, pose_index, err)


def _tip_position(page, id="tip"):
    return page.evaluate(
        f"""() => {{ const o = window.threejsViewer._objects.get('{id}');
            o.updateWorldMatrix(true, false);
            const e = o.matrixWorld.elements; return [e[12], e[13], e[14]]; }}"""
    )


@pytest.mark.browser
def test_joints_channel_lerps_in_joint_space(viewer_client, viewer_page):
    """Between keyframes the arm turns about its joint, so the tip stays on the
    arc where a per-link matrix lerp would cut the chord. The chain wins over a
    transforms channel on the same id, an object added later wears the current
    pose, and unload_animation returns to the set_joints pose."""
    viewer_client.set_kinematic_chain("arm", _arm(), object_ids={"tip": "tip"})
    viewer_client.set_joints("arm", [30])
    viewer_client.add_box("tip", width=0.2, height=0.2, depth=0.2)
    settle(viewer_client)
    x, y, _ = _tip_position(viewer_page)
    assert math.isclose(x, 2 * math.cos(math.radians(30)), abs_tol=1e-6)
    assert math.isclose(y, 2 * math.sin(math.radians(30)), abs_tol=1e-6)

    anim = Animation(loop=False)
    anim.set_frame_times(np.array([0.0, 1.0]))
    anim.set_joint_data({"arm": [[0.0], [90.0]]})
    junk = np.tile(np.eye(4, dtype=np.float32).T.reshape(1, 16), (2, 1))
    junk[:, 12] = 50.0
    anim.set_transform_data(["tip"], junk)
    viewer_client.load_animation(anim, autoplay=False)
    deadline = time.time() + 5
    while not viewer_page.evaluate("() => window.threejsViewer._animation != null"):
        assert time.time() < deadline, "animation never loaded"
        time.sleep(0.05)
    viewer_page.evaluate("() => window.threejsViewer.seekAnimationTime(0.5)")
    frames(viewer_page)
    x, y, _ = _tip_position(viewer_page)
    r = math.sqrt(2.0)
    assert math.isclose(x, r, abs_tol=1e-5) and math.isclose(y, r, abs_tol=1e-5)

    viewer_client.unload_animation()
    settle(viewer_client)
    x, y, _ = _tip_position(viewer_page)
    assert math.isclose(x, 2 * math.cos(math.radians(30)), abs_tol=1e-6)
    assert math.isclose(y, 2 * math.sin(math.radians(30)), abs_tol=1e-6)


def test_set_kinematic_chain_refuses_unknown_format_and_units():
    client = ViewerClient()
    with pytest.raises(ValueError, match="format"):
        client.set_kinematic_chain(
            "a", {**_arm(), "format": "other.description"}, object_ids={}
        )
    with pytest.raises(ValueError, match="units"):
        client.set_kinematic_chain(
            "a", {**_arm(), "units": {"length": "mm", "angle": "deg"}}, object_ids={}
        )
    assert client._kinematic_chains == {}


@pytest.mark.browser
def test_query_returns_the_viewer_fk_for_a_producer_to_check(
    viewer_client, viewer_page
):
    """query_kinematic_chain is how a producer checks the viewer against its own
    FK on a real description: evaluated poses come back exactly as the viewer's
    float64 FK computed them, and the current pose reports each posed object's
    actual world matrix (mapping, scale and parenting included)."""
    chain = _random_chain(7)
    desc = chain["description"]
    names = [link["name"] for link in desc["links"]]
    child = next(link for link in desc["links"] if link["parent"] == "L0")["name"]
    viewer_client.add_group("L0")
    viewer_client.add_group(child, parent="L0")
    ids = {"L0": "L0", child: child}
    viewer_client.set_kinematic_chain(
        "cell", desc, object_ids=ids, object_scale={child: 2.0}
    )

    poses = chain["poses"]
    res = viewer_client.query_kinematic_chain(
        "cell",
        [p["axes"] for p in poses],
        [p["solved"] for p in poses] if desc["solved"] else None,
    )
    assert res["names"] == names
    assert res["world"].shape == (len(poses), len(names), 4, 4)
    for k, pose in enumerate(poses):
        for i, n in enumerate(names):
            ref = np.asarray(pose["world"][n][:12]).reshape(3, 4)
            assert np.abs(res["world"][k, i, :3] - ref).max() < 1e-9

    before = viewer_client.query_kinematic_chain("cell")
    assert before["posed"] is False and before["world"] is None

    viewer_client.set_joints("cell", poses[3]["axes"], poses[3]["solved"])
    settle(viewer_client)
    now = viewer_client.query_kinematic_chain("cell")
    assert now["posed"] and now["object_ids"] == ids
    np.testing.assert_allclose(now["world"][0], res["world"][3], atol=1e-12)
    for link, obj in ids.items():
        expected = res["world"][3, names.index(link)].copy()
        if link == child:
            expected[:3, :3] *= 2.0
        assert np.abs(now["objects"][obj] - expected).max() < 1e-5, link

    with pytest.raises(ValueError, match="expected 4 values"):
        viewer_client.query_kinematic_chain("cell", [[0.0]])
    with pytest.raises(ValueError, match="no kinematic chain"):
        viewer_client.query_kinematic_chain("nope")
