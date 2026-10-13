"""Build a pinned, local comparison of production tube captures.

.venv/bin/python bench/tube_lab/production_compare.py
.venv/bin/python -m http.server 8767 --bind 127.0.0.1 --directory bench/tube_lab/out/production_compare

Input captures stay local and unmodified; output is gitignored. No application
code is patched. Colour modes insert only the optional packed-colour block.
"""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "examples"), str(Path(__file__).parent)]
from bead_texture_maker import texture_set  # noqa: E402

# Production compose pins ribweaver:0.1.35; that tag installs this viewer
# from uv.lock using uv sync --frozen --group prod. Keep it separate from
# the newer viewer release (#279), already present in the comparison.
BUILDS = {
    "production": (
        "Production pin · v0.0.67",
        "a5ec37d06e2d1962a15d39165fdf89bfdf36e503",
    ),
    "baseline": ("v0.0.69 · #279", "6752f6b92836f46b2422f83feaa9bc8a80582237"),
    "local": ("#282 · local joins", "3591e88c1053ffb7b1a158e6090618fe449ae6d8"),
    "round": ("#283 · round joins", "682a0cf94d34efe7e04b5ec25c2e6ac88632c1e7"),
}
OUT = Path(__file__).parent / "out/production_compare"


def example_cases(dest):
    """Reuse actual example generators, without running their viewer side effects."""
    from local_joins import scenes, section_values

    cases = list(scenes())
    source = ROOT / "examples/38_bead_texture.py"
    tree = ast.parse(source.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) or (
            isinstance(node, ast.Assign)
            and all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets)
        ):
            nodes.append(node)
    scope = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    spine = scope["make_toolpath"]()
    cases.append(
        dict(
            name="example_38",
            title="Example 38 · textured wall and ribs",
            spine=spine,
            w=scope["BEAD_W"],
            h=scope["BEAD_H"],
            up=[0, 0, 1],
            anchor="top",
        )
    )
    dest.mkdir(exist_ok=True)
    for case in cases:
        name = case["name"]
        spine = np.asarray(case["spine"], dtype="<f4") / 1000
        w, h = [
            np.asarray(section_values(case, key), dtype="<f4") / 1000
            for key in ("w", "h")
        ]
        breaks = case.get("breaks")
        has_breaks = breaks is not None and bool(np.any(breaks))
        blob = spine.tobytes() + w.tobytes() + h.tobytes()
        if has_breaks:
            blob += np.asarray(breaks, dtype="u1").tobytes()
        (dest / f"{name}.tube.bin").write_bytes(blob)
        meta = dict(
            numSpinePoints=len(spine),
            hasColors=False,
            hasOrientations=False,
            hasBreakMask=has_breaks,
            upVector=case["up"],
            heightOffset=-0.5 if case.get("anchor") == "top" else 0,
            lod=False,
            units="m",
            description=case["title"],
            source="existing viewer example/fixture; mm converted to m",
        )
        (dest / f"{name}.tube.json").write_text(json.dumps(meta))


def describe(meta_path, root, output, prefix):
    meta = json.loads(meta_path.read_text())
    blob_path = meta_path.with_suffix(".bin")
    blob = blob_path.read_bytes()
    n = meta["numSpinePoints"]
    assert not meta["hasColors"] and not meta["hasOrientations"]
    assert len(blob) == n * (21 if meta["hasBreakMask"] else 20), meta_path
    spine = np.frombuffer(blob, "<f4", n * 3).reshape(n, 3)
    w = np.frombuffer(blob, "<f4", n, n * 12)
    h = np.frombuffer(blob, "<f4", n, n * 16)
    breaks = (
        np.frombuffer(blob, "u1", n, n * 20)
        if meta["hasBreakMask"]
        else np.zeros(n, "u1")
    )
    assert np.isfinite(spine).all() and np.isfinite(w).all() and np.isfinite(h).all()
    name = str(meta_path.relative_to(root)).removesuffix(".tube.json")
    key = prefix + "/" + name
    extent = np.ptp(spine.astype(float), axis=0)
    center = (spine.min(axis=0).astype(float) + spine.max(axis=0)) / 2
    radius = max(float(np.linalg.norm(extent) / 2 + w.max()), 0.001)
    npz = meta_path.with_name(meta_path.name.replace(".tube.json", ".npz"))
    scalars = None
    if npz.exists():
        with np.load(npz) as data:
            # NPZ is metadata only: geometry is always decoded from the wire blob.
            assert len(data["spine"]) == n
            layer = np.asarray(data["layer_ids"], dtype="<f4")
            line = (
                np.asarray(data["src_line"], dtype="<f4")
                if "src_line" in data
                else np.full(n, -1, dtype="<f4")
            )
            scalars = np.concatenate([layer, line])
    scalar_url = None
    if scalars is not None:
        scalar_url = "scalars/" + key.replace("/", "__") + ".bin"
        (output / scalar_url).write_bytes(scalars.tobytes())
    # Camera-only close-ups: never crop/rebuild the input geometry.
    dirs = np.diff(spine.astype(float), axis=0)
    lengths = np.linalg.norm(dirs, axis=1)
    dirs /= np.maximum(lengths[:, None], 1e-12)
    turns = np.sum(dirs[:-1] * dirs[1:], axis=1)
    valid = (
        (w[1:-1] > 0)
        & (h[1:-1] > 0)
        & (breaks[1:-1] == 0)
        & (breaks[2:] == 0)
        & (lengths[:-1] > 1e-8)
        & (lengths[1:] > 1e-8)
    )
    corner = int(np.argmin(np.where(valid, turns, np.inf))) + 1
    wide_candidates = np.flatnonzero(w >= 0.99 * w.max())
    widest = int(
        wide_candidates[
            np.argmin(np.linalg.norm(spine[wide_candidates] - center, axis=1))
        ]
    )
    presets = [
        dict(name="Whole path", center=center.tolist(), radius=radius, progress=1)
    ]
    for label, point in [("Sharpest bend", corner), ("Widest bead / crossing", widest)]:
        bead = max(float(w[point]), float(h[point]), 0.001)
        # A short amount of following deposition helps reveal the completed join.
        end = min(n - 1, point + 10)
        presets.append(
            dict(
                name=label,
                center=spine[point].tolist(),
                radius=bead * 1.8,
                progress=1 if label.startswith("Widest") else end / max(1, n - 1),
                point=point,
                line=float(scalars[n + point])
                if scalars is not None and scalars[n + point] >= 0
                else None,
            )
        )
    return dict(
        key=key,
        name=name,
        group=prefix,
        meta=meta,
        points=n,
        breaks=int(np.count_nonzero(breaks)),
        blob=f"{prefix}/{name}.tube.bin",
        sha256=hashlib.sha256(blob).hexdigest(),
        scalars=scalar_url,
        presets=presets,
        ranges={
            "width": [float(w.min() * 1000), float(w.max() * 1000)],
            "height": [float(h.min() * 1000), float(h.max() * 1000)],
            "layer": [float(scalars[:n].min()), float(scalars[:n].max())]
            if scalars is not None
            else [0, 0],
            "station": [0, n - 1],
            "elevation": [
                float(spine[:, 2].min() * 1000),
                float(spine[:, 2].max() * 1000),
            ],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "tmp/toolpath_testdata")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scalars").mkdir(exist_ok=True)
    data = args.data.resolve()
    link = OUT / "captures"
    if not link.exists():
        link.symlink_to(data, target_is_directory=True)
    elif link.resolve() != data:
        raise ValueError(f"{link} already points elsewhere")
    example_cases(OUT / "examples")
    cases = [
        describe(p, data, OUT, "captures") for p in sorted(data.glob("*/*.tube.json"))
    ]
    cases += [
        describe(p, OUT / "examples", OUT, "examples")
        for p in sorted((OUT / "examples").glob("*.tube.json"))
    ]
    builds = []
    child = Path(__file__).with_name("production_compare_frame.js").read_text()
    for key, (label, rev) in BUILDS.items():
        html = subprocess.check_output(
            ["git", "show", f"{rev}:src/threejs_viewer/viewer.html"],
            cwd=ROOT,
            text=True,
        )
        html = html.replace("<head>", '<head><link rel="icon" href="data:,">')
        html = html.replace(
            "window.threejsViewer = new ThreeJSViewer(container, {",
            "window.threejsViewer = new ThreeJSViewer(container, {\n autoConnect:false,",
        )
        marker = "    </script>\n</body>"
        assert html.count(marker) == 1
        html = html.replace(
            marker, child.replace('"__BUILD__"', json.dumps(key)) + "\n" + marker
        )
        (OUT / f"{key}.html").write_text(html)
        builds.append(dict(key=key, label=label, rev=rev))
    maps = texture_set(
        color="#698d9e", width=10, height=2.5, length=25, smoothness=0.6, fibers=True
    )
    for key in ("color", "normal", "roughness"):
        (OUT / f"{key}.png").write_bytes(maps[key])
    (OUT / "manifest.json").write_text(
        json.dumps(
            dict(
                builds=builds,
                cases=cases,
                harnessSha256=hashlib.sha256(
                    (
                        child
                        + Path(__file__).with_name("production_compare.js").read_text()
                        + Path(__file__)
                        .with_name("production_compare.html")
                        .read_text()
                    ).encode()
                ).hexdigest(),
            ),
            indent=2,
        )
    )
    for source, target in [
        ("production_compare.html", "index.html"),
        ("production_compare.js", "compare.js"),
    ]:
        (OUT / target).write_text(Path(__file__).with_name(source).read_text())
    print(
        f"Built {len(cases)} cases ({len(cases) - 25} production, 25 earlier examples), four pinned viewers in {OUT}"
    )


if __name__ == "__main__":
    main()
