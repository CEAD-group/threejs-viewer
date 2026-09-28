"""Unit tests for viewer/build.py (no browser)."""

import base64
import gzip
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

VIEWER_DIR = Path(__file__).parent.parent / "src" / "threejs_viewer" / "viewer"


def _load_build():
    """Import build.py as a module (it is a script, not part of the package)."""
    spec = importlib.util.spec_from_file_location(
        "viewer_build", VIEWER_DIR / "build.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def build_mod():
    return _load_build()


def test_gzip_b64_reuses_previous_when_payload_equal(build_mod):
    """An already-committed encoding that inflates to the same bytes is kept
    verbatim (gzip output differs per zlib implementation); a changed payload
    or an unparseable previous value recompresses."""
    raw = bytes(range(256)) * 200
    # A different but valid encoding of the same bytes (compresslevel 1 vs 9).
    previous = base64.b64encode(gzip.compress(raw, compresslevel=1, mtime=0)).decode()
    fresh = build_mod._gzip_b64(raw)
    assert fresh != previous
    assert build_mod._gzip_b64(raw, previous) == previous
    changed = build_mod._gzip_b64(raw + b"x", previous)
    assert changed != previous
    assert gzip.decompress(base64.b64decode(changed)) == raw + b"x"
    assert build_mod._gzip_b64(raw, "not base64!") == fresh
    assert build_mod._gzip_b64(raw, None) == fresh


def test_previous_blobs_finds_every_inlined_slot(build_mod):
    """The regexes that read the committed viewer.html back find the Draco pair
    and one blob per cubemap face."""
    blobs = build_mod._previous_blobs()
    assert {"draco.wasm", "draco.wrapper"} <= set(blobs)
    for face in build_mod.CUBEMAP_FACES:
        assert f"cubemap.{build_mod.DEFAULT_CUBEMAP}.{face}" in blobs
    for value in blobs.values():
        gzip.decompress(base64.b64decode(value, validate=True))


def test_cubemap_faces_are_flat_rgbe(build_mod):
    """Committed .hdr faces are flat RGBE, not RLE: run-length coding gzips
    worse on HDR noise (+15% in viewer.html for the same pixels), so a
    re-export that turns RLE back on is caught here."""
    checked = 0
    for d in build_mod.CUBEMAPS_DIR.iterdir():
        for face in build_mod.CUBEMAP_FACES:
            p = d / f"{face}.hdr"
            if not p.is_file():
                continue
            raw = p.read_bytes()
            i = raw.index(b"\n\n") + 2
            j = raw.index(b"\n", i) + 1
            dims = raw[i : j - 1].decode().split()
            assert dims[0] == "-Y" and dims[2] == "+X", (p, dims)
            h, w = int(dims[1]), int(dims[3])
            assert raw[j : j + 2] != b"\x02\x02", f"{p} is RLE-encoded"
            assert len(raw) - j == w * h * 4, p
            checked += 1
    assert checked >= 6


def test_build_is_idempotent(build_mod, tmp_path, monkeypatch):
    """Two builds over the same sources produce byte-identical outputs, and a
    build over a committed viewer.html keeps its blob encodings."""
    viewer_dir = tmp_path / "viewer"
    shutil.copytree(
        VIEWER_DIR, viewer_dir, ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copy(VIEWER_DIR.parent / "viewer.html", tmp_path / "viewer.html")
    monkeypatch.setattr(build_mod, "VIEWER_DIR", viewer_dir)
    monkeypatch.setattr(build_mod, "OUTPUT_HTML", tmp_path / "viewer.html")
    monkeypatch.setattr(build_mod, "STATIC_DIR", viewer_dir / "static")
    monkeypatch.setattr(build_mod, "DRACO_DIR", viewer_dir / "static" / "draco")
    monkeypatch.setattr(build_mod, "CUBEMAPS_DIR", viewer_dir / "static" / "cubemaps")
    committed = build_mod._previous_blobs()
    build_mod.build()
    first = (tmp_path / "viewer.html").read_bytes()
    first_index = (viewer_dir / "static" / "cubemaps" / "index.json").read_bytes()
    assert build_mod._previous_blobs() == committed
    build_mod.build()
    assert (tmp_path / "viewer.html").read_bytes() == first
    assert (
        viewer_dir / "static" / "cubemaps" / "index.json"
    ).read_bytes() == first_index
