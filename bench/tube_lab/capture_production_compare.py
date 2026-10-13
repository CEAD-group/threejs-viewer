"""Capture production comparison sheets and exercise synchronized playback.

Requires production_compare.py and its HTTP server on localhost:8767.
Outputs images, validation results and a linked gallery beside the local demo.
"""

import argparse
import html
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent / "out/production_compare"


def gallery(results):
    cards = []
    for row in results:
        label = f"{row['case']} · {row['mode']} · {row['focus']}"
        error = row.get("error") or "; ".join(row.get("pageErrors", []))
        cards.append(
            f'<article><h2><a href="{html.escape(row["url"])}">{html.escape(label)}</a></h2>'
            f'<p>{html.escape(error)}</p><a href="{row["image"]}"><img loading="lazy" src="{row["image"]}" alt="{html.escape(label)}"></a></article>'
        )
    (OUT / "screenshots.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>Saved toolpath comparisons</title>
<style>body{background:#111923;color:#eef;font:14px system-ui;margin:20px}a{color:#93ccff}img{width:100%;max-width:2400px}article{margin:30px 0;border-top:1px solid #456}h2{font-size:16px}</style>
<h1>Production toolpath comparisons</h1><label>Filter <input id="filter" type="search" placeholder="case or appearance"></label><script>document.addEventListener("input",e=>{if(e.target.id==="filter")for(const a of document.querySelectorAll("article"))a.hidden=!a.querySelector("h2").textContent.toLowerCase().includes(e.target.value.toLowerCase());});</script><p><a href="index.html">Open interactive comparison</a> · Click a heading to reproduce its settings; click an image for full size.</p>
<p>Left to right: Ribweaver production pin v0.0.67, v0.0.69 / #279, #282, #283. Captures use the full original blob; close-ups change camera and playback only. Flat means unlit vertex colours. These are visual comparisons, not GPU performance measurements.</p>"""
        + "\n".join(cards)
    )
    (OUT / "capture_results.json").write_text(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", action="append", help="Only keys containing this text"
    )
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((OUT / "manifest.json").read_text())
    # Fail once before replacing results if the demo server is unavailable.
    with urlopen("http://127.0.0.1:8767/manifest.json", timeout=10) as response:
        assert json.load(response) == manifest, "Server is serving a different build"
    results = (
        json.loads((OUT / "capture_results.json").read_text())
        if args.resume and (OUT / "capture_results.json").exists()
        else []
    )
    results = [
        r
        for r in results
        if r.get("revisions") == manifest["builds"]
        and r.get("harnessSha256") == manifest["harnessSha256"]
    ]
    complete = {
        (r["case"], r["mode"], r["focus"]) for r in results if not r.get("error")
    }
    (OUT / "screenshots").mkdir(exist_ok=True)
    jobs = []
    for entry in manifest["cases"]:
        if args.case and not any(part in entry["key"] for part in args.case):
            continue
        for mode in ("texture", "width", "station"):
            jobs.append((entry, mode, 0))
        if args.details:
            for mode in ("texture", "width", "height", "contrast"):
                jobs.append((entry, mode, 2 if "/panel/" in entry["key"] else 1))
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-angle=metal"])
        asset_cache = {}

        def cached_asset(route):
            url = route.request.url
            if url not in asset_cache:
                response = route.fetch(timeout=30000)
                asset_cache[url] = (response.status, response.headers, response.body())
            status, headers, body = asset_cache[url]
            route.fulfill(status=status, headers=headers, body=body)

        for entry, mode, focus in jobs:
            if (entry["key"], mode, focus) in complete:
                continue
            context = browser.new_context(
                viewport={"width": 600 * len(manifest["builds"]), "height": 1050},
                device_scale_factor=1,
            )
            context.route("https://unpkg.com/**", cached_asset)
            page = context.new_page()
            page_errors = []
            page.on("pageerror", lambda e: page_errors.append(str(e)))
            query = urlencode(
                dict(case=entry["key"], mode=mode, focus=focus, shading="flat")
            )
            url = "index.html?" + query
            name = entry["key"].replace("/", "__") + f"__{mode}__{focus}.png"
            row = dict(
                case=entry["key"],
                mode=mode,
                focus=focus,
                url=url,
                image="screenshots/" + name,
                sha256=entry["sha256"],
                revisions=manifest["builds"],
                harnessSha256=manifest["harnessSha256"],
            )
            try:
                page.goto(
                    "http://127.0.0.1:8767/" + url,
                    wait_until="domcontentloaded",
                    timeout=120000,
                )
                page.wait_for_function("window.comparisonReady", timeout=240000)
                row["builds"] = page.evaluate("window.comparisonResults")
                assert not any("error" in b for b in row["builds"]), row["builds"]
                assert len(row["builds"]) == len(manifest["builds"])
                for b in row["builds"]:
                    assert not b.get("unsupported"), b
                    assert b["geometryPreserved"]
                    assert (
                        (b["maps"] and b["normalMap"] and b["roughnessMap"])
                        if mode == "texture"
                        else b["colorVertices"] > 0
                    )
                    if b["build"] == "local" and b["repairs"] and mode != "texture":
                        assert b["patchColorVertices"] > 0, b
                original = float(page.locator("#progress").input_value())
                # Forward, backward and repeated cap/frontier restoration.
                for fraction in (0, 0.25, 0.5, 0.99, 1, 0.2, 0, 1):
                    page.evaluate("(f)=>comparison.setProgress(f)", fraction)
                row["playbackStats"] = page.evaluate("comparison.stats()")
                assert all(
                    not b["contextLost"]
                    and b["drawCalls"] > 0
                    and b["submittedTriangles"] > 0
                    for b in row["playbackStats"]
                )
                assert all(
                    0.0099 <= b["cameraNear"] / b["cameraDistance"] <= 0.0101
                    and b["cameraFar"] > b["cameraNear"]
                    for b in row["playbackStats"]
                )
                page.evaluate("(f)=>comparison.setProgress(f)", original)
                page.screenshot(path=str(OUT / row["image"]), timeout=120000)
                row["pageErrors"] = page_errors
                assert not page_errors, page_errors
                print("OK", entry["key"], mode, focus, flush=True)
            except Exception as exc:
                row["error"] = str(exc)
                row["pageErrors"] = page_errors
                print("FAIL", entry["key"], mode, focus, str(exc)[:300], flush=True)
            finally:
                context.close()
            previous = [
                r
                for r in results
                if (r["case"], r["mode"], r["focus"])
                == (row["case"], row["mode"], row["focus"])
            ]
            if previous:
                row["previousErrors"] = [r["error"] for r in previous if r.get("error")]
            results = [
                r
                for r in results
                if (r["case"], r["mode"], r["focus"])
                != (row["case"], row["mode"], row["focus"])
            ]
            results.append(row)
            gallery(results)
        browser.close()
    errors = [r for r in results if r.get("error")]
    print(f"{len(results)} comparisons, {len(errors)} failures", flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
