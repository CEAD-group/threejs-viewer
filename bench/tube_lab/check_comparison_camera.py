"""Check view-scaled clipping under linked zoom, orbit and material rebuilds.

Run against a generated comparison server, e.g. --port 8768 for sectors.
"""

import argparse
import json
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-angle=metal"])
        page = browser.new_page(viewport={"width": 2400, "height": 1000})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(
            f"http://127.0.0.1:{args.port}/?case=captures/panel/lissajous&mode=plain&focus=2&lod=off"
        )
        page.wait_for_function("comparisonReady", timeout=120000)

        def check():
            stats = page.evaluate("comparison.stats()")
            assert len(stats) in (4, 6)
            for s in stats:
                assert 0.0099 <= s["cameraNear"] / s["cameraDistance"] <= 0.0101, s
                assert s["cameraFar"] > s["cameraNear"] and not s["contextLost"], s
            assert (
                max(s["cameraNear"] for s in stats)
                - min(s["cameraNear"] for s in stats)
                < 1e-8
            )
            return stats[0]["cameraNear"]

        before = check()
        rect = page.locator("iframe").first.bounding_box()
        page.mouse.move(rect["x"] + rect["width"] / 2, rect["y"] + rect["height"] / 2)
        page.mouse.wheel(0, -500)
        page.wait_for_timeout(250)
        after = check()
        assert after < before, (before, after)
        page.mouse.down()
        page.mouse.move(
            rect["x"] + rect["width"] / 2 + 60,
            rect["y"] + rect["height"] / 2 + 30,
            steps=8,
        )
        page.mouse.up()
        check()
        page.select_option("#mode", "texture")
        page.wait_for_function("comparisonReady", timeout=120000)
        check()
        page.select_option("#focus", "0")
        page.wait_for_function("comparisonReady", timeout=120000)
        check()
        assert not errors, errors
        print(
            json.dumps(
                {
                    "port": args.port,
                    "passed": True,
                    "nearBeforeZoom": before,
                    "nearAfterZoom": after,
                }
            )
        )
        browser.close()


if __name__ == "__main__":
    main()
