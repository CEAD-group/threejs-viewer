"""Pixel regression for thin CAD panels that cast shadows onto themselves."""

from pathlib import Path

import pytest


@pytest.mark.browser
def test_thin_panel_shadow_acne(viewer_page):
    scene = Path(__file__).parents[1] / "bench/shadow_lab/scene.js"
    viewer_page.add_script_tag(type="module", content=scene.read_text())
    viewer_page.wait_for_function("() => !!window.shadowLab")
    result = viewer_page.evaluate(
        """() => ({
            baseline: shadowLab.acneScore('baseline', -46, 45),
            production: [15, 45, 80].map(el => shadowLab.acneScore('production', -46, el)),
        })"""
    )
    # Positive control: the fixture must actually reproduce the old artifact.
    assert result["baseline"]["darkFraction"] > 0.2, result
    for score in result["production"]:
        assert score["samples"] > 1000, score
        assert score["darkFraction"] < 0.01, score
        assert score["meanLoss"] < 1, score
