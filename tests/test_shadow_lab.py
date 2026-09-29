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
            narrowEdge: shadowLab.edgeScore('narrow'),
            softEdge: shadowLab.edgeScore('production'),
        })"""
    )
    # Positive control: the fixture must actually reproduce the old artifact.
    assert result["baseline"]["darkFraction"] > 0.2, result
    for score in result["production"]:
        assert score["samples"] > 1000, score
        assert score["darkFraction"] < 0.01, score
        assert score["meanLoss"] < 1, score
    assert result["softEdge"]["meanWidth"] > 1.5 * result["narrowEdge"]["meanWidth"], (
        result
    )


@pytest.mark.browser
def test_soft_shadow_preserves_material_hooks_and_global_chunks(viewer_page):
    errors = []
    viewer_page.on(
        "console", lambda msg: errors.append(msg.text) if msg.type == "error" else None
    )
    result = viewer_page.evaluate(
        """async () => {
            const T = await import('three'), v = window.threejsViewer;
            const chunks = JSON.stringify(T.ShaderChunk), calls = {};
            const materials = [new T.MeshStandardMaterial(), new T.MeshPhysicalMaterial(),
                new T.MeshPhongMaterial(), new T.MeshLambertMaterial(), new T.ShadowMaterial()];
            const group = new T.Group();
            materials.forEach((m, i) => {
                m.onBeforeCompile = function(shader) {
                    calls[i] = (calls[i] || 0) + 1;
                    shader.uniforms.userUniform = {value: 1};
                };
                m.customProgramCacheKey = () => 'user-key-' + i;
                const mesh = new T.Mesh(new T.BoxGeometry(0.8, 0.8, 0.8), m);
                mesh.position.x = i - 2; group.add(mesh);
            });
            const add = (id, object) => { v._scene.add(object); v._registerObject(id, object); };
            add('hooks', group);
            v._sceneBoundsDirty = true;
            v.setCameraPose({position: [5, -8, 5], target: [0, 0, 0]});
            v._updateSun();
            v._renderer.render(v._scene, v._camera);
            // Three's clone does not copy callbacks: it must receive its own patch.
            const clone = materials[0].clone();
            add('clone', new T.Mesh(new T.BoxGeometry(), clone));
            v._updateSun(); v._renderer.render(v._scene, v._camera);
            // Non-PCF fallback still compiles (no sampler-type mismatch).
            v._renderer.shadowMap.type = T.BasicShadowMap;
            v._sun.shadow.map.dispose(); v._sun.shadow.map = null;
            for (const m of [...materials, clone]) m.needsUpdate = true;
            v.requestShadowUpdate(); v._updateSun();
            v._renderer.render(v._scene, v._camera);
            return {calls, keys: materials.map(m => m.customProgramCacheKey()),
                cloneKey: clone.customProgramCacheKey(),
                globalUnchanged: chunks === JSON.stringify(T.ShaderChunk),
                outsideKey: new T.MeshStandardMaterial().customProgramCacheKey()};
        }"""
    )
    assert result["globalUnchanged"]
    assert "tjsv-sun-tent5" not in result["outsideKey"]
    assert "tjsv-sun-tent5" in result["cloneKey"]
    for i, key in enumerate(result["keys"]):
        assert key.startswith(f"user-key-{i}|tjsv-sun-tent5")
        assert result["calls"][str(i)] >= 1
    assert not errors, errors
