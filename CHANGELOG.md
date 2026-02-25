# Changelog

## 0.0.5

- Add `roughness` and `metalness` parameters to `add_box`, `add_sphere`, `add_cylinder`, `add_capsule`
- New example: `13_material_properties.py` — PBR material property grid
- Sync `__init__.py` version with `pyproject.toml`

## 0.0.4

- Add transparency support: `set_opacity()`, `set_color()` with optional opacity
- Fix opacity on GLB models (needsUpdate, depthWrite, multi-material)
- Add initial transform support for binary models
- New example: `12_transparency.py`

## 0.0.3

- Add generic binary animation channels (`add_channel()` API)
- Supported channel types: transforms, draw_ranges, colors, visibility, opacity
- Supported dtypes: float32, uint32, uint8 (indexed colors with colormap)
- Update examples to showcase binary animation channels

## 0.0.2

- Add pre-built mesh display (`add_mesh()`) and bead extrusion (`add_bead()`)
- Add `draw_range` support for polylines and meshes
- Switch binary transfers from WebSocket to HTTP sidecar for performance
- Lighting overhaul for GLB/PBR and embedded animation support
- New examples: GLB models, animated GLB, animation stress test, toolpath

## 0.0.1

- Initial release
- Primitives: box, sphere, cylinder, plane, cone, torus, capsule
- Polylines with gradient colormaps (viridis, plasma, turbo)
- 3D model loading (GLTF/GLB, STL, OBJ, FBX, DAE, PLY, 3DS)
- Frame-based animation with looping playback
- Real-time streaming mode with batch updates
- WebSocket communication between Python and browser
