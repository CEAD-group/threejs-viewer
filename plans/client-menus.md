# Client-defined menus: one menu mechanism for the viewer and its embedders

> **Status: decided and implemented (PR after #204, branch `client-menus`);
> the ribweaver migration is open.** Decision artifact from the 2026-09-15
> discussion with Thijs. The viewer's own options menu (#204) and any menu an
> embedder or a Python script adds now share one `MenuController` in
> `viewer.js`; this file records why, the shape that was chosen, and what
> still has to move out of ribweaver.

## Problem

ribweaver had three overlay families over the same canvas, each with its own
CSS, positioning and behaviour:

- `web/static/display_menu.js`: a rail of vertical tabs folding off the right
  edge (Display, Edit, Reach Analysis). Declarative content (eye toggles bound
  to viewer object ids or prefixes with localStorage persistence, buttons,
  pill toggles, captions, dividers, a raw `body(el)` hatch), auto-stacking
  and mutual exclusion between mounts. Re-applies eye state on timers
  (350 ms, 900 ms) because it cannot know when objects arrive.
- `.viewer-3d-controls` at top-right in sim, boat and panel: label + select
  rows (colour mode), a colour legend with min/max inputs, a Bead|Line
  segmented control, a Graph button. Copied verbatim between the three lanes.
- `#viewerOverlayTL` / `#placeBar` at top-left: segmented Move/Rotate/Snap
  clusters, offset 44 px by hand to clear the viewer's own toolbar.

The rail module's own header already states the split: the mechanism is
lane-agnostic, only the content is lane knowledge. It lived in the wrong repo.

## Decision

The viewer owns mechanism, styling, layout, open/close, persistence, eye
visibility and shortcut display. The client owns items and callbacks.

- `viewer.addMenu(spec)` (JS) and `ViewerClient.add_menu(...)` (Python).
  The built-in options menu is the first caller of the same code path, so it
  cannot drift from client menus.
- One visual idiom, not two: a top-right bar of dropdown buttons. A menu that
  must stay open (a legend) uses `mode: 'panel'` and stacks below the bar.
  Segmented clusters use `mode: 'bar'` and go top-left. The slide-out rail
  is retired.
- Eye items are re-applied from `_registerObject`, the viewer's single
  registration choke point, so a hidden layer stays hidden when its objects
  are re-pushed or stream in later. The timer hack goes away.
- Every item can show a shortcut chip; `bindKey` also binds it, unless the
  key is one the viewer's own handler consumes (refused with a warning).
- Styling follows ribweaver's overlay palette, exposed as `--tjsv-*` tokens
  on `.threejs-viewer` so ribweaver can override by setting variables.

## Shapes

Menu: `{id, label, icon, title, placement: 'top-right'|'top-left',
mode: 'dropdown'|'panel'|'bar', items, storageKey, hidden}`.

Item: `{type: 'button'|'toggle'|'eye'|'select'|'segmented'|'label'|'divider'|'custom',
id, label, hint, shortcut, bindKey, state, active, checked, value, options,
ids, prefix, disabled, hidden, onClick, onChange, render}`. Function-valued
`state`/`active`/`checked`/`value`/`hidden`/`disabled` are re-evaluated by
`refresh()`; `custom` (JS only) renders into a bare element for widgets like
the colour legend.

Handle: `setItem(id, patch)`, `getValue(id)`, `open/close/isOpen`,
`refresh`, `setHidden`, `remove`, `el`. Viewer-level: `onMenuAction(cb)`
(the Python bridge sends `menu_action` over the socket), `hidingEyeFor(objId)`.

Python: `add_menu`, `update_menu_item`, `remove_menu`, `on_menu_action`.
Menus are replayed on reconnect with their latest patched state.

## Open: ribweaver migration

Per lane, replace the `DisplayMenu.mount` calls and the copied
`.viewer-3d-controls` markup with `viewer.addMenu` calls and delete the
lane CSS. Suggested order: cells (Display and Edit menus, the pilot), panel,
boat, sim, mill, calib. The `display_menu.js` and `toggle_switch.js` modules
are removed once the last lane is migrated. The `KIND` escape hatch (custom
`match`/`apply`/`isOn`/`toggle`) maps onto function-valued `checked` plus an
`onChange` that talks to lane state; `hidingEyeFor` keeps its name.
