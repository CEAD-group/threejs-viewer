"""Client-defined menu: the viewer draws it, this script owns the contents.

Adds a "Demo" tab on the viewer's right-edge rail; clicking it slides out a panel with
one of each item type in two foldable sections. Interactions return to Python through
``on_menu_action`` and are applied to the scene from here (browser -> Python ->
browser), so the viewer never needs to know what "Recolour" means. G recolours and K
toggles the spin.

Run:  uv run python examples/35_client_menu.py
"""

import math
import random
import time

import numpy as np

from threejs_viewer import ViewerClient

COLORS = [0x3388FF, 0xFF8833, 0x44CC66, 0xCC44AA, 0xEEEE44]
SIZES = {"small": 0.5, "medium": 1.0, "large": 1.6}

client = ViewerClient(toolbar=True)
client.connect()
client.add_grid("grid", cell_size=1.0, extent=30.0, center_color=0x888888)
for i in range(5):
    client.add_box(f"box_{i}", 1, 1, 1, color=COLORS[i], position=[2.0 * i - 4, 0, 0.5])
client.set_view("iso", animate=False)
client.frame_points(
    np.array(np.meshgrid(*zip([-5, -1, 0], [5, 3.5, 1]))).reshape(3, -1).T, fill=0.8
)
client.add_sphere("marker", 0.4, color=0xFFFFFF, position=[0, 3, 0.4])

client.add_menu(
    "demo",
    label="Demo",
    title="Example menu driven from Python",
    storage_key="tjsv-example.demo",
    items=[
        # A collapsible label folds the rows up to the next divider.
        {"type": "label", "id": "scene", "label": "Scene", "collapsible": True},
        {
            "type": "eye",
            "id": "boxes",
            "label": "Boxes",
            "prefix": "box_",
            "hint": "Show or hide every box_* object",
        },
        {"type": "eye", "id": "marker", "label": "Marker", "ids": ["marker"]},
        {"type": "divider"},
        {
            "type": "label",
            "id": "controls",
            "label": "Controls",
            "collapsible": True,
        },
        {
            "type": "button",
            "id": "recolour",
            "label": "Recolour boxes",
            "shortcut": "G",
            "bind_key": True,
        },
        {
            "type": "toggle",
            "id": "spin",
            "label": "Spin marker",
            "checked": False,
            "shortcut": "K",
            "bind_key": True,
        },
        {
            "type": "select",
            "id": "size",
            "label": "Box size",
            "options": [{"value": k, "label": k} for k in SIZES],
            "value": "medium",
        },
        {
            "type": "segmented",
            "id": "side",
            "label": "Marker side",
            "options": ["left", "right"],
            "value": "right",
        },
    ],
)

state = {"spin": False, "side": 1.0}


def on_action(action):
    print("menu action:", action)
    item, value = action["item"], action.get("value")
    if item == "recolour":
        for i in range(5):
            client.set_color(f"box_{i}", random.choice(COLORS))
    elif item == "spin":
        state["spin"] = bool(value)
    elif item == "size":
        s = SIZES[value]
        client.batch_update(
            {
                f"box_{i}": {"position": [2.0 * i - 4, 0, s / 2], "scale": [s, s, s]}
                for i in range(5)
            }
        )
    elif item == "side":
        state["side"] = -1.0 if value == "left" else 1.0
        client.update_menu_item(
            "demo", "side", state="flipped" if value == "left" else ""
        )


client.on_menu_action(on_action)

print("Open the Demo tab on the right edge. G recolours, K spins. Ctrl+C to stop.")
t0 = time.time()
try:
    while True:
        if state["spin"]:
            a = (time.time() - t0) * 1.5
            client.batch_update(
                {
                    "marker": {
                        "position": [
                            3 * state["side"] * math.cos(a),
                            3 * math.sin(a),
                            0.4,
                        ]
                    }
                }
            )
        time.sleep(1 / 30)
except KeyboardInterrupt:
    pass
finally:
    client.disconnect()
