"""
Animation Basics Demo

Demonstrates the looping animation mode with pre-computed frames.
Creates a simple solar system with orbiting planets.

Run: uv run python examples/03_animation_basics.py
"""

import math


from threejs_viewer import Animation, viewer


def make_transform_matrix(position, rotation_z=0, scale=1.0):
    """Create a 4x4 transform matrix (column-major for Three.js)."""
    c, s = math.cos(rotation_z), math.sin(rotation_z)

    # Column-major order
    return [
        scale * c,
        scale * s,
        0,
        0,
        -scale * s,
        scale * c,
        0,
        0,
        0,
        0,
        scale,
        0,
        position[0],
        position[1],
        position[2],
        1,
    ]


v = viewer()
v.clear()

# Create the sun (stationary)
v.add_sphere(
    "sun", radius=1.0, color=0xFFDD00, position=[0, 0, 0], roughness=0.3, metalness=0.8
)

# Create planets
planets = [
    {
        "id": "mercury",
        "radius": 0.15,
        "color": 0x888888,
        "orbit_radius": 2.0,
        "period": 2.0,
        "roughness": 0.6,
        "metalness": 0.4,
    },
    {
        "id": "venus",
        "radius": 0.25,
        "color": 0xFFAA55,
        "orbit_radius": 3.0,
        "period": 3.5,
        "roughness": 0.8,
        "metalness": 0.1,
    },
    {
        "id": "mars",
        "radius": 0.2,
        "color": 0xFF4422,
        "orbit_radius": 6.0,
        "period": 7.0,
        "roughness": 0.9,
        "metalness": 0.05,
    },
]

for planet in planets:
    v.add_sphere(
        planet["id"],
        radius=planet["radius"],
        color=planet["color"],
        roughness=planet["roughness"],
        metalness=planet["metalness"],
    )

# Earth system group — earth and moon are both children.
# Animating the group moves both; moon only needs its local orbit offset.
v.add_group("earth_system")
v.add_sphere(
    "earth",
    radius=0.3,
    color=0x4488FF,
    roughness=0.5,
    metalness=0.2,
    parent="earth_system",
)
v.add_sphere(
    "moon",
    radius=0.08,
    color=0xCCCCCC,
    roughness=0.9,
    metalness=0.0,
    parent="earth_system",
)

# Create animation frames
duration = 10.0  # seconds
fps = 30
n_frames = int(duration * fps)

animation = Animation(loop=True)

for i in range(n_frames):
    t = i / fps
    transforms = {}

    # Sun stays still (but we include it for completeness)
    transforms["sun"] = make_transform_matrix([0, 0, 0])

    # Animate each planet
    for planet in planets:
        angle = 2 * math.pi * t / planet["period"]
        x = planet["orbit_radius"] * math.cos(angle)
        y = planet["orbit_radius"] * math.sin(angle)
        transforms[planet["id"]] = make_transform_matrix([x, y, 0])

    # Earth system group — moves earth + moon together
    earth_angle = 2 * math.pi * t / 5.0
    earth_x = 4.5 * math.cos(earth_angle)
    earth_y = 4.5 * math.sin(earth_angle)
    transforms["earth_system"] = make_transform_matrix([earth_x, earth_y, 0])
    # (earth sphere has no separate transform — it sits at origin within the group)

    # Moon only needs its LOCAL orbit offset — the group handles the rest
    moon_angle = 2 * math.pi * t / 0.8  # Fast orbit around earth
    moon_x = 0.6 * math.cos(moon_angle)
    moon_y = 0.6 * math.sin(moon_angle)
    transforms["moon"] = make_transform_matrix([moon_x, moon_y, 0])

    animation.add_frame(time=t, transforms=transforms)

# Add timeline markers for notable events
animation.add_marker(0.0, "Animation start", color=0x00FF00)
animation.add_marker(5.0, "Halfway point", color=0xFFFF00)

# Load the animation
v.load_animation(animation)

print(f"Animation loaded: {animation.n_frames} frames, {animation.duration:.1f}s")
v.wait_for_assets()
