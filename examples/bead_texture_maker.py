"""
Generate a tiling texture set with an extruded-thermoplastic look for beads:
a colour map, a normal map, a roughness map and a height (bump) map.

The maps are laid out the way ``add_parametric_tube(texture=...)`` reads
them: the horizontal axis runs along the bead and tiles (left edge matches
right edge), the vertical axis runs across it from one side over the top to
the other. The image height is sized from the bead's width and height, so
surface features come out the same size in both directions.

What goes into the surface, all sized in millimetres:

- die lines: fine streaks running along the extrusion direction,
- sharkskin: the fine transverse ripple of melt fracture,
- grain: fine isotropic roughness, plus a slow waviness,
- fibres (optional): short chopped fibres lying mostly along the flow, as in
  a glass- or carbon-filled pellet material.

Main options:

  --color        base colour of the plastic, e.g. "#3a3d42"
  --fibers / --no-fibers, --fiber-density, --fiber-length, --fiber-color
  --roughness    how matte the surface is (0 mirror-like .. 1 fully diffuse);
                 this sets the level of the roughness map
  --smoothness   how flat the surface is (0 heavy ripple .. 1 no relief);
                 this scales the height, normal and shading detail

Run: uv run python examples/bead_texture_maker.py --color "#3a3d42" --show
     uv run python examples/bead_texture_maker.py --no-fibers --smoothness 0.9
         --roughness 0.25 --color "#c8452c" -o glossy_red

Use the result:

    v.add_parametric_tube("bead", spine, widths, heights,
                          texture="bead_color.png", normal_map="bead_normal.png",
                          roughness_map="bead_roughness.png", roughness=1.0,
                          texture_length=25)

The height map is a 16-bit PNG for a Bump or Displacement node in Blender;
its black-to-white range in millimetres is printed when the script runs.
"""

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- PNG output


def png_bytes(pixels):
    """Encode (H, W) gray or (H, W, 3) RGB, uint8 or uint16, as a PNG."""
    pixels = np.ascontiguousarray(pixels)
    depth = pixels.dtype.itemsize * 8
    color_type = 2 if pixels.ndim == 3 else 0
    h, w = pixels.shape[:2]
    rows = pixels.astype(">u2" if depth == 16 else "u1").reshape(h, -1).view(np.uint8)
    raw = np.hstack([np.zeros((h, 1), np.uint8), rows]).tobytes()  # filter 0 per row

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, color_type, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def parse_color(text):
    text = text.strip().lstrip("#")
    if text.lower().startswith("0x"):
        text = text[2:]
    if len(text) != 6:
        raise argparse.ArgumentTypeError(f"colour must be RRGGBB hex, got {text!r}")
    return np.array([int(text[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0


# ------------------------------------------------------------- noise helpers


class Canvas:
    """A W x H pixel grid covering ``length`` x ``across`` mm. Noise is shaped
    in the frequency domain, so every field tiles exactly in both axes."""

    def __init__(self, width_px, length_mm, across_mm, seed):
        self.w = int(width_px)
        self.h = max(8, int(round(width_px * across_mm / length_mm / 4)) * 4)
        self.dx = length_mm / self.w  # mm per pixel along the bead
        self.dy = across_mm / self.h  # mm per pixel across it
        self.rng = np.random.default_rng(seed)
        self.fx = np.fft.fftfreq(self.w, d=self.dx)[None, :]  # cycles per mm
        self.fy = np.fft.fftfreq(self.h, d=self.dy)[:, None]

    def noise(self, spectrum):
        """White noise shaped by ``spectrum(fx, fy)``, zero mean, unit std."""
        white = np.fft.fft2(self.rng.standard_normal((self.h, self.w)))
        field = np.fft.ifft2(white * spectrum(self.fx, self.fy)).real
        field -= field.mean()
        return field / (field.std() + 1e-12)

    def blobs(self, along_mm, across_mm):
        """Smooth noise with the given correlation lengths."""
        return self.noise(
            lambda fx, fy: np.exp(-((fx * along_mm) ** 2) - (fy * across_mm) ** 2)
        )

    def ripple(self, period_mm, spread, across_mm):
        """Ridges across the bead with a loose period along it."""
        f0 = 1.0 / period_mm
        return self.noise(
            lambda fx, fy: (
                np.exp(-(((np.abs(fx) - f0) / (spread * f0)) ** 2))
                * np.exp(-((fy * across_mm) ** 2))
            )
        )

    def blur(self, field, sigma_px):
        kx = np.fft.fftfreq(self.w)[None, :]
        ky = np.fft.fftfreq(self.h)[:, None]
        kernel = np.exp(-2 * (np.pi * sigma_px) ** 2 * (kx**2 + ky**2))
        return np.fft.ifft2(np.fft.fft2(field) * kernel).real

    def fibers(self, density, length_mm, angle_deg):
        """Coverage mask (0..1) of short fibres lying mostly along the bead."""
        area = self.w * self.dx * self.h * self.dy
        n = int(density * area)
        if n == 0:
            return np.zeros((self.h, self.w))
        rng = self.rng
        cx = rng.uniform(0, self.w * self.dx, n)
        cy = rng.uniform(0, self.h * self.dy, n)
        angle = np.radians(rng.normal(0, angle_deg, n))
        length = length_mm * rng.lognormal(0, 0.35, n)
        weight = rng.uniform(0.45, 1.0, n)
        steps = int(np.ceil(length.max() / min(self.dx, self.dy) * 2)) + 1
        t = np.linspace(-0.5, 0.5, steps)[None, :]
        x = (cx[:, None] + t * (length * np.cos(angle))[:, None]) / self.dx
        y = (cy[:, None] + t * (length * np.sin(angle))[:, None]) / self.dy
        mask = np.zeros((self.h, self.w))
        np.maximum.at(
            mask,
            (y.astype(int).ravel() % self.h, x.astype(int).ravel() % self.w),
            np.repeat(weight, steps),
        )
        return np.clip(self.blur(mask, 0.8) * 1.5, 0, 1)


# ----------------------------------------------------------------- the maps


def half_perimeter(width, height):
    """Surface length from one side tip over the top to the other, for the
    viewer's chamfered hexagonal bead section: what texture v 0..1 spans."""
    c = min(width, height) / 2
    return 2 * c * np.sqrt(2) + abs(width - height)


def top_band(cv, width, height):
    """Per image row, how far the row lies on the bead's flat top face
    (0 on the chamfered sides, 1 on the top, eased over ~0.5 mm). The top
    spans the middle of v, between the two chamfers; a bead taller than
    wide has no flat top."""
    if width <= height:
        return np.zeros(cv.h)
    across = half_perimeter(width, height)
    chamfer = min(width, height) / 2 * np.sqrt(2)
    s = (np.arange(cv.h) + 0.5) * cv.dy  # mm from one side tip
    inside = np.minimum(s - chamfer, (across - chamfer) - s)  # mm into the top
    t = np.clip(inside / 0.5 + 0.5, 0, 1)
    return t * t * (3 - 2 * t)


def make_maps(args):
    across = half_perimeter(args.width, args.height)
    cv = Canvas(args.size, args.length, across, args.seed)
    relief = (1.0 - args.smoothness) * args.bump_strength

    # Surface components, each zero mean and unit std.
    die_lines = 0.7 * cv.blobs(60.0, 0.12) + 0.3 * cv.blobs(8.0, 0.3)
    sharkskin = cv.ripple(0.55, 0.45, 2.5) * (0.85 + 0.15 * cv.blobs(3.0, 3.0))
    grain = cv.blobs(0.07, 0.07)
    waviness = cv.blobs(20.0, 1.5)  # slow, and stretched along the flow
    fibers = (
        cv.fibers(args.fiber_density, args.fiber_length, 10.0)
        if args.fibers
        else np.zeros((cv.h, cv.w))
    )

    # Height in mm. A filled material extrudes rougher than a neat one.
    fill = 1.5 if args.fibers else 1.0
    # The streaks along the flow lead; the transverse ripple stays faint.
    height = relief * (
        0.006 * args.ripple * fill * sharkskin + 0.006 * fill * grain + 0.020 * waviness
    )
    height += (0.3 + 0.7 * relief) * 0.022 * args.streaks * die_lines
    height += (0.3 + 0.7 * relief) * 0.012 * fibers

    # The nozzle wipes the flat top of the bead as it lays it: flatter, less
    # streaky and a little glossier there, while the free sides keep the full
    # extruded surface. `keep` is 1 on the sides and 1 - wipe on the top.
    keep = 1 - args.wipe * top_band(cv, args.width, args.height)[:, None]
    height *= keep

    # Tangent-space normals (OpenGL convention: +X along u, +Y along v, and
    # v runs up the image). u tiles; v is mirrored at both edges by the viewer.
    dh_du = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) / (2 * cv.dx)
    padded = np.pad(height, ((1, 1), (0, 0)), mode="symmetric")
    dh_dv = -(padded[2:] - padded[:-2]) / (2 * cv.dy)
    normal = np.dstack([-dh_du, -dh_dv, np.ones_like(height)])
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)

    # Roughness: matte in the ripple valleys and grain, glossier on fibres.
    detail = height - cv.blur(height, 0.6 / cv.dx)
    cavity = np.clip(-detail / (detail.std() * 2 + 1e-9), 0, 1)
    roughness = args.roughness * (1 + (0.12 * grain + 0.08 * die_lines) * keep)
    roughness *= 0.7 + 0.3 * keep
    roughness += 0.15 * cavity * (1 - args.smoothness) - 0.25 * fibers * args.roughness
    roughness = np.clip(roughness, 0.04, 1.0)

    # Colour: the base tint, streaked along the flow, darker in the valleys,
    # with the fibres showing through.
    base = np.asarray(args.color)
    streak_tint = 0.6 * cv.blobs(80.0, 0.5) + 0.4 * die_lines
    tint = 1 + (0.10 * args.streaks * streak_tint + 0.02 * waviness) * keep
    tint -= 0.22 * cavity * (1 - args.smoothness)
    color = base[None, None, :] * tint[:, :, None]
    if args.fibers:
        if args.fiber_color is not None:
            fiber_color = np.asarray(args.fiber_color)
        elif base @ [0.2126, 0.7152, 0.0722] < 0.35:
            fiber_color = 0.7 * base + 0.3 * np.array([0.62, 0.62, 0.60])
        else:
            fiber_color = 0.6 * base + 0.4 * np.array([0.12, 0.12, 0.12])
        cover = (0.5 * fibers)[:, :, None]
        color = color * (1 - cover) + fiber_color[None, None, :] * cover
    color = np.clip(color, 0, 1)

    lo, hi = height.min(), height.max()
    return (
        {
            "color": (color * 255 + 0.5).astype(np.uint8),
            "normal": ((normal * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8),
            "roughness": (roughness * 255 + 0.5).astype(np.uint8),
            "height": ((height - lo) / max(hi - lo, 1e-12) * 65535 + 0.5).astype(
                np.uint16
            ),
        },
        (hi - lo),
        cv,
    )


def show(paths, args):
    """Put the texture set on the two sample beads in the viewer."""
    from bead_uv_template import corners_bead, wavy_bead

    from threejs_viewer import viewer

    v = viewer()
    v.clear()
    v.add_grid("floor", cell_size=25.0, extent=600.0, position=[75, -30, -3])
    for name, (spine, widths, heights) in (
        ("bead_wavy", wavy_bead()),
        ("bead_corners", corners_bead()),
    ):
        v.add_parametric_tube(
            name,
            spine.astype(np.float32),
            widths.astype(np.float32),
            heights.astype(np.float32),
            texture=paths["color"],
            normal_map=paths["normal"],
            roughness_map=paths["roughness"],
            roughness=1.0,
            metalness=0.0,
            texture_length=args.length,
        )
    v.wait_for_assets()


def texture_set(**options):
    """The maps as PNG bytes, for use from another script. Options are the
    command-line ones by keyword, e.g. ``texture_set(color="#d0d0d0",
    fibers=False, smoothness=0.7)``; returns ``{"color", "normal",
    "roughness", "height"}``."""
    args = build_parser().parse_args([])
    for key, value in options.items():
        if not hasattr(args, key):
            raise TypeError(f"unknown option {key!r}")
        is_color = key in ("color", "fiber_color") and isinstance(value, str)
        setattr(args, key, parse_color(value) if is_color else value)
    return {name: png_bytes(pixels) for name, pixels in make_maps(args)[0].items()}


def build_parser():
    ap = argparse.ArgumentParser(
        description="Generate an extruded-thermoplastic texture set for beads."
    )
    ap.add_argument(
        "--color",
        type=parse_color,
        default=parse_color("3a3d42"),
        help='base colour, RRGGBB hex (default "#3a3d42")',
    )
    ap.add_argument(
        "--fibers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="chopped-fibre filled look (default on)",
    )
    ap.add_argument(
        "--fiber-density",
        type=float,
        default=14.0,
        help="visible fibres per mm^2 (default 14)",
    )
    ap.add_argument(
        "--fiber-length",
        type=float,
        default=0.4,
        help="typical fibre length in mm (default 0.4)",
    )
    ap.add_argument(
        "--fiber-color",
        type=parse_color,
        default=None,
        help="fibre colour, RRGGBB hex (default: picked against --color)",
    )
    ap.add_argument(
        "--roughness",
        type=float,
        default=0.6,
        help="0 mirror-like .. 1 fully matte (default 0.6)",
    )
    ap.add_argument(
        "--smoothness",
        type=float,
        default=0.4,
        help="0 heavy ripple .. 1 no relief (default 0.4)",
    )
    ap.add_argument(
        "--streaks",
        type=float,
        default=1.0,
        help="strength of the streaks along the bead (default 1)",
    )
    ap.add_argument(
        "--ripple",
        type=float,
        default=1.0,
        help="strength of the fine bands across the bead; 0 removes them (default 1)",
    )
    ap.add_argument(
        "--wipe",
        type=float,
        default=0.7,
        help="how much the nozzle smooths the bead's flat top: 0 leaves it "
        "like the sides, 1 wipes it flat (default 0.7)",
    )
    ap.add_argument(
        "--bump-strength",
        type=float,
        default=1.0,
        help="extra multiplier on the relief (default 1)",
    )
    ap.add_argument(
        "--length",
        type=float,
        default=25.0,
        help="bead length one tile covers, mm: texture_length (default 25)",
    )
    ap.add_argument(
        "--width", type=float, default=8.0, help="bead width, mm (default 8)"
    )
    ap.add_argument(
        "--height", type=float, default=4.0, help="bead height, mm (default 4)"
    )
    ap.add_argument(
        "--size",
        type=int,
        default=1024,
        help="image width in pixels; the height follows (default 1024)",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "-o",
        "--output",
        default="bead",
        help='file prefix: writes <prefix>_color.png etc. (default "bead")',
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="show the result on two sample beads in the viewer",
    )
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    for name in ("roughness", "smoothness"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            ap.error(f"--{name} must be between 0 and 1")

    maps, height_range, cv = make_maps(args)
    paths = {}
    for name, pixels in maps.items():
        paths[name] = Path(f"{args.output}_{name}.png")
        paths[name].write_bytes(png_bytes(pixels))
        print(f"Wrote {paths[name]}")
    print(
        f"{cv.w} x {cv.h} px covering {args.length:g} x {cv.h * cv.dy:.2f} mm; "
        f"height map black-to-white = {height_range:.4f} mm"
    )
    if args.show:
        show(paths, args)


if __name__ == "__main__":
    main()
