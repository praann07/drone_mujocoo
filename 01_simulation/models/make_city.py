"""Deterministic generator for the city's two artifacts.

Reads the single source of truth in city.py and writes, in this directory:

  map_texture.png  1024x1024 printed top-down map (draws the road grid,
                   every building footprint, parks, the home pad, the
                   obstacle tower's warning zone, and a compass) - applied
                   to the scene.xml floor.
  city.xml         worldbody fragment of non-collidable (contype=0
                   conaffinity=0) box/cylinder geoms standing exactly on
                   the footprints drawn in the map, so map and 3D landmarks
                   can never drift apart. <include>d by scene.xml.

Everything is cosmetic/visualization-only - zero mass, no joints, no
actuators - so loading scene.xml instead of quad.xml remains physics-identical
(re-verified at the end of any change by the full test suite plus a
bit-for-bit position comparison).

Usage:  python make_city.py      (run from anywhere; paths are __file__-relative)
Output: 01_simulation/models/map_texture.png
        01_simulation/models/city.xml
"""
from __future__ import annotations

import pathlib

from PIL import Image, ImageDraw, ImageFont

from city import CITY_OBJECTS, OBSTACLE_FACE_X, OBSTACLE_RING_R, OBSTACLE_X

HERE = pathlib.Path(__file__).resolve().parent
MAP_PNG = HERE / "map_texture.png"
CITY_XML = HERE / "city.xml"

SIZE = 1024
HALF_WORLD = 10.0  # scene.xml floor half-size
PX_PER_M = SIZE / (2.0 * HALF_WORLD)  # 51.2 px per meter

# Map style (printed-map look)
PAPER = (235, 232, 220)
ROAD = (250, 250, 248)
ROAD_WORLD_W = 1.4
STROKE = (110, 104, 94)
PARK_GREEN = (120, 172, 116)
TREE_GREEN = (96, 150, 100)
PAD_CYAN = (86, 200, 231)
TOWER_ORANGE = (222, 122, 80)
WARN_RED = (214, 61, 61)


def to_px(x: float, y: float) -> tuple[float, float]:
    """World (x, y) -> pixel position, y up (PIL canvas y is down)."""
    px = (x + HALF_WORLD) * PX_PER_M
    py = (HALF_WORLD - y) * PX_PER_M
    return px, py


def rect(world: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = world
    (px0, py0), (px1, py1) = to_px(x0, y0), to_px(x1, y1)
    return int(min(px0, px1)), int(min(py0, py1)), int(max(px0, px1)), int(max(py0, py1))


def lighten(c: tuple[int, int, int], f: float = 0.35) -> tuple[int, int, int]:
    return tuple(int(ch + (255 - ch) * f) for ch in c)


def draw_map() -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE), PAPER)
    d = ImageDraw.Draw(img)

    # Road grid: two vertical (x=+-4) + two horizontal (y=+-4) corridors,
    # drawn before buildings so blocks overpaint where they meet.
    hw = ROAD_WORLD_W / 2.0
    for rx in (-4.0, 4.0):
        d.rectangle(rect((rx - hw, -HALF_WORLD, rx + hw, HALF_WORLD)), fill=ROAD)
    for ry in (-4.0, 4.0):
        d.rectangle(rect((-HALF_WORLD, ry - hw, HALF_WORLD, ry + hw)), fill=ROAD)

    # Parks + trees + home pad first (flat, beneath buildings' strokes).
    for o in CITY_OBJECTS:
        if o.kind == "park":
            d.rectangle(rect((o.x - o.half_x, o.y - o.half_y,
                              o.x + o.half_x, o.y + o.half_y)), fill=PARK_GREEN)
            d.rectangle(rect((o.x - o.half_x, o.y - o.half_y,
                              o.x + o.half_x, o.y + o.half_y)), outline=STROKE, width=2)
        elif o.kind == "tree":
            cx, cy = to_px(o.x, o.y)
            r = int(o.half_x * PX_PER_M)
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=TREE_GREEN)
        elif o.kind == "pad":
            cx, cy = to_px(o.x, o.y)
            r = int(o.half_x * PX_PER_M)
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=PAD_CYAN, outline=STROKE, width=2)

    # Every standing building footprint (hue stays matched to the 3D geoms).
    for o in CITY_OBJECTS:
        if o.kind not in ("building", "tower"):
            continue
        c = tuple(min(255, int(ch * 255)) for ch in o.rgba[:3])
        if o.kind == "tower":
            c = TOWER_ORANGE
        else:
            c = lighten(c)
        d.rectangle(rect((o.x - o.half_x, o.y - o.half_y,
                          o.x + o.half_x, o.y + o.half_y)), fill=c, outline=STROKE, width=2)

    # Obstacle tower's warning zone ring (dashed) so the "stay clear of the
    # forward building" story is legible on the map itself.
    cx, cy = to_px(OBSTACLE_X, 0.0)
    r = int(OBSTACLE_RING_R * PX_PER_M)
    for start in range(0, 360, 18):
        d.arc((cx - r, cy - r, cx + r, cy + r), start, start + 12, fill=WARN_RED, width=3)

    # Labels: "H" on the home pad, "N" compass at the bottom-left corner.
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None
    px, py = to_px(0.0, 0.0)
    bbox = font.getbbox("H") if font else (0, 0, 8, 11)
    d.text((px - bbox[2] / 2, py - bbox[3] / 2), "H", fill=(0, 62, 84), font=font)
    nx, ny = to_px(-8.9, -8.95)
    d.arc((nx - 10, ny - 10, nx + 10, ny + 10), 225, 45, fill=(70, 66, 60), width=3)
    d.line((nx, ny - 14, nx, ny + 14), fill=(70, 66, 60), width=3)
    d.line((nx, ny - 14, nx - 5, ny - 7), fill=(70, 66, 60), width=3)
    d.line((nx, ny - 14, nx + 5, ny - 7), fill=(70, 66, 60), width=3)
    if font:
        d.text((nx - 4, ny + 14), "N", fill=(70, 66, 60), font=font)

    return img


def city_xml() -> str:
    lines = ["<worldbody>",
             '  <!-- Generated by make_city.py from city.py - do not edit by hand. -->',
             '  <!-- Visualization-only: contype/conaffinity=0, zero mass, no joints or',
             '       actuators; cannot affect physics or identification data. -->']
    for o in CITY_OBJECTS:
        name = o.name
        r, g, b, a = o.rgba
        rgba = f"{r:.2f} {g:.2f} {b:.2f} {a:.2f}"
        if o.kind in ("building", "tower"):
            sz = f"{o.half_x:.3f} {o.half_y:.3f} {o.height / 2:.3f}"
            lines.append(f'  <geom name="{name}" type="box" pos="{o.x:.3f} {o.y:.3f} {o.height / 2:.3f}" '
                         f'size="{sz}" rgba="{rgba}" contype="0" conaffinity="0"/>')
        elif o.kind == "tree":
            r_m = (o.half_x + o.half_y) / 2.0
            lines.append(f'  <geom name="{name}" type="cylinder" pos="{o.x:.3f} {o.y:.3f} {o.height / 2:.3f}" '
                         f'size="{r_m:.3f} {o.height / 2:.3f}" rgba="{rgba}" contype="0" conaffinity="0"/>')
        elif o.kind == "park":
            lines.append(f'  <geom name="{name}" type="box" pos="{o.x:.3f} {o.y:.3f} {o.height / 2:.3f}" '
                         f'size="{o.half_x:.3f} {o.half_y:.3f} {o.height / 2:.3f}" rgba="{rgba}" '
                         f'contype="0" conaffinity="0"/>')
        elif o.kind == "pad":
            r_m = (o.half_x + o.half_y) / 2.0
            lines.append(f'  <geom name="{name}" type="cylinder" pos="{o.x:.3f} {o.y:.3f} {o.height / 2:.3f}" '
                         f'size="{r_m:.3f} {o.height / 2:.3f}" rgba="{rgba}" contype="0" conaffinity="0"/>')
    lines.append("</worldbody>")
    return "\n".join(lines) + "\n"


def main() -> None:
    img = draw_map()
    img.save(MAP_PNG)
    CITY_XML.write_text(city_xml(), encoding="utf-8")
    n_geo = len(CITY_OBJECTS)
    print(f"wrote {MAP_PNG.name} ({img.size[0]}x{img.size[1]}) and {CITY_XML.name} "
          f"({n_geo} geoms)")
    obs = next(o for o in CITY_OBJECTS if o.kind == "tower")
    print(f"obstacle tower face at x = {OBSTACLE_FACE_X:.2f} m "
          f"(center {obs.x:.2f}, half {obs.half_x:.2f}); forward waypoint +1.5 m leaves "
          f"{OBSTACLE_FACE_X - 1.5:.2f} m clearance")


if __name__ == "__main__":
    main()