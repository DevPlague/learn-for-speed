"""Draws the README banner GIF and the pixel icons of its headings.

The banner is a Pillow port of the arcade replay in dashboard/static/app.js, same palette,
curbs, sprite and lidar, driven by a trained PPO model on validation tracks.

Run it as a module from the repo root so sim, evaluation and dashboard import, and leave out
--model to redraw only the icons.

make run CMD="python -m docs.media.make_media --model runs/final_s0/best_model.zip"
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from dashboard.record import record_episode, track_geometry
from evaluation.evaluate import make_policy
from sim import car, lidar
from sim.env import RacingEnv

OUT = Path(__file__).parent
W, H, PX = 960, 192, 3  # banner size and the size of one art pixel
SCALE = 7.5  # screen pixels per meter
CLIP_S = 6.0  # seconds of driving in the banner
THEME = {
    "grass": "#9fc27a",
    "grass2": "#93b76e",
    "tuft": "#6c9150",
    "road": "#4a4548",
    "edge": "#f7f0e1",
    "wall": "#2d2a2e",
    "curb": "#d9483f",
    "curb2": "#f7f0e1",
    "center": "#8f8a85",
}
ACCENT, YELLOW, INK, CREAM = "#d9483f", "#ffd23f", "#1b191c", "#f7f0e1"

CAR_ROWS = [
    "..tttttt.......ttttt..",
    "..trrrrt.......trrrt..",
    "ww..bbbbbbbbbbbbb.....",
    "wwddbbblllllbbbbbbbw..",
    "wwdbbbgghhbbwwwwbbbbww",
    "wwdbbbgghhbbwwwwbbbbww",
    "wwddddbbbbbbbbbbbbbw..",
    "ww..ddddbbbbbbbbb.....",
    "..trrrrt.......trrrt..",
    "..tttttt.......ttttt..",
]
REAR_AXLE_PX = 6


def rgb(color):
    return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))


def mix(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(rgb(a), rgb(b)))


def outlined(rows):
    """Adds a one pixel outline `k` around the filled cells, like outlined() in app.js."""
    w = max(map(len, rows)) + 2
    grid = ["." * w] + ["." + r.ljust(w - 2, ".") + "." for r in rows] + ["." * w]
    h = len(grid)

    def filled(i, j):
        return 0 <= j < h and 0 <= i < w and grid[j][i] not in ".k"

    near = ((-1, 0), (1, 0), (0, -1), (0, 1))
    return [
        "".join(
            "k" if c == "." and any(filled(i + a, j + b) for a, b in near) else c
            for i, c in enumerate(row)
        )
        for j, row in enumerate(grid)
    ]


def sprite(rows, colors, scale=1):
    img = Image.new("RGBA", (max(map(len, rows)), len(rows)))
    for j, row in enumerate(rows):
        for i, c in enumerate(row):
            if c in colors:
                img.putpixel((i, j), colors[c] + (255,))
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


# ---------- track ----------


def curbs(center):
    """Centerline points in corners tighter than 28 m get red and white curbs."""
    a, b = np.roll(center, 1, axis=0), np.roll(center, -1, axis=0)
    h1 = np.arctan2(center[:, 1] - a[:, 1], center[:, 0] - a[:, 0])
    h2 = np.arctan2(b[:, 1] - center[:, 1], b[:, 0] - center[:, 0])
    turn = np.abs(np.arctan2(np.sin(h2 - h1), np.cos(h2 - h1)))
    return np.hypot(*(b - center).T) / np.maximum(turn, 1e-6) < 28


def tufts(geo):
    """Same Park Miller generator and seed as decor() in app.js."""
    walls = np.concatenate([geo["left_wall"], geo["right_wall"]])
    (x0, y0), (x1, y1) = walls.min(axis=0), walls.max(axis=0)
    seed = round(geo["length"] * 1000) % 2147483647 or 7
    out = []
    for _ in range(140):
        vals = []
        for _ in range(3):
            seed = seed * 16807 % 2147483647
            vals.append(seed / 2147483647)
        out.append((x0 + vals[0] * (x1 - x0), y0 + vals[1] * (y1 - y0), vals[2] * 6.28))
    return out


def dashed(draw, pts, dash, fill, width):
    """Draws an open polyline as dashes of `dash` pixels with gaps of the same length."""
    seg = np.diff(pts, axis=0)
    run = np.concatenate([[0], np.cumsum(np.hypot(*seg.T))])
    for i in range(len(pts) - 1):
        if run[i] % (2 * dash) < dash:
            draw.line([tuple(pts[i]), tuple(pts[i + 1])], fill=fill, width=width)


def draw_track(geo, deco, cx, cy, ms):
    """Track on the low resolution canvas, camera centered on (cx, cy), `ms` the time in ms."""
    lw, lh, s = W // PX, H // PX, SCALE / PX

    def to_low(p):
        p = np.asarray(p, dtype=float)
        x = W / 2 + (p[..., 0] - cx) * SCALE
        y = H / 2 - (p[..., 1] - cy) * SCALE
        return np.round(np.stack([x, y], axis=-1) / PX)

    def closed(name):
        q = to_low(geo[name])
        return np.vstack([q, q[:1]])

    img = Image.new("RGB", (lw, lh), THEME["grass"])
    d = ImageDraw.Draw(img)
    tile = max(2, round(6 * s))
    ax, ay = to_low((0, 0))
    sx = (ax % (2 * tile)) - 2 * tile
    sy = (ay % tile) - tile
    y = sy
    while y < lh:
        odd = round((y - ay) / tile) & 1
        x = sx + (tile if odd else 0)
        while x < lw:
            d.rectangle([x, y, x + tile - 1, y + tile - 1], fill=THEME["grass2"])
            x += 2 * tile
        y += tile
    for tx, ty, ph in deco["tufts"]:
        u, v = to_low((tx, ty))
        if -3 <= u <= lw + 3 and -6 <= v <= lh + 2:
            sway = round(math.sin(ms / 420 + ph))
            d.rectangle([u + sway, v - 1, u + sway, v], fill=THEME["tuft"])
            d.point((u + 2 + sway, v), fill=THEME["tuft"])

    # tire walls with a dashed red band
    for name in ("left_wall", "right_wall"):
        d.line([tuple(p) for p in closed(name)], fill=THEME["wall"], width=4, joint="curve")
        dashed(d, closed(name), max(2, 2 * s), THEME["curb"], 2)

    # asphalt between the two edges, even odd like the canvas fill
    masks = []
    for name in ("left", "right"):
        m = Image.new("1", (lw, lh))
        ImageDraw.Draw(m).polygon([tuple(p) for p in to_low(geo[name])], fill=1)
        masks.append(m)
    img.paste(THEME["road"], mask=ImageChops.logical_xor(*masks))

    # edges, then two color curbs every other point on tight corners
    for name in ("left", "right"):
        edge = closed(name)
        d.line([tuple(p) for p in edge], fill=THEME["edge"], width=max(1, round(s * 0.8)))
        for i in range(0, len(edge) - 1, 2):
            if deco["curb"][i]:
                color = THEME["curb"] if (i >> 1) % 2 else THEME["curb2"]
                d.line(
                    [tuple(edge[i]), tuple(edge[i + 1])],
                    fill=color,
                    width=max(2, math.floor(s * 1.4 + 0.5)),
                )
    dashed(d, closed("centerline"), max(1, 3 * s), THEME["center"], 1)

    # checkered start line
    a, b = np.array(geo["right"][0]), np.array(geo["left"][0])
    c0, c1 = np.array(geo["centerline"][0]), np.array(geo["centerline"][1])
    hd = math.atan2(*(c1 - c0)[::-1])
    fwd = np.array([math.cos(hd), math.sin(hd)])
    for k in range(6):
        for r in range(2):
            p0 = a + (b - a) * k / 6 + fwd * r * 1.2
            p1 = a + (b - a) * (k + 1) / 6 + fwd * r * 1.2
            color = CREAM if (k + r) % 2 else THEME["wall"]
            d.line([tuple(to_low(p0)), tuple(to_low(p1))], fill=color, width=max(1, round(s * 1.2)))
    return img.resize((W, H), Image.NEAREST)


# ---------- car and lidar ----------


def car_sprite(on_grass):
    body = YELLOW
    colors = {
        "k": rgb(ACCENT if on_grass else INK),
        "b": rgb(body),
        "d": mix(body, "#000000", 0.28),
        "l": mix(body, "#ffffff", 0.45),
        "w": rgb(CREAM),
        "t": rgb(INK),
        "r": rgb("#77727a"),
        "g": rgb("#2b3550"),
        "h": rgb(ACCENT),
    }
    art = max(1, math.floor(4.4 * max(SCALE, 2.6) / 22 + 0.5))
    img = sprite(outlined(CAR_ROWS), colors, art)
    # pad so the rear axle sits in the middle, then rotation turns around the axle
    c = img.width + 8
    pad = Image.new("RGBA", (2 * c, 2 * c))
    pad.paste(img, (c - REAR_AXLE_PX * art, c - img.height // 2))
    return pad


def draw_car(frame, x, y, yaw, hits, on_grass):
    px, py = W / 2, H / 2  # follow camera
    over = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(over)
    for ang, dist in zip(lidar.ANGLES, hits):
        ex = px + dist * math.cos(yaw + ang) * SCALE
        ey = py - dist * math.sin(yaw + ang) * SCALE
        d.line([(px, py), (ex, ey)], fill=rgb(ACCENT) + (200,), width=1)
        if dist < lidar.RANGE:
            d.rectangle([ex - 2, ey - 2, ex + 1, ey + 1], fill=rgb(ACCENT) + (255,))
    frame = Image.alpha_composite(frame.convert("RGBA"), over)

    body = car_sprite(on_grass).rotate(math.degrees(yaw), resample=Image.NEAREST)
    alpha = (
        body.getchannel("A").filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(5))
    )
    glow = Image.new("RGBA", body.size, rgb(YELLOW) + (0,))
    glow.putalpha(alpha.point(lambda a: int(a * 0.9)))
    at = (round(px) - body.width // 2, round(py) - body.height // 2)
    frame.alpha_composite(glow, at)
    frame.alpha_composite(body, at)
    return frame.convert("RGB")


# ---------- banner ----------


def best_clip(model, seeds, difficulty):
    """Lapped episode and window with the most sideways turning over CLIP_S seconds."""
    best = None
    n = round(CLIP_S / car.DT)
    for seed in seeds:
        env = RacingEnv(difficulty=difficulty)
        ep = record_episode(make_policy("ppo", env, model), seed, difficulty, env)
        if ep["outcome"] != "lap":
            continue
        yaw = np.unwrap(ep["frames"]["yaw"])
        # turning, weighted by how much the car moves across the wide banner
        turn = np.abs(np.diff(yaw)) * np.abs(np.cos(yaw[1:]))
        score = np.convolve(turn, np.ones(n), mode="valid")
        start = int(score.argmax())
        print(f"track seed {seed}  lap {ep['lap_time']:.1f} s  turning {score[start]:.1f} rad")
        if best is None or score[start] > best[0]:
            best = (score[start], seed, ep, track_geometry(env.track), start)
    return best


def banner(model, seeds, difficulty, step):
    _, seed, ep, geo, start = best_clip(model, seeds, difficulty)
    deco = {"curb": curbs(np.array(geo["centerline"])), "tufts": tufts(geo)}
    f = ep["frames"]
    frames = []
    for i in range(start, start + round(CLIP_S / car.DT), step):
        x, y, yaw = f["x"][i], f["y"][i], f["yaw"][i]
        img = draw_track(geo, deco, x, y, i * car.DT * 1000)
        frames.append(draw_car(img, x, y, yaw, f["hits"][i], f["on_grass"][i]))

    # one shared palette keeps colors steady from frame to frame
    sample = Image.new("RGB", (W, H * 4))
    for k, fr in enumerate(frames[:: max(1, len(frames) // 4)][:4]):
        sample.paste(fr, (0, H * k))
    pal = sample.quantize(colors=64, method=Image.MEDIANCUT)
    frames = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    path = OUT / "banner.gif"
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=round(step * car.DT * 1000),
        loop=0,
        optimize=True,
    )
    print(f"{path}  track seed {seed}  {len(frames)} frames  {path.stat().st_size / 1e6:.2f} MB")


# ---------- icons ----------

ICON_COLORS = {
    "k": rgb(INK),
    "w": rgb(CREAM),
    "a": rgb(THEME["road"]),
    "s": rgb("#77727a"),
    "m": rgb("#a39d97"),
    "r": rgb(ACCENT),
    "R": rgb("#a8322b"),
    "y": rgb(YELLOW),
    "l": mix(YELLOW, "#ffffff", 0.45),
    "d": mix(YELLOW, "#000000", 0.28),
    "g": rgb(THEME["grass"]),
    "e": rgb(CREAM),
    "h": rgb(ACCENT),
    "n": rgb("#2b3550"),
}

ICONS = {
    "flag": [
        ".s..........",
        ".kkkkkkkkkk.",
        ".kwwaawwaak.",
        ".kwwaawwaak.",
        ".kaawwaawwk.",
        ".kaawwaawwk.",
        ".kkkkkkkkkk.",
        ".s..........",
        ".s..........",
        ".s..........",
        ".s..........",
        "sss.........",
    ],
    "pitboard": [
        ".kkkkkkkkkk.",
        ".kwwwwwwwwk.",
        ".kwrrrrrrwk.",
        ".kwwwwwwwwk.",
        ".kwaaaawwwk.",
        ".kwwwwwwwwk.",
        ".kwaaaaawwk.",
        ".kwwwwwwwwk.",
        ".kkkkkkkkkk.",
        ".....kk.....",
        ".....kk.....",
        "....kkkk....",
    ],
    "lidar": [
        ".....RR.....",
        ".....rr.....",
        ".....rr.....",
        ".....rr.....",
        "R....rr....R",
        ".r...rr...r.",
        "..r..rr..r..",
        "...r.rr.r...",
        "Rr..rrrr..rR",
        "..rrkkkkrr..",
        "....kyyk....",
        "....kkkk....",
    ],
    "floppy": [
        "kkkkkkkkkkk.",
        "kaaksssskaak",
        "kaaksksskaak",
        "kaaksksskaak",
        "kaakkkkkkaak",
        "kaaaaaaaaaak",
        "kawwwwwwwwak",
        "kawrrrrrrwak",
        "kawwwwwwwwak",
        "kawaaaawwwak",
        "kawwwwwwwwak",
        "kkkkkkkkkkkk",
    ],
    "wheel": [
        "....kkkk....",
        "..kkrrrrkk..",
        ".kmmkkkkmmk.",
        ".kmk....kmk.",
        "kmk......kmk",
        "kmkkkkkkkkmk",
        "kmssswwsssmk",
        "kmkkkwwkkkmk",
        ".kmkksskkmk.",
        ".kmmkkkkmmk.",
        "..kkmmmmkk..",
        "....kkkk....",
    ],
    "trophy": [
        "..kkkkkkkk..",
        "kkkylyyyykkk",
        "k.kylyyyyk.k",
        "k.kylyyyyk.k",
        ".kkylyyyykk.",
        "..kdyyyydk..",
        "...kddddk...",
        "....kyyk....",
        "....kddk....",
        "...kkkkkk...",
        "..kaaaaaak..",
        "..kkkkkkkk..",
    ],
    "car": [
        "..kkkkkkkk..",
        "..kwwwwwwk..",
        "..kkkyykkk..",
        "kkk.kyyk.kkk",
        "ksk.kyyk.ksk",
        "kkkkyyyykkkk",
        "...kyhhyk...",
        "kkkkyhhykkkk",
        "ksk.kyyk.ksk",
        "kkk.kddk.kkk",
        ".kwwwwwwwwk.",
        ".kkkkkkkkkk.",
    ],
    "track": [
        "..kkkkkkkk..",
        ".kaaaaaaaak.",
        "kaaaaaaaaaak",
        "kaaeeeeeeaak",
        "kaaeggggeaak",
        "kwkeggggeaak",
        "kkweggggeaak",
        "kaaeggggeaak",
        "kaaeeeeeeaak",
        "kaaaaaaaaaak",
        ".kaaaaaaaak.",
        "..kkkkkkkk..",
    ],
}


def icons():
    folder = OUT / "icons"
    folder.mkdir(exist_ok=True)
    for name, rows in ICONS.items():
        assert all(len(r) == 12 for r in rows) and len(rows) == 12, name
        sprite(rows, ICON_COLORS, 4).save(folder / f"{name}.png", optimize=True)
    print(f"{len(ICONS)} icons in {folder}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", help="PPO zip that drives the banner, skip to only draw icons")
    p.add_argument(
        "--seeds", type=int, nargs=2, default=[1000, 1020], help="validation track seed range"
    )
    p.add_argument("--difficulty", type=float, default=1.0)
    p.add_argument("--step", type=int, default=1, help="env steps per GIF frame")
    args = p.parse_args()
    icons()
    if args.model:
        banner(args.model, range(*args.seeds), args.difficulty, args.step)


if __name__ == "__main__":
    main()
