"""Watch a run drive several tracks at once, checkpoint after checkpoint. Run with
python -m learning.watch --run runs/mvp_s0 --tracks 6

Each phase is one checkpoint of the run, in training order, then best_model.zip. In a phase
every panel drives one episode on its own validation track. The road is colored by the speed
the car may still brake from (allowed_speed in learning/reward.py), the trail is green below
that speed and red above it, and a cross marks where an episode failed.
Keys are space to pause, right and left for the next and previous phase, up and down for
the playback speed and esc to quit.
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pygame
from stable_baselines3 import PPO

from evaluation.evaluate import split_seeds
from learning.reward import allowed_speed
from sim import car
from sim.env import RacingEnv, outcome

WINDOW = (1500, 860)
HEADER = 70
COLS = 3
WHITE, GREY, ROAD, GRASS = (235, 235, 235), (150, 150, 160), (70, 70, 75), (40, 90, 45)
GREEN, RED, YELLOW = (60, 220, 90), (235, 60, 60), (255, 220, 0)


def phases(run):
    """(label steps, model path) in training order, best_model last."""
    ckpts = sorted(
        (int(re.search(r"_(\d+)_steps", p.name).group(1)), p)
        for p in (run / "checkpoints").glob("rl_model_*_steps.zip")
    )
    return [*ckpts, (None, run / "best_model.zip")]


def level_at(run, steps):
    """Training difficulty at a number of steps, from val_curve.csv."""
    path = run / "val_curve.csv"
    if not path.exists():
        return 0.0
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return 0.0
    if steps is None:
        return float(rows[-1]["difficulty"])
    done = [r for r in rows if int(r["timesteps"]) <= steps]
    return float((done or rows[:1])[-1]["difficulty"])


def speed_color(v, top):
    """Red for slow allowed speeds, white for the top speed."""
    t = np.clip((v - 8.0) / (top - 8.0), 0, 1)
    return (235, int(60 + 175 * t), int(60 + 175 * t))


class Panel:
    def __init__(self, rect, episode, seed, difficulty):
        self.rect = pygame.Rect(rect)
        self.episode = episode
        self.seed = seed
        self.env = RacingEnv(seed, difficulty)
        self.obs, _ = self.env.reset()
        t = self.env.track
        self.allowed = allowed_speed(t)
        walls = np.vstack([t.left_wall, t.right_wall])
        lo, hi = walls.min(axis=0), walls.max(axis=0)
        inner = self.rect.inflate(-20, -50)
        self.scale = min(inner.w / (hi - lo)[0], inner.h / (hi - lo)[1])
        self.origin = np.array(inner.center) - (lo + hi) / 2 * [self.scale, -self.scale]
        self.trail = []
        self.done = None
        self.background = self._draw_track()

    def to_px(self, p):
        p = np.asarray(p)
        return self.origin + p * [self.scale, -self.scale] - self.rect.topleft

    def _draw_track(self):
        t = self.env.track
        surf = pygame.Surface(self.rect.size)
        surf.fill(GRASS)
        n = len(t.centerline)
        for i in range(n):
            j = (i + 1) % n
            quad = [self.to_px(p) for p in (t.left[i], t.left[j], t.right[j], t.right[i])]
            pygame.draw.polygon(surf, ROAD, quad)
        pygame.draw.lines(surf, GREY, True, [self.to_px(p) for p in t.left_wall], 1)
        pygame.draw.lines(surf, GREY, True, [self.to_px(p) for p in t.right_wall], 1)
        for i in range(n):  # centerline colored by the speed the car may still brake from
            j = (i + 1) % n
            color = speed_color(self.allowed[i], car.MAX_SPEED)
            pygame.draw.line(
                surf, color, self.to_px(t.centerline[i]), self.to_px(t.centerline[j]), 2
            )
        a, b = self.to_px(t.left[0]), self.to_px(t.right[0])
        pygame.draw.line(surf, WHITE, a, b, 3)  # start line
        return surf

    def step(self, action):
        self.obs, _, term, trunc, info = self.env.step(action)
        x, y, *_ = self.env.state
        over = info["speed"] > self.allowed[self.env.idx]
        self.trail.append((self.to_px((x, y)), over))
        if term or trunc:
            self.done = outcome(info)

    def draw(self, screen, font):
        surf = self.background.copy()
        for k in range(1, len(self.trail)):
            pygame.draw.line(
                surf, RED if self.trail[k][1] else GREEN, self.trail[k - 1][0], self.trail[k][0], 2
            )
        x, y, yaw, v, _, _ = self.env.state
        c, s = np.cos(yaw), np.sin(yaw)
        size = max(3.0, 2.0 / self.scale * 3)  # at least a few pixels long
        body = [(size * 1.5, 0), (-size, -size * 0.8), (-size, size * 0.8)]
        pygame.draw.polygon(
            surf,
            YELLOW,
            [self.to_px((x + bx * c - by * s, y + bx * s + by * c)) for bx, by in body],
        )
        if self.done and self.done != "lap":
            px, py = self.to_px((x, y))
            for d in (-7, 7):
                pygame.draw.line(surf, RED, (px - 7, py + d), (px + 7, py - d), 3)
        env = self.env
        status = self.done or "driving"
        color = GREEN if status == "lap" else RED if self.done else WHITE
        lines = [
            (f"episode {self.episode}  track {self.seed}  step {env.steps}", WHITE),
            (f"{3.6 * v:5.1f} km/h  lap {100 * env.progress:5.1f}%  {status}", color),
        ]
        for k, (text, col) in enumerate(lines):
            surf.blit(font.render(text, True, col), (8, 6 + 18 * k))
        screen.blit(surf, self.rect.topleft)
        pygame.draw.rect(screen, (20, 20, 20), self.rect, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="runs/<config>_s<training seed>")
    parser.add_argument("--tracks", type=int, default=6, help="validation track seeds from 1000")
    parser.add_argument(
        "--difficulty", type=float, help="fixed difficulty, default the training level"
    )
    parser.add_argument("--phase", type=int, default=0, help="phase to start from")
    args = parser.parse_args()

    run = Path(args.run)
    if not (run / "best_model.zip").exists():
        parser.error(f"{run} has no best_model.zip, train it first")
    plan = phases(run)
    seeds = split_seeds("val", args.tracks)
    rows = -(-len(seeds) // COLS)
    w, h = WINDOW[0] // COLS, (WINDOW[1] - HEADER) // rows

    pygame.init()
    screen = pygame.display.set_mode(WINDOW)
    pygame.display.set_caption(f"watch {run.name}")
    font = pygame.font.SysFont("monospace", 15)
    big = pygame.font.SysFont("monospace", 20, bold=True)
    clock = pygame.time.Clock()
    k, speed, paused = args.phase, 2, False

    while 0 <= k < len(plan):
        steps, path = plan[k]
        model = PPO.load(path, device="cpu")
        level = args.difficulty if args.difficulty is not None else level_at(run, steps)
        panels = [
            Panel((i % COLS * w, HEADER + i // COLS * h, w, h), i + 1, s, level)
            for i, s in enumerate(seeds)
        ]
        jump = None
        while jump is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_RIGHT:
                        jump = 1
                    elif event.key == pygame.K_LEFT:
                        jump = -1
                    elif event.key == pygame.K_UP:
                        speed = min(speed * 2, 64)
                    elif event.key == pygame.K_DOWN:
                        speed = max(speed // 2, 1)
            live = [p for p in panels if p.done is None]
            if not paused and live:
                for _ in range(speed):
                    live = [p for p in live if p.done is None]
                    if not live:
                        break
                    actions = model.predict(np.array([p.obs for p in live]), deterministic=True)[0]
                    for p, a in zip(live, actions, strict=True):
                        p.step(a)
            screen.fill((25, 25, 28))
            laps = sum(p.done == "lap" for p in panels)
            name = f"{steps:,} steps".replace(",", ".") if steps else "best_model"
            head = f"phase {k + 1}/{len(plan)}  {name}  difficulty {level:.2f}   laps {laps} of {len(panels)}"
            screen.blit(big.render(head, True, WHITE), (12, 10))
            keys = f"x{speed} playback   space pause   left/right phase   up/down speed   esc quit"
            legend = "road line: white fast, red brake zone   trail: green within, red over the allowed speed"
            screen.blit(font.render(keys + ("   PAUSED" if paused else ""), True, GREY), (12, 36))
            screen.blit(font.render(legend, True, GREY), (12, 52))
            for p in panels:
                p.draw(screen, font)
            pygame.display.flip()
            clock.tick(round(1 / car.DT))
            if not live and not paused:
                pygame.time.wait(1500)
                jump = 1
        k = max(k + jump, 0)
    pygame.quit()


if __name__ == "__main__":
    main()
