"""Demo videos. Several policies drive the same track side by side, lidar rays included.

Examples
    make run CMD="python -m evaluation.demo --policies ppo pure_pursuit --model runs/mvp_s0/best_model.zip --seed 1000"
    make run CMD="python -m evaluation.demo --policies ppo --model runs/mvp_s1/best_model.zip --seed 1003 --out videos/failure.mp4"
"""

import argparse
import os
from pathlib import Path

import numpy as np

from evaluation.evaluate import check_seeds, make_policy
from sim.env import RacingEnv, outcome

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # no window, works headless and in Docker
_FONT = None


def label(frame, text):
    """Frame with `text` in a dark box at the bottom centre, so each view names its policy."""
    import pygame  # lazy like the renderer, the font needs pygame initialised

    global _FONT
    if _FONT is None or not pygame.font.get_init():  # env.close() may quit pygame
        pygame.font.init()
        _FONT = pygame.font.SysFont("monospace", 22, bold=True)
    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    img = _FONT.render(text, True, (255, 255, 255))
    x = (surf.get_width() - img.get_width()) // 2
    y = surf.get_height() - img.get_height() - 12  # bottom, clear of the renderer HUD on top
    pygame.draw.rect(surf, (0, 0, 0), (x - 8, y - 3, img.get_width() + 16, img.get_height() + 6))
    surf.blit(img, (x, y))
    return pygame.surfarray.array3d(surf).swapaxes(0, 1)


def record(policies, seed, difficulty, max_steps=2400):
    """Frames of all policies driving track `seed` at once, joined left to right.

    `policies` maps a name to a factory that takes the env and returns the policy, so pure
    pursuit reads its own env. A finished car keeps its last frame until all are done.
    """
    check_seeds([seed])
    drivers = []
    for name, factory in policies.items():
        env = RacingEnv(track_seed=seed, difficulty=difficulty, render_mode="rgb_array")
        obs, _ = env.reset(seed=seed)
        drivers.append({"name": name, "env": env, "policy": factory(env), "obs": obs})
    frames, outcomes = [], {}
    for _ in range(max_steps):
        row = []
        for d in drivers:
            if d["name"] not in outcomes:
                d["obs"], _, terminated, truncated, info = d["env"].step(d["policy"](d["obs"]))
                d["frame"] = d["env"].render()
                if terminated or truncated:
                    outcomes[d["name"]] = describe(info)
                d["frame"] = label(
                    d["frame"], f"{d['name']}  {outcomes.get(d['name'], '')}".strip()
                )
            row.append(d["frame"])
        frames.append(np.hstack(row))
        if len(outcomes) == len(drivers):
            break
    for d in drivers:
        d["env"].close()
    return frames, outcomes


def describe(info):
    end = outcome(info)
    return f"lap {info['lap_time']:.1f} s" if end == "lap" else end.replace("_", " ")


def save_video(frames, out):
    import imageio.v2 as imageio  # lazy so the rest of evaluation never needs ffmpeg

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=RacingEnv.metadata["render_fps"], macro_block_size=8)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--policies", nargs="+", choices=["random", "pure_pursuit", "ppo"], required=True
    )
    p.add_argument("--model", default=None, help="PPO zip, needed when ppo is listed")
    p.add_argument("--seed", type=int, default=1000, help="validation or test track")
    p.add_argument("--difficulty", type=float, default=0.0)
    p.add_argument("--out", default=None, help="default videos/<policies>_<seed>.mp4")
    args = p.parse_args()
    if "ppo" in args.policies and not args.model:
        p.error("ppo needs --model")

    policies = {n: (lambda env, n=n: make_policy(n, env, args.model)) for n in args.policies}
    frames, outcomes = record(policies, args.seed, args.difficulty)
    out = args.out or f"videos/{'_vs_'.join(args.policies)}_{args.seed}.mp4"
    print(f"saved {save_video(frames, out)}, left to right {outcomes}")


if __name__ == "__main__":
    main()
