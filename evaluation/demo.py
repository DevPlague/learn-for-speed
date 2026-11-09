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
from sim.env import RacingEnv

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # no window, works headless and in Docker
FPS = 20  # one frame per env step, the env runs at 20 steps per second


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
                    outcomes[d["name"]] = outcome(info)
            row.append(d["frame"])
        frames.append(np.hstack(row))
        if len(outcomes) == len(drivers):
            break
    for d in drivers:
        d["env"].close()
    return frames, outcomes


def outcome(info):
    if info["lap_done"]:
        return f"lap {info['lap_time']:.1f} s"
    if info["off_road"]:
        return "off road"
    return "stuck" if info["stuck"] else "timeout"


def save_video(frames, out):
    import imageio.v2 as imageio  # lazy so the rest of evaluation never needs ffmpeg

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=FPS, macro_block_size=8)
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
