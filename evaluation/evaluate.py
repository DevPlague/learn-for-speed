"""Run any policy on held-out tracks and report the metrics from docs/SPEC.md.

Example
    make run CMD="python -m evaluation.evaluate --policy pure_pursuit --split val --n 20"
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.baselines import PurePursuitPolicy, RandomPolicy
from sim.env import RacingEnv

VAL_SEEDS = range(1000, 2000)
TEST_SEEDS = range(10_000, 100_000)
METRICS = (
    "lap_success_rate",
    "off_track_rate",
    "stuck_rate",
    "mean_lap_time",
    "mean_speed",
    "mean_progress",
)


def split_seeds(split, n):
    """First n seeds of the validation or test split, so every policy sees the same tracks."""
    seeds = {"val": VAL_SEEDS, "test": TEST_SEEDS}[split]
    return list(seeds[:n])


def check_seeds(seeds):
    """Only validation and test seeds are allowed, never training seeds."""
    bad = [s for s in seeds if s not in VAL_SEEDS and s not in TEST_SEEDS]
    if bad:
        raise ValueError(f"seeds outside validation and test splits {bad[:5]}")


def evaluate(policy, seeds, difficulty, env=None):
    """One episode per track seed. Pass `env` when the policy reads from it, like pure pursuit."""
    check_seeds(seeds)
    env = RacingEnv(difficulty=difficulty) if env is None else env
    env.difficulty = difficulty
    laps, offs, stucks, lap_times, speeds, progress = [], [], [], [], [], []
    for seed in seeds:
        env.track_seed = seed
        obs, info = env.reset(seed=seed)
        ep_speeds = []
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            ep_speeds.append(abs(info["speed"]))
            done = terminated or truncated
        laps.append(info["lap_done"])
        offs.append(info["off_road"])
        stucks.append(info["stuck"])
        if info["lap_done"]:
            lap_times.append(info["lap_time"])
        speeds.append(np.mean(ep_speeds))
        progress.append(info["progress"])
    return {
        "n_episodes": len(seeds),
        "lap_success_rate": float(np.mean(laps)),
        "off_track_rate": float(np.mean(offs)),
        "stuck_rate": float(np.mean(stucks)),
        "mean_lap_time": float(np.mean(lap_times)) if lap_times else None,
        "mean_speed": float(np.mean(speeds)),
        "mean_progress": float(np.mean(progress)),
    }


def aggregate(results):
    """Mean and std of each metric over several training seeds of the same agent."""
    out = {"n_episodes": sum(r["n_episodes"] for r in results), "n_models": len(results)}
    for key in METRICS:
        values = [r[key] for r in results if r[key] is not None]
        out[key] = float(np.mean(values)) if values else None
        out[key + "_std"] = float(np.std(values)) if values else None
    return out


def to_markdown(rows):
    """Table with one row per policy. Shows mean +- std when the row has std values."""

    def cell(row, key):
        if row[key] is None:
            return "-"
        std = row.get(key + "_std")
        return f"{row[key]:.2f} ± {std:.2f}" if std is not None else f"{row[key]:.2f}"

    lines = ["| policy | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for name, row in rows.items():
        lines.append(f"| {name} | " + " | ".join(cell(row, k) for k in METRICS) + " |")
    return "\n".join(lines)


def ppo_policy(model_path):
    """Deterministic PPO actions. VecNormalize only scales the reward, so obs need no stats."""
    from stable_baselines3 import PPO  # lazy so baselines run without loading torch

    model = PPO.load(model_path, device="cpu")
    return lambda obs: model.predict(obs, deterministic=True)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=["random", "pure_pursuit", "ppo"], required=True)
    p.add_argument("--model", nargs="+", default=[], help="one zip per training seed, ppo only")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--difficulty", type=float, default=0.5)
    p.add_argument("--out", default=None, help="output folder, default runs/eval_<policy>_<split>")
    args = p.parse_args()

    seeds = split_seeds(args.split, args.n)
    if args.policy == "ppo":
        if not args.model:
            p.error("--policy ppo needs --model")
        row = aggregate([evaluate(ppo_policy(m), seeds, args.difficulty) for m in args.model])
    elif args.policy == "pure_pursuit":
        env = RacingEnv(difficulty=args.difficulty)
        row = evaluate(PurePursuitPolicy(env), seeds, args.difficulty, env=env)
    else:
        row = evaluate(RandomPolicy(seed=0), seeds, args.difficulty)

    out = Path(args.out or f"runs/eval_{args.policy}_{args.split}")
    out.mkdir(parents=True, exist_ok=True)
    summary = {"args": vars(args), "seeds": seeds, "metrics": row}
    (out / "eval.json").write_text(json.dumps(summary, indent=2))
    print(to_markdown({args.policy: row}))
    print(f"\nsaved {out / 'eval.json'}")


if __name__ == "__main__":
    main()
