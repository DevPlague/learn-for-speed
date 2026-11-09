"""Report figures. Learning curves from val_curve.csv, results bars and failure trajectories.

Examples
    make run CMD="python -m evaluation.plots curve --runs runs/mvp_s0 runs/mvp_s2 --label mvp"
    make run CMD="python -m evaluation.plots bars --evals pp=runs/eval_pp/eval.json"
    make run CMD="python -m evaluation.plots failures --policy ppo --model runs/mvp_s1/model.zip"
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, works in Docker
import matplotlib.pyplot as plt
import numpy as np

from evaluation.evaluate import check_seeds, make_policy, run_episode, split_seeds
from sim.env import RacingEnv

CURVE_KEYS = ("lap_success_rate", "mean_progress")


def load_curve(run_dir):
    """Columns of runs/<run>/val_curve.csv as float arrays. Empty cells become NaN."""
    with open(Path(run_dir) / "val_curve.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) if r[k] else np.nan for r in rows]) for k in rows[0]}


def curve_stats(curves, key):
    """Mean and std over seeds, cut to the checks every seed reached."""
    n = min(len(c[key]) for c in curves)
    values = np.array([c[key][:n] for c in curves])
    return curves[0]["timesteps"][:n], values.mean(axis=0), values.std(axis=0)


def plot_learning_curves(groups, out):
    """One line per config with a mean +- std band over its seeds."""
    fig, axes = plt.subplots(1, len(CURVE_KEYS), figsize=(10, 3.5))
    for label, run_dirs in groups.items():
        curves = [load_curve(d) for d in run_dirs]
        for ax, key in zip(axes, CURVE_KEYS, strict=True):
            steps, mean, std = curve_stats(curves, key)
            seeds = "seed" if len(curves) == 1 else "seeds"
            ax.plot(steps, mean, label=f"{label} ({len(curves)} {seeds})")
            ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
    for ax, key in zip(axes, CURVE_KEYS, strict=True):
        ax.set(xlabel="training steps", title=key.replace("_", " "), ylim=(-0.05, 1.05))
        ax.grid(alpha=0.3)
    axes[0].legend()
    return save(fig, out)


def plot_results(evals, out):
    """Lap, off-track and stuck rates per policy from eval.json files, std bars for PPO seeds."""
    keys = ("lap_success_rate", "off_track_rate", "stuck_rate")
    names = list(evals)
    x = np.arange(len(names))
    width = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 3, 3.5))
    for i, key in enumerate(keys):
        metrics = [json.loads(Path(evals[n]).read_text())["metrics"] for n in names]
        values = [m[key] for m in metrics]
        errors = [m.get(key + "_std") or 0.0 for m in metrics]
        ax.bar(x + i * width, values, width, yerr=errors, capsize=3, label=key.replace("_", " "))
    ax.set_xticks(x + width * (len(keys) - 1) / 2, names)
    ax.set(ylabel="share of tracks", ylim=(0, 1.05))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    return save(fig, out)


def plot_trajectory(episode, track, out):
    """Track, car path colored by speed and a cross where the episode ended."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for edge, style in (("left", "k-"), ("right", "k-"), ("left_wall", "r:"), ("right_wall", "r:")):
        pts = getattr(track, edge)
        ax.plot(*np.vstack([pts, pts[:1]]).T, style, lw=0.8)
    path = episode["path"]
    sc = ax.scatter(path[:, 0], path[:, 1], c=episode["speeds"], s=2, cmap="viridis")
    ax.plot(*track.centerline[0], "g^", ms=8, label="start")
    ax.plot(*path[-1], "rx", ms=12, mew=3, label=episode["outcome"])
    fig.colorbar(sc, ax=ax, label="speed m/s", shrink=0.7)
    progress = episode["info"]["progress"]
    ax.set(title=f"seed {episode['seed']}, {episode['outcome']}, progress {progress:.0%}")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    return save(fig, out)


def find_failures(policy, seeds, difficulty, count, env):
    """First `count` episodes that did not finish a lap, with the track they ran on."""
    check_seeds(seeds)
    env.difficulty = difficulty
    failures = []
    for seed in seeds:
        ep = run_episode(policy, seed, env)
        if ep["outcome"] != "lap":
            failures.append((ep, env.track))
            if len(failures) == count:
                break
    return failures


def save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curve", help="learning curve from val_curve.csv")
    c.add_argument("--runs", nargs="+", required=True, help="run folders, one per seed")
    c.add_argument("--label", default="ppo")
    c.add_argument("--out", default="runs/figures/curve.png")
    b = sub.add_parser("bars", help="results bar chart from eval.json files")
    b.add_argument("--evals", nargs="+", required=True, help="name=path/to/eval.json")
    b.add_argument("--out", default="runs/figures/results.png")
    f = sub.add_parser("failures", help="trajectory plots of episodes that did not lap")
    f.add_argument("--policy", choices=["random", "pure_pursuit", "ppo"], required=True)
    f.add_argument("--model", default=None)
    f.add_argument("--split", choices=["val", "test"], default="val")
    f.add_argument("--n", type=int, default=50, help="tracks to search")
    f.add_argument("--count", type=int, default=3)
    f.add_argument("--difficulty", type=float, default=0.5)
    f.add_argument("--out", default="runs/figures/failures")
    args = p.parse_args()

    if args.cmd == "curve":
        print(f"saved {plot_learning_curves({args.label: args.runs}, args.out)}")
        return
    if args.cmd == "bars":
        evals = dict(item.split("=", 1) for item in args.evals)
        print(f"saved {plot_results(evals, args.out)}")
        return
    env = RacingEnv(difficulty=args.difficulty)
    policy = make_policy(args.policy, env, args.model)
    seeds = split_seeds(args.split, args.n)
    failures = find_failures(policy, seeds, args.difficulty, args.count, env)
    for ep, track in failures:
        out = plot_trajectory(ep, track, Path(args.out) / f"{args.policy}_{ep['seed']}.png")
        info = ep["info"]
        print(f"seed {ep['seed']} {ep['outcome']} progress {info['progress']:.0%} saved {out}")
    print(f"{len(failures)} failures in {len(seeds)} tracks")


if __name__ == "__main__":
    main()
