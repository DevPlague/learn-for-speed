"""Builds the local dashboard from runs/ into runs/dashboard/. Open index.html, no server needed.

Example
    uv run python -m dashboard.build --runs mvp_s0 probe_nojitter_s0
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import yaml

from dashboard.record import record_episode, track_geometry
from evaluation.baselines import PurePursuitPolicy, RandomPolicy
from evaluation.evaluate import evaluate, ppo_policy, split_seeds
from sim import car, lidar
from sim.env import RacingEnv

STATIC = Path(__file__).parent / "static"
PROGRESS_COLUMNS = {
    "time/total_timesteps": "timesteps",
    "rollout/ep_rew_mean": "ep_rew_mean",
    "rollout/ep_len_mean": "ep_len_mean",
    "train/entropy_loss": "entropy_loss",
    "train/approx_kl": "approx_kl",
    "train/value_loss": "value_loss",
    "train/explained_variance": "explained_variance",
    "train/clip_fraction": "clip_fraction",
}


def number(text):
    return float(text) if text not in ("", None) else None


def read_csv(path, columns=None):
    """Columns as lists of floats. `columns` maps file names to output names."""
    if not path.exists():
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    columns = columns or {key: key for key in rows[0]}
    return {out: [number(row.get(key)) for row in rows] for key, out in columns.items()}


def read_run(run_dir):
    """Config and learning curves of one training run."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8-sig"))
    return {
        "name": run_dir.name,
        "config": cfg,
        "progress": read_csv(run_dir / "progress.csv", PROGRESS_COLUMNS),
        "val_curve": read_csv(run_dir / "val_curve.csv"),
    }


def model_path(run_dir):
    best = run_dir / "best_model.zip"
    return best if best.exists() else run_dir / "model.zip"


def make_policies(runs_dir, names, env):
    """Policy callables by display name. Pure pursuit reads the shared env."""
    policies = {f"ppo:{name}": ppo_policy(model_path(runs_dir / name)) for name in names}
    policies["pure_pursuit"] = PurePursuitPolicy(env)
    policies["random"] = RandomPolicy(seed=0)
    return policies


def read_test_results(runs_dir):
    """Test numbers only come from S3 eval folders, never from this script."""
    results = {}
    for path in sorted(runs_dir.glob("eval_*_test/eval.json")):
        results[path.parent.name] = json.loads(path.read_text())["metrics"]
    return results


def default_difficulty(runs, names):
    """Difficulty the first selected run trained on, so it is judged on its own level."""
    configs = {run["name"]: run["config"] for run in runs}
    return float(configs[names[0]].get("difficulty", 0.0)) if names else 0.0


def build(runs_dir, names, difficulty, n_eval, n_replay, out):
    runs_dir, out = Path(runs_dir), Path(out)
    runs = [read_run(p.parent) for p in sorted(runs_dir.glob("*/config.yaml"))]
    if difficulty is None:
        difficulty = default_difficulty(runs, names)
    env = RacingEnv(difficulty=difficulty)
    policies = make_policies(runs_dir, names, env)

    validation, replays = {}, {}
    for label, policy in policies.items():
        print(f"evaluating {label}")
        validation[label] = evaluate(policy, split_seeds("val", n_eval), difficulty, env=env)
        replays[label] = [
            record_episode(policy, seed, difficulty, env=env)
            for seed in split_seeds("val", n_replay)
        ]

    tracks = {}
    for seed in split_seeds("val", n_replay):
        env.track_seed = seed
        env.reset(seed=seed)
        tracks[seed] = track_geometry(env.track)

    data = {
        "meta": {
            "difficulty": difficulty,
            "eval_tracks": n_eval,
            "lidar_angles": [round(float(a), 4) for a in lidar.ANGLES],
            "lidar_range": float(lidar.RANGE),
            "max_speed": float(car.MAX_SPEED),
            "selected": [f"ppo:{name}" for name in names],
        },
        "runs": runs,
        "validation": validation,
        "test": read_test_results(runs_dir),
        "tracks": tracks,
        "replays": replays,
    }
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(STATIC, out)
    payload = "window.DATA = " + json.dumps(data, separators=(",", ":"))
    (out / "data.js").write_text(payload, encoding="utf-8")
    return out / "index.html"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="*", default=[], help="run folders to evaluate and replay")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--difficulty", type=float, default=None, help="default from the first run")
    p.add_argument("--eval-tracks", type=int, default=20)
    p.add_argument("--replay-tracks", type=int, default=6)
    p.add_argument("--out", default="runs/dashboard")
    args = p.parse_args()
    index = build(
        args.runs_dir, args.runs, args.difficulty, args.eval_tracks, args.replay_tracks, args.out
    )
    print(f"dashboard ready, open {index.resolve()}")


if __name__ == "__main__":
    main()
