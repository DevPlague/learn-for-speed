"""Builds the local dashboard from runs/ into runs/dashboard/. Open index.html, no server needed.

Example
    uv run python -m dashboard.build --runs mvp_s0 probe_nojitter_s0
"""

import argparse
import csv
import json
import math
import random
import re
import shutil
from pathlib import Path

import yaml

from dashboard.record import record_episode, track_geometry
from dashboard.serve import serve
from evaluation import stats
from evaluation.evaluate import evaluate, make_policy, split_seeds
from sim import car, lidar
from sim.env import GRASS_TIME, MAX_TIME, RacingEnv
from sim.track import WALL_OFFSET, generate_track


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


FINAL_RUNS = ["final_s0", "final_s1", "final_s2"]

# names shown on the page, in Spanish because the page is
CONFIG_LABELS = {
    "final": "Final",
    "mvp": "MVP inicial",
    "mvp_crawl": "MVP + anti-arrastre",
    "mvp_curriculum": "Curriculum 2M",
    "mvp_curriculum_4m": "Curriculum 4M",
    "exp_overspeed_4m": "Curriculum + exceso de velocidad",
    "probe": "Sonda 200k",
    "probe_nojitter": "Sonda sin jitter",
    "probe_mvp": "Sonda MVP 300k",
    "probe_ent": "Sonda con entropía",
    "probe_crawl": "Sonda anti-arrastre",
}


def run_label(name):
    """Readable name of a run, `Final · semilla 2` for final_s2."""
    match = re.match(r"^(.*)_s(\d+)$", name)
    config, seed = (match.group(1), match.group(2)) if match else (name, None)
    label = CONFIG_LABELS.get(config, config)
    return f"{label} · semilla {seed}" if seed is not None else label


def require_runs(runs_dir, names):
    """Names of the runs without a trained model, so the build can refuse to start."""
    return [n for n in names if not (Path(runs_dir) / n / "best_model.zip").exists()]


def read_run(run_dir):
    """Config and learning curves of one training run."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8-sig"))
    return {
        "name": run_dir.name,
        "label": run_label(run_dir.name),
        "config": cfg,
        "progress": read_csv(
            run_dir / "progress.csv",
            {
                "time/total_timesteps": "timesteps",
                "rollout/ep_rew_mean": "ep_rew_mean",
                "rollout/ep_len_mean": "ep_len_mean",
                "train/entropy_loss": "entropy_loss",
                "train/approx_kl": "approx_kl",
                "train/value_loss": "value_loss",
                "train/explained_variance": "explained_variance",
                "train/clip_fraction": "clip_fraction",
            },
        ),
        "val_curve": read_csv(run_dir / "val_curve.csv"),
    }


def model_path(run_dir):
    best = run_dir / "best_model.zip"
    return best if best.exists() else run_dir / "model.zip"


def make_policies(runs_dir, names, env):
    """Policy callables by display name. Pure pursuit reads the shared env."""
    policies = {f"ppo:{n}": make_policy("ppo", env, model_path(runs_dir / n)) for n in names}
    for name in ("pure_pursuit", "random"):
        policies[name] = make_policy(name, env)
    return policies


def read_test_results(runs_dir):
    """Test numbers only come from S3 eval folders, never from this script."""
    results = {}
    for path in sorted(runs_dir.glob("eval_*_test*/eval.json")):
        results[path.parent.name] = json.loads(path.read_text())["metrics"]
    return results


def race_tracks(policy, difficulties, seeds, env):
    """Validation tracks the policy lapped, each with its lap as a ghost to race against."""
    tracks = []
    for difficulty in difficulties:
        for seed in seeds:
            ep = record_episode(policy, seed, difficulty, env=env)
            if ep["outcome"] != "lap":
                continue
            frames = ep["frames"]
            tracks.append(
                {
                    "seed": seed,
                    "difficulty": difficulty,
                    "lap_time": round(ep["lap_time"], 2),
                    "track": track_geometry(generate_track(seed, difficulty)),
                    "ghost": {k: frames[k] for k in ("x", "y", "yaw", "progress")},
                }
            )
    return tracks


def car_constants():
    """Physics constants for the race view, so the browser car matches rust/src/car.rs."""
    names = [n for n in car.__all__ if n.isupper()]
    return {
        **{n: float(getattr(car, n)) for n in names},
        "WALL_OFFSET": WALL_OFFSET,
        "GRASS_TIME": GRASS_TIME,
        "MAX_TIME": MAX_TIME,
    }


def finite(value):
    """Plain JSON values, NaN becomes None and tuples become lists."""
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return None if isinstance(value, float) and math.isnan(value) else value


def level_stats(episodes):
    """Intervals and paired tests from evaluation.stats for one difficulty.

    Several seeds of one config are also pooled into `pooled:<config>`, which adds the t interval
    over seeds. Tests compare pooled configs, single runs and baselines, never a seed with its pool.
    """
    evals = {key: {"episodes": [eps]} for key, eps in episodes.items()}
    configs = {}
    for key in episodes:
        if key.startswith("ppo:"):
            configs.setdefault(re.sub(r"_s\d+$", "", key[4:]), []).append(key)
    units = dict(evals)
    for config, keys in configs.items():
        if len(keys) > 1:
            evals[f"pooled:{config}"] = units[f"pooled:{config}"] = {
                "episodes": [episodes[k] for k in keys]
            }
            for k in keys:
                units.pop(k)
    out = {"intervals": {key: stats.summarize(e) for key, e in evals.items()}}
    try:
        out["tests"] = stats.pairwise_tests(units)
    except ValueError as err:  # pooled configs with different seed counts cannot be paired
        out["tests"], out["note"] = [], str(err)
    out["friedman"] = stats.friedman(units) if len(units) >= 3 else None
    return finite(out)


def build(runs_dir, names, difficulties, n_eval, n_replay, out, n_race=6):
    runs_dir, out = Path(runs_dir), Path(out)
    runs = [
        read_run(p.parent)
        for p in sorted(runs_dir.glob("*/config.yaml"))
        if (p.parent / "val_curve.csv").exists()
        and (p.parent.name in names or not p.parent.name.startswith("tiny"))
    ]
    env = RacingEnv(difficulty=difficulties[0])
    policies = make_policies(runs_dir, names, env)

    levels = []
    for difficulty in difficulties:
        validation = {}
        for label, policy in policies.items():
            print(f"evaluating {label} at difficulty {difficulty}")
            validation[label] = evaluate(policy, split_seeds("val", n_eval), difficulty, env=env)
        episodes = {label: m.pop("episodes") for label, m in validation.items()}
        levels.append(
            {"difficulty": difficulty, "validation": validation, "stats": level_stats(episodes)}
        )

    difficulty = difficulties[0]
    # replays and race tracks change on every build, validation keeps its fixed first tracks
    rng = random.Random()
    replay_seeds = sorted(rng.sample(range(1000, 2000), n_replay))
    race_seeds = sorted(rng.sample(range(1000, 2000), n_race))
    replays = {
        label: [record_episode(policy, s, difficulty, env=env) for s in replay_seeds]
        for label, policy in policies.items()
    }
    tracks = {seed: track_geometry(generate_track(seed, difficulty)) for seed in replay_seeds}
    race_key = f"ppo:{names[0]}" if names else None
    race = race_tracks(policies[race_key], difficulties, race_seeds, env) if race_key else []

    data = {
        "meta": {
            "difficulty": difficulty,
            "difficulties": difficulties,
            "eval_tracks": n_eval,
            "lidar_angles": [round(float(a), 4) for a in lidar.ANGLES],
            "lidar_range": float(lidar.RANGE),
            "max_speed": float(car.MAX_SPEED),
            "selected": [f"ppo:{name}" for name in names],
            "replay_seeds": replay_seeds,
            "car": car_constants(),
        },
        "race": {"policy": race_key, "tracks": race},
        "runs": runs,
        "validation": levels[0]["validation"],
        "levels": levels,
        "test": read_test_results(runs_dir),
        "tracks": tracks,
        "replays": replays,
    }
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(Path(__file__).parent / "static", out)
    payload = "window.DATA = " + json.dumps(data, separators=(",", ":"))
    (out / "data.js").write_text(payload, encoding="utf-8")
    return out / "index.html"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs", nargs="*", default=FINAL_RUNS, help="run folders to evaluate and replay"
    )
    p.add_argument("--runs-dir", default="runs")
    p.add_argument(
        "--difficulties",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
        help="validation levels, replays use the first one",
    )
    p.add_argument("--eval-tracks", type=int, default=20)
    p.add_argument("--replay-tracks", type=int, default=6)
    p.add_argument(
        "--race-tracks",
        type=int,
        default=6,
        help="validation tracks per difficulty for the race view, only the ones the first run laps",
    )
    p.add_argument("--out", default="runs/dashboard")
    p.add_argument("--serve", action="store_true", help="serve it at a local link after the build")
    args = p.parse_args()
    missing = require_runs(args.runs_dir, args.runs)
    if missing:
        print(f"Faltan modelos entrenados para {', '.join(missing)}, el dashboard no se construye.")
        print("Entrena las runs finales primero, las 3 a la vez tardan unos 17 minutos")
        print("    make train-seeds CONFIG=final")
        raise SystemExit(1)
    index = build(
        args.runs_dir,
        args.runs,
        args.difficulties,
        args.eval_tracks,
        args.replay_tracks,
        args.out,
        args.race_tracks,
    )
    print(f"dashboard ready, open {index.resolve()}")
    if args.serve:
        serve(index.parent)


if __name__ == "__main__":
    main()
