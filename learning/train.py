"""PPO training from one YAML config. Run with
python -m learning.train --config configs/tiny.yaml --seed 0
Outputs go to runs/<config>_s<seed>/ (config.yaml, model.zip, best_model.zip, checkpoints/,
val_curve.csv, progress.csv and TensorBoard events).
"""

import argparse
import csv
from functools import partial
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from evaluation.evaluate import METRICS, evaluate, split_seeds
from learning.reward import RaceReward, RewardWeights
from sim.env import RacingEnv

RUNS_DIR = Path("runs")


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))  # tolerates a Windows BOM
    cfg["name"] = path.stem
    return cfg


def make_env(weights, track_seed=None, difficulty=0.0):
    return Monitor(RaceReward(RacingEnv(track_seed, difficulty), weights))


def make_train_env(cfg):
    factory = partial(make_env, RewardWeights(**cfg["reward"]), difficulty=cfg["difficulty"])
    vec_cls = SubprocVecEnv if cfg["n_envs"] > 1 else DummyVecEnv
    env = vec_cls([factory] * cfg["n_envs"])
    gamma = cfg["ppo"].get("gamma", 0.99)
    return VecNormalize(env, norm_obs=False, norm_reward=True, gamma=gamma)


class ValCurveCallback(BaseCallback):
    """Drives the validation tracks with evaluation.evaluate, so training and S3 share metrics.

    Appends one row per check to val_curve.csv and keeps best_model.zip by lap rate,
    then by mean progress.
    """

    def __init__(self, cfg, run_dir):
        super().__init__()
        self.freq = cfg["eval"]["freq"]
        self.seeds = split_seeds("val", cfg["eval"]["tracks"])
        self.difficulty = cfg["difficulty"]
        self.run_dir = run_dir
        self.best = (-1.0, -1.0)
        self.last = None
        self.next_check = self.freq

    def _on_training_start(self):
        with open(self.run_dir / "val_curve.csv", "w", newline="") as f:
            csv.writer(f).writerow(["timesteps", *METRICS])

    def _on_step(self):
        # num_timesteps grows by n_envs per call, so compare against a threshold
        if self.num_timesteps >= self.next_check:
            self.next_check += self.freq
            self.validate()
        return True

    def validate(self):
        model = self.model
        row = evaluate(
            lambda obs: model.predict(obs, deterministic=True)[0], self.seeds, self.difficulty
        )
        with open(self.run_dir / "val_curve.csv", "a", newline="") as f:
            csv.writer(f).writerow([self.num_timesteps, *(row[k] for k in METRICS)])
        score = (row["lap_success_rate"], row["mean_progress"])
        if score > self.best:
            self.best = score
            model.save(self.run_dir / "best_model")
        self.last = row


def make_callbacks(cfg, run_dir):
    # SB3 counts the checkpoint frequency in vec env steps, each one is n_envs env steps
    save_freq = max(cfg["checkpoint_freq"] // cfg["n_envs"], 1)
    checkpoint = CheckpointCallback(save_freq, str(run_dir / "checkpoints"))
    return checkpoint, ValCurveCallback(cfg, run_dir)


def train(cfg, seed, runs_dir=RUNS_DIR):
    run_dir = Path(runs_dir) / f"{cfg['name']}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "seed": seed}, sort_keys=False))

    env = make_train_env(cfg)
    model = PPO("MlpPolicy", env, seed=seed, device="cpu", **cfg["ppo"])
    # progress.csv gives S3 the learning curves without reading TensorBoard files
    model.set_logger(configure(str(run_dir), ["csv", "tensorboard"]))
    checkpoint, validation = make_callbacks(cfg, run_dir)
    model.learn(cfg["total_timesteps"], callback=[checkpoint, validation])
    model.save(run_dir / "model")
    env.close()
    print_summary(run_dir, validation)
    return run_dir


def print_summary(run_dir, validation):
    print(f"Run saved in {run_dir}")
    if validation.last:
        laps, progress = validation.best
        print(
            f"Last validation lap rate {validation.last['lap_success_rate']:.0%}, "
            f"progress {validation.last['mean_progress']:.2f}. "
            f"Best lap rate {laps:.0%}, progress {progress:.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(load_config(args.config), args.seed)


if __name__ == "__main__":
    main()
