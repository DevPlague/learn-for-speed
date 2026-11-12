"""PPO training from one YAML config. Run with
python -m learning.train --config configs/tiny.yaml --seed 0
Outputs go to runs/<config>_s<seed>/ (config.yaml, model.zip, best_model.zip, checkpoints/,
val_curve.csv, progress.csv and TensorBoard events).
"""

import argparse
import csv
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from evaluation.evaluate import METRICS, evaluate, split_seeds
from learning.reward import RaceReward, RewardWeights
from sim.env import RacingEnv


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))  # tolerates a Windows BOM
    cfg["name"] = path.stem
    return cfg


def make_env(weights, track_seed=None, difficulty=0.0, mix=0.0):
    env = RaceReward(RacingEnv(track_seed, difficulty), weights)
    if mix:
        env = MixedDifficulty(env, mix)
    return Monitor(env)


class MixedDifficulty(gym.Wrapper):
    """Curriculum level in `difficulty`. A share `mix` of the episodes draws its track
    difficulty uniformly below the level, the rest uses the level itself, so the agent keeps
    the easier tracks while it learns the harder ones.
    """

    def __init__(self, env, mix):
        super().__init__(env)
        self.mix = mix
        self.difficulty = env.unwrapped.difficulty  # the curriculum sets it, see ValCurveCallback
        self.rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        below = self.rng.random() < self.mix
        level = self.rng.uniform(0, self.difficulty) if below else self.difficulty
        self.env.unwrapped.difficulty = level
        return self.env.reset(seed=seed, options=options)


def make_train_env(cfg):
    mix = (cfg.get("curriculum") or {}).get("mix", 0.0)
    weights = RewardWeights(**cfg["reward"])
    factory = partial(make_env, weights, difficulty=cfg["difficulty"], mix=mix)
    # one env step costs about 60 us, so subprocess messaging costs more than it saves.
    # One process per run leaves the other cores free to train several training seeds at once
    env = DummyVecEnv([factory] * cfg["n_envs"])
    gamma = cfg["ppo"].get("gamma", 0.99)
    return VecNormalize(env, norm_obs=False, norm_reward=True, gamma=gamma)


class ValCurveCallback(BaseCallback):
    """Drives the validation tracks with evaluation.evaluate, so training and S3 share metrics.

    Appends one row per check to val_curve.csv and keeps best_model.zip by difficulty, then
    lap rate, then mean progress. With a `curriculum` in the config the training difficulty
    rises by `step` each time the validation lap rate at the current difficulty reaches
    `threshold`, up to 1.
    """

    def __init__(self, cfg, run_dir):
        super().__init__()
        self.freq = cfg["eval"]["freq"]
        self.seeds = split_seeds("val", cfg["eval"]["tracks"])
        self.difficulty = cfg["difficulty"]
        self.curriculum = cfg.get("curriculum")
        self.run_dir = run_dir
        self.best = (-1.0, -1.0, -1.0)
        self.last = None
        self.next_check = self.freq

    def _on_training_start(self):
        with open(self.run_dir / "val_curve.csv", "w", newline="") as f:
            csv.writer(f).writerow(["timesteps", "difficulty", *METRICS])

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
            row_values = (row[k] for k in METRICS)
            csv.writer(f).writerow([self.num_timesteps, self.difficulty, *row_values])
        score = (self.difficulty, row["lap_success_rate"], row["mean_progress"])
        if score > self.best:
            self.best = score
            model.save(self.run_dir / "best_model")
        self.last = row
        c = self.curriculum
        if c and row["lap_success_rate"] >= c["threshold"] and self.difficulty < 1:
            self.difficulty = min(1.0, round(self.difficulty + c["step"], 6))
            # set_wrapper_attr reaches RacingEnv through Monitor and RaceReward
            self.training_env.env_method("set_wrapper_attr", "difficulty", self.difficulty)


def make_callbacks(cfg, run_dir):
    # SB3 counts the checkpoint frequency in vec env steps, each one is n_envs env steps
    save_freq = max(cfg["checkpoint_freq"] // cfg["n_envs"], 1)
    checkpoint = CheckpointCallback(save_freq, str(run_dir / "checkpoints"))
    return checkpoint, ValCurveCallback(cfg, run_dir)


def train(cfg, seed, runs_dir="runs"):
    run_dir = Path(runs_dir) / f"{cfg['name']}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "seed": seed}, sort_keys=False))

    # a 64x64 network gains nothing from several torch threads, they only add sync cost
    torch.set_num_threads(1)
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
        difficulty, laps, progress = validation.best
        print(
            f"Last validation lap rate {validation.last['lap_success_rate']:.0%}, "
            f"progress {validation.last['mean_progress']:.2f}. "
            f"Best lap rate {laps:.0%}, progress {progress:.2f} at difficulty {difficulty}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(load_config(args.config), args.seed)


if __name__ == "__main__":
    main()
