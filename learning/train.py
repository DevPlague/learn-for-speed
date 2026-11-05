"""PPO training from one YAML config. Run with
python -m learning.train --config configs/tiny.yaml --seed 0
Outputs go to runs/<config>_s<seed>/ (config.yaml, model.zip, best_model.zip, checkpoints/,
progress.csv and TensorBoard events).
"""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from learning.reward import RaceReward, RewardWeights
from sim.env import RacingEnv

VAL_SEED_MIN = 1000
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


def make_val_env(cfg):
    """One sub-env per validation track, so each evaluation drives every track once."""
    weights = RewardWeights(**cfg["reward"])
    seeds = range(VAL_SEED_MIN, VAL_SEED_MIN + cfg["eval"]["tracks"])
    env = DummyVecEnv([partial(make_env, weights, seed, cfg["difficulty"]) for seed in seeds])

    return VecNormalize(env, training=False, norm_obs=False, norm_reward=False)


def make_callbacks(cfg, run_dir):
    # SB3 counts callback frequencies in vec env steps, each one is n_envs env steps
    def per_env(steps):
        return max(steps // cfg["n_envs"], 1)

    checkpoint = CheckpointCallback(per_env(cfg["checkpoint_freq"]), str(run_dir / "checkpoints"))
    evaluation = EvalCallback(
        make_val_env(cfg),
        n_eval_episodes=cfg["eval"]["tracks"],
        eval_freq=per_env(cfg["eval"]["freq"]),
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        deterministic=True,
    )
    return checkpoint, evaluation


def train(cfg, seed, runs_dir=RUNS_DIR):
    run_dir = Path(runs_dir) / f"{cfg['name']}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "seed": seed}, sort_keys=False))

    env = make_train_env(cfg)
    model = PPO("MlpPolicy", env, seed=seed, device="cpu", **cfg["ppo"])
    # progress.csv gives S3 the learning curves without reading TensorBoard files
    model.set_logger(configure(str(run_dir), ["csv", "tensorboard"]))
    checkpoint, evaluation = make_callbacks(cfg, run_dir)
    model.learn(cfg["total_timesteps"], callback=[checkpoint, evaluation])
    model.save(run_dir / "model")
    env.close()
    evaluation.eval_env.close()
    print_summary(run_dir, evaluation)
    return run_dir


def print_summary(run_dir, evaluation):
    print(f"Run saved in {run_dir}")
    if evaluation.evaluations_successes:
        laps = np.mean(evaluation.evaluations_successes[-1])
        print(
            f"Last validation lap rate {laps:.0%}, best mean reward {evaluation.best_mean_reward:.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(load_config(args.config), args.seed)


if __name__ == "__main__":
    main()
