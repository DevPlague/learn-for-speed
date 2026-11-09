import csv

import numpy as np

from evaluation.evaluate import METRICS
from learning.policy import load_policy
from learning.train import load_config, train
from sim.env import RacingEnv


def test_tiny_run_saves_a_loadable_policy_and_a_validation_curve(tmp_path):
    cfg = load_config("configs/tiny.yaml")
    cfg.update(total_timesteps=128, n_envs=1, checkpoint_freq=10**9)
    cfg["eval"].update(freq=64, tracks=1)  # one short validation pass per rollout
    cfg["ppo"].update(n_steps=64, batch_size=64)

    run_dir = train(cfg, seed=0, runs_dir=tmp_path)

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "progress.csv").exists()
    assert (run_dir / "best_model.zip").exists()
    with open(run_dir / "val_curve.csv") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["timesteps", *METRICS] and len(rows) == 2
    policy = load_policy(run_dir / "model.zip")
    obs, _ = RacingEnv(track_seed=1000).reset(seed=0)
    action = policy(obs)
    assert action.shape == (2,) and np.all(np.abs(action) <= 1)
