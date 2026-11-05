import numpy as np

from learning.policy import load_policy
from learning.train import load_config, train
from sim.env import RacingEnv


def test_tiny_run_saves_a_loadable_policy(tmp_path):
    cfg = load_config("configs/tiny.yaml")
    cfg.update(total_timesteps=128, n_envs=1, checkpoint_freq=10**9)
    cfg["eval"]["freq"] = 10**9  # never evaluate, keeps the test fast
    cfg["ppo"].update(n_steps=64, batch_size=64)

    run_dir = train(cfg, seed=0, runs_dir=tmp_path)

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "progress.csv").exists()
    policy = load_policy(run_dir / "model.zip")
    obs, _ = RacingEnv(track_seed=1000).reset(seed=0)
    action = policy(obs)
    assert action.shape == (2,) and np.all(np.abs(action) <= 1)
