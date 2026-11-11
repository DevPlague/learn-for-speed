import pytest

from learning.train import load_config, train


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory):
    """One short PPO run shared by the training and dashboard tests, two validation passes."""
    cfg = load_config("configs/tiny.yaml")
    cfg.update(total_timesteps=128, n_envs=1, checkpoint_freq=10**9)
    cfg["eval"].update(freq=64, tracks=1)
    cfg["ppo"].update(n_steps=64, batch_size=64)
    return train(cfg, seed=0, runs_dir=tmp_path_factory.mktemp("runs"))
