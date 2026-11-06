import json

import pytest

from dashboard.build import build
from dashboard.record import record_episode
from evaluation.baselines import PurePursuitPolicy
from learning.train import load_config, train
from sim.env import RacingEnv


def test_record_episode_has_equal_length_frames_and_is_json():
    env = RacingEnv()
    episode = record_episode(PurePursuitPolicy(env), 1000, 0.0, env=env)

    lengths = {len(values) for values in episode["frames"].values()}
    assert len(lengths) == 1
    assert episode["outcome"] in ("lap", "off_road", "stuck", "timeout")
    json.dumps(episode)


def test_record_episode_rejects_training_seeds():
    with pytest.raises(ValueError):
        record_episode(lambda obs: [0.0, 0.0], 100_000, 0.0)


def test_build_writes_a_page_with_data(tmp_path):
    cfg = load_config("configs/tiny.yaml")
    cfg.update(total_timesteps=64, n_envs=1, checkpoint_freq=10**9)
    cfg["eval"].update(freq=64, tracks=1)
    cfg["ppo"].update(n_steps=64, batch_size=64)
    run_dir = train(cfg, seed=0, runs_dir=tmp_path)

    index = build(tmp_path, [run_dir.name], 0.0, 1, 1, tmp_path / "dashboard")

    assert index.exists()
    data = (index.parent / "data.js").read_text()
    payload = json.loads(data.removeprefix("window.DATA = "))
    assert set(payload["validation"]) == {f"ppo:{run_dir.name}", "pure_pursuit", "random"}
    assert payload["runs"][0]["val_curve"]["timesteps"] == [64.0]
