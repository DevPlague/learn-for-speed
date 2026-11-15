import csv

import numpy as np
import pytest
import yaml

import learning.train as train_module
from evaluation.evaluate import METRICS, ppo_policy
from learning.reward import RewardWeights
from learning.train import ValCurveCallback, load_config, make_env, make_train_env
from sim.env import RacingEnv


def test_tiny_run_saves_config_curves_and_a_loadable_policy(tiny_run):
    cfg = yaml.safe_load((tiny_run / "config.yaml").read_text())
    assert cfg["seed"] == 0 and cfg["name"] == "tiny"
    assert (tiny_run / "progress.csv").exists()
    with open(tiny_run / "val_curve.csv") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["timesteps", "difficulty", *METRICS]
    assert [r["timesteps"] for r in rows] == [
        "64",
        "128",
    ]
    obs, _ = RacingEnv(track_seed=1000).reset(seed=0)
    assert ppo_policy(tiny_run / "model.zip")(obs).shape == (2,)


class FakeModel:
    def __init__(self, env=None):
        self.saved = 0
        self.env = env

    def save(self, path):
        self.saved += 1

    def get_env(self):
        return self.env


def test_best_model_is_highest_lap_rate_then_progress(tmp_path, monkeypatch):
    rows = iter([(0.5, 0.6), (0.5, 0.4), (0.7, 0.1), (0.7, 0.1), (0.7, 0.3)])

    def scripted(policy, seeds, difficulty):
        laps, progress = next(rows)
        return {**dict.fromkeys(METRICS, 0.0), "lap_success_rate": laps, "mean_progress": progress}

    monkeypatch.setattr(train_module, "evaluate", scripted)
    cfg = load_config("configs/tiny.yaml")
    callback = ValCurveCallback(cfg, tmp_path)
    callback.model = FakeModel()
    callback._on_training_start()
    saves = []
    for _ in range(5):
        callback.validate()
        saves.append(callback.model.saved)
    assert saves == [1, 1, 2, 2, 3]  # a lower or equal score never replaces the best model
    assert callback.best == (0.0, 0.7, 0.3)
    with open(tmp_path / "val_curve.csv") as f:
        assert len(list(csv.DictReader(f))) == 5


def test_curriculum_raises_difficulty_in_the_inner_env_and_stops_at_one(tmp_path, monkeypatch):
    laps = iter([0.5, 0.7, 0.9, 0.2, 1.0, 1.0, 1.0])

    def scripted(policy, seeds, difficulty):
        return {**dict.fromkeys(METRICS, 0.0), "lap_success_rate": next(laps)}

    monkeypatch.setattr(train_module, "evaluate", scripted)
    cfg = load_config("configs/mvp_curriculum.yaml")
    cfg["n_envs"] = 1
    env = make_train_env(cfg)
    callback = ValCurveCallback(cfg, tmp_path)
    callback.model = FakeModel(env)
    callback._on_training_start()
    levels = []
    for _ in range(7):
        callback.validate()
        levels.append(env.venv.envs[0].unwrapped.difficulty)
    env.close()
    assert levels == [0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 1.0]  # under 70% laps it stays
    with open(tmp_path / "val_curve.csv") as f:
        logged = [float(r["difficulty"]) for r in csv.DictReader(f)]
    assert logged == [0.0, 0.0, 0.25, 0.5, 0.5, 0.75, 1.0]  # each row is the level it ran at
    assert callback.model.saved == 7  # a harder level beats any lap rate at an easier one


def test_only_the_reward_is_normalized():
    cfg = load_config("configs/tiny.yaml")
    cfg["n_envs"] = 1
    env = make_train_env(cfg)
    assert env.norm_reward and not env.norm_obs  # evaluation feeds raw observations
    env.close()


@pytest.mark.parametrize(
    "name",
    [
        "tiny",
        "mvp",
        "mvp_crawl",
        "mvp_curriculum",
        "final",
        "probe_mvp",
        "probe_ent",
        "probe_crawl",
    ],
)
def test_configs_build_their_reward_weights(name):
    cfg = load_config(f"configs/{name}.yaml")
    train_module.RewardWeights(**cfg["reward"])
    assert np.isfinite(cfg["ppo"]["learning_rate"])


def test_mixed_difficulty_keeps_the_level_for_part_of_the_episodes_and_draws_below_it():
    env = make_env(RewardWeights(), difficulty=0.0, mix=0.5)
    env.set_wrapper_attr("difficulty", 0.5)  # what the curriculum calls through Monitor
    levels = []
    for i in range(40):
        env.reset(seed=i)
        levels.append(env.unwrapped.difficulty)
    at_level = sum(level == 0.5 for level in levels)
    assert 10 <= at_level <= 30
    assert all(0.0 <= level <= 0.5 for level in levels)
    assert env.get_wrapper_attr("difficulty") == 0.5
