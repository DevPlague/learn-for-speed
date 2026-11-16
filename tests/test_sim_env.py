import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import sim.env as env_module
from evaluation.baselines import PurePursuitPolicy
from sim.env import OBS_DIM, TRAIN_SEED_MIN, RacingEnv, outcome
from sim.track import WALL_OFFSET


def test_check_env():
    check_env(RacingEnv(track_seed=1000, difficulty=0.5), skip_render_check=True)


def test_info_has_every_key_the_reward_and_evaluation_read():
    env = RacingEnv(track_seed=1000)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert {"progress", "lap_done", "off_road", "stuck", "on_grass", "speed"} <= info.keys()


def rollout(seed):
    env = RacingEnv(difficulty=0.5)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    out = [obs]
    for _ in range(50):
        obs, *_ = env.step(rng.uniform(-1, 1, 2))
        out.append(obs)
    return np.array(out)


def test_same_seed_same_trajectory():
    assert np.array_equal(rollout(5), rollout(5))
    assert not np.array_equal(rollout(5), rollout(6))


def test_default_tracks_come_from_the_training_range(monkeypatch):
    seeds = []
    real = env_module.generate_track
    monkeypatch.setattr(env_module, "generate_track", lambda s, d: seeds.append(s) or real(s, d))
    env = RacingEnv()
    for seed in range(3):
        env.reset(seed=seed)
    assert min(seeds) >= TRAIN_SEED_MIN and len(set(seeds)) == 3


def on_grass_start(env, offset):
    """Car standing still `offset` meters left of the start, which is on the grass for 4 to 12."""
    env.reset(seed=0)
    h = env.track.heading[0]
    x, y = env.track.centerline[0] + offset * np.array([-np.sin(h), np.cos(h)])
    env.state = np.array([x, y, h, 0.0, 0.0, 0.0])


def test_grass_is_allowed_until_the_grass_timer_runs_out(monkeypatch):
    monkeypatch.setattr(env_module, "GRASS_TIME", 1.0)
    env = RacingEnv(track_seed=1000)
    on_grass_start(env, 6.0)
    steps = 0
    terminated = False
    while not terminated:
        _, _, terminated, _, info = env.step([0.0, 0.0])
        steps += 1
    assert info["off_road"] and not info["stuck"] and steps == 21  # one step past 1 s
    assert abs(env.lateral) < WALL_OFFSET


def test_reaching_the_barrier_ends_at_once():
    env = RacingEnv(track_seed=1000)
    on_grass_start(env, 11.0)
    env.state[2] += np.pi / 2  # facing the left barrier
    steps = 0
    terminated = False
    while not terminated:
        _, _, terminated, _, info = env.step([0.0, 1.0])
        steps += 1
    assert info["off_road"] and steps < 20 and abs(env.lateral) > WALL_OFFSET


def test_timeout_is_truncated_not_terminated(monkeypatch):
    monkeypatch.setattr(env_module, "MAX_TIME", 1.0)
    env = RacingEnv(track_seed=1000)
    obs, _ = env.reset(seed=0)
    policy = PurePursuitPolicy(env)
    for _ in range(20):
        obs, _, terminated, truncated, info = env.step(policy(obs))
    assert truncated and not terminated and outcome(info) == "timeout"


@pytest.mark.parametrize(
    "flags,expected",
    [
        ({"lap_done": True, "off_road": False, "stuck": False}, "lap"),
        ({"lap_done": False, "off_road": True, "stuck": False}, "off_road"),
        ({"lap_done": False, "off_road": False, "stuck": True}, "stuck"),
        ({"lap_done": False, "off_road": False, "stuck": False}, "timeout"),
    ],
)
def test_outcome(flags, expected):
    assert outcome(flags) == expected
