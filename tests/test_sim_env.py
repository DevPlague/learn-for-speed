import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from sim import car, lidar
from sim.env import LOOKAHEAD, OBS_DIM, RacingEnv

AHEAD_10M = lidar.N_RAYS + 4 + 2 * LOOKAHEAD.index(10.0)


def controller(obs):
    """Tiny pure-pursuit-like driver that only uses the observation."""
    x, y = obs[AHEAD_10M] * LOOKAHEAD[-1], obs[AHEAD_10M + 1] * LOOKAHEAD[-1]
    v = (obs[lidar.N_RAYS] + 1) / 2 * car.MAX_SPEED
    alpha = np.arctan2(y, x)
    steer = np.arctan(2 * car.WHEELBASE * np.sin(alpha) / np.hypot(x, y)) / car.MAX_STEER
    return np.array([steer, np.clip(0.5 * (8.0 - v), -1, 1)], dtype=np.float32)


def test_check_env():
    check_env(RacingEnv(track_seed=1000, difficulty=0.5), skip_render_check=True)


def test_spaces_and_info():
    env = RacingEnv(track_seed=1000)
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) == env.observation_space.shape
    assert env.observation_space.contains(obs)
    obs, _, _, _, info = env.step(env.action_space.sample())
    assert {"progress", "lap_done", "off_road", "stuck"} <= info.keys()
    assert env.observation_space.contains(obs)


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


def test_default_tracks_come_from_the_training_range():
    env = RacingEnv()
    env.reset(seed=0)
    first = env.track.centerline.copy()
    env.reset(seed=1)
    assert not np.array_equal(first, env.track.centerline)


def test_driving_off_road_ends_the_episode():
    env = RacingEnv(track_seed=1000)
    env.reset(seed=0)
    for _ in range(400):
        _, _, terminated, _, info = env.step([1, 1])
        if terminated:
            break
    assert terminated and info["off_road"]


def test_standing_still_is_stuck():
    env = RacingEnv(track_seed=1000)
    env.reset(seed=0)
    for _ in range(100):
        _, _, terminated, _, info = env.step([0, 0])
        if terminated:
            break
    assert terminated and info["stuck"]


@pytest.mark.parametrize("difficulty", [0.0, 0.5])
def test_scripted_controller_laps(difficulty):
    laps = 0
    for seed in range(1000, 1005):
        env = RacingEnv(track_seed=seed, difficulty=difficulty)
        obs, _ = env.reset(seed=0)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(controller(obs))
            done = terminated or truncated
        laps += info["lap_done"]
    assert laps >= 4


def test_registered_as_plain_gym_env():
    assert isinstance(RacingEnv(), gym.Env)
