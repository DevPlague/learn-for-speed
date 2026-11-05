import pytest

from learning.reward import DEFAULT_WEIGHTS, RaceReward, RewardWeights
from sim.env import RacingEnv


def make_env(weights=DEFAULT_WEIGHTS):
    env = RaceReward(RacingEnv(track_seed=1000), weights)
    env.reset(seed=0)
    return env


def reward_after_20_steps(action):
    env = make_env()
    for _ in range(20):  # let the car pick up speed
        _, reward, *_ = env.step(action)
    return reward


def test_driving_forward_beats_standing_still():
    assert reward_after_20_steps([0.0, 1.0]) > reward_after_20_steps([0.0, 0.0])


def test_standing_still_costs_the_time_penalty():
    assert reward_after_20_steps([0.0, 0.0]) == pytest.approx(-DEFAULT_WEIGHTS.time)


def test_steering_jitter_is_penalized():
    env = make_env(RewardWeights(progress=0, time=0))
    _, steady, *_ = env.step([0.0, 0.0])
    _, jitter, *_ = env.step([1.0, 0.0])
    assert steady == 0 and jitter == pytest.approx(-DEFAULT_WEIGHTS.jitter)


def test_episode_end_reports_success():
    env = make_env()
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step([0.0, 0.0])
        done = terminated or truncated
    assert info["is_success"] is False
