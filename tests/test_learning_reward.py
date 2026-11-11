import pytest

from evaluation.baselines import PurePursuitPolicy
from learning.reward import DEFAULT_WEIGHTS, RaceReward, RewardWeights
from sim.env import RacingEnv

NONE = dict.fromkeys(RewardWeights.__dataclass_fields__, 0.0)


def make_env(**weights):
    env = RaceReward(RacingEnv(track_seed=1000), RewardWeights(**weights))
    env.reset(seed=0)
    return env


def test_progress_term_is_the_lap_fraction_gained():
    env = make_env(**{**NONE, "progress": 10.0, "time": 0.01})
    for _ in range(60):  # past 8.8 m/s, where progress pays for the time
        before = env.unwrapped.progress
        _, reward, *_ = env.step([0.0, 1.0])
    assert reward == pytest.approx(10.0 * (env.unwrapped.progress - before) - 0.01)
    assert reward > 0


def test_standing_still_costs_time_idle_and_crawl():
    env = make_env(crawl=0.02)
    for _ in range(20):
        _, reward, *_ = env.step([0.0, 0.0])
    assert reward == pytest.approx(-(DEFAULT_WEIGHTS.time + DEFAULT_WEIGHTS.idle + 0.02))


def test_reversing_beats_standing_still():
    rewards = {}
    for throttle in (0.0, -1.0):
        env = make_env()
        for _ in range(20):
            _, rewards[throttle], *_ = env.step([0.0, throttle])
    assert rewards[-1.0] > rewards[0.0]


def test_crawl_applies_below_2_mps_along_the_track_only():
    env = make_env(**{**NONE, "crawl": 1.0})
    _, first, *_ = env.step([0.0, 1.0])  # 0.3 m/s after one step
    for _ in range(40):
        _, fast, *_ = env.step([0.0, 1.0])
    assert first == -1.0 and fast == 0.0


def test_steering_jitter_is_penalized_and_reset_between_episodes():
    env = make_env(**{**NONE, "jitter": 0.05})
    _, steady, *_ = env.step([0.0, 0.0])
    _, jitter, *_ = env.step([1.0, 0.0])
    env.reset(seed=0)
    _, after_reset, *_ = env.step([0.0, 0.0])
    assert steady == 0 and jitter == pytest.approx(-0.05) and after_reset == 0


def test_getting_stuck_ends_with_the_stuck_penalty():
    env = make_env(**{**NONE, "stuck": 10.0})
    done = False
    while not done:
        _, reward, terminated, truncated, info = env.step([0.0, 0.0])
        done = terminated or truncated
    assert info["stuck"] and info["is_success"] is False
    assert reward == pytest.approx(-10.0)
    assert DEFAULT_WEIGHTS.stuck > DEFAULT_WEIGHTS.off_road


def test_a_lap_pays_the_bonus_and_reports_success():
    env = make_env(**{**NONE, "lap": 5.0})
    policy = PurePursuitPolicy(env.unwrapped)
    obs, done = env.unwrapped._obs(), False
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        done = terminated or truncated
    assert info["is_success"] is True and reward == 5.0
