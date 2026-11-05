import numpy as np

from evaluation.baselines import PurePursuitPolicy, RandomPolicy
from sim.env import RacingEnv


def run(env, policy, seed):
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(policy(obs))
        done = term or trunc
    return info


def test_random_policy_range_and_seed():
    obs = np.zeros(4, dtype=np.float32)
    a = [RandomPolicy(seed=3)(obs) for _ in range(2)]
    b = RandomPolicy(seed=3)
    assert np.array_equal(a[0], b(obs))
    acts = np.array([b(obs) for _ in range(200)])
    assert acts.shape == (200, 2)
    assert acts.min() >= -1 and acts.max() <= 1


def test_pure_pursuit_action_in_range():
    env = RacingEnv(track_seed=1000)
    obs, _ = env.reset(seed=0)
    pp = PurePursuitPolicy(env)
    for _ in range(50):
        a = pp(obs)
        assert a.shape == (2,) and env.action_space.contains(a)
        obs, *_ = env.step(a)


def test_pure_pursuit_laps_validation_tracks():
    for seed in (1000, 1001, 1002):
        env = RacingEnv(track_seed=seed, difficulty=0.5)
        assert run(env, PurePursuitPolicy(env), seed=0)["lap_done"]
