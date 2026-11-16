from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.baselines import PurePursuitPolicy, RandomPolicy
from evaluation.evaluate import run_episode
from sim.env import RacingEnv
from sim.track import _build


def test_random_policy_is_seeded_and_uniform():
    obs = np.zeros(4, dtype=np.float32)
    a, b = RandomPolicy(seed=3), RandomPolicy(seed=3)
    assert np.array_equal(a(obs), b(obs))
    assert not np.array_equal(RandomPolicy(seed=4)(obs), RandomPolicy(seed=3)(obs))
    acts = np.array([a(obs) for _ in range(2000)])
    assert acts.min() >= -1 and acts.max() <= 1 and abs(acts.mean()) < 0.05


def on_circle(radius, speed):
    """Pure pursuit on the start of a circular road, driving at `speed`."""
    angles = np.linspace(0, 2 * np.pi, round(2 * np.pi * radius), endpoint=False)
    track = _build(radius * np.column_stack([np.cos(angles), np.sin(angles)]))
    env = SimpleNamespace(track=track, idx=0)
    env.state = np.array([*track.centerline[0], track.heading[0], speed, 0.0, 0.0])
    return PurePursuitPolicy(env)(None)


def test_pure_pursuit_brakes_for_a_tight_corner_and_steers_into_it():
    tight, wide = on_circle(15.0, 14.0), on_circle(200.0, 14.0)
    assert tight[1] < 0 < wide[1]  # sqrt(GRIP * 15) is about 11 m/s, a wide curve allows 16
    assert tight[0] > wide[0] > 0  # left circle, more steering when tighter


@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_pure_pursuit_laps_every_validation_track(difficulty):
    env = RacingEnv(difficulty=difficulty)
    outcomes = [run_episode(PurePursuitPolicy(env), s, env)["outcome"] for s in range(1000, 1005)]
    assert outcomes == ["lap"] * 5  # the env is solvable, Gate 1 reference
