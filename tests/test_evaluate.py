import numpy as np
import pytest

from evaluation.baselines import RandomPolicy
from evaluation.evaluate import (
    METRICS,
    aggregate,
    check_seeds,
    evaluate,
    make_policy,
    split_seeds,
    to_markdown,
)

# track seed to scripted outcome and episode length in steps
OUTCOMES = {1000: ("lap", 4), 1001: ("off_road", 2), 1002: ("stuck", 3), 1003: ("timeout", 5)}


class MockEnv:
    """Ends each episode as scripted for its track seed. Speed is always 2."""

    def __init__(self):
        self.track_seed = None
        self.difficulty = 0.0
        self.used = []

    def reset(self, seed=None):
        self.used.append(self.track_seed)
        self.kind, self.length = OUTCOMES[self.track_seed]
        self.steps = 0
        self.state = np.zeros(6)
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        end = self.steps == self.length
        info = {
            "lap_done": end and self.kind == "lap",
            "off_road": end and self.kind == "off_road",
            "stuck": end and self.kind == "stuck",
            "progress": 1.0 if end and self.kind == "lap" else 0.5,
            "speed": -2.0 if self.steps == 1 else 2.0,  # reverse counts as speed too
        }
        if info["lap_done"]:
            info["lap_time"] = self.steps * 0.05
        terminated = end and self.kind != "timeout"
        return np.zeros(3, dtype=np.float32), 0.0, terminated, end and not terminated, info


def zero_policy(obs):
    return np.zeros(2, dtype=np.float32)


def test_metrics_from_scripted_episodes():
    env = MockEnv()
    r = evaluate(zero_policy, list(OUTCOMES), 0.3, env=env)
    assert env.used == list(OUTCOMES) and env.difficulty == 0.3
    assert r["n_episodes"] == 4
    assert r["lap_success_rate"] == r["off_track_rate"] == r["stuck_rate"] == 0.25
    assert r["mean_lap_time"] == pytest.approx(0.2)
    assert r["mean_speed"] == pytest.approx(2.0)
    assert r["mean_progress"] == pytest.approx((1.0 + 0.5 * 3) / 4)


def test_no_lap_gives_no_lap_time():
    assert evaluate(zero_policy, [1001, 1002], 0.0, env=MockEnv())["mean_lap_time"] is None


@pytest.mark.parametrize("seeds", [[0], [999], [2000], [9999], [100_000], [1000, 5]])
def test_rejects_training_and_unknown_seeds(seeds):
    with pytest.raises(ValueError):
        evaluate(zero_policy, seeds, 0.0, env=MockEnv())


def test_accepts_the_edges_of_both_splits():
    check_seeds([1000, 1999, 10_000, 99_999])


def test_unknown_policy_name_is_an_error():
    with pytest.raises(ValueError):
        make_policy("pure_persuit", env=None)


def test_split_seeds():
    assert split_seeds("val", 3) == [1000, 1001, 1002]
    assert split_seeds("test", 2) == [10_000, 10_001]


def test_aggregate_mean_and_std():
    a = {"n_episodes": 2, **dict.fromkeys(METRICS, 1.0)}
    b = {"n_episodes": 2, **dict.fromkeys(METRICS, 3.0), "mean_lap_time": None}
    agg = aggregate([a, b])
    assert agg["n_models"] == 2 and agg["n_episodes"] == 4
    assert agg["lap_success_rate"] == 2.0 and agg["lap_success_rate_std"] == 1.0
    assert agg["mean_lap_time"] == 1.0 and agg["mean_lap_time_std"] == 0.0


def test_to_markdown():
    plain = {"n_episodes": 1, **dict.fromkeys(METRICS, 0.5), "mean_lap_time": None}
    table = to_markdown({"random": plain, "ppo": aggregate([plain, plain])})
    lines = table.splitlines()
    assert len(lines) == 4 and lines[0].startswith("| policy |")
    assert "| random | 0.50 |" in lines[2] and "-" in lines[2]
    assert "0.50 ± 0.00" in lines[3]


def test_real_env_smoke():
    r = evaluate(RandomPolicy(seed=0), [1000], 0.0)
    assert r["lap_success_rate"] == 0.0 and r["episodes"][0]["outcome"] in ("stuck", "off_road")
