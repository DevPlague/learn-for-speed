import json

import numpy as np
import pytest

from evaluation.baselines import PurePursuitPolicy, RandomPolicy
from evaluation.evaluate import run_episode
from evaluation.plots import (
    curve_stats,
    find_failures,
    load_curve,
    plot_learning_curves,
    plot_results,
    plot_trajectory,
)
from sim.env import RacingEnv
from tests.test_evaluate import OUTCOMES, MockEnv, zero_policy

HEADER = "timesteps,lap_success_rate,mean_progress,mean_lap_time\n"


def write_curve(folder, rows):
    folder.mkdir()
    lines = [f"{t},{lap},{prog},{'' if lt is None else lt}" for t, lap, prog, lt in rows]
    (folder / "val_curve.csv").write_text(HEADER + "\n".join(lines) + "\n")
    return folder


@pytest.fixture
def two_seeds(tmp_path):
    a = write_curve(tmp_path / "a_s0", [(100, 0.0, 0.2, None), (200, 0.5, 0.6, 30.0)])
    b = write_curve(
        tmp_path / "a_s1", [(100, 1.0, 0.4, 25.0), (200, 1.0, 1.0, 24.0), (300, 1, 1, 23)]
    )
    return [a, b]


def test_load_curve_reads_empty_cells_as_nan(two_seeds):
    c = load_curve(two_seeds[0])
    assert list(c["timesteps"]) == [100, 200]
    assert np.isnan(c["mean_lap_time"][0]) and c["mean_lap_time"][1] == 30.0


def test_curve_stats_cuts_to_shared_checks(two_seeds):
    steps, mean, std = curve_stats([load_curve(d) for d in two_seeds], "lap_success_rate")
    assert list(steps) == [100, 200]
    assert np.allclose(mean, [0.5, 0.75]) and np.allclose(std, [0.5, 0.25])


def test_plot_learning_curves_writes_png(two_seeds, tmp_path):
    out = plot_learning_curves({"a": two_seeds}, tmp_path / "fig" / "curve.png")
    assert out.exists() and out.stat().st_size > 0


def test_run_episode_outcomes():
    env = MockEnv()
    for seed, (kind, length) in OUTCOMES.items():
        ep = run_episode(zero_policy, seed, env)
        assert ep["outcome"] == kind
        assert ep["path"].shape == (length, 2) and len(ep["speeds"]) == length


def test_find_failures_skips_laps():
    env = MockEnv()
    env.track = "mock track"
    found = find_failures(zero_policy, list(OUTCOMES), 0.0, 2, env)
    assert [ep["outcome"] for ep, _ in found] == ["off_road", "stuck"]


def test_real_trajectory_plot(tmp_path):
    env = RacingEnv(difficulty=0.0)
    ep = run_episode(PurePursuitPolicy(env), 1000, env)
    assert ep["outcome"] == "lap"
    assert plot_trajectory(ep, env.track, tmp_path / "pp.png").stat().st_size > 0
    failures = find_failures(RandomPolicy(seed=0), [1000], 0.0, 1, env)
    assert failures[0][0]["outcome"] != "lap"


def test_plot_results_writes_png(tmp_path):
    plain = {"lap_success_rate": 1.0, "off_track_rate": 0.0, "stuck_rate": 0.0}
    ppo = {**plain, "lap_success_rate": 0.6, "lap_success_rate_std": 0.2}
    paths = {}
    for name, m in (("pp", plain), ("ppo", ppo)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps({"metrics": m}))
    assert plot_results(paths, tmp_path / "bars.png").stat().st_size > 0
