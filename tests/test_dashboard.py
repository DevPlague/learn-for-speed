import json

import numpy as np
import pytest

from dashboard.build import build, level_stats
from dashboard.record import record_episode
from evaluation.baselines import PurePursuitPolicy
from evaluation.evaluate import run_episode
from sim.env import RacingEnv


def test_record_episode_matches_the_evaluation_episode():
    env = RacingEnv()
    episode = record_episode(PurePursuitPolicy(env), 1000, 0.0, env=env)
    reference = run_episode(PurePursuitPolicy(env), 1000, env)
    frames = episode["frames"]
    assert {len(values) for values in frames.values()} == {len(reference["path"])}
    assert np.allclose(frames["x"], reference["path"][:, 0], atol=0.01)
    assert episode["outcome"] == reference["outcome"] == "lap"
    json.dumps(episode)


def test_level_stats_pools_seeds_of_one_config_and_never_tests_a_seed_against_its_pool():
    lap = {"seed": 1000, "outcome": "lap", "progress": 1.0, "lap_time": 30.0}
    stuck = {"seed": 1000, "outcome": "stuck", "progress": 0.0, "lap_time": None}
    s = level_stats({"ppo:a_s0": [lap], "ppo:a_s1": [stuck], "random": [stuck]})
    assert s["intervals"]["pooled:a"]["lap_rate_per_seed"] == [1.0, 0.0]
    assert {(t["a"], t["b"]) for t in s["tests"]} == {("random", "pooled:a")}
    assert s["friedman"] is None  # needs three units
    json.dumps(s)  # NaN intervals were turned into None


def test_build_writes_validation_replays_and_test_results(tiny_run, tmp_path):
    runs_dir = tiny_run.parent
    test_eval = runs_dir / "eval_ppo_test" / "eval.json"
    test_eval.parent.mkdir(exist_ok=True)
    test_eval.write_text(json.dumps({"metrics": {"lap_success_rate": 0.5}}))

    index = build(runs_dir, [tiny_run.name], [0.0, 1.0], 3, 1, tmp_path / "dashboard")

    payload = json.loads((index.parent / "data.js").read_text().removeprefix("window.DATA = "))
    policies = {f"ppo:{tiny_run.name}", "pure_pursuit", "random"}
    assert set(payload["validation"]) == policies and payload["meta"]["difficulty"] == 0.0
    assert [level["difficulty"] for level in payload["levels"]] == [0.0, 1.0]
    for level in payload["levels"]:
        assert "episodes" not in level["validation"]["random"]
        intervals = level["stats"]["intervals"]
        assert set(intervals) == policies and intervals["pure_pursuit"]["lap_rate"] == 1.0
        lo, hi = intervals["random"]["lap_rate_ci"]
        assert lo == pytest.approx(0.0) and 0.0 < hi < 1.0  # Wilson over 3 tracks, never laps
        tests = level["stats"]["tests"]
        assert {(t["a"], t["b"]) for t in tests} == {
            (f"ppo:{tiny_run.name}", "pure_pursuit"),
            (f"ppo:{tiny_run.name}", "random"),
            ("pure_pursuit", "random"),
        }
        assert all(t["p_holm"] is None or t["p_holm"] >= t["p"] for t in tests)
    assert list(payload["tracks"]) == ["1000"] and len(payload["replays"]["random"]) == 1
    assert payload["test"] == {"eval_ppo_test": {"lap_success_rate": 0.5}}
    assert payload["runs"][0]["val_curve"]["timesteps"] == [64.0, 128.0]
