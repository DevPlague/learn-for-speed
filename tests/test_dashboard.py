import json

import numpy as np

from dashboard.build import build
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


def test_build_writes_validation_replays_and_test_results(tiny_run, tmp_path):
    runs_dir = tiny_run.parent
    test_eval = runs_dir / "eval_ppo_test" / "eval.json"
    test_eval.parent.mkdir(exist_ok=True)
    test_eval.write_text(json.dumps({"metrics": {"lap_success_rate": 0.5}}))

    index = build(runs_dir, [tiny_run.name], None, 1, 1, tmp_path / "dashboard")

    payload = json.loads((index.parent / "data.js").read_text().removeprefix("window.DATA = "))
    assert set(payload["validation"]) == {f"ppo:{tiny_run.name}", "pure_pursuit", "random"}
    assert payload["meta"]["difficulty"] == 0.0  # taken from the run config when not given
    assert list(payload["tracks"]) == ["1000"] and len(payload["replays"]["random"]) == 1
    assert payload["test"] == {"eval_ppo_test": {"lap_success_rate": 0.5}}
    assert payload["runs"][0]["val_curve"]["timesteps"] == [64.0, 128.0]
