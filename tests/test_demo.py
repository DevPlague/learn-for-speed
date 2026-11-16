import numpy as np

from evaluation.baselines import PurePursuitPolicy, RandomPolicy
from evaluation.demo import describe, label, record


def test_record_side_by_side_freezes_a_finished_car():
    policies = {"pp": PurePursuitPolicy, "random": lambda env: RandomPolicy(seed=0)}
    frames, outcomes = record(policies, 1000, 0.0, max_steps=300)
    h, w, c = frames[0].shape
    assert (h, w, c) == (600, 2 * 800, 3) and len(frames) == 300
    assert list(outcomes) == ["random"]  # random ends within 10 s, pure pursuit still drives
    before, last = frames[-2], frames[-1]
    assert (before[:, :800] != last[:, :800]).any() and (before[:, 800:] == last[:, 800:]).all()


def test_describe_names_each_outcome():
    base = {"lap_done": False, "off_road": False, "stuck": False}
    assert describe({**base, "lap_done": True, "lap_time": 25.14}) == "lap 25.1 s"
    assert describe({**base, "off_road": True}) == "off road"
    assert describe(base) == "timeout"


def test_label_draws_on_a_copy():
    frame = np.full((600, 800, 3), 40, dtype=np.uint8)
    out = label(frame, "ppo  lap 25.1 s")
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert (out != frame).any() and (frame == 40).all()  # text drawn, input untouched
