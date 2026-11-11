import numpy as np
import pytest

from evaluation.baselines import PurePursuitPolicy, RandomPolicy
from evaluation.demo import label, record


def test_record_side_by_side_until_all_finish():
    policies = {"pp": PurePursuitPolicy, "random": lambda env: RandomPolicy(seed=0)}
    frames, outcomes = record(policies, 1000, 0.0, max_steps=300)
    h, w, c = frames[0].shape
    assert c == 3 and w == 2 * 800 and h == 600
    assert outcomes == {"random": "stuck"}  # random gives up within 10 s, pure pursuit drives on


def test_record_rejects_training_seed():
    with pytest.raises(ValueError):
        record({"random": lambda env: RandomPolicy()}, 100_000, 0.0, max_steps=1)


def test_label_draws_on_a_copy():
    frame = np.full((600, 800, 3), 40, dtype=np.uint8)
    out = label(frame, "ppo  lap 25.1 s")
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert (out != frame).any() and (frame == 40).all()  # text drawn, input untouched
