import numpy as np
import pytest

from sim.track import frenet, generate_track, locate, lookahead

DIFFICULTIES = [0.0, 0.5, 1.0]


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_same_seed_same_track(difficulty):
    a, b = generate_track(7, difficulty), generate_track(7, difficulty)
    assert np.array_equal(a.centerline, b.centerline)
    assert not np.array_equal(a.centerline, generate_track(8, difficulty).centerline)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_closed_loop_with_even_spacing(difficulty):
    for seed in range(20):
        t = generate_track(seed, difficulty)
        seg = np.linalg.norm(np.roll(t.centerline, -1, axis=0) - t.centerline, axis=1)
        assert np.allclose(seg, 1.0, atol=0.15)
        assert t.length == pytest.approx(seg.sum())
        assert np.allclose(np.linalg.norm(t.left - t.right, axis=1), t.width)


def test_edges_do_not_overlap_the_rest_of_the_road():
    t = generate_track(3, 1.0)
    d = np.linalg.norm(t.left[:, None] - t.centerline[None], axis=2)
    n = len(t.centerline)
    gap = np.abs(np.arange(n)[:, None] - np.arange(n)[None])
    far = np.minimum(gap, n - gap) > 20
    assert d[far].min() > t.width / 2


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_barriers_never_stand_on_the_road(difficulty):
    for seed in range(10):
        t = generate_track(seed, difficulty)
        for wall in (t.left_wall, t.right_wall):
            d = np.linalg.norm(wall[:, None] - t.centerline[None], axis=2)
            assert d.min() > t.width / 2 + 0.5


def test_frenet_and_locate():
    t = generate_track(1, 0.5)
    i = 100
    left_point = t.centerline[i] + 2.0 * np.array([-np.sin(t.heading[i]), np.cos(t.heading[i])])
    idx = locate(t, left_point)
    progress, lateral = frenet(t, left_point, idx)
    assert idx == i
    assert lateral == pytest.approx(2.0, abs=0.05)
    assert progress == pytest.approx(t.s[i] / t.length, abs=0.01)
    assert locate(t, left_point, last=i - 5) == i  # window search finds the same point


def test_lookahead_in_car_frame():
    t = generate_track(1, 0.0)
    i = 50
    pts = lookahead(t, i, t.centerline[i], t.heading[i], [5.0])
    assert pts[0, 0] == pytest.approx(5.0, abs=0.2)  # straight ahead
    assert abs(pts[0, 1]) < 1.0
