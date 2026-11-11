import numpy as np
import pytest

from sim.track import (
    WALL_OFFSET,
    _ease_tight_corners,
    _radius,
    _spline_loop,
    generate_track,
    lookahead,
    make_core,
)

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
    core = make_core(t)
    idx = core.locate(*left_point)
    progress, lateral = core.frenet(*left_point, idx)
    assert idx == i
    assert lateral == pytest.approx(2.0, abs=0.05)
    assert progress == pytest.approx(t.s[i] / t.length, abs=0.01)
    assert core.locate(*left_point, last=i - 5) == i  # window search finds the same point


def test_lookahead_in_car_frame():
    t = generate_track(1, 0.0)
    i = 50
    pts = lookahead(t, i, t.centerline[i], t.heading[i], [5.0])
    assert pts[0, 0] == pytest.approx(5.0, abs=0.2)  # straight ahead
    assert abs(pts[0, 1]) < 1.0
    left = t.centerline[i] + 3.0 * np.array([-np.sin(t.heading[i]), np.cos(t.heading[i])])
    assert lookahead(t, i, left, t.heading[i], [0.0])[0, 1] < -2.5  # centre is to the right


def turning(difficulty):
    """Total absolute turning in laps and share of points tighter than 25 m, over 10 tracks."""
    totals, tight = [], []
    for seed in range(1000, 1010):
        t = generate_track(seed, difficulty)
        totals.append(np.abs(np.angle(np.exp(1j * np.diff(t.heading)))).sum() / (2 * np.pi))
        tight.append((_radius(t.centerline) < 25).mean())
    return np.mean(totals), np.mean(tight)


def test_harder_tracks_have_more_and_tighter_corners():
    stats = [turning(d) for d in (0.0, 0.5, 1.0)]
    assert stats[0][0] < stats[1][0] < stats[2][0] and stats[0][1] < stats[1][1] < stats[2][1]


def total_turn(loop):
    step = np.roll(loop, -1, axis=0) - loop
    heading = np.arctan2(step[:, 1], step[:, 0])
    return np.angle(np.exp(1j * np.diff(heading, append=heading[0]))).sum()


def test_easing_opens_tight_corners_and_keeps_the_loop():
    # hardest raw splines, most break the 12 m rule, the few that still do are redrawn
    eased = tight = 0
    for seed in range(1000, 1040):
        raw = _spline_loop(np.random.default_rng(seed), 12, 0.5)
        if _radius(raw).min() >= WALL_OFFSET:
            continue
        tight += 1
        out = _ease_tight_corners(raw)
        eased += _radius(out).min() >= WALL_OFFSET
        assert total_turn(out) == pytest.approx(2 * np.pi)
        assert len(out) == pytest.approx(len(raw), rel=0.15)  # same loop, wider corners
    assert tight >= 35 and eased >= 0.9 * tight
