"""The Rust core must give the same numbers as the plain NumPy reference."""

import numpy as np
import physics_reference as ref
import pytest

from sim import car, lidar
from sim.track import generate_track, make_core

TOL = 1e-9


def test_constants_match():
    for name in car.__all__[:13]:
        assert getattr(car, name) == getattr(ref, name), name
    assert lidar.N_RAYS == ref.N_RAYS and lidar.RANGE == ref.RANGE
    assert np.allclose(lidar.ANGLES, ref.ANGLES, atol=TOL)


def test_car_step_on_random_states():
    rng = np.random.default_rng(0)
    low = [-200, -200, -10, -5, -0.5, -15]
    high = [200, 200, 10, 20, 0.5, 15]
    for _ in range(5000):
        state = rng.uniform(low, high)
        action = rng.uniform(-1.2, 1.2, 2)
        on_grass = bool(rng.integers(2))
        rust = car.step_car(state, action, on_grass)
        assert rust == pytest.approx(ref.step_car(state, action, on_grass), abs=TOL)


def test_braking_edge_cases():
    for v in (0.0, 1e-4, -1e-4, 0.2, -0.2, 20.0, -5.0):
        for throttle in (-1.0, 0.0, 1.0):
            state = np.array([0.0, 0.0, 0.0, v, 0.1, 0.0])
            rust = car.step_car(state, [0.3, throttle])
            assert rust == pytest.approx(ref.step_car(state, [0.3, throttle]), abs=TOL)


@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_full_trajectory_with_lidar_and_queries(difficulty):
    track = generate_track(1000, difficulty)
    core = make_core(track)
    rng = np.random.default_rng(1)
    state = ref_state = car.initial_state(*track.centerline[0], track.heading[0])
    idx = ref_idx = 0
    on_grass = False
    for step in range(1500):
        # held steering so the car visits road, grass and drift
        if step % 20 == 0:
            action = np.array([rng.uniform(-0.6, 0.6), rng.uniform(0.2, 1.0)])
        state, hits, idx, frac, lateral = core.step(state, action, idx, on_grass)
        ref_state = ref.step_car(ref_state, action, on_grass)
        ref_idx = ref.locate(track, ref_state[:2], ref_idx)
        ref_frac, ref_lateral = ref.frenet(track, ref_state[:2], ref_idx)
        assert state == pytest.approx(ref_state, abs=TOL)
        assert idx == ref_idx
        assert frac == pytest.approx(ref_frac, abs=TOL)
        assert lateral == pytest.approx(ref_lateral, abs=TOL)
        assert hits == pytest.approx(ref.cast(track, ref_state[:2], ref_state[2]), abs=TOL)
        on_grass = abs(lateral) > track.width / 2


def test_full_search_and_wrap_around():
    track = generate_track(3, 0.5)
    core = make_core(track)
    n = len(track.centerline)
    rng = np.random.default_rng(2)
    for _ in range(300):
        pos = track.centerline[rng.integers(n)] + rng.uniform(-14, 14, 2)
        assert core.locate(*pos) == ref.locate(track, pos)
        last = int(rng.integers(n))
        assert core.locate(*pos, last=last) == ref.locate(track, pos, last)


def test_wrong_shapes_raise():
    with pytest.raises(ValueError):
        car.step_car(np.zeros(5), [0, 0])
