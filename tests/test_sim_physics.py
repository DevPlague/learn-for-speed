"""Every number stated in docs/PHYSICS.md, checked against the real code."""

import numpy as np
import pytest

from sim import car, lidar
from sim.env import MAX_TIME, OBS_DIM, SLIP_SCALE, RacingEnv
from sim.track import WALL_OFFSET, WIDTH, generate_track


def state(v=0.0, steer=0.0, v_lat=0.0):
    return np.array([0.0, 0.0, 0.0, v, steer, v_lat])


def run(start, action, steps, on_grass=False):
    s = start
    for _ in range(steps):
        s = car.step_car(s, action, on_grass)
    return s


def test_steering_reaches_full_lock_in_a_quarter_second():
    s = run(state(), [1, 0], 4)
    assert s[4] == pytest.approx(0.4)
    assert run(s, [1, 0], 1)[4] == pytest.approx(car.MAX_STEER)
    assert run(s, [1, 0], 5)[4] == pytest.approx(car.MAX_STEER)


def test_top_speeds():
    assert run(state(), [0, 1], 400)[3] == pytest.approx(20.0)
    assert run(state(), [0, 1], 800, on_grass=True)[3] == pytest.approx(7.06, abs=0.01)
    assert run(state(), [0, -1], 400)[3] == pytest.approx(-5.0)


def test_time_from_zero_to_top_speed():
    s, steps = state(), 0
    while s[3] < 19.99:
        s, steps = car.step_car(s, [0, 1]), steps + 1
    assert steps * car.DT == pytest.approx(3.65, abs=0.1)


def test_stopping_time_and_distance_with_drag():
    s, steps = state(20.0), 0
    while s[3] > 0:
        s, steps = car.step_car(s, [0, -1]), steps + 1
    assert steps * car.DT == pytest.approx(1.91, abs=0.06)
    assert s[0] == pytest.approx(18.2, abs=0.1)  # exact answer 18.8, less half a step of travel
    assert s[0] < 20.0  # the no drag estimate v^2 / 2B is an upper bound


def test_coasting_follows_the_drag_formula():
    assert run(state(10.0), [0, 0], 100)[3] == pytest.approx(10 * (1 - car.DRAG * car.DT) ** 100)


def test_brakes_stop_before_reverse_starts():
    s = car.step_car(state(0.2), [0, -1])
    assert s[3] == 0.0
    assert car.step_car(s, [0, -1])[3] < 0.0


def test_tightest_turn_radius():
    s = run(state(3.0, car.MAX_STEER), [1, 0], 1)
    assert 3.0 * (1 - car.DRAG * car.DT) / (s[2] / car.DT) == pytest.approx(4.58, abs=0.01)


def test_reverse_turns_the_other_way():
    assert run(state(-3.0, car.MAX_STEER), [1, -0.1], 5)[2] < 0.0


def test_yaw_limit_on_road_and_grass():
    for on_grass, limit in ((False, 14.0), (True, 7.0)):
        s = car.step_car(state(20.0, car.MAX_STEER), [1, 0], on_grass)
        v_new = 20.0 * (1 - (car.DRAG + car.GRASS_DRAG * on_grass) * car.DT)
        assert v_new * s[2] / car.DT == pytest.approx(limit)


@pytest.mark.parametrize("on_grass,limit", [(False, 6.05), (True, 4.28)])
def test_grip_ends_at_the_stated_speed_at_full_lock(on_grass, limit):
    # drag acts first in the step, so start slightly above the speed of interest
    assert car.step_car(state(limit * 0.98, car.MAX_STEER), [1, 0], on_grass)[5] == 0.0
    assert car.step_car(state(limit * 1.05, car.MAX_STEER), [1, 0], on_grass)[5] < 0.0


def test_nose_reaches_its_limit_at_8_mps():
    def lateral_acc(v):
        s = car.step_car(state(v, car.MAX_STEER), [1, 0])
        return v * (1 - car.DRAG * car.DT) * s[2] / car.DT

    assert lateral_acc(7.9) < 14.0
    assert lateral_acc(8.2) == pytest.approx(14.0)


@pytest.mark.parametrize("radius,speed", [(15.0, 11.0), (30.0, 15.5)])
def test_corner_speed_without_sliding(radius, speed):
    assert np.sqrt(car.GRIP * radius) == pytest.approx(speed, abs=0.05)
    steer = np.arctan(car.WHEELBASE / radius)
    assert car.step_car(state(speed * 0.98, steer), [steer / car.MAX_STEER, 0])[5] == 0.0
    assert car.step_car(state(speed * 1.05, steer), [steer / car.MAX_STEER, 0])[5] < 0.0


def test_gripping_corner_loses_only_drag():
    end = run(state(5.0, car.MAX_STEER), [1, 0], 60)
    assert end[5] == 0.0
    assert end[3] == pytest.approx(5.0 * (1 - car.DRAG * car.DT) ** 60)


def test_slip_growth_is_the_difference_of_limits():
    assert run(state(20.0, car.MAX_STEER), [1, 0], 20)[5] == pytest.approx(-6.0, abs=0.1)
    # throttle holds the speed, grass drag would otherwise drop it below the grip speed
    grass = run(state(7.0, car.MAX_STEER), [1, 1], 20, on_grass=True)
    assert grass[5] == pytest.approx(-3.0, abs=0.1)


def test_left_turn_slides_to_the_outside():
    assert car.step_car(state(20.0, car.MAX_STEER), [1, 0])[5] < 0.0
    assert car.step_car(state(20.0, -car.MAX_STEER), [-1, 0])[5] > 0.0


def test_slip_recovers_at_the_grip_rate():
    assert run(state(10.0, 0.0, -6.0), [0, 0], 14)[5] < 0.0  # 0.70 s
    assert run(state(10.0, 0.0, -6.0), [0, 0], 15)[5] == 0.0  # 0.75 s


def test_position_includes_the_slip():
    s = np.array([0.0, 0.0, np.pi / 2, 10.0, 0.0, 1.0])  # pointing up, sliding to the left
    x, y = car.step_car(s, [0, 0])[:2]
    assert x == pytest.approx(-(1.0 - car.GRIP * car.DT) * car.DT)  # friction acts first
    assert y == pytest.approx(10.0 * (1 - car.DRAG * car.DT) * car.DT)


def test_speed_cap_and_total_speed_after_counter_steer():
    s = run(state(20.0, car.MAX_STEER), [1, 1], 25)
    for _ in range(40):
        before = np.hypot(s[3], s[5])
        s = car.step_car(s, [-1, 0])
        assert -car.REVERSE_SPEED <= s[3] <= car.MAX_SPEED
        assert np.hypot(s[3], s[5]) <= before * 1.004  # guide says under 0.4 percent


def spin(throttle, steps=1200):
    s, peak, top = state(20.0, car.MAX_STEER), 0.0, 0.0
    for _ in range(steps):
        s = car.step_car(s, [1, throttle])
        peak, top = max(peak, abs(s[5])), max(top, np.hypot(s[3], s[5]))
    return s, peak, top


def test_spin_with_full_throttle_settles_into_a_circle():
    s, peak, top = spin(1.0)
    assert peak == pytest.approx(18.9, abs=0.1)
    assert peak < SLIP_SCALE
    assert s[3] == pytest.approx(5.8, abs=0.05) and s[5] == pytest.approx(-4.3, abs=0.05)
    assert top == pytest.approx(22.0, abs=0.3)  # the cap acts on the forward speed only


def test_spin_without_throttle_dies_out():
    s, peak, _ = spin(0.0)
    assert peak == pytest.approx(11.7, abs=0.1)
    assert s[5] == 0.0
    assert np.degrees(s[2]) == pytest.approx(240, abs=5)


def test_lidar_layout():
    degrees = np.degrees(lidar.ANGLES)
    assert lidar.N_RAYS == 25 and lidar.RANGE == 30.0
    assert degrees[12] == 0.0
    assert degrees[13] == pytest.approx(4.07, abs=0.01)
    assert degrees[-1] == pytest.approx(169.3, abs=0.1)
    assert 360 - 2 * degrees[-1] == pytest.approx(21.4, abs=0.2)
    assert np.allclose(degrees, -degrees[::-1])
    assert (np.diff(degrees) > 0).all()


def test_track_constants_and_validity_rules():
    assert WIDTH == 8.0 and WALL_OFFSET == 12.0
    for difficulty in (0.0, 0.5, 1.0):
        for seed in range(10):
            t = generate_track(seed, difficulty)
            turn = np.abs(np.angle(np.exp(1j * np.diff(t.heading, append=t.heading[0]))))
            seg = np.linalg.norm(np.roll(t.centerline, -1, axis=0) - t.centerline, axis=1)
            assert (seg / np.maximum(turn, 1e-9)).min() >= WALL_OFFSET
            n = len(t.centerline)
            idx = np.arange(n)
            gap = np.abs(idx[:, None] - idx[None])
            dist = np.linalg.norm(t.centerline[:, None] - t.centerline[None], axis=2)
            assert dist[np.minimum(gap, n - gap) > 32].min() > 17


def test_observation_layout():
    env = RacingEnv(track_seed=1000)
    obs, _ = env.reset(seed=0)
    assert OBS_DIM == 28
    env.state = np.array([*env.state[:3], 10.0, 0.25, -5.0])
    obs = env._obs()
    assert obs[25:] == pytest.approx([0.5, 0.5, -0.25])
    assert obs[:25] == pytest.approx(2 * env.hits / lidar.RANGE - 1)


def test_search_window_finds_the_nearest_point():
    rng = np.random.default_rng(0)
    for seed in range(1000, 1010):
        env = RacingEnv(track_seed=seed, difficulty=1.0)
        env.reset(seed=0)
        done = False
        while not done:
            if env.steps % 15 == 0:
                action = rng.uniform([-1, 0.3], [1, 1])
            *_, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            assert env.idx == env.core.locate(*env.state[:2])


def test_end_rules():
    assert MAX_TIME == 120.0
    env = RacingEnv(track_seed=1000)
    env.reset(seed=0)
    for step in range(1, 400):
        _, _, terminated, _, info = env.step([0, 0])
        if terminated:
            break
    assert info["stuck"] and step * car.DT == pytest.approx(10.0)


def test_barrier_overshoot_is_small():
    worst = 0.0
    for sign in (1, -1):
        env = RacingEnv(track_seed=1000, difficulty=0.5)
        env.reset(seed=0)
        lateral, terminated = 0.0, False
        for _ in range(100):
            env.step([0, 0.8])
        while not terminated:
            _, _, terminated, _, info = env.step([sign * 0.25, 1])
            lateral = env.core.frenet(*env.state[:2], env.idx)[1]
        assert info["off_road"]
        worst = max(worst, abs(lateral) - WALL_OFFSET)
    assert worst < 1.1  # one step of travel at the highest total speed
