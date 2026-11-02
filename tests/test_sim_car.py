import numpy as np
import pytest

from sim import car, lidar
from sim.track import Track


def run(action, steps, state=None, on_grass=False):
    state = car.initial_state(0, 0, 0) if state is None else state
    for _ in range(steps):
        state = car.step_car(state, action, on_grass)
    return state


def test_straight_line_accelerates_and_caps_speed():
    state = run([0, 1], 20)
    assert 0 < state[3] < car.MAX_SPEED
    assert state[0] > 0 and state[1] == pytest.approx(0)
    assert run([0, 1], 600)[3] == pytest.approx(car.MAX_SPEED, abs=0.5)


def test_brakes_to_a_stop_then_reverses():
    state = np.array([0.0, 0.0, 0.0, 10.0, 0.0])
    speeds = []
    for _ in range(200):
        state = car.step_car(state, [0, -1])
        speeds.append(state[3])
    assert 0.0 in speeds  # stops before going backwards
    assert speeds[-1] == pytest.approx(-car.REVERSE_SPEED)
    assert state[0] < max(run([0, -1], i, np.array([0.0, 0, 0, 10, 0]))[0] for i in range(40))


def test_steering_turns_left_for_positive_action():
    assert run([1, 0.3], 40)[2] > 0.1


def test_reverse_steering_turns_the_other_way():
    assert run([1, -0.3], 40)[2] < -0.05


def test_grip_limits_lateral_acceleration():
    state = np.array([0.0, 0.0, 0.0, 18.0, car.MAX_STEER])
    after = car.step_car(state, [1, 0])
    lateral_acc = after[3] * (after[2] - state[2]) / car.DT
    assert lateral_acc <= car.GRIP + 1e-6


def test_grass_is_slow():
    assert run([0, 1], 400, on_grass=True)[3] < 8.0 < run([0, 1], 400)[3]


def straight_corridor():
    x = np.linspace(-100, 100, 201)
    zeros = np.zeros_like(x)
    return Track(
        centerline=np.column_stack([x, zeros]),
        width=8.0,
        left=np.column_stack([x, zeros + 4]),
        right=np.column_stack([x, zeros - 4]),
        left_wall=np.column_stack([x, zeros + 12]),
        right_wall=np.column_stack([x, zeros - 12]),
        s=x + 100,
        heading=zeros,
        length=200.0,
    )


def test_lidar_in_a_corridor():
    segs = lidar.segments(straight_corridor())
    on_road = lidar.cast(segs, np.array([0.0, 0.0]), 0.0)
    expected = np.minimum(lidar.RANGE, 4.0 / np.maximum(np.abs(np.sin(lidar.ANGLES)), 1e-9))
    assert np.allclose(on_road, expected)
    assert on_road[lidar.N_RAYS // 2] == lidar.RANGE  # straight ahead sees nothing in range


def test_lidar_sees_the_barrier_from_the_grass():
    segs = lidar.segments(straight_corridor())
    on_grass = lidar.cast(segs, np.array([0.0, 8.0]), np.pi / 2)  # facing the left barrier
    assert on_grass[lidar.N_RAYS // 2] == pytest.approx(4.0)
