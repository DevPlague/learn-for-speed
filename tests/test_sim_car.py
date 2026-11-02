import numpy as np
import pytest

from sim import car, lidar
from sim.track import Track


def run(action, steps, state=None):
    state = car.initial_state(0, 0, 0) if state is None else state
    for _ in range(steps):
        state = car.step_car(state, action)
    return state


def test_straight_line_accelerates_and_caps_speed():
    state = run([0, 1], 20)
    assert 0 < state[3] < car.MAX_SPEED
    assert state[0] > 0 and state[1] == pytest.approx(0)
    assert run([0, 1], 600)[3] == pytest.approx(car.MAX_SPEED, abs=0.5)


def test_brake_stops_without_reversing():
    fast = np.array([0.0, 0.0, 0.0, 10.0, 0.0])
    assert run([0, -1], 100, fast)[3] == 0.0


def test_steering_turns_left_for_positive_action():
    assert run([1, 0.3], 40)[2] > 0.1


def test_grip_limits_lateral_acceleration():
    state = np.array([0.0, 0.0, 0.0, 18.0, car.MAX_STEER])
    after = car.step_car(state, [1, 0])
    lateral_acc = after[3] * (after[2] - state[2]) / car.DT
    assert lateral_acc <= car.GRIP + 1e-6


def straight_corridor():
    x = np.linspace(-100, 100, 201)
    zeros = np.zeros_like(x)
    return Track(
        centerline=np.column_stack([x, zeros]),
        width=8.0,
        left=np.column_stack([x, zeros + 4]),
        right=np.column_stack([x, zeros - 4]),
        s=x + 100,
        heading=zeros,
        length=200.0,
    )


def test_lidar_in_a_corridor():
    hits = lidar.cast(straight_corridor(), np.array([0.0, 0.0]), 0.0)
    assert hits[lidar.N_RAYS // 2] == lidar.RANGE  # straight ahead sees nothing in range
    assert hits[0] == pytest.approx(4.0)  # +-90 degrees hit the walls at half width
    assert hits[-1] == pytest.approx(4.0)
    assert hits.min() >= 4.0 - 1e-9
