"""Kinematic bicycle car with a grip limit. State is [x, y, yaw, v, steer]."""

import numpy as np

DT = 0.05
WHEELBASE = 2.5
MAX_STEER = 0.5  # rad
STEER_RATE = 2.0  # rad/s
MAX_SPEED = 20.0
ACCEL = 6.0  # m/s^2 at full throttle
BRAKE = 10.0  # m/s^2 at full brake
DRAG = 0.05  # 1/s
GRIP = 8.0  # max lateral acceleration, m/s^2


def initial_state(x, y, yaw):
    return np.array([x, y, yaw, 0.0, 0.0])


def step_car(state, action):
    """Advance one DT. action = [steer, throttle/brake] in [-1, 1]."""
    x, y, yaw, v, steer = state
    steer_cmd = np.clip(action[0], -1, 1) * MAX_STEER
    steer += np.clip(steer_cmd - steer, -STEER_RATE * DT, STEER_RATE * DT)

    throttle = np.clip(action[1], -1, 1)
    acc = (ACCEL if throttle > 0 else BRAKE) * throttle - DRAG * v
    v_new = np.clip(v + acc * DT, 0.0, MAX_SPEED)

    # past the grip limit the car understeers, so it turns less than the wheels point
    tan_steer = np.tan(steer)
    if v_new > 1e-3:
        limit = GRIP * WHEELBASE / v_new**2
        tan_steer = np.clip(tan_steer, -limit, limit)
    yaw += v_new * tan_steer / WHEELBASE * DT
    x += v_new * np.cos(yaw) * DT
    y += v_new * np.sin(yaw) * DT
    return np.array([x, y, yaw, v_new, steer])
