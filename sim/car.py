"""Kinematic bicycle car with a grip limit. State is [x, y, yaw, v, steer], v < 0 is reverse."""

import numpy as np

DT = 0.05
WHEELBASE = 2.5
MAX_STEER = 0.5  # rad
STEER_RATE = 2.0  # rad/s
MAX_SPEED = 20.0
REVERSE_SPEED = 5.0
ACCEL = 6.0  # m/s^2 at full throttle
BRAKE = 10.0  # m/s^2 at full brake
DRAG = 0.05  # 1/s
GRIP = 8.0  # max lateral acceleration, m/s^2
GRASS_DRAG = 0.8  # extra drag on grass, caps speed near 7 m/s
GRASS_GRIP = 4.0


def initial_state(x, y, yaw):
    return np.array([x, y, yaw, 0.0, 0.0])


def step_car(state, action, on_grass=False):
    """Advance one DT. action = [steer, throttle/brake] in [-1, 1].

    Pushing against the motion brakes. From standstill a negative throttle drives backwards.
    """
    x, y, yaw, v, steer = state
    steer_cmd = np.clip(action[0], -1, 1) * MAX_STEER
    steer += np.clip(steer_cmd - steer, -STEER_RATE * DT, STEER_RATE * DT)

    throttle = np.clip(action[1], -1, 1)
    braking = v * throttle < 0
    drag = DRAG + GRASS_DRAG * on_grass
    v_new = v + ((BRAKE if braking else ACCEL) * throttle - drag * v) * DT
    if braking and v_new * v < 0:
        v_new = 0.0  # brakes stop the car, reverse starts on the next step
    v_new = np.clip(v_new, -REVERSE_SPEED, MAX_SPEED)

    # past the grip limit the car understeers, so it turns less than the wheels point
    tan_steer = np.tan(steer)
    if abs(v_new) > 1e-3:
        limit = (GRASS_GRIP if on_grass else GRIP) * WHEELBASE / v_new**2
        tan_steer = np.clip(tan_steer, -limit, limit)
    yaw += v_new * tan_steer / WHEELBASE * DT
    x += v_new * np.cos(yaw) * DT
    y += v_new * np.sin(yaw) * DT
    return np.array([x, y, yaw, v_new, steer])
