"""Bicycle car that grips in slow corners and drifts in fast ones.

State is [x, y, yaw, v, steer, v_lat]. v is the forward speed (v < 0 is reverse) and v_lat the
sideways slip speed in the car frame (positive is to the left).
"""

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
GRIP = 8.0  # max sideways acceleration the tires hold, m/s^2
YAW_GRIP = 14.0  # the nose may turn this hard, past GRIP the car slides
GRASS_DRAG = 0.8  # extra drag on grass, caps speed near 7 m/s
GRASS_GRIP = 4.0


def initial_state(x, y, yaw):
    return np.array([x, y, yaw, 0.0, 0.0, 0.0])


def step_car(state, action, on_grass=False):
    """Advance one DT. action = [steer, throttle/brake] in [-1, 1].

    Pushing against the motion brakes. From standstill a negative throttle drives backwards.
    """
    x, y, yaw, v, steer, v_lat = state
    steer_cmd = np.clip(action[0], -1, 1) * MAX_STEER
    steer += np.clip(steer_cmd - steer, -STEER_RATE * DT, STEER_RATE * DT)
    grip = GRASS_GRIP if on_grass else GRIP

    throttle = np.clip(action[1], -1, 1)
    braking = v * throttle < 0
    drag = DRAG + GRASS_DRAG * on_grass
    v_new = v + ((BRAKE if braking else ACCEL) * throttle - drag * v) * DT
    if braking and v_new * v < 0:
        v_new = 0.0  # brakes stop the car, reverse starts on the next step
    v_new = np.clip(v_new, -REVERSE_SPEED, MAX_SPEED)

    tan_steer = np.tan(steer)
    if abs(v_new) > 1e-3:
        limit = YAW_GRIP * grip / GRIP * WHEELBASE / v_new**2
        tan_steer = np.clip(tan_steer, -limit, limit)
    dyaw = v_new * tan_steer / WHEELBASE * DT

    # the body turns but the velocity keeps its direction, so part of it becomes sideways slip
    c, s = np.cos(dyaw), np.sin(dyaw)
    v_long = v_new * c + v_lat * s
    v_lat = -v_new * s + v_lat * c
    # tires pull the slip back to zero, at most at the grip limit
    v_lat -= np.clip(v_lat, -grip * DT, grip * DT)

    yaw += dyaw
    x += (v_long * np.cos(yaw) - v_lat * np.sin(yaw)) * DT
    y += (v_long * np.sin(yaw) + v_lat * np.cos(yaw)) * DT
    return np.array([x, y, yaw, v_long, steer, v_lat])
