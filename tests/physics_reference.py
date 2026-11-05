"""Plain NumPy version of the per-step physics. Test-only reference for the Rust core."""

import numpy as np

DT = 0.05
WHEELBASE = 2.5
MAX_STEER = 0.5
STEER_RATE = 2.0
MAX_SPEED = 20.0
REVERSE_SPEED = 5.0
ACCEL = 6.0
BRAKE = 10.0
DRAG = 0.05
GRIP = 8.0
YAW_GRIP = 14.0
GRASS_DRAG = 0.8
GRASS_GRIP = 4.0

N_RAYS = 25
RANGE = 30.0
_u = np.arange(-(N_RAYS // 2), N_RAYS // 2 + 1) / (N_RAYS / 2)
ANGLES = np.pi * np.sign(_u) * np.abs(_u) ** 1.5


def step_car(state, action, on_grass=False):
    x, y, yaw, v, steer, v_lat = state
    steer_cmd = np.clip(action[0], -1, 1) * MAX_STEER
    steer += np.clip(steer_cmd - steer, -STEER_RATE * DT, STEER_RATE * DT)
    grip = GRASS_GRIP if on_grass else GRIP

    throttle = np.clip(action[1], -1, 1)
    braking = v * throttle < 0
    drag = DRAG + GRASS_DRAG * on_grass
    v_new = v + ((BRAKE if braking else ACCEL) * throttle - drag * v) * DT
    if braking and v_new * v < 0:
        v_new = 0.0
    v_new = np.clip(v_new, -REVERSE_SPEED, MAX_SPEED)

    tan_steer = np.tan(steer)
    if abs(v_new) > 1e-3:
        limit = YAW_GRIP * grip / GRIP * WHEELBASE / v_new**2
        tan_steer = np.clip(tan_steer, -limit, limit)
    dyaw = v_new * tan_steer / WHEELBASE * DT

    v_long = np.clip(v_new + dyaw * v_lat, -REVERSE_SPEED, MAX_SPEED)
    v_lat = v_lat - dyaw * v_new
    v_lat -= np.clip(v_lat, -grip * DT, grip * DT)

    yaw += dyaw
    x += (v_long * np.cos(yaw) - v_lat * np.sin(yaw)) * DT
    y += (v_long * np.sin(yaw) + v_lat * np.cos(yaw)) * DT
    return np.array([x, y, yaw, v_long, steer, v_lat])


def cast(track, pos, yaw):
    lines = (track.left, track.right, track.left_wall, track.right_wall)
    a = np.concatenate(lines)
    e = np.concatenate([np.roll(p, -1, axis=0) - p for p in lines])
    length = np.linalg.norm(e, axis=1)
    near = np.linalg.norm(a - pos, axis=1) < RANGE + length
    a, e = a[near], e[near]
    ang = yaw + ANGLES
    d = np.column_stack([np.cos(ang), np.sin(ang)])
    ao = a - pos
    denom = d[:, 0:1] * e[:, 1] - d[:, 1:2] * e[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ao[:, 0] * e[:, 1] - ao[:, 1] * e[:, 0]) / denom
        u = (ao[:, 0] * d[:, 1:2] - ao[:, 1] * d[:, 0:1]) / denom
    ok = (t > 0) & (u >= 0) & (u <= 1)
    return np.minimum(RANGE, np.where(ok, t, RANGE).min(axis=1, initial=RANGE))


def locate(track, pos, last=None, window=20):
    n = len(track.centerline)
    cand = np.arange(n) if last is None else (last + np.arange(-window, window + 1)) % n
    d = np.linalg.norm(track.centerline[cand] - pos, axis=1)
    return int(cand[np.argmin(d)])


def frenet(track, pos, idx):
    h = track.heading[idx]
    delta = pos - track.centerline[idx]
    along = delta[0] * np.cos(h) + delta[1] * np.sin(h)
    lateral = -delta[0] * np.sin(h) + delta[1] * np.cos(h)
    return ((track.s[idx] + along) % track.length) / track.length, lateral
