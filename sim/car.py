"""Car state and constants for Python. The physics is in rust/src/car.rs, see docs/PHYSICS.md.

State is [x, y, yaw, v, steer, v_lat]. v is the forward speed (v < 0 is reverse) and v_lat the
sideways slip speed in the car frame (positive is to the left).
"""

import numpy as np
import racing_core
from racing_core import (
    ACCEL,
    BRAKE,
    DRAG,
    DT,
    GRASS_DRAG,
    GRASS_GRIP,
    GRIP,
    MAX_SPEED,
    MAX_STEER,
    REVERSE_SPEED,
    STEER_RATE,
    WHEELBASE,
    YAW_GRIP,
)

__all__ = [
    "ACCEL",
    "BRAKE",
    "DRAG",
    "DT",
    "GRASS_DRAG",
    "GRASS_GRIP",
    "GRIP",
    "MAX_SPEED",
    "MAX_STEER",
    "REVERSE_SPEED",
    "STEER_RATE",
    "WHEELBASE",
    "YAW_GRIP",
    "initial_state",
    "step_car",
]


def initial_state(x, y, yaw):
    return np.array([x, y, yaw, 0.0, 0.0, 0.0])


def step_car(state, action, on_grass=False):
    """Advance one DT. action = [steer, throttle/brake] in [-1, 1]."""
    state, action = np.asarray(state, np.float64), np.asarray(action, np.float64)
    return racing_core.step_car(state, action, on_grass)
