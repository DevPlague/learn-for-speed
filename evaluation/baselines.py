"""Baseline policies. Each one is a callable that maps an observation to an action in [-1, 1]."""

import numpy as np

from sim import car
from sim.track import lookahead


class RandomPolicy:
    """Uniform random actions, seeded for reproducibility."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs):
        return self.rng.uniform(-1.0, 1.0, 2).astype(np.float32)


class PurePursuitPolicy:
    """Steers toward a centerline point ahead and slows down before sharp corners.

    It reads the track and car state from the env, which a trained agent never sees.
    """

    def __init__(self, env, lookahead_dist=10.0, max_speed=16.0, min_speed=6.0, gain=0.5):
        self.env = env
        self.lookahead_dist = lookahead_dist
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.gain = gain
        self.preview = np.array([10.0, 20.0, 30.0])  # meters ahead to sense corners

    def __call__(self, obs):
        env = self.env
        pos, yaw, v = env.state[:2], env.state[2], env.state[3]
        x, y = lookahead(env.track, env.idx, pos, yaw, [self.lookahead_dist])[0]
        alpha = np.arctan2(y, x)
        steer = np.arctan(2 * car.WHEELBASE * np.sin(alpha) / max(np.hypot(x, y), 1e-6))

        # Curvature from the heading change between the car and points ahead
        pts = lookahead(env.track, env.idx, pos, yaw, self.preview)
        turn = np.abs(np.arctan2(pts[:, 1], pts[:, 0]))
        curvature = np.max(2 * np.sin(turn) / self.preview)
        target = np.sqrt(car.GRIP / max(curvature, 1e-6))
        target = np.clip(target, self.min_speed, self.max_speed)
        throttle = np.clip(self.gain * (target - v), -1.0, 1.0)
        return np.array([np.clip(steer / car.MAX_STEER, -1, 1), throttle], dtype=np.float32)
