"""Baseline policies. Each one is a callable that maps an observation to an action in [-1, 1]."""

import numpy as np

from sim import car
from sim.track import lookahead, pursuit_steer


class RandomPolicy:
    """Uniform random actions, seeded for reproducibility."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs):
        return self.rng.uniform(-1.0, 1.0, 2).astype(np.float32)


class PurePursuitPolicy:
    """Steers toward the centerline 10 m ahead and slows down before sharp corners.

    It reads the track and car state from the env, which a trained agent never sees.
    """

    def __init__(self, env):
        self.env = env

    def __call__(self, obs):
        env = self.env
        preview = np.array([10.0, 20.0, 30.0])
        pts = lookahead(env.track, env.idx, env.state[:2], env.state[2], preview)
        # curvature of the arc to each preview point, the sharpest sets the speed the tires hold
        curvature = np.max(2 * np.sin(np.abs(np.arctan2(pts[:, 1], pts[:, 0]))) / preview)
        target = np.clip(np.sqrt(car.GRIP / max(curvature, 1e-6)), 6.0, 16.0)
        throttle = np.clip(0.5 * (target - env.state[3]), -1.0, 1.0)
        return np.array([pursuit_steer(pts[0]), throttle], dtype=np.float32)
