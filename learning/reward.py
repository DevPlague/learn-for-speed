"""Training reward, a wrapper around the env so sim/ keeps its own placeholder."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from sim import car, lidar
from sim.track import SPACING, _radius


@dataclass(frozen=True)
class RewardWeights:
    progress: float = 10.0  # per lap of progress along the centerline
    time: float = 0.01
    jitter: float = 0.01  # per unit of steering change between steps
    grass: float = 0.05  # per step on grass
    idle: float = 0.03  # per step with forward speed under 1 m/s
    crawl: float = 0.0  # per step gaining under 2 m/s along the track, creeping and circling too
    overspeed: float = 0.0  # per step, per MAX_SPEED above the speed it could still brake from
    off_road: float = 5.0  # once, when the attempt fails off road
    stuck: float = 10.0  # once, when the attempt fails stuck, worse than crashing
    lap: float = 5.0  # once, when the lap is done


DEFAULT_WEIGHTS = RewardWeights()


def allowed_speed(track):
    """Fastest speed at each centerline point from which the car can still brake in time for
    every corner within lidar range. A corner of radius R holds sqrt(GRIP * R) and the car may
    use half the road, so R grows by width / 2. Braking over d meters loses v^2 - 2 BRAKE d.
    """
    curvature = 1 / _radius(track.centerline)
    curvature = np.convolve(np.pad(curvature, 3, mode="wrap"), np.ones(7) / 7, "valid")
    corner = np.minimum(np.sqrt(car.GRIP * (1 / curvature + track.width / 2)), car.MAX_SPEED)
    ahead = np.arange(round(lidar.RANGE / SPACING) + 1)
    points = (np.arange(len(corner))[:, None] + ahead) % len(corner)
    return np.sqrt(corner[points] ** 2 + 2 * car.BRAKE * ahead * SPACING).min(axis=1)


class RaceReward(gym.Wrapper):
    def __init__(self, env, weights=DEFAULT_WEIGHTS):
        super().__init__(env)
        self.weights = weights
        self._progress = 0.0
        self._steer = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._progress = self.env.unwrapped.progress
        self._steer = 0.0
        if self.weights.overspeed:
            self._allowed = allowed_speed(self.env.unwrapped.track)
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        progress = self.env.unwrapped.progress
        steer = float(action[0])
        reward = self._reward(progress - self._progress, abs(steer - self._steer), info)
        self._progress, self._steer = progress, steer

        if terminated or truncated:
            info["is_success"] = info["lap_done"]  # SB3 logs it as rollout/success_rate
        return obs, reward, terminated, truncated, info

    def _reward(self, dprogress, dsteer, info):
        w = self.weights
        env = self.env.unwrapped
        along = dprogress * env.track.length / car.DT  # m/s along the centerline
        over = 0.0
        if w.overspeed:
            over = max(0.0, info["speed"] - self._allowed[env.idx]) / car.MAX_SPEED
        return (
            w.progress * dprogress
            - w.time
            - w.jitter * dsteer
            - w.grass * info["on_grass"]
            - w.idle * (abs(info["speed"]) < 1.0)
            - w.crawl * (along < 2.0)
            - w.overspeed * over
            - w.off_road * info["off_road"]
            - w.stuck * info["stuck"]
            + w.lap * info["lap_done"]
        )
