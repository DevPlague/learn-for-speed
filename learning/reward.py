"""Training reward, a wrapper around the env so sim/ keeps its own placeholder."""

from dataclasses import dataclass

import gymnasium as gym

from sim import car


@dataclass(frozen=True)
class RewardWeights:
    progress: float = 10.0  # per lap of progress along the centerline
    time: float = 0.01
    jitter: float = 0.01  # per unit of steering change between steps
    grass: float = 0.05  # per step on grass
    idle: float = 0.03  # per step with forward speed under 1 m/s
    crawl: float = 0.0  # per step gaining under 2 m/s along the track, creeping and circling too
    off_road: float = 5.0  # once, when the attempt fails off road
    stuck: float = 10.0  # once, when the attempt fails stuck, worse than crashing
    lap: float = 5.0  # once, when the lap is done


DEFAULT_WEIGHTS = RewardWeights()


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
        along = dprogress * self.env.unwrapped.track.length / car.DT  # m/s along the centerline
        return (
            w.progress * dprogress
            - w.time
            - w.jitter * dsteer
            - w.grass * info["on_grass"]
            - w.idle * (abs(info["speed"]) < 1.0)
            - w.crawl * (along < 2.0)
            - w.off_road * info["off_road"]
            - w.stuck * info["stuck"]
            + w.lap * info["lap_done"]
        )
