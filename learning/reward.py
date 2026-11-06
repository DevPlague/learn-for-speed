"""Training reward wrapper"""

from dataclasses import dataclass

import gymnasium as gym


@dataclass(frozen=True)
class RewardWeights:
    progress: float = 10.0  # per lap of progress along the centerline
    time: float = 0.01
    jitter: float = 0.05  # per unit of steering change between steps
    grass: float = 0.05  # per step on grass
    off_road: float = 5.0  # once, when the attempt fails off road
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
            info["is_success"] = info["lap_done"]  # read by EvalCallback as success rate
        return obs, reward, terminated, truncated, info

    def _reward(self, dprogress, dsteer, info):
        w = self.weights
        return (
            w.progress * dprogress
            - w.time
            - w.jitter * dsteer
            - w.grass * info["on_grass"]
            - w.off_road * info["off_road"]
            + w.lap * info["lap_done"]
        )
