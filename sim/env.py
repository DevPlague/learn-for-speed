"""Gymnasium environment. One step = one car step, lidar, track queries and reward.

The agent only observes its own sensors (lidar, speed, steering, slip). The track geometry is
used for progress, failure checks and reward, never as an observation.
"""

from typing import ClassVar

import gymnasium as gym
import numpy as np

from sim import car, lidar
from sim.track import WALL_OFFSET, generate_track, make_core

OBS_DIM = lidar.N_RAYS + 3
SLIP_SCALE = car.MAX_SPEED  # m/s of sideways slip that maps to 1 in the observation

MAX_TIME = 120.0  # s, then truncated
GRASS_TIME = 10.0  # s in a row on grass, then failed
TRAIN_SEED_MIN = 100_000


def default_reward(dprogress, on_grass, off_road, lap_done):
    """Reward of the bare env. Training replaces it with learning.reward.RaceReward."""
    return 10.0 * dprogress - 0.01 - 0.05 * on_grass - 5.0 * off_road + 5.0 * lap_done


def outcome(info):
    """How a finished episode ended, lap, off_road, stuck or timeout."""
    if info["lap_done"]:
        return "lap"
    if info["off_road"]:
        return "off_road"
    return "stuck" if info["stuck"] else "timeout"


class RacingEnv(gym.Env):
    metadata: ClassVar[dict] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": round(1 / car.DT),
    }

    def __init__(self, track_seed=None, difficulty=0.0, render_mode=None):
        self.track_seed = track_seed
        self.difficulty = difficulty  # curriculum may change it between episodes
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.track_seed is None:
            seed_ = TRAIN_SEED_MIN + int(self.np_random.integers(0, 1_000_000))
        else:
            seed_ = self.track_seed
        self.track = generate_track(seed_, self.difficulty)
        t = self.track
        self.core = make_core(t)
        self.state = car.initial_state(*t.centerline[0], t.heading[0])  # on the start line
        self.idx = 0
        self.steps = 0
        self.progress = 0.0  # cumulative laps, can go negative
        self.last_frac = 0.0
        self.ref_progress = 0.0
        self.ref_step = 0
        self.on_grass = False
        self.grass_steps = 0
        self.lateral = 0.0
        self.hits = self.core.cast(*self.state[:3])
        return self._obs(), self._info(False, False, False)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self.state, self.hits, self.idx, frac, self.lateral = self.core.step(
            self.state, action, self.idx, self.on_grass
        )
        self.steps += 1
        t = self.track
        dprogress = (frac - self.last_frac + 0.5) % 1.0 - 0.5  # unwrap across the start line
        self.last_frac = frac
        self.progress += dprogress

        self.on_grass = abs(self.lateral) > t.width / 2
        self.grass_steps = self.grass_steps + 1 if self.on_grass else 0
        hit_wall = abs(self.lateral) > WALL_OFFSET
        off_road = hit_wall or self.grass_steps * car.DT > GRASS_TIME
        lap_done = self.progress >= 1.0
        if self.progress - self.ref_progress >= 2.0 / t.length:  # 2 m gained resets the clock
            self.ref_progress, self.ref_step = self.progress, self.steps
        stuck = (self.steps - self.ref_step) * car.DT >= 10.0
        terminated = bool(off_road or lap_done or stuck)
        truncated = bool(not terminated and self.steps * car.DT >= MAX_TIME)

        reward = default_reward(dprogress, self.on_grass, off_road, lap_done)
        return (
            self._obs(),
            float(reward),
            terminated,
            truncated,
            self._info(off_road, lap_done, stuck),
        )

    def _obs(self):
        _, _, _, v, steer, v_lat = self.state
        own = [v / car.MAX_SPEED, steer / car.MAX_STEER, v_lat / SLIP_SCALE]
        obs = np.concatenate([2 * self.hits / lidar.RANGE - 1, own])
        return np.clip(obs, -1, 1).astype(np.float32)

    def _info(self, off_road, lap_done, stuck):
        info = {
            "progress": float(np.clip(self.progress, 0.0, 1.0)),
            "lap_done": bool(lap_done),
            "off_road": bool(off_road),
            "stuck": bool(stuck),
            "on_grass": bool(self.on_grass),
            "speed": float(self.state[3]),
        }
        if lap_done:
            info["lap_time"] = self.steps * car.DT
        return info

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from sim.render import Renderer  # lazy so training never imports pygame

            self._renderer = Renderer(self.render_mode)
        return self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
