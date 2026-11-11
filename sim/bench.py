"""Print env speed. Run with `make bench`. Only env.step and reset are timed, not the driver."""

import time

import numpy as np

from sim.env import TRAIN_SEED_MIN, RacingEnv
from sim.track import generate_track, lookahead, pursuit_steer


def driver(env):
    point = lookahead(env.track, env.idx, env.state[:2], env.state[2], [10.0])[0]
    return np.array([pursuit_steer(point), np.clip(0.5 * (8.0 - env.state[3]), -1, 1)])


def main(steps=20_000):
    env = RacingEnv(track_seed=1000, difficulty=0.5)
    env.reset(seed=0)
    total = 0.0
    for _ in range(steps):
        action = driver(env)
        start = time.perf_counter()
        _, _, terminated, truncated, _ = env.step(action)
        total += time.perf_counter() - start
        if terminated or truncated:
            env.reset()
    print(f"env.step {1e6 * total / steps:.0f} us, {steps / total:.0f} steps/s on one core")
    for difficulty in (0.0, 0.5, 1.0):
        start = time.perf_counter()
        for seed in range(20):
            generate_track(TRAIN_SEED_MIN + seed, difficulty)
        ms = 1e3 * (time.perf_counter() - start) / 20
        print(f"generate_track difficulty {difficulty} {ms:.0f} ms")


if __name__ == "__main__":
    main()
