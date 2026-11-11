"""Drive by keyboard. Arrows steer and accelerate, brake or reverse, Esc quits.

A failed attempt or a finished lap puts the car back on the start line.
Run with `make run CMD="python -m sim.play --seed 1000 --difficulty 0.5"`.
"""

import argparse

import pygame

from sim.env import RacingEnv, outcome


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--difficulty", type=float, default=0.0)
    args = parser.parse_args()

    env = RacingEnv(args.seed, args.difficulty, render_mode="human")
    env.reset()
    env.render()  # creates the window, pygame is only initialized from here
    clock = pygame.time.Clock()
    while True:
        if any(e.type == pygame.QUIT for e in pygame.event.get()):
            break
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            break
        steer = keys[pygame.K_LEFT] - keys[pygame.K_RIGHT]
        throttle = keys[pygame.K_UP] - keys[pygame.K_DOWN]
        _, _, terminated, truncated, info = env.step([steer, throttle])
        env.render()
        clock.tick(env.metadata["render_fps"])
        if terminated or truncated:
            end = outcome(info)
            print(f"lap in {info['lap_time']:.1f} s" if end == "lap" else f"failed, {end}")
            pygame.time.wait(1000)
            env.reset()
    env.close()


if __name__ == "__main__":
    main()
