"""Drive by keyboard. Arrows steer and accelerate or brake, Esc quits.

Run with `make run CMD="python -m sim.play --seed 1000 --difficulty 0.5"`.
"""

import argparse

import pygame

from sim.env import RacingEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--difficulty", type=float, default=0.0)
    args = parser.parse_args()

    env = RacingEnv(args.seed, args.difficulty, render_mode="human")
    env.reset()
    clock = pygame.time.Clock()
    done = False
    while not done:
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE] or pygame.event.peek(pygame.QUIT):
            break
        steer = keys[pygame.K_LEFT] - keys[pygame.K_RIGHT]
        throttle = keys[pygame.K_UP] - keys[pygame.K_DOWN]
        _, _, terminated, truncated, info = env.step([steer, throttle])
        env.render()
        clock.tick(env.metadata["render_fps"])
        done = terminated or truncated
    print(info)
    env.close()


if __name__ == "__main__":
    main()
