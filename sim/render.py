"""Top-down pygame view with a camera that follows the car. Imported lazily by the env."""

import numpy as np
import pygame

from sim import car, lidar

SIZE = (800, 600)
SCALE = 8  # pixels per meter
GRASS, ROAD, EDGE = (40, 110, 50), (70, 70, 75), (235, 235, 235)


class Renderer:
    def __init__(self, mode):
        pygame.init()
        self.mode = mode
        flags = pygame.SHOWN if mode == "human" else pygame.HIDDEN
        self.screen = pygame.display.set_mode(SIZE, flags)
        self.surface = pygame.Surface(SIZE)
        self.font = pygame.font.SysFont("monospace", 18)

    def close(self):
        pygame.quit()

    def draw(self, track, state, hits, progress, steps):
        x, y, yaw, v, _ = state

        def screen_pos(p):
            return SIZE[0] / 2 + (p[0] - x) * SCALE, SIZE[1] / 2 - (p[1] - y) * SCALE

        surf = self.surface
        surf.fill(GRASS)
        near = np.linalg.norm(track.centerline - (x, y), axis=1) < 70
        n = len(track.centerline)
        for i in np.flatnonzero(near):
            j = (i + 1) % n
            quad = [track.left[i], track.left[j], track.right[j], track.right[i]]
            pygame.draw.polygon(surf, ROAD, [screen_pos(p) for p in quad])
            pygame.draw.line(surf, EDGE, screen_pos(track.left[i]), screen_pos(track.left[j]), 2)
            pygame.draw.line(surf, EDGE, screen_pos(track.right[i]), screen_pos(track.right[j]), 2)

        for ang, dist in zip(yaw + lidar.ANGLES, hits):
            end = (x + dist * np.cos(ang), y + dist * np.sin(ang))
            pygame.draw.line(surf, (255, 220, 0), screen_pos((x, y)), screen_pos(end), 1)

        c, s = np.cos(yaw), np.sin(yaw)
        body = [(-1.0, -0.9), (3.0, -0.9), (3.0, 0.9), (-1.0, 0.9)]  # rear axle at the origin
        corners = [(x + bx * c - by * s, y + bx * s + by * c) for bx, by in body]
        pygame.draw.polygon(surf, (220, 40, 40), [screen_pos(p) for p in corners])

        text = f"{v * 3.6:5.1f} km/h  lap {100 * progress:5.1f}%  t {steps * car.DT:5.1f}s"
        surf.blit(self.font.render(text, True, (255, 255, 255)), (10, 10))

        if self.mode == "human":
            self.screen.blit(surf, (0, 0))
            pygame.display.flip()
            return None
        return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))
