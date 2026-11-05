"""Top-down pygame view with a camera that follows the car. Imported lazily by the env."""

import numpy as np
import pygame

from sim import car, lidar
from sim.env import GRASS_TIME

SIZE = (800, 600)
SCALE = 8  # pixels per meter
GRASS, ROAD, EDGE = (40, 110, 50), (70, 70, 75), (235, 235, 235)
WALL = (150, 150, 160)
RAY, HIT = (255, 220, 0, 70), (255, 220, 0, 170)  # alpha keeps the lidar see-through
CAR, WARN = (220, 40, 40), (255, 60, 60)
START_SQUARES = 8
MAP_SIZE, MAP_MARGIN = 170, 10


class Renderer:
    def __init__(self, mode):
        pygame.init()
        self.mode = mode
        flags = pygame.SHOWN if mode == "human" else pygame.HIDDEN
        self.screen = pygame.display.set_mode(SIZE, flags)
        self.surface = pygame.Surface(SIZE)
        self.overlay = pygame.Surface(SIZE, pygame.SRCALPHA)
        self.font = pygame.font.SysFont("monospace", 18)
        self.map_track = None

    def close(self):
        pygame.quit()

    def _minimap(self, track):
        """Whole track scaled into a corner box, rebuilt only when the track changes."""
        if self.map_track is not track:
            walls = np.vstack([track.left_wall, track.right_wall])
            lo, hi = walls.min(axis=0), walls.max(axis=0)
            scale = (MAP_SIZE - 16) / (hi - lo).max()
            origin = np.array([SIZE[0] - MAP_SIZE - MAP_MARGIN, MAP_MARGIN])
            center = (MAP_SIZE - (hi - lo) * scale) / 2

            def to_map(p):
                q = (np.asarray(p) - lo) * scale
                return origin[0] + center[0] + q[..., 0], origin[1] + MAP_SIZE - center[1] - q[
                    ..., 1
                ]

            self.map_to = to_map
            self.map_line = list(zip(*to_map(track.centerline)))
            self.map_track = track
        box = pygame.Surface((MAP_SIZE, MAP_SIZE), pygame.SRCALPHA)
        box.fill((0, 0, 0, 120))
        self.surface.blit(box, (SIZE[0] - MAP_SIZE - MAP_MARGIN, MAP_MARGIN))
        pygame.draw.lines(self.surface, EDGE, True, self.map_line, 3)
        pygame.draw.circle(self.surface, (20, 20, 20), self.map_line[0], 4)  # start line

    def draw(self, env):
        track, state, hits = env.track, env.state, env.hits
        x, y, yaw, v, _, v_lat = state

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
            for wall in (track.left_wall, track.right_wall):
                pygame.draw.line(surf, WALL, screen_pos(wall[i]), screen_pos(wall[j]), 4)

        # checkered start line across the road at the first centerline point
        for k in range(START_SQUARES):
            a = track.right[0] + (track.left[0] - track.right[0]) * k / START_SQUARES
            b = track.right[0] + (track.left[0] - track.right[0]) * (k + 1) / START_SQUARES
            color = (255, 255, 255) if k % 2 else (20, 20, 20)
            pygame.draw.line(surf, color, screen_pos(a), screen_pos(b), 6)

        self.overlay.fill((0, 0, 0, 0))
        for ang, dist in zip(yaw + lidar.ANGLES, hits):
            end = screen_pos((x + dist * np.cos(ang), y + dist * np.sin(ang)))
            pygame.draw.line(self.overlay, RAY, screen_pos((x, y)), end, 1)
            if dist < lidar.RANGE:
                pygame.draw.circle(self.overlay, HIT, end, 3)
        surf.blit(self.overlay, (0, 0))

        c, s = np.cos(yaw), np.sin(yaw)
        body = [(-1.0, -0.9), (3.0, -0.9), (3.0, 0.9), (-1.0, 0.9)]  # rear axle at the origin
        corners = [(x + bx * c - by * s, y + bx * s + by * c) for bx, by in body]
        pygame.draw.polygon(surf, CAR, [screen_pos(p) for p in corners])

        speed = np.hypot(v, v_lat) * 3.6
        text = f"{speed:5.1f} km/h  lap {100 * env.progress:5.1f}%  t {env.steps * car.DT:5.1f}s"
        surf.blit(self.font.render(text, True, (255, 255, 255)), (10, 10))
        if abs(v_lat) > 0.5:
            surf.blit(
                self.font.render(f"drift {v_lat * 3.6:+5.1f} km/h", True, (255, 220, 0)), (10, 32)
            )
        if env.grass_steps:
            grass = f"grass {env.grass_steps * car.DT:4.1f} / {GRASS_TIME:.0f} s"
            surf.blit(self.font.render(grass, True, WARN), (10, 54))

        if self.mode == "human":  # recorded videos should not show the keys
            keys = "arrows drive   down brakes, then reverses   esc quits"
            surf.blit(self.font.render(keys, True, (255, 255, 255)), (10, SIZE[1] - 28))

        self._minimap(track)
        pygame.draw.circle(surf, CAR, self.map_to((x, y)), 5)

        if self.mode == "human":
            self.screen.blit(surf, (0, 0))
            pygame.display.flip()
            return None
        return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))
