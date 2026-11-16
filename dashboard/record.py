"""Records episodes and track geometry as plain lists for the dashboard replay."""

import numpy as np

from evaluation.evaluate import check_seeds, run_episode
from sim import car
from sim.env import RacingEnv


def rounded(values, digits=2):
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def track_geometry(track, stride=2):
    """Road edges, barriers and centerline, every `stride` points to keep the file small."""
    parts = ("centerline", "left", "right", "left_wall", "right_wall")
    out = {name: rounded(getattr(track, name)[::stride]) for name in parts}
    out["length"] = round(float(track.length), 1)
    out["width"] = float(track.width)
    return out


def record_episode(policy, seed, difficulty, env=None):
    """One episode on a validation or test track, driven by evaluation.run_episode.

    Returns per step car state, lateral offset, progress, grass flag, action and lidar,
    plus the outcome. Pass `env` when the policy reads from it, like pure pursuit.
    """
    check_seeds([seed])
    env = RacingEnv(difficulty=difficulty) if env is None else env
    env.difficulty = difficulty
    steps = []

    def keep(action):
        action = np.asarray(action, dtype=float)
        steps.append((env.state, action, env.hits, env.lateral, env.progress, env.on_grass))

    ep = run_episode(policy, seed, env, keep)
    states, actions, hits, lateral, progress, grass = (np.array(v) for v in zip(*steps))
    frames = {
        key: rounded(states[:, i]) for i, key in enumerate(("x", "y", "yaw", "v", "steer", "v_lat"))
    }
    frames.update(
        lateral=rounded(lateral),
        progress=rounded(progress, 4),
        on_grass=grass.astype(int).tolist(),
        steer_cmd=rounded(actions[:, 0], 3),
        throttle=rounded(actions[:, 1], 3),
        hits=rounded(hits, 1),
    )
    return {
        "seed": seed,
        "dt": car.DT,
        "frames": frames,
        "outcome": ep["outcome"],
        "lap_time": ep["info"].get("lap_time"),
        "progress": round(float(ep["info"]["progress"]), 4),
    }
