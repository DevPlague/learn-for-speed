"""Records episodes and track geometry as plain lists for the dashboard replay."""

import numpy as np

from evaluation.evaluate import check_seeds
from sim import car
from sim.env import RacingEnv

FRAME_KEYS = ("x", "y", "yaw", "v", "steer", "v_lat")


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
    """One episode on a validation or test track, the same loop as evaluation.evaluate.

    Returns per step car state, lateral offset, progress, grass flag, action and lidar,
    plus the outcome. Pass `env` when the policy reads from it, like pure pursuit.
    """
    check_seeds([seed])
    env = RacingEnv(difficulty=difficulty) if env is None else env
    env.difficulty, env.track_seed = difficulty, seed
    obs, info = env.reset(seed=seed)
    rows, actions, hits, lateral, progress, grass = [], [], [], [], [], []
    done = False
    while not done:
        action = np.asarray(policy(obs), dtype=float)
        obs, _, terminated, truncated, info = env.step(action)
        rows.append(env.state)
        actions.append(action)
        hits.append(env.hits)
        lateral.append(env.core.frenet(env.state[0], env.state[1], env.idx)[1])
        progress.append(env.progress)
        grass.append(env.on_grass)
        done = terminated or truncated
    states = np.array(rows)
    actions = np.array(actions)
    frames = {key: rounded(states[:, i]) for i, key in enumerate(FRAME_KEYS)}
    frames.update(
        lateral=rounded(lateral),
        progress=rounded(progress, 4),
        on_grass=[int(g) for g in grass],
        steer_cmd=rounded(actions[:, 0], 3),
        throttle=rounded(actions[:, 1], 3),
        hits=rounded(hits, 1),
    )
    outcome = "lap" if info["lap_done"] else "off_road" if info["off_road"] else "stuck"
    if truncated and not terminated:
        outcome = "timeout"
    return {
        "seed": seed,
        "dt": car.DT,
        "frames": frames,
        "outcome": outcome,
        "lap_time": info.get("lap_time"),
        "progress": round(float(info["progress"]), 4),
    }
