"""Lidar rays against the road edges and the barriers. This is all the agent sees of the track."""

import numpy as np

N_RAYS = 25
RANGE = 30.0  # meters
# full circle, denser in front, one ray straight ahead
_u = np.arange(-(N_RAYS // 2), N_RAYS // 2 + 1) / (N_RAYS / 2)
ANGLES = np.pi * np.sign(_u) * np.abs(_u) ** 1.5


def segments(track):
    """Start points, direction vectors and lengths of every edge and barrier segment."""
    lines = (track.left, track.right, track.left_wall, track.right_wall)
    a = np.concatenate(lines)
    e = np.concatenate([np.roll(p, -1, axis=0) - p for p in lines])
    return a, e, np.linalg.norm(e, axis=1)


def cast(segs, pos, yaw):
    """Distance in meters to the nearest edge or barrier along each ray, capped at RANGE."""
    a, e, length = segs
    near = np.linalg.norm(a - pos, axis=1) < RANGE + length  # others cannot be hit
    a, e = a[near], e[near]
    ang = yaw + ANGLES
    d = np.column_stack([np.cos(ang), np.sin(ang)])  # (R, 2)
    ao = a - pos
    # solve pos + t*d = a + u*e for every ray and segment
    denom = d[:, 0:1] * e[:, 1] - d[:, 1:2] * e[:, 0]  # (R, M)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ao[:, 0] * e[:, 1] - ao[:, 1] * e[:, 0]) / denom
        u = (ao[:, 0] * d[:, 1:2] - ao[:, 1] * d[:, 0:1]) / denom
    ok = (t > 0) & (u >= 0) & (u <= 1)
    return np.minimum(RANGE, np.where(ok, t, RANGE).min(axis=1, initial=RANGE))
