"""Lidar rays against the road edges."""

import numpy as np

N_RAYS = 11
RANGE = 30.0  # meters
FOV = np.pi  # rays spread over +-90 degrees around the heading
ANGLES = np.linspace(-FOV / 2, FOV / 2, N_RAYS)


def cast(track, pos, yaw):
    """Distance in meters to the road edge along each ray, capped at RANGE."""
    ang = yaw + ANGLES
    d = np.column_stack([np.cos(ang), np.sin(ang)])  # (R, 2)
    hits = np.full(N_RAYS, RANGE)
    for edge in (track.left, track.right):
        a = edge
        e = np.roll(edge, -1, axis=0) - edge  # (M, 2)
        ao = a - pos
        # solve pos + t*d = a + u*e for every ray and segment
        denom = d[:, 0:1] * e[:, 1] - d[:, 1:2] * e[:, 0]  # (R, M)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ao[:, 0] * e[:, 1] - ao[:, 1] * e[:, 0]) / denom
            u = (ao[:, 0] * d[:, 1:2] - ao[:, 1] * d[:, 0:1]) / denom
        ok = (t > 0) & (u >= 0) & (u <= 1)
        hits = np.minimum(hits, np.where(ok, t, RANGE).min(axis=1))
    return hits
