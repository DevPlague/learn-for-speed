"""Closed spline tracks from a seed and a difficulty in [0, 1]. Queries are in the Rust core."""

from dataclasses import dataclass

import numpy as np
import racing_core
from scipy.interpolate import CubicSpline

SPACING = 1.0  # meters between centerline points
WIDTH = 8.0
WALL_GAP = WIDTH  # grass between the road edge and the barrier
WALL_OFFSET = WIDTH / 2 + WALL_GAP  # centerline to barrier
BASE_RADIUS = 70.0
MAX_TRIES = 50


@dataclass
class Track:
    centerline: np.ndarray  # (N, 2) closed loop, meters
    width: float
    left: np.ndarray  # (N, 2) road edges
    right: np.ndarray  # (N, 2)
    left_wall: np.ndarray  # (N, 2) barriers past the grass
    right_wall: np.ndarray  # (N, 2)
    s: np.ndarray  # (N,) arclength of each centerline point
    heading: np.ndarray  # (N,) direction of travel, radians
    length: float


def generate_track(seed, difficulty):
    """Same seed and difficulty always give the same track."""
    rng = np.random.default_rng(seed)
    k = round(6 + 6 * difficulty)
    amp = 0.1 + 0.4 * difficulty
    for attempt in range(MAX_TRIES):
        # shrink the noise on every rejection so the loop always ends with a valid track
        center = _spline_loop(rng, k, amp * 0.9**attempt)
        if _is_valid(center):
            return _build(center)
    raise RuntimeError(f"no valid track for seed={seed} difficulty={difficulty}")


def _spline_loop(rng, k, amp):
    angles = 2 * np.pi * (np.arange(k) + rng.uniform(-0.3, 0.3, k)) / k
    radii = BASE_RADIUS * (1 + amp * rng.uniform(-1, 1, k))
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    pts = np.vstack([pts, pts[:1]])
    dense = CubicSpline(np.arange(k + 1), pts, bc_type="periodic")(np.linspace(0, k, 4000))
    # resample at an even spacing along the arc
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    n = round(arc[-1] / SPACING)
    arc_new = np.linspace(0, arc[-1], n, endpoint=False)
    return np.column_stack([np.interp(arc_new, arc, dense[:, i]) for i in range(2)])


def _headings(center):
    tangent = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    return np.arctan2(tangent[:, 1], tangent[:, 0])


def _is_valid(center):
    n = len(center)
    seg = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
    turn = np.angle(np.exp(1j * np.diff(_headings(center), append=_headings(center)[0])))
    if (seg / np.maximum(np.abs(turn), 1e-9)).min() < WALL_OFFSET:
        return False  # the inner barrier would fold on itself
    # distant parts of the loop must stay apart so no barrier stands on another part of the road
    idx = np.arange(n)
    gap = np.abs(idx[:, None] - idx[None, :])
    far = np.minimum(gap, n - gap) * SPACING > 2 * WALL_OFFSET + WIDTH
    dist = np.linalg.norm(center[:, None] - center[None, :], axis=2)
    return bool((dist[far] > WALL_OFFSET + WIDTH / 2 + 1).all())


def _build(center):
    heading = _headings(center)
    normal = np.column_stack([-np.sin(heading), np.cos(heading)])
    seg = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
    s = np.concatenate([[0], np.cumsum(seg)[:-1]])
    return Track(
        centerline=center,
        width=WIDTH,
        left=center + normal * WIDTH / 2,
        right=center - normal * WIDTH / 2,
        left_wall=center + normal * WALL_OFFSET,
        right_wall=center - normal * WALL_OFFSET,
        s=s,
        heading=heading,
        length=float(seg.sum()),
    )


def make_core(track):
    """Hand the track to Rust, once per reset."""
    return racing_core.TrackCore(
        track.centerline,
        track.heading,
        track.s,
        track.length,
        track.left,
        track.right,
        track.left_wall,
        track.right_wall,
    )


def lookahead(track, idx, pos, yaw, dists):
    """Centerline points `dists` meters ahead, in the car frame (x forward, y left)."""
    steps = np.round(np.asarray(dists) / (track.length / len(track.centerline))).astype(int)
    rel = track.centerline[(idx + steps) % len(track.centerline)] - pos
    c, s = np.cos(yaw), np.sin(yaw)
    return np.column_stack([rel[:, 0] * c + rel[:, 1] * s, -rel[:, 0] * s + rel[:, 1] * c])
