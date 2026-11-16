"""Closed spline tracks from a seed and a difficulty in [0, 1]. Queries are in the Rust core."""

from dataclasses import dataclass

import numpy as np
import racing_core
from scipy.interpolate import CubicSpline

from sim import car

SPACING = 1.0  # meters between centerline points
WIDTH = 8.0
WALL_OFFSET = WIDTH / 2 + WIDTH  # centerline to barrier, the grass is as wide as the road


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
    for attempt in range(50):
        # tight corners are eased in place, so a redraw is only needed when two far parts of
        # the loop come too close, and the noise shrinks then so the loop always ends
        center = _ease_tight_corners(_spline_loop(rng, k, amp * 0.9**attempt))
        if _is_valid(center):
            return _build(center)
    raise RuntimeError(f"no valid track for seed={seed} difficulty={difficulty}")


def _spline_loop(rng, k, amp):
    angles = 2 * np.pi * (np.arange(k) + rng.uniform(-0.3, 0.3, k)) / k
    radii = 70.0 * (1 + amp * rng.uniform(-1, 1, k))
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    pts = np.vstack([pts, pts[:1]])
    return _resample(
        CubicSpline(np.arange(k + 1), pts, bc_type="periodic")(np.linspace(0, k, 4000))
    )


def _resample(loop):
    """Points at an even SPACING along a closed polyline whose last point repeats the first."""
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))])
    s = np.linspace(0, arc[-1], round(arc[-1] / SPACING), endpoint=False)
    return np.column_stack([np.interp(s, arc, loop[:, i]) for i in range(2)])


def _headings(center):
    tangent = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    return np.arctan2(tangent[:, 1], tangent[:, 0])


def _radius(center):
    """Turning radius at each point, segment length over the heading change."""
    seg = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
    heading = _headings(center)
    turn = np.angle(np.exp(1j * np.diff(heading, append=heading[:1])))
    return seg / np.maximum(np.abs(turn), 1e-9)


def _ease_tight_corners(center):
    """Spreads the turn of corners tighter than the barrier offset over more road.

    A curve offset by more than its radius folds on itself, which would break the inner barrier.
    Each corner keeps its total angle and only its radius grows. The small gap this opens at the
    start line is spread over the whole loop.
    """
    step = np.roll(center, -1, axis=0) - center
    seg = np.linalg.norm(step, axis=1)
    angle = np.arctan2(step[:, 1], step[:, 0])
    turn = np.angle(np.exp(1j * (np.roll(angle, -1) - angle)))
    limit = seg / (WALL_OFFSET + 2)  # 2 m margin for the resampling and the tolerance below
    excess = np.sign(turn) * np.maximum(np.abs(turn) - limit, 0)
    if not excess.any():
        return center
    for _ in range(5000):
        turn += (np.roll(excess, 1) + np.roll(excess, -1)) / 2 - excess
        excess = np.sign(turn) * np.maximum(np.abs(turn) - limit, 0)
        if np.abs(excess).max() < 1e-4:
            break
    heading = angle[0] + np.concatenate([[0], np.cumsum(turn[:-1])])
    steps = seg[:, None] * np.column_stack([np.cos(heading), np.sin(heading)])
    loop = center[0] + np.vstack([[0, 0], np.cumsum(steps, axis=0)])
    loop -= (loop[-1] - loop[0]) * np.linspace(0, 1, len(loop))[:, None]
    return _resample(loop)


def _is_valid(center):
    n = len(center)
    if _radius(center).min() < WALL_OFFSET:
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


def pursuit_steer(point):
    """Pure pursuit steering in [-1, 1], an arc from the rear axle through `point` (car frame)."""
    x, y = point
    curvature = 2 * np.sin(np.arctan2(y, x)) / max(np.hypot(x, y), 1e-6)
    return float(np.clip(np.arctan(car.WHEELBASE * curvature) / car.MAX_STEER, -1, 1))
