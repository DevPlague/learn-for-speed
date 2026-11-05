"""Lidar constants for Python. The rays are cast in rust/src/lidar.rs, see docs/PHYSICS.md.

The lidar is all the agent sees of the track.
"""

from racing_core import ANGLES, N_RAYS, RANGE

__all__ = ["ANGLES", "N_RAYS", "RANGE"]
