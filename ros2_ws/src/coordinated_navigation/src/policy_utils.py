"""Local copy of policy-related helpers from :mod:`multi_robot.renderer.utils`."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

__all__ = [
    "ensure_repo_root_on_path",
    "process_lidar_for_observation",
]


def ensure_repo_root_on_path() -> Path:
    """Ensure the repository root is present on :data:`sys.path`.

    The upstream helper from :mod:`multi_robot.renderer.utils` simply climbs two
    directories up from the renderer package.  Because the ROS package lives in a
    different subtree, we replicate the search here by walking up until we find
    a folder containing ``.git`` or the ``multi_robot`` sources, and then make
    sure that location is importable.
    """

    current = Path(__file__).resolve()
    repo_root = current.parent
    for candidate in current.parents:
        if (candidate / ".git").exists() or (candidate / "multi_robot").exists():
            repo_root = candidate
            break
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def process_lidar_for_observation(
    sensor_data: Sequence[Tuple[float, float]] | None,
    n_l: int,
    n_e: int,
    max_range: float,
) -> np.ndarray:
    """Convert LiDAR measurements into the policy observation format.

    This function is copied from :func:`multi_robot.renderer.utils.process_lidar_for_observation`
    so that the ROS integration can reuse the exact same preprocessing routine
    without depending on the renderer package at runtime.
    """

    if not sensor_data:
        return np.full(n_l + n_e, 1.0, dtype=np.float32)

    processed: list[float] = []
    for distance, _angle in sensor_data:
        if distance < max_range:
            processed.append(distance / max_range)
        else:
            processed.append(100.0) # I fogot why it is converted to 100

    # if len(processed) < n_l:
    #     fill_value = processed[-1] if processed else 1.0
    #     processed.extend([fill_value] * (n_l - len(processed)))

    extended = processed[:n_l]
    if n_e > 0:
        extended += processed[:n_e]
        # if len(extended) < n_l + n_e:
        #     fill_value = extended[-1] if extended else 1.0
        #     extended.extend([fill_value] * (n_l + n_e - len(extended)))

    return np.asarray(extended, dtype=np.float32)
