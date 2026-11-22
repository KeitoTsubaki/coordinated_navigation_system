"""Helper utilities shared across the renderer implementation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np

from .types import Color

__all__ = [
    "ensure_repo_root_on_path",
    "to_rgb",
    "brighten_color",
    "with_alpha",
    "color_from_palette",
    "process_lidar_for_observation",
]


def ensure_repo_root_on_path() -> Path:
    """Ensure the repository root is available on :data:`sys.path`."""

    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def to_rgb(color: Union[Color, str]) -> Color:
    from .env import utils_module

    if isinstance(color, str):
        return utils_module.hex_to_rgb(color)
    return color


def brighten_color(color: Color) -> Color:
    return tuple(min(255, int(channel * 1.2 + 30)) for channel in color)


def with_alpha(color: Color, alpha: float) -> Color:
    r, g, b = color
    return (int(r * alpha), int(g * alpha), int(b * alpha))


def color_from_palette(index: int) -> Color:
    palette: Sequence[Color] = [
        (0, 128, 255),
        (255, 99, 71),
        (60, 179, 113),
        (238, 130, 238),
        (255, 215, 0),
        (0, 206, 209),
    ]
    return palette[index % len(palette)]


def process_lidar_for_observation(
    sensor_data: Sequence[Tuple[float, float]] | None,
    n_l: int,
    n_e: int,
    max_range: float,
) -> np.ndarray:
    """Convert LiDAR measurements into the policy observation format.

    The helper mirrors the preprocessing used by the formation module so that
    the policy controller can share the exact same LiDAR representation for
    both the current scan and the stacked history.
    """

    if not sensor_data:
        return np.full(n_l + n_e, 1.0, dtype=np.float32)

    processed: list[float] = []
    for distance, _angle in sensor_data:
        if distance < max_range:
            processed.append(distance / max_range)
        else:
            processed.append(1.0)

    if len(processed) < n_l:
        fill_value = processed[-1] if processed else 1.0
        processed.extend([fill_value] * (n_l - len(processed)))

    extended = processed[:n_l]
    if n_e > 0:
        extended += processed[:n_e]
        if len(extended) < n_l + n_e:
            fill_value = extended[-1] if extended else 1.0
            extended.extend([fill_value] * (n_l + n_e - len(extended)))

    return np.asarray(extended, dtype=np.float32)

