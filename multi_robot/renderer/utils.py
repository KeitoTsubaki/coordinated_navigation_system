"""Helper utilities shared across the renderer implementation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence, Union

from .types import Color

__all__ = [
    "ensure_repo_root_on_path",
    "to_rgb",
    "brighten_color",
    "with_alpha",
    "color_from_palette",
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

