"""Small helper functions shared across renderer modules."""
from __future__ import annotations

from typing import Sequence, Union

from .environment import hex_to_rgb
from .state import Color

ColorInput = Union[Color, str]


def to_rgb(color: ColorInput) -> Color:
    """Return an RGB tuple for ``color``.

    The renderer accepts either tuples or ``"#RRGGBB"`` strings in its
    configuration.  Centralising the conversion keeps the other modules free
    from the implementation details of :mod:`simulation_core`.
    """

    if isinstance(color, str):
        return hex_to_rgb(color)
    return color


def with_alpha(color: Color, alpha: float) -> Color:
    """Return ``color`` darkened by ``alpha``.

    Pygame does not natively handle per-primitive transparency.  The renderer
    therefore emulates it by scaling the RGB components.
    """

    r, g, b = color
    return (int(r * alpha), int(g * alpha), int(b * alpha))


def brighten_color(color: Color) -> Color:
    """Return a slightly brighter variant of ``color`` for highlights."""

    r, g, b = color
    return (
        min(255, int(r * 1.2 + 30)),
        min(255, int(g * 1.2 + 30)),
        min(255, int(b * 1.2 + 30)),
    )


def color_from_palette(index: int) -> Color:
    """Return a deterministic color from the renderer palette."""

    palette: Sequence[Color] = [
        (0, 128, 255),
        (255, 99, 71),
        (60, 179, 113),
        (238, 130, 238),
        (255, 215, 0),
        (0, 206, 209),
    ]
    return palette[index % len(palette)]


__all__ = [
    "ColorInput",
    "to_rgb",
    "with_alpha",
    "brighten_color",
    "color_from_palette",
]
