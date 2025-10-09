"""Domain models used by the multi robot renderer."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

from .environment import LiDARSensor

Color = Tuple[int, int, int]
Position = Tuple[float, float]


@dataclass
class RobotState:
    """Container describing the visual state of a single robot."""

    name: str
    position: Position
    theta: float = 0.0
    color: Color = (0, 128, 255)
    target: Optional[Position] = None
    path: Deque[Tuple[Position, float]] = field(default_factory=deque)
    sensor: Optional[LiDARSensor] = None
    sensor_data: List[Tuple[float, float]] = field(default_factory=list)
    filtered_sensor_data: List[Tuple[float, float]] = field(default_factory=list)
    closest_obstacle: Tuple[float, float] = (float("inf"), 0.0)
    lidar_enabled: bool = True
    last_update_time: Optional[float] = None
    velocity: float = 0.0


__all__ = ["Color", "Position", "RobotState"]
