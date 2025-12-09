"""Shared type definitions for the multi-robot renderer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

Color = Tuple[int, int, int]
Position = Tuple[float, float]

__all__ = ["Color", "Position", "RobotState", "FormationPlan"]


@dataclass
class RobotState:
    """State container used by the renderer to draw a robot."""

    name: str
    position: Position
    theta: float = 0.0
    color: Color = (0, 128, 255)
    target: Optional[Position] = None
    target_theta: Optional[float] = None
    path: Deque[Tuple[Position, float]] = field(default_factory=deque)
    sensor: Optional["LiDARSensor"] = None
    sensor_data: List[Tuple[float, float]] = field(default_factory=list)
    filtered_sensor_data: List[Tuple[float, float]] = field(default_factory=list)
    closest_obstacle: Tuple[float, float] = (float("inf"), 0.0)
    lidar_enabled: bool = True
    path_enabled: bool = True
    last_update_time: Optional[float] = None
    velocity: float = 0.0
    braking: bool = False
    executable_velocity: float = 0.0
    executable_angular_velocity: float = 0.0
    output_velocity: float = 0.0
    output_angular_velocity: float = 0.0
    trail_points = []
    trail_step_counter = 0


@dataclass
class FormationPlan:
    """Result of the constrained formation optimisation routine."""

    theta: float
    targets: List[Position]
    assignment: List[int]
    cost: float
    per_robot_costs: Dict[int, float]
    formation_type: str
    spacing: float
    center: Position

