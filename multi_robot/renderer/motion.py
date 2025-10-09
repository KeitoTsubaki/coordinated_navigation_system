"""Motion related helpers for :class:`MultiRobotSimulator`."""
from __future__ import annotations

import math
from typing import Optional

import pygame

from .state import Position, RobotState


def update_robot_motion(simulator: "MultiRobotSimulator", robot: RobotState, new_position: Position) -> None:
    """Update ``robot`` with a new position while tracking velocity estimates."""

    previous_position = robot.position
    dx = new_position[0] - previous_position[0]
    dy = new_position[1] - previous_position[1]
    distance_pixels = math.hypot(dx, dy)

    current_time = pygame.time.get_ticks() / 1000.0 if pygame.get_init() else None

    velocity = 0.0
    if (
        current_time is not None
        and robot.last_update_time is not None
        and distance_pixels > 0.0
    ):
        dt = current_time - robot.last_update_time
        if dt > 0:
            distance_m = distance_pixels / simulator.env.PIXELS_PER_METER
            velocity = distance_m / dt

    robot.velocity = velocity
    robot.last_update_time = current_time
    robot.position = (float(new_position[0]), float(new_position[1]))


def attempt_move(simulator: "MultiRobotSimulator", robot: RobotState, step: float) -> bool:
    """Move ``robot`` forward by ``step`` pixels if no collision occurs."""

    dx = step * math.sin(robot.theta)
    dy = step * math.cos(robot.theta)
    next_position = (robot.position[0] + dx, robot.position[1] + dy)

    if is_collision(simulator, next_position, ignore_robot=robot):
        return False

    update_robot_motion(simulator, robot, next_position)
    return True


def is_collision(
    simulator: "MultiRobotSimulator",
    position: Position,
    *,
    ignore_robot: Optional[RobotState] = None,
) -> bool:
    """Return ``True`` if ``position`` collides with an obstacle or robot."""

    for i in range(simulator.env.NUM_CHECK_POINTS):
        angle = 2 * math.pi * i / simulator.env.NUM_CHECK_POINTS
        check_x = int(position[0] + simulator.env.ROBOT_RADIUS_PIXELS * math.cos(angle))
        check_y = int(position[1] + simulator.env.ROBOT_RADIUS_PIXELS * math.sin(angle))

        if not (0 <= check_x < simulator.env.SCREEN_W and 0 <= check_y < simulator.env.SCREEN_H):
            return True

        pixel_color = simulator.env.obstacle_layer.get_at((check_x, check_y))
        if (pixel_color[0], pixel_color[1], pixel_color[2]) == simulator.obstacle_color_rgb:
            return True

    for other in simulator.robot_states:
        if other is ignore_robot:
            continue

        distance = math.hypot(position[0] - other.position[0], position[1] - other.position[1])
        min_clearance = simulator.robot_frame_radius * 2
        if distance < min_clearance:
            return True

    return False


__all__ = ["update_robot_motion", "attempt_move", "is_collision"]
