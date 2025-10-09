"""Keyboard control helpers."""
from __future__ import annotations

import math

import pygame

from . import motion
from .environment import switch_map_type


def cycle_map(simulator: "MultiRobotSimulator") -> None:
    """Switch to the next map using the :mod:`simulation_core` helper."""

    next_map_type = switch_map_type(
        simulator.env.map_type, simulator.env.np_random, switch_mode="cycle"
    )
    if next_map_type is not None:
        simulator.set_map(next_map_type)


def operate_moving(simulator: "MultiRobotSimulator") -> None:
    """Handle manual motion controls for the currently selected robot."""

    robot = simulator.robot_states[simulator.selected_robot_index]
    keys = pygame.key.get_pressed()
    moved = False
    translated = False

    if keys[pygame.K_w]:
        move_success = motion.attempt_move(simulator, robot, -simulator.manual_step_pixels)
        moved |= move_success
        translated |= move_success
    if keys[pygame.K_q]:
        robot.theta += simulator.manual_rotation_step
        moved = True
    if keys[pygame.K_e]:
        robot.theta -= simulator.manual_rotation_step
        moved = True

    if moved:
        robot.theta = (robot.theta + math.pi) % (2 * math.pi) - math.pi

    if moved and simulator.path_history != 0:
        robot.path.append((robot.position, robot.theta))
    if not translated:
        robot.velocity = 0.0


def select_next_robot(simulator: "MultiRobotSimulator") -> None:
    """Select the next robot when the user presses TAB."""

    if not simulator.robot_states:
        return
    simulator.selected_robot_index = (simulator.selected_robot_index + 1) % len(
        simulator.robot_states
    )


def handle_keydown(simulator: "MultiRobotSimulator", event: pygame.event.Event) -> None:
    """React to a pygame KEYDOWN event."""

    if event.key == pygame.K_ESCAPE:
        simulator.running = False
    elif event.key == pygame.K_c:
        cycle_map(simulator)
    elif event.key == pygame.K_SPACE:
        simulator.set_map(simulator.env.map_type)
    elif event.key == pygame.K_TAB:
        select_next_robot(simulator)
    elif pygame.K_1 <= event.key <= pygame.K_9:
        index = event.key - pygame.K_1
        if index < len(simulator.robot_states):
            simulator.selected_robot_index = index
    elif event.key == pygame.K_0 and len(simulator.robot_states) >= 10:
        simulator.selected_robot_index = 9
    elif event.key == pygame.K_l and simulator.robot_states:
        robot = simulator.robot_states[simulator.selected_robot_index]
        robot.lidar_enabled = not robot.lidar_enabled


__all__ = ["cycle_map", "operate_moving", "select_next_robot", "handle_keydown"]
