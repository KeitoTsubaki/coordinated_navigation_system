"""Rendering primitives used by :class:`MultiRobotSimulator`."""
from __future__ import annotations

import math
import textwrap
from typing import Iterable

import pygame

from .state import RobotState
from . import utils


def draw_robot(simulator: "MultiRobotSimulator", robot: RobotState) -> None:
    """Draw the robot body together with the heading indicator."""

    position = robot.position
    pygame.draw.circle(
        simulator.env.robot_and_lidar_layer,
        robot.color,
        (int(position[0]), int(position[1])),
        int(simulator.robot_radius_pixels),
    )
    head_position = (
        position[0] - simulator.robot_frame_radius * math.sin(robot.theta),
        position[1] - simulator.robot_frame_radius * math.cos(robot.theta),
    )
    pygame.draw.circle(
        simulator.env.robot_and_lidar_layer,
        robot.color,
        (int(position[0]), int(position[1])),
        int(simulator.robot_frame_radius),
        width=2,
    )
    pygame.draw.line(
        simulator.env.robot_and_lidar_layer,
        (255, 255, 255),
        (int(position[0]), int(position[1])),
        (int(head_position[0]), int(head_position[1])),
        width=2,
    )
    if robot is simulator.robot_states[simulator.selected_robot_index]:
        pygame.draw.circle(
            simulator.env.robot_and_lidar_layer,
            (0, 255, 0),
            (int(position[0]), int(position[1])),
            int(simulator.robot_frame_radius + 6),
            width=2,
        )


def draw_target(simulator: "MultiRobotSimulator", robot: RobotState) -> None:
    """Draw the navigation target indicator for ``robot``."""

    assert robot.target is not None
    color = utils.brighten_color(robot.color)
    pygame.draw.circle(
        simulator.env.robot_and_lidar_layer,
        color,
        (int(robot.target[0]), int(robot.target[1])),
        int(simulator.robot_radius_pixels * 0.6),
        width=2,
    )


def draw_path(simulator: "MultiRobotSimulator", robot: RobotState) -> None:
    """Render the travelled path for ``robot``."""

    points = list(robot.path)
    for (start_pos, _), (end_pos, _) in zip(points[:-1], points[1:]):
        pygame.draw.line(
            simulator.env.robot_and_lidar_layer,
            utils.with_alpha(robot.color, 0.6),
            (int(start_pos[0]), int(start_pos[1])),
            (int(end_pos[0]), int(end_pos[1])),
            width=2,
        )


def draw_label(simulator: "MultiRobotSimulator", robot: RobotState) -> None:
    """Render the textual label displayed above the robot."""

    font = pygame.font.SysFont("arial", 16)
    lines = [robot.name]
    if robot.target is not None:
        tx, ty = robot.target
        distance = math.hypot(tx - robot.position[0], ty - robot.position[1]) / simulator.env.PIXELS_PER_METER
        lines.append(f"target: {distance:.1f}m")
    if robot.closest_obstacle[0] != float("inf"):
        lines.append(f"obstacle: {robot.closest_obstacle[0]:.1f}m")
    for line_index, text in enumerate(lines):
        label = font.render(text, True, (0, 0, 0))
        text_position = (
            int(robot.position[0] - label.get_width() / 2),
            int(
                robot.position[1]
                - simulator.robot_frame_radius
                - label.get_height() * (len(lines) - line_index)
                - 4,
            ),
        )
        simulator.env.robot_and_lidar_layer.blit(label, text_position)


def draw_hud(simulator: "MultiRobotSimulator") -> None:
    """Render general simulation controls in the top left corner."""

    if not simulator.robot_states:
        return
    font = pygame.font.SysFont("arial", 16)
    hud_lines = [
        "Controls: TAB/0-9 select, W move forward, Q/E rotate, C cycle map, SPACE reset, L toggle LiDAR",
    ]
    for index, text in enumerate(hud_lines):
        label = font.render(text, True, (0, 0, 0))
        simulator.env.robot_and_lidar_layer.blit(label, (10, 10 + index * 20))


def draw_parallel_formation(simulator: "MultiRobotSimulator") -> None:
    """Visualise pairwise distances when robots are aligned in parallel."""

    if len(simulator.robot_states) < 2:
        return

    formation_color = (30, 144, 255)
    center_color = (255, 140, 0)
    text_color = (25, 25, 25)
    font = pygame.font.SysFont("arial", 14)

    sorted_states = sorted(simulator.robot_states, key=lambda robot: robot.position[0])
    positions = [robot.position for robot in sorted_states]

    center_x = sum(position[0] for position in positions) / len(positions)
    center_y = sum(position[1] for position in positions) / len(positions)
    pygame.draw.circle(
        simulator.env.robot_and_lidar_layer,
        center_color,
        (int(center_x), int(center_y)),
        6,
    )

    for first, second in zip(sorted_states[:-1], sorted_states[1:]):
        start = first.position
        end = second.position

        pygame.draw.line(
            simulator.env.robot_and_lidar_layer,
            formation_color,
            (int(start[0]), int(start[1])),
            (int(end[0]), int(end[1])),
            width=2,
        )

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance_pixels = math.hypot(dx, dy)
        if distance_pixels == 0:
            continue

        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        normal_x = -dy / distance_pixels
        normal_y = dx / distance_pixels

        distance_m = distance_pixels / simulator.env.PIXELS_PER_METER
        distance_label = font.render(f"{distance_m:.2f}m", True, text_color)

        angle_rad = math.atan2(-(dy), dx)
        angle_deg = (math.degrees(angle_rad) + 360.0) % 360.0
        angle_label = font.render(f"{angle_deg:.1f}°", True, text_color)

        offset = 12
        distance_position = (
            int(midpoint[0] + normal_x * offset - distance_label.get_width() / 2),
            int(midpoint[1] + normal_y * offset - distance_label.get_height() / 2),
        )
        angle_position = (
            int(midpoint[0] - normal_x * offset - angle_label.get_width() / 2),
            int(midpoint[1] - normal_y * offset - angle_label.get_height() / 2),
        )

        simulator.env.robot_and_lidar_layer.blit(distance_label, distance_position)
        simulator.env.robot_and_lidar_layer.blit(angle_label, angle_position)


def draw_info_panel(simulator: "MultiRobotSimulator") -> None:
    """Render the right-hand information panel."""

    panel_left = simulator.env.SCREEN_W
    panel_rect = pygame.Rect(
        panel_left, 0, simulator.info_panel_width, simulator.env.SCREEN_H
    )
    pygame.draw.rect(
        simulator.display_surface, simulator.info_panel_background, panel_rect
    )
    pygame.draw.line(
        simulator.display_surface,
        simulator.info_panel_divider,
        (panel_left, 0),
        (panel_left, simulator.env.SCREEN_H),
        width=2,
    )

    title_font = pygame.font.SysFont("arial", 20, bold=True)
    name_font = pygame.font.SysFont("arial", 18, bold=True)
    value_font = pygame.font.SysFont("consolas", 16)

    x_offset = panel_left + simulator.info_padding
    y_offset = simulator.info_padding

    title = title_font.render(
        "Simulation Info", True, simulator.info_panel_text_color
    )
    simulator.display_surface.blit(title, (x_offset, y_offset))
    y_offset += title.get_height() + 10

    selected_name = (
        simulator.robot_states[simulator.selected_robot_index].name
        if simulator.robot_states
        else "None"
    )
    general_lines = [
        f"FPS: {simulator._fps:5.1f}",
        f"Robots: {len(simulator.robot_states)}",
        f"Selected: {selected_name}",
    ]
    for line in general_lines:
        label = value_font.render(line, True, simulator.info_panel_text_color)
        simulator.display_surface.blit(label, (x_offset, y_offset))
        y_offset += label.get_height() + 4

    y_offset += 10

    map_header = title_font.render("Map", True, simulator.info_panel_text_color)
    simulator.display_surface.blit(map_header, (x_offset, y_offset))
    y_offset += map_header.get_height() + 8

    map_width_m = simulator.env.SCREEN_W / simulator.env.PIXELS_PER_METER
    map_height_m = simulator.env.SCREEN_H / simulator.env.PIXELS_PER_METER
    map_lines = [
        f"Type: {simulator.env.map_type}",
        f"Size: {map_width_m:.1f}m x {map_height_m:.1f}m",
    ]
    for line in map_lines:
        label = value_font.render(line, True, simulator.info_panel_text_color)
        simulator.display_surface.blit(label, (x_offset, y_offset))
        y_offset += label.get_height() + 4

    description = simulator.map_descriptions.get(simulator.env.map_type)
    if description:
        desc_label = value_font.render(
            "Description:", True, simulator.info_panel_text_color
        )
        simulator.display_surface.blit(desc_label, (x_offset, y_offset))
        y_offset += desc_label.get_height() + 4

        wrap_width = max(
            1, (simulator.info_panel_width - 2 * simulator.info_padding) // 8
        )
        for line in textwrap.wrap(description, width=wrap_width):
            desc_line = value_font.render(
                line, True, simulator.info_panel_text_color
            )
            simulator.display_surface.blit(desc_line, (x_offset + 10, y_offset))
            y_offset += desc_line.get_height() + 2

        y_offset += 6

    robots_header = title_font.render("Robots", True, simulator.info_panel_text_color)
    simulator.display_surface.blit(robots_header, (x_offset, y_offset))
    y_offset += robots_header.get_height() + 8

    if not simulator.robot_states:
        empty_label = value_font.render(
            "No robots", True, simulator.info_panel_text_color
        )
        simulator.display_surface.blit(empty_label, (x_offset, y_offset))
        return

    for index, robot in enumerate(simulator.robot_states):
        robot_name = name_font.render(
            robot.name, True, simulator.info_panel_text_color
        )
        metrics = list(_robot_status_lines(simulator, robot))

        line_height = value_font.get_height() + 4
        block_height = (
            robot_name.get_height() + 4 + len(metrics) * line_height + 12
        )
        block_rect = pygame.Rect(
            panel_left + 8,
            y_offset - 6,
            simulator.info_panel_width - 16,
            block_height,
        )
        background_color = (
            simulator.info_panel_highlight
            if index == simulator.selected_robot_index
            else simulator.info_panel_background
        )
        pygame.draw.rect(
            simulator.display_surface,
            background_color,
            block_rect,
            border_radius=8,
        )
        pygame.draw.rect(
            simulator.display_surface,
            simulator.info_panel_divider,
            block_rect,
            width=1,
            border_radius=8,
        )

        text_x = block_rect.x + 10
        text_y = y_offset
        simulator.display_surface.blit(robot_name, (text_x, text_y))
        text_y += robot_name.get_height() + 4

        for metric in metrics:
            metric_label = value_font.render(
                metric, True, simulator.info_panel_text_color
            )
            simulator.display_surface.blit(metric_label, (text_x, text_y))
            text_y += line_height

        y_offset = block_rect.bottom + 10


def _robot_status_lines(simulator: "MultiRobotSimulator", robot: RobotState) -> Iterable[str]:
    """Yield formatted lines describing ``robot`` for the info panel."""

    px, py = robot.position
    x_m = (px - simulator.env.SCREEN_CENTER_X) / simulator.env.PIXELS_PER_METER
    y_m = (simulator.env.SCREEN_CENTER_Y - py) / simulator.env.PIXELS_PER_METER
    yield f"position: ({x_m:+.2f}, {y_m:+.2f}) m"
    yield f"heading: {math.degrees(robot.theta):+.1f}°"
    yield f"velocity: {robot.velocity:.2f} m/s"
    if robot.target is not None:
        tx_m, ty_m = simulator.pixels_to_meters(*robot.target)
        yield f"target: ({tx_m:+.2f}, {ty_m:+.2f}) m"
    if robot.closest_obstacle[0] != float("inf"):
        distance, angle = robot.closest_obstacle
        yield f"closest obstacle: {distance:.2f}m @ {math.degrees(angle):.1f}°"
    yield f"LiDAR: {'ON' if robot.lidar_enabled else 'OFF'}"
    yield f"path length: {len(robot.path)}"


__all__ = [
    "draw_robot",
    "draw_target",
    "draw_path",
    "draw_label",
    "draw_hud",
    "draw_parallel_formation",
    "draw_info_panel",
]
