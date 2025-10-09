"""LiDAR helpers used by the renderer."""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import pygame

from .environment import AD2pos, LiDARSensor
from .state import RobotState


def create_sensor(simulator: "MultiRobotSimulator") -> LiDARSensor:
    """Instantiate a :class:`LiDARSensor` configured for ``simulator``."""

    return LiDARSensor(
        Range=simulator.env.LIDAR_RANGE_M,
        uncertainty=(0, 0),
        pixels_per_meter=simulator.env.PIXELS_PER_METER,
        lidar_resolution=simulator.env.LIDAR_RESOLUTION,
        screen_w=simulator.env.SCREEN_W,
        screen_h=simulator.env.SCREEN_H,
    )


def update_lidar(simulator: "MultiRobotSimulator", robot: RobotState) -> None:
    """Update LiDAR readings for ``robot`` and render the points."""

    if not robot.lidar_enabled or robot.sensor is None:
        robot.sensor_data = []
        robot.closest_obstacle = (float("inf"), 0.0)
        return

    robot.sensor.position = (
        int(robot.position[0]),
        int(robot.position[1]),
    )
    robot.sensor.theta = robot.theta
    detection_surface = simulator._lidar_detection_surface
    detection_surface.fill((0, 0, 0, 0))
    detection_surface.blit(simulator.env.obstacle_layer, (0, 0))

    for other in simulator.robot_states:
        if other is robot:
            continue

        pygame.draw.circle(
            detection_surface,
            simulator.obstacle_color_rgb,
            (int(other.position[0]), int(other.position[1])),
            int(simulator.robot_frame_radius),
        )

    raw_sensor_data = robot.sensor.sense_obstacles(
        detection_surface, simulator.env.OBSTACLE_COLOR
    )
    robot.sensor_data = list(raw_sensor_data) if raw_sensor_data else []
    robot.filtered_sensor_data = []
    if robot.sensor_data:
        filter_robot_reflections(simulator, robot)

    if robot.sensor_data:
        filtered_counts: Dict[float, int] = {}
        if robot.filtered_sensor_data:
            for _distance, relative_angle in robot.filtered_sensor_data:
                filtered_counts[relative_angle] = (
                    filtered_counts.get(relative_angle, 0) + 1
                )

        for distance, relative_angle in robot.sensor_data:
            color = robot.color
            if filtered_counts.get(relative_angle):
                filtered_counts[relative_angle] -= 1
                color = simulator.lidar_filtered_points_color

            obstacle_angle = robot.theta - math.pi / 2 - relative_angle
            point = AD2pos(
                distance,
                obstacle_angle,
                robot.position,
                simulator.env.PIXELS_PER_METER,
            )
            pygame.draw.circle(
                simulator.env.robot_and_lidar_layer,
                color,
                (int(point[0]), int(point[1])),
                2,
            )
        robot.closest_obstacle = robot.sensor.get_closest_obstacle_info(
            robot.sensor_data
        )
    else:
        robot.closest_obstacle = (float("inf"), 0.0)


def filter_robot_reflections(simulator: "MultiRobotSimulator", robot: RobotState) -> None:
    """Filter LiDAR reflections caused by other robots."""

    if robot.sensor is None:
        return

    robot_radius_m = simulator.env.ROBOT_RADIUS_M
    lidar_range = robot.sensor.LIDAR_RANGE_M
    pixels_per_meter = simulator.env.PIXELS_PER_METER
    other_robot_filters = []

    for other in simulator.robot_states:
        if other is robot:
            continue

        dx_pixels = other.position[0] - robot.position[0]
        dy_pixels = other.position[1] - robot.position[1]
        distance_pixels = math.hypot(dx_pixels, dy_pixels)
        if distance_pixels == 0:
            continue

        distance_m = distance_pixels / pixels_per_meter
        if distance_m > lidar_range + robot_radius_m:
            continue

        global_angle = math.atan2(-dx_pixels, -dy_pixels)
        if global_angle < 0:
            global_angle += 2 * math.pi

        if distance_m <= robot_radius_m:
            angle_half_width = math.pi
        else:
            ratio = min(1.0, robot_radius_m / distance_m)
            angle_half_width = math.asin(ratio)

        distance_min = max(0.0, distance_m - robot_radius_m)
        distance_max = distance_m + robot_radius_m
        other_robot_filters.append(
            (global_angle, angle_half_width, distance_min, distance_max)
        )

    if not other_robot_filters:
        return

    filtered_data: List[Tuple[float, float]] = []
    replaced_measurements: List[Tuple[float, float]] = []
    theta = robot.theta
    buffer = 1.1
    for distance, relative_angle in robot.sensor_data:
        global_angle = (theta - math.pi / 2 - relative_angle) % (2 * math.pi)
        replaced = False

        for angle_center, angle_half_width, distance_min, distance_max in other_robot_filters:
            angle_diff = abs(global_angle - angle_center)
            angle_diff = min(angle_diff, 2 * math.pi - angle_diff)

            if angle_diff <= angle_half_width * buffer and distance_min * (2 - buffer) <= distance <= distance_max * buffer:
                replaced_measurements.append((distance, relative_angle))
                filtered_data.append((robot.sensor.out_of_range_value, relative_angle))
                replaced = True
                break

        if not replaced:
            filtered_data.append((distance, relative_angle))

    robot.sensor_data = filtered_data
    robot.filtered_sensor_data = replaced_measurements


__all__ = ["create_sensor", "update_lidar", "filter_robot_reflections"]
