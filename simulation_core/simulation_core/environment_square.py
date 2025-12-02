import importlib.resources
import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pygame
import yaml

from . import utils
from .environment import SimCoreEnvironment


class SimCoreSquareEnvironment(SimCoreEnvironment):
    """Simulation environment that models the robot with a rectangular footprint."""

    def __init__(
        self,
        render_mode: Optional[str] = None,
        config_path: Optional[Path] = None,
        min_target_clearance_m: float = 0.0,
    ) -> None:
        if config_path is None:
            config_path = (
                importlib.resources.files("simulation_core")
                .joinpath("config")
                .joinpath("simulator_config_square.yaml")
            )

        config_text = config_path.read_text(encoding="utf-8")
        config = yaml.safe_load(config_text)
        robot_cfg = config["robot"]

        wheel_base_m = float(robot_cfg.get("wheel_base_m", 0.0) or 0.0)
        wheel_track_m = float(robot_cfg.get("wheel_track_m", 0.0) or 0.0)
        wheel_radius_m = float(robot_cfg.get("wheel_radius_m", 0.0) or 0.0)

        drivetrain_length_m = wheel_base_m + 2 * wheel_radius_m
        drivetrain_width_m = wheel_track_m + 2 * wheel_radius_m
        base_diameter_m = float(robot_cfg.get("radius_m", 0.0)) * 2.0

        width_candidates = [
            robot_cfg.get("width_m"),
            # drivetrain_width_m if drivetrain_width_m > 0 else None,
            # robot_cfg.get("side_length_m"),
            # base_diameter_m if base_diameter_m > 0 else None,
        ]
        height_candidates = [
            robot_cfg.get("height_m"),
            # drivetrain_length_m if drivetrain_length_m > 0 else None,
            # robot_cfg.get("side_length_m"),
            # base_diameter_m if base_diameter_m > 0 else None,
        ]

        valid_widths = [float(value) for value in width_candidates if value is not None]
        valid_heights = [float(value) for value in height_candidates if value is not None]

        footprint_width_m = float(max(valid_widths)) if valid_widths else 0.0
        footprint_height_m = float(max(valid_heights)) if valid_heights else 0.0

        super().__init__(
            render_mode=render_mode,
            config_path=config_path,
            min_target_clearance_m=min_target_clearance_m,
        )

        self.ROBOT_WIDTH_M = footprint_width_m
        self.ROBOT_HEIGHT_M = footprint_height_m

        self.ROBOT_HALF_WIDTH_M = self.ROBOT_WIDTH_M / 2.0
        self.ROBOT_HALF_HEIGHT_M = self.ROBOT_HEIGHT_M / 2.0
        self.ROBOT_HALF_WIDTH_PIXELS = self.ROBOT_HALF_WIDTH_M * self.PIXELS_PER_METER
        self.ROBOT_HALF_HEIGHT_PIXELS = (
            self.ROBOT_HALF_HEIGHT_M * self.PIXELS_PER_METER
        )

        self.ROBOT_HALF_DIAGONAL_M = math.hypot(
            self.ROBOT_HALF_WIDTH_M, self.ROBOT_HALF_HEIGHT_M
        )
        self.ROBOT_HALF_DIAGONAL_PIXELS = (
            self.ROBOT_HALF_DIAGONAL_M * self.PIXELS_PER_METER
        )

        self.ROBOT_WIDTH_PIXELS = self.ROBOT_WIDTH_M * self.PIXELS_PER_METER
        self.ROBOT_HEIGHT_PIXELS = self.ROBOT_HEIGHT_M * self.PIXELS_PER_METER
        self.ROBOT_RADIUS_M = self.ROBOT_HALF_DIAGONAL_M
        self.ROBOT_SAFETY_RADIUS_M = self.ROBOT_HALF_DIAGONAL_M
        self.ROBOT_BODY_RADIUS_M = self.ROBOT_HALF_DIAGONAL_M
        self.ROBOT_RADIUS_PIXELS = self.ROBOT_HALF_DIAGONAL_PIXELS
        self.ROBOT_SAFETY_RADIUS_PIXELS = self.ROBOT_HALF_DIAGONAL_PIXELS

        self.map_creator.ROBOT_RADIUS_PIXELS = self.ROBOT_RADIUS_PIXELS

        self.WHEEL_HALF_BASE_PIXELS = (self.WHEEL_BASE_M / 2.0) * self.PIXELS_PER_METER
        self.WHEEL_HALF_TRACK_PIXELS = (
            self.WHEEL_TRACK_M / 2.0
        ) * self.PIXELS_PER_METER
        self.WHEEL_RADIUS_PIXELS = self.WHEEL_RADIUS_M * self.PIXELS_PER_METER

    def _rectangle_vertices(
        self, position: Tuple[float, float], theta: float
    ) -> List[Tuple[int, int]]:
        position = self._axle_to_body_center(position, theta)
        half_width = self.ROBOT_HALF_WIDTH_PIXELS
        half_height = self.ROBOT_HALF_HEIGHT_PIXELS
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        local_vertices = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        ]

        vertices: List[Tuple[int, int]] = []
        for dx, dy in local_vertices:
            rotated_x = dx * cos_theta - dy * sin_theta
            rotated_y = dx * sin_theta + dy * cos_theta
            vertices.append((int(position[0] + rotated_x), int(position[1] + rotated_y)))
        return vertices

    def _axle_to_body_center(
        self, position: Tuple[float, float], theta: float
    ) -> Tuple[float, float]:
        forward_offset = self.WHEEL_HALF_BASE_PIXELS
        if forward_offset == 0:
            return position

        offset_x = -forward_offset * math.sin(theta)
        offset_y = -forward_offset * math.cos(theta)

        return position[0] + offset_x, position[1] + offset_y

    def _rectangle_perimeter_points(
        self, position: Tuple[float, float], theta: float
    ) -> Iterable[Tuple[int, int]]:
        vertices = self._rectangle_vertices(position, theta)
        samples_per_edge = max(1, self.NUM_CHECK_POINTS // 4)

        for i in range(4):
            start_vertex = vertices[i]
            end_vertex = vertices[(i + 1) % 4]
            for step in range(samples_per_edge + 1):
                t = step / samples_per_edge
                x = int(round(start_vertex[0] * (1 - t) + end_vertex[0] * t))
                y = int(round(start_vertex[1] * (1 - t) + end_vertex[1] * t))
                yield x, y

    def _project_polygon(self, vertices: List[Tuple[int, int]], axis: Tuple[float, float]):
        projections = [vertex[0] * axis[0] + vertex[1] * axis[1] for vertex in vertices]
        return min(projections), max(projections)

    def _rectangles_intersect(
        self, first_vertices: List[Tuple[int, int]], second_vertices: List[Tuple[int, int]]
    ) -> bool:
        axes: List[Tuple[float, float]] = []
        for vertices in (first_vertices, second_vertices):
            for i in range(4):
                start_vertex = vertices[i]
                end_vertex = vertices[(i + 1) % 4]
                edge = (end_vertex[0] - start_vertex[0], end_vertex[1] - start_vertex[1])
                axis = (-edge[1], edge[0])
                axis_length = math.hypot(axis[0], axis[1])
                if axis_length == 0:
                    continue
                axes.append((axis[0] / axis_length, axis[1] / axis_length))

        for axis in axes:
            min_a, max_a = self._project_polygon(first_vertices, axis)
            min_b, max_b = self._project_polygon(second_vertices, axis)
            if max_a < min_b or max_b < min_a:
                return False
        return True

    def visualize_search_point(self, position, theta, color=(255, 0, 0)):
        for check_x, check_y in self._rectangle_perimeter_points(position, theta):
            if 0 <= check_x < self.SCREEN_W and 0 <= check_y < self.SCREEN_H:
                pygame.draw.circle(
                    self.robot_and_lidar_layer,
                    color,
                    (check_x, check_y),
                    1,
                )

    def _draw_rectangle(self, position: Tuple[float, float], theta: float, color):
        vertices = self._rectangle_vertices(position, theta)
        pygame.draw.polygon(self.robot_and_lidar_layer, color, vertices)

    def _draw_wheels(self):
        wheel_radius_pixels = self.WHEEL_RADIUS_PIXELS
        half_track = self.WHEEL_HALF_TRACK_PIXELS
        half_base = self.WHEEL_HALF_BASE_PIXELS

        if wheel_radius_pixels <= 0 or (half_track == 0 and half_base == 0):
            return

        half_side = max(1, int(round(wheel_radius_pixels)))

        cos_theta = math.cos(self.robot_theta)
        sin_theta = math.sin(self.robot_theta)

        local_wheel_centers = [
            (-half_track, -half_base),
            (half_track, -half_base),
            (-half_track, half_base),
            (half_track, half_base),
        ]

        wheel_color = utils.hex_to_rgb(self.ROBOT_COLOR)

        robot_body_center = self._axle_to_body_center(
            self.robot_position, self.robot_theta
        )

        for offset_x, offset_y in local_wheel_centers:
            rotated_x = offset_x * cos_theta - offset_y * sin_theta
            rotated_y = offset_x * sin_theta + offset_y * cos_theta
            wheel_center = (
                robot_body_center[0] + rotated_x,
                robot_body_center[1] + rotated_y,
            )

            local_corners = [
                (-half_side, -half_side),
                (half_side, -half_side),
                (half_side, half_side),
                (-half_side, half_side),
            ]

            wheel_vertices: List[Tuple[int, int]] = []
            for corner_x, corner_y in local_corners:
                rotated_corner_x = corner_x * cos_theta - corner_y * sin_theta
                rotated_corner_y = corner_x * sin_theta + corner_y * cos_theta
                wheel_vertices.append(
                    (
                        int(wheel_center[0] + rotated_corner_x),
                        int(wheel_center[1] + rotated_corner_y),
                    )
                )

            pygame.draw.polygon(self.robot_and_lidar_layer, wheel_color, wheel_vertices)

    def show_robot_and_lidar(self, sensor_data=None):
        if sensor_data is not False:
            lidar_point_color = utils.hex_to_rgb(self.ROBOT_COLOR)
            for element in sensor_data:
                obstacle_angle = self.robot_theta - math.pi / 2 - element[1]
                point = utils.AD2pos(
                    element[0],
                    obstacle_angle,
                    self.robot_position,
                    self.PIXELS_PER_METER,
                )
                pygame.draw.circle(
                    self.robot_and_lidar_layer,
                    lidar_point_color,
                    (int(point[0]), int(point[1])),
                    2,
                )

            frame_extent = self.ROBOT_HALF_DIAGONAL_PIXELS

            target_body_position = self._axle_to_body_center(
                self.target_position, self.target_theta
            )

            self.visualize_search_point(
                self.target_position, self.target_theta, self.TARGET_COLOR
            )
            self._draw_rectangle(
                self.target_position, self.target_theta, self.TARGET_COLOR
            )
            target_head_position = self._compute_orientation_endpoint(
                target_body_position, self.target_theta, frame_extent
            )
            pygame.draw.line(
                self.robot_and_lidar_layer,
                (255, 0, 0),
                target_body_position,
                target_head_position,
                2,
            )

            self.visualize_path()
            robot_body_position = self._axle_to_body_center(
                self.robot_position, self.robot_theta
            )
            self.visualize_search_point(
                self.robot_position, self.robot_theta, self.ROBOT_COLOR
            )
            self.visualize_goal_info(robot_body_position)
            self.visualize_closest_obstacle_info(robot_body_position)
            self._draw_rectangle(self.robot_position, self.robot_theta, self.ROBOT_COLOR)
            self._draw_wheels()
            robot_head_position = self._compute_orientation_endpoint(
                robot_body_position, self.robot_theta, frame_extent
            )
            pygame.draw.line(
                self.robot_and_lidar_layer,
                (255, 0, 0),
                robot_body_position,
                robot_head_position,
                2,
            )

            self.screen.blit(self.robot_and_lidar_layer, (0, 0))

    def is_collision(self, position):
        for check_x, check_y in self._rectangle_perimeter_points(
            position, self.robot_theta
        ):
            if not (0 <= check_x < self.SCREEN_W and 0 <= check_y < self.SCREEN_H):
                return True

            try:
                pixel_color = self.obstacle_layer.get_at((check_x, check_y))
            except IndexError:
                return True

            if (pixel_color[0], pixel_color[1], pixel_color[2]) == self._obstacle_color_rgb:
                return True
        return False

    def is_target_reached(self):
        robot_vertices = self._rectangle_vertices(self.robot_position, self.robot_theta)
        target_vertices = self._rectangle_vertices(
            self.target_position, self.target_theta
        )
        return self._rectangles_intersect(robot_vertices, target_vertices)
