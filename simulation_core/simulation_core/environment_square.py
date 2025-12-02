import importlib.resources
import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pygame
import yaml

from . import utils
from .environment import SimCoreEnvironment


class SimCoreSquareEnvironment(SimCoreEnvironment):
    """Simulation environment that models the robot as a square body."""

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

        square_side_m = robot_cfg.get("side_length_m")
        if square_side_m is None:
            width_m = robot_cfg.get("width_m")
            height_m = robot_cfg.get("height_m")
            if width_m is not None and height_m is not None:
                square_side_m = max(float(width_m), float(height_m))
            else:
                square_side_m = float(robot_cfg.get("radius_m", 0.0)) * 2.0
        self.ROBOT_SQUARE_SIDE_M = float(square_side_m)

        super().__init__(
            render_mode=render_mode,
            config_path=config_path,
            min_target_clearance_m=min_target_clearance_m,
        )

        self.map_creator.ROBOT_RADIUS_PIXELS = self.ROBOT_RADIUS_PIXELS

        self.ROBOT_HALF_SIDE_M = self.ROBOT_SQUARE_SIDE_M / 2.0
        self.ROBOT_HALF_DIAGONAL_M = math.sqrt(2.0) * self.ROBOT_HALF_SIDE_M
        self.ROBOT_HALF_SIDE_PIXELS = self.ROBOT_HALF_SIDE_M * self.PIXELS_PER_METER
        self.ROBOT_HALF_DIAGONAL_PIXELS = (
            self.ROBOT_HALF_DIAGONAL_M * self.PIXELS_PER_METER
        )

        self.ROBOT_WIDTH_M = self.ROBOT_SQUARE_SIDE_M
        self.ROBOT_HEIGHT_M = self.ROBOT_SQUARE_SIDE_M
        self.ROBOT_WIDTH_PIXELS = self.ROBOT_SQUARE_SIDE_M * self.PIXELS_PER_METER
        self.ROBOT_HEIGHT_PIXELS = self.ROBOT_SQUARE_SIDE_M * self.PIXELS_PER_METER
        self.ROBOT_RADIUS_M = self.ROBOT_HALF_DIAGONAL_M
        self.ROBOT_SAFETY_RADIUS_M = self.ROBOT_HALF_DIAGONAL_M
        self.ROBOT_BODY_RADIUS_M = self.ROBOT_HALF_DIAGONAL_M
        self.ROBOT_RADIUS_PIXELS = self.ROBOT_HALF_DIAGONAL_PIXELS
        self.ROBOT_SAFETY_RADIUS_PIXELS = self.ROBOT_HALF_DIAGONAL_PIXELS

    def _square_vertices(
        self, position: Tuple[float, float], theta: float
    ) -> List[Tuple[int, int]]:
        half_side = self.ROBOT_HALF_SIDE_PIXELS
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        local_vertices = [
            (-half_side, -half_side),
            (half_side, -half_side),
            (half_side, half_side),
            (-half_side, half_side),
        ]

        vertices: List[Tuple[int, int]] = []
        for dx, dy in local_vertices:
            rotated_x = dx * cos_theta - dy * sin_theta
            rotated_y = dx * sin_theta + dy * cos_theta
            vertices.append((int(position[0] + rotated_x), int(position[1] + rotated_y)))
        return vertices

    def _square_perimeter_points(
        self, position: Tuple[float, float], theta: float
    ) -> Iterable[Tuple[int, int]]:
        vertices = self._square_vertices(position, theta)
        samples_per_edge = max(1, self.NUM_CHECK_POINTS // 4)

        for i in range(4):
            start_vertex = vertices[i]
            end_vertex = vertices[(i + 1) % 4]
            for step in range(samples_per_edge + 1):
                t = step / samples_per_edge
                x = int(round(start_vertex[0] * (1 - t) + end_vertex[0] * t))
                y = int(round(start_vertex[1] * (1 - t) + end_vertex[1] * t))
                yield x, y

    def visualize_search_point(self, position, color=(255, 0, 0)):
        for check_x, check_y in self._square_perimeter_points(
            position, self.robot_theta
        ):
            if 0 <= check_x < self.SCREEN_W and 0 <= check_y < self.SCREEN_H:
                pygame.draw.circle(
                    self.robot_and_lidar_layer,
                    color,
                    (check_x, check_y),
                    1,
                )

    def _draw_square(self, position: Tuple[float, float], theta: float, color):
        vertices = self._square_vertices(position, theta)
        pygame.draw.polygon(self.robot_and_lidar_layer, color, vertices)

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

            self.visualize_search_point(self.target_position, self.TARGET_COLOR)
            self._draw_square(self.target_position, self.target_theta, self.TARGET_COLOR)
            target_head_position = self._compute_orientation_endpoint(
                self.target_position, self.target_theta, frame_extent
            )
            pygame.draw.line(
                self.robot_and_lidar_layer,
                (255, 0, 0),
                self.target_position,
                target_head_position,
                2,
            )

            self.visualize_path()
            self.visualize_search_point(self.robot_position, self.ROBOT_COLOR)
            self.visualize_goal_info(self.robot_position)
            self.visualize_closest_obstacle_info(self.robot_position)
            self._draw_square(self.robot_position, self.robot_theta, self.ROBOT_COLOR)
            robot_head_position = self._compute_orientation_endpoint(
                self.robot_position, self.robot_theta, frame_extent
            )
            pygame.draw.line(
                self.robot_and_lidar_layer,
                (255, 0, 0),
                self.robot_position,
                robot_head_position,
                2,
            )

            self.screen.blit(self.robot_and_lidar_layer, (0, 0))

    def is_collision(self, position):
        for check_x, check_y in self._square_perimeter_points(
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
        distance_to_target = math.sqrt(
            (self.target_position[0] - self.robot_position[0]) ** 2
            + (self.target_position[1] - self.robot_position[1]) ** 2
        )
        return distance_to_target < self.ROBOT_HALF_DIAGONAL_PIXELS
