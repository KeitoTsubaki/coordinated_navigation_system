"""Pygame based renderer for a multi-robot simulator.

The module reuses the configuration and map generation utilities provided by
:mod:`simulation_core`.  It is inspired by the single robot renderer inside
``simulation_core.environment`` and the ROS2 pygame example located at
``ros2_ws/src/matplotllib_2d_simulator``.  The code is adapted to support an
arbitrary number of robots while keeping the API simple so that it can be used
from reinforcement learning experiments or ROS2 nodes.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple, Union

import importlib
import importlib.resources
import importlib.util
import pygame
import textwrap

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment, minimize_scalar

_SPEC = importlib.util.find_spec("simulation_core.environment")
if _SPEC is None:
    _SPEC = importlib.util.find_spec("simulation_core.simulation_core.environment")
    if _SPEC is None:
        raise ModuleNotFoundError("Unable to locate simulation_core.environment")
    _environment_module = importlib.import_module("simulation_core.simulation_core.environment")
else:
    _environment_module = importlib.import_module("simulation_core.environment")

SimCoreEnvironment = _environment_module.SimCoreEnvironment

_sensor_module = importlib.import_module(f"{_environment_module.__package__}.sensor")
_utils_module = importlib.import_module(f"{_environment_module.__package__}.utils")

LiDARSensor = _sensor_module.LiDARSensor

Color = Tuple[int, int, int]
Position = Tuple[float, float]


def _to_rgb(color: Union[Color, str]) -> Color:
    if isinstance(color, str):
        return _utils_module.hex_to_rgb(color)
    return color


@dataclass
class RobotState:
    """State container used by :class:`MultiRobotSimulator` to render a robot."""

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


@dataclass
class FormationPlan:
    """Result of the constrained formation optimisation routine."""

    theta: float
    targets: List[Position]
    assignment: List[int]
    cost: float
    per_robot_costs: Dict[int, float]


class MultiRobotSimulator:
    """Utility that renders multiple differential drive robots using pygame."""

    def __init__(
        self,
        num_robots: int = 1,
        *,
        render_mode: str = "human",
        config_path: Optional[str] = None,
        path_history: int = 256,
        show_labels: bool = True,
        manual_control: bool = True,
    ) -> None:
        if config_path is None:
            config_path = str(
                importlib.resources.files("simulation_core").parent / "config" / "simulator_config.yaml"
            )

        self.env = SimCoreEnvironment(render_mode=render_mode, config_path=config_path)
        self.path_history = max(0, path_history)
        self.show_labels = show_labels
        self.running = True
        self.manual_control = manual_control
        self.selected_robot_index = 0
        self.manual_step_pixels = 5
        self.manual_rotation_step = 0.1
        self.info_panel_width = max(280, int(self.env.SCREEN_W * 0.4))
        self.info_panel_background = (245, 245, 245)
        self.info_panel_text_color = (32, 32, 32)
        self.info_panel_highlight = (220, 235, 250)
        self.info_panel_divider = (200, 200, 200)
        self.info_padding = 16
        self.map_descriptions = {
            "trap_shaped": "Trap-shaped course with a narrow passage leading to the target zone.",
            "zigzag": "Zigzag corridors that require repeated tight turns to reach the goal.",
            "random": "Randomly distributed obstacles that create unpredictable navigation challenges.",
            "empty": "Open field without obstacles for free navigation and debugging.",
        }
        self._clock = pygame.time.Clock()
        self._fps = 0.0

        self.robot_radius_pixels = (
            self.env.ROBOT_RADIUS_M * 0.8 * self.env.PIXELS_PER_METER
        )
        self.robot_frame_radius = self.env.ROBOT_RADIUS_M * self.env.PIXELS_PER_METER
        self.obstacle_color_rgb = _to_rgb(self.env.OBSTACLE_COLOR)
        self.lidar_line_color = _to_rgb(self.env.LIDAR_LINE_COLOR)
        self.lidar_filtered_points_color: Color = (64, 224, 208)
        self._lidar_detection_surface = pygame.Surface(
            (self.env.SCREEN_W, self.env.SCREEN_H), pygame.SRCALPHA
        )

        self._obstacle_mask: Optional[np.ndarray] = None
        self._obstacle_distance_map: Optional[np.ndarray] = None

        self.robot_states: List[RobotState] = []
        for i in range(num_robots):
            name = f"robot_{i}"
            color = self._color_from_palette(i)
            self.robot_states.append(
                RobotState(
                    name=name,
                    position=(self.env.SCREEN_CENTER_X, self.env.SCREEN_CENTER_Y),
                    theta=0.0,
                    color=color,
                    path=deque(maxlen=self.path_history or None),
                )
            )
            self.robot_states[-1].sensor = self._create_sensor()

        total_width = self.env.SCREEN_W + self.info_panel_width
        if self.env.render_mode == "human":
            pygame.display.set_mode((total_width, self.env.SCREEN_H))
            self.display_surface = pygame.display.get_surface()
            self.env.screen = pygame.Surface((self.env.SCREEN_W, self.env.SCREEN_H))
            self.env.map_creator.map = self.env.screen
            self.env.clear_all_layers()
        else:
            self.display_surface = pygame.Surface((total_width, self.env.SCREEN_H))
        self.simulation_surface = self.env.screen

        # Create the first map.  By default we use the same map type as the
        # single robot simulator so that existing scenarios work out of the box.
        self.set_map(self.env.map_type)

    # ------------------------------------------------------------------
    # Coordinate conversion helpers
    # ------------------------------------------------------------------
    def meters_to_pixels(self, x_m: float, y_m: float) -> Position:
        """Convert a position expressed in meters to screen coordinates."""

        px = self.env.SCREEN_CENTER_X + x_m * self.env.PIXELS_PER_METER
        py = self.env.SCREEN_CENTER_Y - y_m * self.env.PIXELS_PER_METER
        return px, py

    def pixels_to_meters(self, px: float, py: float) -> Position:
        """Convert screen coordinates to metric coordinates."""

        x_m = (px - self.env.SCREEN_CENTER_X) / self.env.PIXELS_PER_METER
        y_m = (self.env.SCREEN_CENTER_Y - py) / self.env.PIXELS_PER_METER
        return x_m, y_m

    # ------------------------------------------------------------------
    # Robot management API
    # ------------------------------------------------------------------
    def add_robot(
        self,
        name: str,
        *,
        color: Optional[Color] = None,
        position: Optional[Position] = None,
        theta: float = 0.0,
    ) -> int:
        """Add a new robot to the simulation and return its index."""

        color = color or self._color_from_palette(len(self.robot_states))
        position = position or (self.env.SCREEN_CENTER_X, self.env.SCREEN_CENTER_Y)
        robot = RobotState(
            name=name,
            color=color,
            position=position,
            theta=theta,
            path=deque(maxlen=self.path_history or None),
        )
        robot.sensor = self._create_sensor()
        self.robot_states.append(robot)
        return len(self.robot_states) - 1

    def set_robot_pose(
        self,
        robot_index: int,
        position: Position,
        theta: float,
        *,
        record_path: bool = True,
        in_meters: bool = False,
    ) -> None:
        """Update the pose of the specified robot."""

        robot = self.robot_states[robot_index]
        if in_meters:
            position = self.meters_to_pixels(*position)

        self._update_robot_motion(robot, (float(position[0]), float(position[1])))
        robot.theta = theta
        if record_path and self.path_history != 0:
            robot.path.append((robot.position, robot.theta))

    def set_robot_target(
        self,
        robot_index: int,
        target: Optional[Position],
        *,
        in_meters: bool = False,
    ) -> None:
        """Assign or clear the target for a robot."""

        robot = self.robot_states[robot_index]
        if target is not None and in_meters:
            target = self.meters_to_pixels(*target)
        robot.target = target

    def clear_paths(self) -> None:
        """Remove stored trajectories for every robot."""

        for robot in self.robot_states:
            robot.path.clear()

    def set_map(self, map_type: str) -> None:
        """Regenerate obstacles using the provided map type."""

        self._invalidate_obstacle_cache()
        self.env.reset(map_type)
        self.clear_paths()
        self._recenter_robots()

    # ------------------------------------------------------------------
    # Rendering utilities
    # ------------------------------------------------------------------
    def draw_scene(self) -> pygame.Surface:
        """Draw the current scene and return the main pygame surface."""

        self.simulation_surface = self.env.screen
        if self.env.render_mode == "human":
            self._clock.tick(60)
        else:
            self._clock.tick()
        self._fps = self._clock.get_fps()

        self.env.clear_robot_and_lidar_layer()
        for robot in self.robot_states:
            self._update_lidar(robot)
            if robot.path:
                self._draw_path(robot)
            if robot.target is not None:
                self._draw_target(robot)
            self._draw_robot(robot)
            if self.show_labels:
                self._draw_label(robot)
        self._draw_parallel_formation()
        self._draw_hud()
        self.env.screen.blit(self.env.robot_and_lidar_layer, (0, 0))

        self.display_surface.fill(self.env.BACKGROUND_COLOR)
        self.display_surface.blit(self.simulation_surface, (0, 0))
        self._draw_info_panel()

        if self.env.render_mode == "human":
            pygame.display.flip()

        return self.display_surface

    def draw_rgb_array(self):
        """Return the rendered surface as a NumPy array."""

        import numpy as np

        surface = self.draw_scene()
        array = pygame.surfarray.array3d(surface)
        return np.transpose(array, (1, 0, 2))

    def process_events(self) -> None:
        """Handle pygame events and update the :attr:`running` flag."""

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_c:
                    self._cycle_map()
                elif event.key == pygame.K_SPACE:
                    self.set_map(self.env.map_type)
                elif event.key == pygame.K_TAB:
                    self._select_next_robot()
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    index = event.key - pygame.K_1
                    if index < len(self.robot_states):
                        self.selected_robot_index = index
                elif event.key == pygame.K_0 and len(self.robot_states) >= 10:
                    self.selected_robot_index = 9
                elif event.key == pygame.K_l:
                    robot = self.robot_states[self.selected_robot_index]
                    robot.lidar_enabled = not robot.lidar_enabled

        if self.manual_control and self.robot_states:
            self._operate_moving()

    def close(self) -> None:
        """Shut down pygame resources."""

        pygame.quit()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _draw_robot(self, robot: RobotState) -> None:
        position = robot.position
        pygame.draw.circle(
            self.env.robot_and_lidar_layer,
            robot.color,
            (int(position[0]), int(position[1])),
            int(self.robot_radius_pixels),
        )
        head_position = (
            position[0] - self.robot_frame_radius * math.sin(robot.theta),
            position[1] - self.robot_frame_radius * math.cos(robot.theta),
        )
        pygame.draw.circle(
            self.env.robot_and_lidar_layer,
            robot.color,
            (int(position[0]), int(position[1])),
            int(self.robot_frame_radius),
            width=2,
        )
        pygame.draw.line(
            self.env.robot_and_lidar_layer,
            (255, 255, 255),
            (int(position[0]), int(position[1])),
            (int(head_position[0]), int(head_position[1])),
            width=2,
        )
        if robot is self.robot_states[self.selected_robot_index]:
            pygame.draw.circle(
                self.env.robot_and_lidar_layer,
                (0, 255, 0),
                (int(position[0]), int(position[1])),
                int(self.robot_frame_radius + 6),
                width=2,
            )

    def _draw_target(self, robot: RobotState) -> None:
        assert robot.target is not None
        color = self._brighten_color(robot.color)
        pygame.draw.circle(
            self.env.robot_and_lidar_layer,
            color,
            (int(robot.target[0]), int(robot.target[1])),
            int(self.robot_radius_pixels * 0.6),
            width=2,
        )

    def _draw_path(self, robot: RobotState) -> None:
        points = list(robot.path)
        for (start_pos, _), (end_pos, _) in zip(points[:-1], points[1:]):
            pygame.draw.line(
                self.env.robot_and_lidar_layer,
                self._with_alpha(robot.color, 0.6),
                (int(start_pos[0]), int(start_pos[1])),
                (int(end_pos[0]), int(end_pos[1])),
                width=2,
            )

    def _draw_label(self, robot: RobotState) -> None:
        font = pygame.font.SysFont("arial", 16)
        lines = [robot.name]
        if robot.target is not None:
            tx, ty = robot.target
            distance = math.hypot(
                tx - robot.position[0], ty - robot.position[1]
            ) / self.env.PIXELS_PER_METER
            lines.append(f"target: {distance:.1f}m")
        if robot.closest_obstacle[0] != float("inf"):
            lines.append(f"obstacle: {robot.closest_obstacle[0]:.1f}m")
        for line_index, text in enumerate(lines):
            label = font.render(text, True, (0, 0, 0))
            text_position = (
                int(robot.position[0] - label.get_width() / 2),
                int(
                    robot.position[1]
                    - self.robot_frame_radius
                    - label.get_height() * (len(lines) - line_index)
                    - 4
                ),
            )
            self.env.robot_and_lidar_layer.blit(label, text_position)

    def _draw_hud(self) -> None:
        if not self.robot_states:
            return
        font = pygame.font.SysFont("arial", 16)
        selected = self.robot_states[self.selected_robot_index]
        hud_lines = [
            f"Controls: TAB/0-9 select, W move forward, Q/E rotate, C cycle map, SPACE reset, L toggle LiDAR",
        ]
        for index, text in enumerate(hud_lines):
            label = font.render(text, True, (0, 0, 0))
            self.env.robot_and_lidar_layer.blit(label, (10, 10 + index * 20))

    def _draw_parallel_formation(self) -> None:
        if len(self.robot_states) < 2:
            return

        formation_color = (30, 144, 255)
        center_color = (255, 140, 0)
        text_color = (25, 25, 25)
        font = pygame.font.SysFont("arial", 14)

        sorted_states = sorted(self.robot_states, key=lambda robot: robot.position[0])
        positions = [robot.position for robot in sorted_states]

        center_x = sum(position[0] for position in positions) / len(positions)
        center_y = sum(position[1] for position in positions) / len(positions)
        pygame.draw.circle(
            self.env.robot_and_lidar_layer,
            center_color,
            (int(center_x), int(center_y)),
            6,
        )

        for first, second in zip(sorted_states[:-1], sorted_states[1:]):
            start = first.position
            end = second.position

            pygame.draw.line(
                self.env.robot_and_lidar_layer,
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

            distance_m = distance_pixels / self.env.PIXELS_PER_METER
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

            self.env.robot_and_lidar_layer.blit(distance_label, distance_position)
            self.env.robot_and_lidar_layer.blit(angle_label, angle_position)

    def _draw_info_panel(self) -> None:
        panel_left = self.env.SCREEN_W
        panel_rect = pygame.Rect(panel_left, 0, self.info_panel_width, self.env.SCREEN_H)
        pygame.draw.rect(self.display_surface, self.info_panel_background, panel_rect)
        pygame.draw.line(
            self.display_surface,
            self.info_panel_divider,
            (panel_left, 0),
            (panel_left, self.env.SCREEN_H),
            width=2,
        )

        title_font = pygame.font.SysFont("arial", 20, bold=True)
        name_font = pygame.font.SysFont("arial", 18, bold=True)
        value_font = pygame.font.SysFont("consolas", 16)

        x_offset = panel_left + self.info_padding
        y_offset = self.info_padding

        title = title_font.render("Simulation Info", True, self.info_panel_text_color)
        self.display_surface.blit(title, (x_offset, y_offset))
        y_offset += title.get_height() + 10

        selected_name = (
            self.robot_states[self.selected_robot_index].name
            if self.robot_states
            else "None"
        )
        general_lines = [
            f"FPS: {self._fps:5.1f}",
            f"Robots: {len(self.robot_states)}",
            f"Selected: {selected_name}",
        ]
        for line in general_lines:
            label = value_font.render(line, True, self.info_panel_text_color)
            self.display_surface.blit(label, (x_offset, y_offset))
            y_offset += label.get_height() + 4

        y_offset += 10

        map_header = title_font.render("Map", True, self.info_panel_text_color)
        self.display_surface.blit(map_header, (x_offset, y_offset))
        y_offset += map_header.get_height() + 8

        map_width_m = self.env.SCREEN_W / self.env.PIXELS_PER_METER
        map_height_m = self.env.SCREEN_H / self.env.PIXELS_PER_METER
        map_lines = [
            f"Type: {self.env.map_type}",
            f"Size: {map_width_m:.1f}m x {map_height_m:.1f}m",
        ]
        for line in map_lines:
            label = value_font.render(line, True, self.info_panel_text_color)
            self.display_surface.blit(label, (x_offset, y_offset))
            y_offset += label.get_height() + 4

        description = self.map_descriptions.get(self.env.map_type)
        if description:
            desc_label = value_font.render("Description:", True, self.info_panel_text_color)
            self.display_surface.blit(desc_label, (x_offset, y_offset))
            y_offset += desc_label.get_height() + 4

            wrap_width = max(1, (self.info_panel_width - 2 * self.info_padding) // 8)
            for line in textwrap.wrap(description, width=wrap_width):
                desc_line = value_font.render(line, True, self.info_panel_text_color)
                self.display_surface.blit(desc_line, (x_offset + 10, y_offset))
                y_offset += desc_line.get_height() + 2

            y_offset += 6

        robots_header = title_font.render("Robots", True, self.info_panel_text_color)
        self.display_surface.blit(robots_header, (x_offset, y_offset))
        y_offset += robots_header.get_height() + 8

        if not self.robot_states:
            empty_label = value_font.render("No robots", True, self.info_panel_text_color)
            self.display_surface.blit(empty_label, (x_offset, y_offset))
            return

        for index, robot in enumerate(self.robot_states):
            robot_name = name_font.render(robot.name, True, self.info_panel_text_color)
            metrics: List[str] = []
            pos_m = self.pixels_to_meters(*robot.position)
            metrics.append(f"Pos: ({pos_m[0]:.2f}, {pos_m[1]:.2f}) m")
            metrics.append(f"Theta: {math.degrees(robot.theta):.1f}°")
            metrics.append(f"Speed: {robot.velocity:.2f} m/s")

            if robot.target is not None:
                target_m = self.pixels_to_meters(*robot.target)
                distance = math.hypot(
                    target_m[0] - pos_m[0], target_m[1] - pos_m[1]
                )
                metrics.append(f"Target dist: {distance:.2f} m")
            else:
                metrics.append("Target: None")

            if robot.closest_obstacle[0] != float("inf"):
                metrics.append(f"Closest obs: {robot.closest_obstacle[0]:.2f} m")
            else:
                metrics.append("Closest obs: --")

            metrics.append(f"LiDAR: {'On' if robot.lidar_enabled else 'Off'}")

            line_height = value_font.get_height() + 4
            block_height = robot_name.get_height() + 4 + len(metrics) * line_height + 12
            block_rect = pygame.Rect(
                panel_left + 8,
                y_offset - 6,
                self.info_panel_width - 16,
                block_height,
            )
            background_color = (
                self.info_panel_highlight
                if index == self.selected_robot_index
                else self.info_panel_background
            )
            pygame.draw.rect(
                self.display_surface,
                background_color,
                block_rect,
                border_radius=8,
            )
            pygame.draw.rect(
                self.display_surface,
                self.info_panel_divider,
                block_rect,
                width=1,
                border_radius=8,
            )

            text_x = block_rect.x + 10
            text_y = y_offset
            self.display_surface.blit(robot_name, (text_x, text_y))
            text_y += robot_name.get_height() + 4

            for metric in metrics:
                metric_label = value_font.render(metric, True, self.info_panel_text_color)
                self.display_surface.blit(metric_label, (text_x, text_y))
                text_y += line_height

            y_offset = block_rect.bottom + 10

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _update_robot_motion(self, robot: RobotState, new_position: Position) -> None:
        previous_position = robot.position
        dx = new_position[0] - previous_position[0]
        dy = new_position[1] - previous_position[1]
        distance_pixels = math.hypot(dx, dy)

        current_time = (
            pygame.time.get_ticks() / 1000.0 if pygame.get_init() else None
        )

        velocity = 0.0
        if (
            current_time is not None
            and robot.last_update_time is not None
            and distance_pixels > 0.0
        ):
            dt = current_time - robot.last_update_time
            if dt > 0:
                distance_m = distance_pixels / self.env.PIXELS_PER_METER
                velocity = distance_m / dt

        robot.velocity = velocity
        robot.last_update_time = current_time
        robot.position = (float(new_position[0]), float(new_position[1]))

    @staticmethod
    def _with_alpha(color: Color, alpha: float) -> Color:
        r, g, b = color
        return (int(r * alpha), int(g * alpha), int(b * alpha))

    @staticmethod
    def _brighten_color(color: Color) -> Color:
        return tuple(min(255, int(channel * 1.2 + 30)) for channel in color)

    @staticmethod
    def _color_from_palette(index: int) -> Color:
        palette: Sequence[Color] = [
            (0, 128, 255),
            (255, 99, 71),
            (60, 179, 113),
            (238, 130, 238),
            (255, 215, 0),
            (0, 206, 209),
        ]
        return palette[index % len(palette)]

    def _create_sensor(self) -> LiDARSensor:
        return LiDARSensor(
            Range=self.env.LIDAR_RANGE_M,
            uncertainty=(0, 0),
            pixels_per_meter=self.env.PIXELS_PER_METER,
            lidar_resolution=self.env.LIDAR_RESOLUTION,
            screen_w=self.env.SCREEN_W,
            screen_h=self.env.SCREEN_H,
        )

    def _recenter_robots(self) -> None:
        if not self.robot_states:
            return

        start_spawn_area, target_spawn_area = self._get_spawn_areas()

        spacing = self.env.ROBOT_RADIUS_PIXELS * 2.5
        robot_positions = self._generate_randomized_formation_positions(
            start_spawn_area, len(self.robot_states), spacing
        )

        for robot, position in zip(self.robot_states, robot_positions):
            robot.position = (float(position[0]), float(position[1]))
            robot.theta = 0.0
            robot.path.clear()
            robot.sensor_data = []
            robot.closest_obstacle = (float("inf"), 0.0)
            robot.last_update_time = None
            robot.velocity = 0.0

        self._assign_random_targets(target_spawn_area)

    def _get_spawn_areas(self) -> Tuple[Optional[Sequence[float]], Optional[Sequence[float]]]:
        screen_center_xy = (self.env.SCREEN_CENTER_X, self.env.SCREEN_CENTER_Y)
        screen_size_wh = (self.env.SCREEN_W, self.env.SCREEN_H)
        start_spawn_area, target_spawn_area = self.env.map_creator.initialize_map_parameters(
            self.env.map_type,
            screen_center_xy,
            screen_size_wh,
            self.env.ROBOT_RADIUS_PIXELS,
        )
        return start_spawn_area, target_spawn_area

    def _assign_random_targets(
        self, target_spawn_area: Optional[Sequence[float]]
    ) -> None:
        if not self.robot_states:
            return

        spacing = self.env.ROBOT_RADIUS_PIXELS * 2.5
        target_positions = self._generate_randomized_formation_positions(
            target_spawn_area, len(self.robot_states), spacing
        )

        if not target_positions:
            return

        for robot, target in zip(self.robot_states, target_positions):
            robot.target = target

    def _invalidate_obstacle_cache(self) -> None:
        self._obstacle_mask = None
        self._obstacle_distance_map = None

    def _get_obstacle_mask(self) -> np.ndarray:
        if self._obstacle_mask is None:
            obstacle_surface = self.env.obstacle_layer
            obstacle_rgb = pygame.surfarray.array3d(obstacle_surface)
            color_array = np.array(self.obstacle_color_rgb, dtype=obstacle_rgb.dtype)
            mask = np.all(obstacle_rgb == color_array[None, None, :], axis=2)
            mask = mask.swapaxes(0, 1)
            mask[0, :] = True
            mask[-1, :] = True
            mask[:, 0] = True
            mask[:, -1] = True
            self._obstacle_mask = mask
        return self._obstacle_mask

    def _get_obstacle_distance_map(self) -> np.ndarray:
        if self._obstacle_distance_map is None:
            mask = self._get_obstacle_mask()
            free_space = ~mask
            distance_pixels = ndimage.distance_transform_edt(free_space)
            self._obstacle_distance_map = distance_pixels / self.env.PIXELS_PER_METER
        return self._obstacle_distance_map

    def _is_point_obstacle(self, point: Position) -> bool:
        x = int(round(point[0]))
        y = int(round(point[1]))
        if not (0 <= x < self.env.SCREEN_W and 0 <= y < self.env.SCREEN_H):
            return True
        mask = self._get_obstacle_mask()
        return bool(mask[y, x])

    def _distance_to_nearest_obstacle(self, point: Position) -> float:
        x = int(round(point[0]))
        y = int(round(point[1]))
        if not (0 <= x < self.env.SCREEN_W and 0 <= y < self.env.SCREEN_H):
            return 0.0
        distance_map = self._get_obstacle_distance_map()
        return float(distance_map[y, x])

    def _is_line_blocked(self, start: Position, end: Position) -> bool:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.hypot(dx, dy)
        steps = max(int(distance), 1)
        for step in range(1, steps + 1):
            t = step / steps
            x = start[0] + dx * t
            y = start[1] + dy * t
            if self._is_point_obstacle((x, y)):
                return True
        return False

    def _formation_offsets(
        self, count: int, formation_type: str, spacing: float
    ) -> List[Tuple[float, float]]:
        if count <= 0:
            return []

        formation_type = formation_type.lower()
        offsets: List[Tuple[float, float]] = []

        if formation_type in {"line", "parallel"}:
            center_index = (count - 1) / 2.0
            offsets = [((i - center_index) * spacing, 0.0) for i in range(count)]
        elif formation_type == "triangle":
            remaining = count
            row = 1
            spacing_y = spacing * math.sin(math.pi / 3)
            while remaining > 0:
                row_count = min(row, remaining)
                x_start = -(row_count - 1) / 2.0 * spacing
                y_offset = (row - 1) * spacing_y
                for i in range(row_count):
                    offsets.append((x_start + i * spacing, y_offset))
                remaining -= row_count
                row += 1
        elif formation_type in {"square", "grid"}:
            rows, columns = self._compute_formation_layout(count, None, spacing)
            for row_index in range(rows):
                y_offset = (row_index - (rows - 1) / 2.0) * spacing
                for column_index in range(columns):
                    if len(offsets) >= count:
                        break
                    x_offset = (column_index - (columns - 1) / 2.0) * spacing
                    offsets.append((x_offset, y_offset))
                if len(offsets) >= count:
                    break
        else:
            raise ValueError(f"Unsupported formation type: {formation_type}")

        if not offsets:
            return []

        mean_x = sum(x for x, _ in offsets) / len(offsets)
        mean_y = sum(y for _, y in offsets) / len(offsets)
        return [(x - mean_x, y - mean_y) for x, y in offsets]

    @staticmethod
    def _formation_angle_bounds(formation_type: str) -> Tuple[float, float]:
        formation_type = formation_type.lower()
        if formation_type in {"line", "parallel"}:
            return 0.0, math.pi
        if formation_type == "triangle":
            return 0.0, 2.0 * math.pi / 3.0
        if formation_type in {"square", "grid"}:
            return 0.0, math.pi / 2.0
        raise ValueError(f"Unsupported formation type: {formation_type}")

    def _generate_formation_targets(
        self,
        center: Position,
        theta: float,
        formation_type: str,
        spacing: float,
    ) -> List[Position]:
        offsets = self._formation_offsets(len(self.robot_states), formation_type, spacing)
        if not offsets:
            return []

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        targets: List[Position] = []
        for offset_x, offset_y in offsets:
            rotated_x = offset_x * cos_theta - offset_y * sin_theta
            rotated_y = offset_x * sin_theta + offset_y * cos_theta
            targets.append((center[0] + rotated_x, center[1] + rotated_y))
        return targets

    def _evaluate_formation_plan(
        self,
        center: Position,
        theta: float,
        formation_type: str,
        spacing: float,
        weights: Sequence[float],
        ego_index: Optional[int],
        penalty: float,
    ) -> Optional[FormationPlan]:
        targets = self._generate_formation_targets(center, theta, formation_type, spacing)
        robot_count = len(self.robot_states)
        if robot_count == 0 or not targets:
            return None

        weights = tuple(weights)
        if len(weights) != 4:
            raise ValueError("weights must contain four elements")

        cost_matrix = np.full((robot_count, robot_count), penalty, dtype=float)
        per_robot_costs: Dict[int, float] = {}

        for robot_index, robot in enumerate(self.robot_states):
            for target_index, target in enumerate(targets):
                if self._is_point_obstacle(target):
                    continue
                if ego_index is None or robot_index == ego_index:
                    if self._is_line_blocked(robot.position, target):
                        continue

                distance_to_obstacle = self._distance_to_nearest_obstacle(target)
                dx = target[0] - robot.position[0]
                dy = target[1] - robot.position[1]
                distance_pixels = math.hypot(dx, dy)
                distance_robot_target = distance_pixels / self.env.PIXELS_PER_METER

                if robot.target is not None:
                    previous_dx = target[0] - robot.target[0]
                    previous_dy = target[1] - robot.target[1]
                    distance_previous = (
                        math.hypot(previous_dx, previous_dy)
                        / self.env.PIXELS_PER_METER
                    )
                else:
                    distance_previous = 0.0

                heading_vector = (-math.sin(robot.theta), -math.cos(robot.theta))
                heading_norm = math.hypot(*heading_vector)
                heading_norm = heading_norm or 1.0
                target_norm = distance_pixels or 1.0
                dot_product = heading_vector[0] * dx + heading_vector[1] * dy
                cos_angle = max(-1.0, min(1.0, dot_product / (heading_norm * target_norm)))
                relative_angle = math.acos(cos_angle)

                c1 = max(2.0 - distance_to_obstacle, 0.0)
                c2 = distance_robot_target
                c3 = distance_previous
                c4 = relative_angle / math.pi

                total_cost = (
                    weights[0] * c1
                    + weights[1] * c2
                    + weights[2] * c3
                    + weights[3] * c4
                )

                cost_matrix[robot_index, target_index] = total_cost

        if np.all(cost_matrix >= penalty):
            return None

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignment = [0] * robot_count
        total_cost = 0.0
        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            if cost >= penalty:
                return None
            assignment[r] = c
            per_robot_costs[r] = cost
            total_cost += cost

        return FormationPlan(
            theta=theta,
            targets=targets,
            assignment=assignment,
            cost=total_cost,
            per_robot_costs=per_robot_costs,
        )

    def optimize_formation(
        self,
        formation_type: str = "line",
        *,
        weights: Optional[Sequence[float]] = None,
        ego_index: Optional[int] = None,
        spacing: Optional[float] = None,
        apply: bool = True,
    ) -> FormationPlan:
        """Solve the constrained optimisation that assigns formation targets."""

        if not self.robot_states:
            raise ValueError("No robots available for formation optimisation")

        center_x = sum(robot.position[0] for robot in self.robot_states) / len(
            self.robot_states
        )
        center_y = sum(robot.position[1] for robot in self.robot_states) / len(
            self.robot_states
        )
        center = (center_x, center_y)

        if spacing is None:
            spacing = self.env.ROBOT_RADIUS_PIXELS * 3.0

        weights = weights or (1.0, 1.0, 0.5, 0.3)
        ego_index = ego_index if ego_index is not None else self.selected_robot_index
        if ego_index is not None and not (0 <= ego_index < len(self.robot_states)):
            raise IndexError("ego_index is out of range")

        angle_min, angle_max = self._formation_angle_bounds(formation_type)
        penalty = 1e6
        best_plan: Dict[str, Optional[FormationPlan]] = {"plan": None}

        def objective(theta: float) -> float:
            plan = self._evaluate_formation_plan(
                center,
                theta,
                formation_type,
                spacing,
                weights,
                ego_index,
                penalty,
            )
            if plan is None:
                return penalty
            if best_plan["plan"] is None or plan.cost < best_plan["plan"].cost:
                best_plan["plan"] = plan
            return plan.cost

        result = minimize_scalar(objective, bounds=(angle_min, angle_max), method="bounded")
        objective(float(result.x))
        objective(angle_min)
        objective(angle_max)

        plan = best_plan["plan"]
        if plan is None:
            raise ValueError("Unable to find a feasible formation configuration")

        if apply:
            for robot_index, target_index in enumerate(plan.assignment):
                target = plan.targets[target_index]
                self.robot_states[robot_index].target = target

        return plan

    def _compute_formation_layout(
        self,
        count: int,
        spawn_area: Optional[Sequence[float]],
        spacing: float,
    ) -> Tuple[int, int]:
        if count <= 0:
            return 0, 0

        if spawn_area is None:
            columns = max(1, math.ceil(math.sqrt(count)))
            rows = math.ceil(count / columns)
        else:
            min_x, max_x, min_y, max_y = spawn_area
            available_width = max(0.0, max_x - min_x)
            available_height = max(0.0, max_y - min_y)

            max_columns = max(1, int(available_width // spacing) + 1)
            max_rows = max(1, int(available_height // spacing) + 1)

            columns = min(count, max_columns)
            rows = math.ceil(count / columns)

            while rows > max_rows and columns > 1:
                columns -= 1
                rows = math.ceil(count / columns)

            rows = max(rows, 1)

        return rows, columns

    def _sample_formation_center(
        self,
        spawn_area: Optional[Sequence[float]],
        spacing: float,
        rows: int,
        columns: int,
    ) -> Position:
        if spawn_area is None:
            min_x = float(self.env.ROBOT_RADIUS_PIXELS)
            max_x = float(self.env.SCREEN_W - self.env.ROBOT_RADIUS_PIXELS)
            min_y = float(self.env.ROBOT_RADIUS_PIXELS)
            max_y = float(self.env.SCREEN_H - self.env.ROBOT_RADIUS_PIXELS)
        else:
            min_x, max_x, min_y, max_y = spawn_area

        half_width = spacing * max(columns - 1, 0) / 2.0
        half_height = spacing * max(rows - 1, 0) / 2.0

        margin_x = half_width + self.robot_frame_radius
        margin_y = half_height + self.robot_frame_radius

        allowed_min_x = min_x + margin_x
        allowed_max_x = max_x - margin_x
        allowed_min_y = min_y + margin_y
        allowed_max_y = max_y - margin_y

        if allowed_min_x > allowed_max_x:
            center_x = (min_x + max_x) / 2.0
        else:
            center_x = float(
                self.env.np_random.uniform(allowed_min_x, allowed_max_x)
            )

        if allowed_min_y > allowed_max_y:
            center_y = (min_y + max_y) / 2.0
        else:
            center_y = float(
                self.env.np_random.uniform(allowed_min_y, allowed_max_y)
            )

        return center_x, center_y

    def _generate_randomized_formation_positions(
        self,
        spawn_area: Optional[Sequence[float]],
        count: int,
        spacing: float,
    ) -> List[Position]:
        if count <= 0:
            return []

        rows, columns = self._compute_formation_layout(count, spawn_area, spacing)
        if rows == 0 or columns == 0:
            return []

        max_attempts = 32 if spawn_area is not None else 8
        fallback_positions: Optional[List[Position]] = None

        for _ in range(max_attempts):
            center = self._sample_formation_center(spawn_area, spacing, rows, columns)
            positions = self._generate_formation_positions(
                spawn_area,
                count,
                spacing,
                rows=rows,
                columns=columns,
                center=center,
                enforce_bounds=False,
            )

            if self._are_spawn_positions_valid(positions, spawn_area):
                return positions

            if fallback_positions is None:
                fallback_positions = positions

        enforced_positions = self._generate_formation_positions(
            spawn_area,
            count,
            spacing,
            rows=rows,
            columns=columns,
            center=None,
            enforce_bounds=True,
        )

        if self._are_spawn_positions_valid(enforced_positions, spawn_area):
            return enforced_positions

        if fallback_positions is not None:
            return fallback_positions

        return enforced_positions

    def _generate_formation_positions(
        self,
        spawn_area: Optional[Sequence[float]],
        count: int,
        spacing: float,
        *,
        rows: Optional[int] = None,
        columns: Optional[int] = None,
        center: Optional[Position] = None,
        enforce_bounds: bool = True,
    ) -> List[Position]:
        if count <= 0:
            return []

        if spawn_area is None:
            min_x = float(self.env.SCREEN_CENTER_X)
            max_x = min_x
            min_y = float(self.env.SCREEN_CENTER_Y)
            max_y = min_y
        else:
            min_x, max_x, min_y, max_y = spawn_area

        if rows is None or columns is None:
            rows, columns = self._compute_formation_layout(count, spawn_area, spacing)

        if center is None:
            if spawn_area is None:
                center_x = self.env.SCREEN_CENTER_X
                center_y = self.env.SCREEN_CENTER_Y
            else:
                center_x = (min_x + max_x) / 2.0
                center_y = (min_y + max_y) / 2.0
        else:
            center_x, center_y = center

        formation_positions: List[Position] = []
        row_offsets = [
            (row - (rows - 1) / 2.0) * spacing for row in range(rows)
        ]
        column_offsets = [
            (col - (columns - 1) / 2.0) * spacing for col in range(columns)
        ]

        idx = 0
        for row_offset in row_offsets:
            for col_offset in column_offsets:
                if idx >= count:
                    break
                x = center_x + col_offset
                y = center_y + row_offset

                if enforce_bounds and spawn_area is not None:
                    x = min(max(x, min_x + self.robot_frame_radius), max_x - self.robot_frame_radius)
                    y = min(max(y, min_y + self.robot_frame_radius), max_y - self.robot_frame_radius)

                formation_positions.append((float(x), float(y)))
                idx += 1
            if idx >= count:
                break

        return formation_positions

    def _are_spawn_positions_valid(
        self, positions: Sequence[Position], spawn_area: Optional[Sequence[float]]
    ) -> bool:
        if not positions:
            return True

        if spawn_area is None:
            min_x = float(self.env.ROBOT_RADIUS_PIXELS)
            max_x = float(self.env.SCREEN_W - self.env.ROBOT_RADIUS_PIXELS)
            min_y = float(self.env.ROBOT_RADIUS_PIXELS)
            max_y = float(self.env.SCREEN_H - self.env.ROBOT_RADIUS_PIXELS)
        else:
            min_x, max_x, min_y, max_y = spawn_area

        for x, y in positions:
            if not (min_x <= x <= max_x and min_y <= y <= max_y):
                return False
            if self.env.is_collision((x, y)):
                return False

        return True

    def _cycle_map(self) -> None:
        next_map_type = _utils_module.switch_map_type(
            self.env.map_type, self.env.np_random, switch_mode="cycle"
        )
        if next_map_type is not None:
            self.set_map(next_map_type)

    def _operate_moving(self) -> None:
        robot = self.robot_states[self.selected_robot_index]
        keys = pygame.key.get_pressed()
        moved = False
        translated = False

        if keys[pygame.K_w]:
            move_success = self._attempt_move(robot, -self.manual_step_pixels)
            moved |= move_success
            translated |= move_success
        # if keys[pygame.K_s]:
        #     moved |= self._attempt_move(robot, self.manual_step_pixels)
        if keys[pygame.K_q]:
            robot.theta += self.manual_rotation_step
            moved = True
        if keys[pygame.K_e]:
            robot.theta -= self.manual_rotation_step
            moved = True

        if moved:
            robot.theta = (robot.theta + math.pi) % (2 * math.pi) - math.pi

        if moved and self.path_history != 0:
            robot.path.append((robot.position, robot.theta))
        if not translated:
            robot.velocity = 0.0

    def _attempt_move(self, robot: RobotState, step: float) -> bool:
        dx = step * math.sin(robot.theta)
        dy = step * math.cos(robot.theta)
        next_position = (robot.position[0] + dx, robot.position[1] + dy)

        if self._is_collision(next_position, ignore_robot=robot):
            return False

        self._update_robot_motion(robot, next_position)
        return True

    def _is_collision(
        self, position: Position, *, ignore_robot: Optional[RobotState] = None
    ) -> bool:
        for i in range(self.env.NUM_CHECK_POINTS):
            angle = 2 * math.pi * i / self.env.NUM_CHECK_POINTS
            check_x = int(position[0] + self.env.ROBOT_RADIUS_PIXELS * math.cos(angle))
            check_y = int(position[1] + self.env.ROBOT_RADIUS_PIXELS * math.sin(angle))

            if not (0 <= check_x < self.env.SCREEN_W and 0 <= check_y < self.env.SCREEN_H):
                return True

            pixel_color = self.env.obstacle_layer.get_at((check_x, check_y))
            if (pixel_color[0], pixel_color[1], pixel_color[2]) == self.obstacle_color_rgb:
                return True

        for other in self.robot_states:
            if other is ignore_robot:
                continue

            distance = math.hypot(position[0] - other.position[0], position[1] - other.position[1])
            min_clearance = self.robot_frame_radius * 2
            if distance < min_clearance:
                return True

        return False

    def _update_lidar(self, robot: RobotState) -> None:
        if not robot.lidar_enabled or robot.sensor is None:
            robot.sensor_data = []
            robot.closest_obstacle = (float("inf"), 0.0)
            return

        robot.sensor.position = (
            int(robot.position[0]),
            int(robot.position[1]),
        )
        robot.sensor.theta = robot.theta
        detection_surface = self._lidar_detection_surface
        detection_surface.fill((0, 0, 0, 0))
        detection_surface.blit(self.env.obstacle_layer, (0, 0))

        for other in self.robot_states:
            if other is robot:
                continue

            pygame.draw.circle(
                detection_surface,
                self.obstacle_color_rgb,
                (int(other.position[0]), int(other.position[1])),
                int(self.robot_frame_radius),
            )

        raw_sensor_data = robot.sensor.sense_obstacles(
            detection_surface, self.env.OBSTACLE_COLOR
        )
        robot.sensor_data = list(raw_sensor_data) if raw_sensor_data else []
        robot.filtered_sensor_data = []
        if robot.sensor_data:
            self._filter_robot_reflections(robot)

        if robot.sensor_data:
            filtered_counts: dict[float, int] = {}
            if robot.filtered_sensor_data:
                for _distance, relative_angle in robot.filtered_sensor_data:
                    filtered_counts[relative_angle] = (
                        filtered_counts.get(relative_angle, 0) + 1
                    )

            for distance, relative_angle in robot.sensor_data:
                color = robot.color
                if filtered_counts.get(relative_angle):
                    filtered_counts[relative_angle] -= 1
                    color = self.lidar_filtered_points_color

                obstacle_angle = robot.theta - math.pi / 2 - relative_angle
                point = _utils_module.AD2pos(
                    distance,
                    obstacle_angle,
                    robot.position,
                    self.env.PIXELS_PER_METER,
                )
                # pygame.draw.line(
                #     self.env.robot_and_lidar_layer,
                #     self.lidar_line_color,
                #     (int(robot.position[0]), int(robot.position[1])),
                #     (int(point[0]), int(point[1])),
                #     1,
                # )
                pygame.draw.circle(
                    self.env.robot_and_lidar_layer,
                    color,
                    (int(point[0]), int(point[1])),
                    2,
                )
            robot.closest_obstacle = robot.sensor.get_closest_obstacle_info(
                robot.sensor_data
            )
        else:
            robot.closest_obstacle = (float("inf"), 0.0)

    def _filter_robot_reflections(self, robot: RobotState) -> None:
        if robot.sensor is None:
            return

        robot_radius_m = self.env.ROBOT_RADIUS_M
        lidar_range = robot.sensor.LIDAR_RANGE_M
        pixels_per_meter = self.env.PIXELS_PER_METER
        other_robot_filters = []

        for other in self.robot_states:
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

                if angle_diff <= angle_half_width * buffer and distance_min * (2-buffer) <= distance <= distance_max * buffer:
                    replaced_measurements.append((distance, relative_angle))
                    filtered_data.append((robot.sensor.out_of_range_value, relative_angle))
                    replaced = True
                    break

            if not replaced:
                filtered_data.append((distance, relative_angle))

        robot.sensor_data = filtered_data
        robot.filtered_sensor_data = replaced_measurements

    def _select_next_robot(self) -> None:
        if not self.robot_states:
            return
        self.selected_robot_index = (self.selected_robot_index + 1) % len(
            self.robot_states
        )


def demo() -> None:
    """Simple interactive demo used for manual testing."""

    simulator = MultiRobotSimulator(num_robots=2, render_mode="human", show_labels=False)
    simulator.set_robot_target(0, simulator.meters_to_pixels(1.0, 1.0))
    simulator.set_robot_target(1, simulator.meters_to_pixels(-1.0, -1.0))

    clock = pygame.time.Clock()
    try:
        while simulator.running:
            simulator.process_events()

            simulator.draw_scene()
            clock.tick(30)
    finally:
        simulator.close()


if __name__ == "__main__":
    demo()
