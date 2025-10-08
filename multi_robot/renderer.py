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
from typing import Deque, List, Optional, Sequence, Tuple, Union

import importlib
import importlib.resources
import importlib.util
import pygame

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
    closest_obstacle: Tuple[float, float] = (float("inf"), 0.0)
    lidar_enabled: bool = True


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

        self.robot_radius_pixels = (
            self.env.ROBOT_RADIUS_M * 0.8 * self.env.PIXELS_PER_METER
        )
        self.robot_frame_radius = self.env.ROBOT_RADIUS_M * self.env.PIXELS_PER_METER
        self.obstacle_color_rgb = _to_rgb(self.env.OBSTACLE_COLOR)
        self.lidar_line_color = _to_rgb(self.env.LIDAR_LINE_COLOR)
        self.lidar_points_color = _to_rgb(self.env.LIDAR_POINTS_COLOR)
        self._lidar_detection_surface = pygame.Surface(
            (self.env.SCREEN_W, self.env.SCREEN_H), pygame.SRCALPHA
        )

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

        robot.position = (float(position[0]), float(position[1]))
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

        self.env.reset(map_type)
        self.clear_paths()
        self._recenter_robots()

    # ------------------------------------------------------------------
    # Rendering utilities
    # ------------------------------------------------------------------
    def draw_scene(self) -> pygame.Surface:
        """Draw the current scene and return the main pygame surface."""

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
        self._draw_hud()
        self.env.screen.blit(self.env.robot_and_lidar_layer, (0, 0))

        if self.env.render_mode == "human":
            pygame.display.flip()

        return self.env.screen

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
            f"Map: {self.env.map_type}",
            f"Selected: {selected.name}",
            "Controls: TAB/0-9 select, W move forward, Q/E rotate, C cycle map, SPACE reset, L toggle LiDAR",
        ]
        for index, text in enumerate(hud_lines):
            label = font.render(text, True, (0, 0, 0))
            self.env.robot_and_lidar_layer.blit(label, (10, 10 + index * 20))

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
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
        robot_positions: List[Position] = []

        for robot in self.robot_states:
            position = self._sample_robot_spawn(start_spawn_area, robot_positions)
            robot.position = position
            robot.theta = 0.0
            robot.path.clear()
            robot.sensor_data = []
            robot.closest_obstacle = (float("inf"), 0.0)
            robot_positions.append(position)

        self._assign_random_targets(target_spawn_area, robot_positions)

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

    def _sample_robot_spawn(
        self,
        spawn_area: Optional[Sequence[float]],
        existing_positions: Sequence[Position],
    ) -> Position:
        min_clearance = self.env.ROBOT_RADIUS_PIXELS * 2.5
        max_attempts = 512

        for _ in range(max_attempts):
            candidate = self.env.randomize_position(spawn_area)
            if all(
                math.hypot(candidate[0] - other[0], candidate[1] - other[1]) >= min_clearance
                for other in existing_positions
            ):
                return float(candidate[0]), float(candidate[1])

        fallback = self.env.randomize_position(spawn_area)
        return float(fallback[0]), float(fallback[1])

    def _assign_random_targets(
        self,
        target_spawn_area: Optional[Sequence[float]],
        robot_positions: Sequence[Position],
    ) -> None:
        if target_spawn_area is None:
            for robot in self.robot_states:
                robot.target = None
            return

        min_target_distance = self._minimum_target_distance_pixels()
        assigned_targets: List[Position] = []

        for robot, robot_position in zip(self.robot_states, robot_positions):
            target = self._sample_target_position(
                target_spawn_area,
                robot_position,
                assigned_targets,
                robot_positions,
                min_target_distance,
            )
            robot.target = target
            assigned_targets.append(target)

    def _sample_target_position(
        self,
        spawn_area: Sequence[float],
        robot_position: Position,
        assigned_targets: Sequence[Position],
        robot_positions: Sequence[Position],
        min_robot_distance: float,
    ) -> Position:
        min_clearance = self.env.ROBOT_RADIUS_PIXELS * 2.0
        max_attempts = 512

        for _ in range(max_attempts):
            candidate = self.env.randomize_position(spawn_area)
            candidate_pos = (float(candidate[0]), float(candidate[1]))

            if math.hypot(candidate_pos[0] - robot_position[0], candidate_pos[1] - robot_position[1]) < min_robot_distance:
                continue

            if any(
                math.hypot(candidate_pos[0] - other[0], candidate_pos[1] - other[1]) < min_clearance
                for other in assigned_targets
            ):
                continue

            if any(
                math.hypot(candidate_pos[0] - other[0], candidate_pos[1] - other[1]) < min_clearance
                for other in robot_positions
                if other is not robot_position
            ):
                continue

            return candidate_pos

        fallback = self.env.randomize_position(spawn_area)
        return float(fallback[0]), float(fallback[1])

    def _minimum_target_distance_pixels(self) -> float:
        map_specific_thresholds = {
            "trap_shaped": self.env.T_DT_M_TRAP_SHAPED,
            "zigzag": self.env.T_DT_M_ZIGZAG,
            "random": self.env.T_DT_M_RANDOM,
            "empty": self.env.T_DT_M_EMPTY,
        }
        return map_specific_thresholds.get(self.env.map_type, 0.0) * self.env.PIXELS_PER_METER

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

        if keys[pygame.K_w]:
            moved |= self._attempt_move(robot, -self.manual_step_pixels)
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

    def _attempt_move(self, robot: RobotState, step: float) -> bool:
        dx = step * math.sin(robot.theta)
        dy = step * math.cos(robot.theta)
        next_position = (robot.position[0] + dx, robot.position[1] + dy)

        if self._is_collision(next_position, ignore_robot=robot):
            return False

        robot.position = next_position
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

        if robot.sensor_data:
            for distance, relative_angle in robot.sensor_data:
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
                    robot.color,
                    (int(point[0]), int(point[1])),
                    2,
                )
            robot.closest_obstacle = robot.sensor.get_closest_obstacle_info(
                robot.sensor_data
            )
        else:
            robot.closest_obstacle = (float("inf"), 0.0)

    def _select_next_robot(self) -> None:
        if not self.robot_states:
            return
        self.selected_robot_index = (self.selected_robot_index + 1) % len(
            self.robot_states
        )


def demo() -> None:
    """Simple interactive demo used for manual testing."""

    simulator = MultiRobotSimulator(num_robots=2, render_mode="human")
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
