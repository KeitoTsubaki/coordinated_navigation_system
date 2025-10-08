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
from typing import Deque, List, Optional, Sequence, Tuple

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

Color = Tuple[int, int, int]
Position = Tuple[float, float]


@dataclass
class RobotState:
    """State container used by :class:`MultiRobotSimulator` to render a robot."""

    name: str
    position: Position
    theta: float = 0.0
    color: Color = (0, 128, 255)
    target: Optional[Position] = None
    path: Deque[Tuple[Position, float]] = field(default_factory=deque)


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
    ) -> None:
        if config_path is None:
            config_path = str(
                importlib.resources.files("simulation_core")
                .joinpath("config", "simulator_config.yaml")
                .resolve()
            )

        self.env = SimCoreEnvironment(render_mode=render_mode, config_path=config_path)
        self.path_history = max(0, path_history)
        self.show_labels = show_labels
        self.running = True

        self.robot_radius_pixels = (
            self.env.ROBOT_RADIUS_M * 0.8 * self.env.PIXELS_PER_METER
        )
        self.robot_frame_radius = self.env.ROBOT_RADIUS_M * self.env.PIXELS_PER_METER

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

        robot.position = position
        robot.theta = theta
        if record_path and self.path_history != 0:
            robot.path.append((position, theta))

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

    # ------------------------------------------------------------------
    # Rendering utilities
    # ------------------------------------------------------------------
    def draw_scene(self) -> pygame.Surface:
        """Draw the current scene and return the main pygame surface."""

        self.env.clear_robot_and_lidar_layer()
        for robot in self.robot_states:
            if robot.path:
                self._draw_path(robot)
            if robot.target is not None:
                self._draw_target(robot)
            self._draw_robot(robot)
            if self.show_labels:
                self._draw_label(robot)
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
        label = font.render(robot.name, True, (255, 255, 255))
        text_position = (
            int(robot.position[0] - label.get_width() / 2),
            int(robot.position[1] - self.robot_frame_radius - label.get_height() - 4),
        )
        self.env.robot_and_lidar_layer.blit(label, text_position)

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


def demo() -> None:
    """Simple interactive demo used for manual testing."""

    simulator = MultiRobotSimulator(num_robots=2, render_mode="human")
    simulator.set_robot_target(0, simulator.meters_to_pixels(1.0, 1.0))
    simulator.set_robot_target(1, simulator.meters_to_pixels(-1.0, -1.0))

    t = 0.0
    clock = pygame.time.Clock()
    try:
        while simulator.running:
            simulator.process_events()
            # Simple circular trajectories for demonstration purposes.
            x0 = math.cos(t) * 2.0
            y0 = math.sin(t) * 2.0
            simulator.set_robot_pose(0, (x0, y0), theta=t / 2, in_meters=True)

            x1 = math.cos(-t) * 1.5
            y1 = math.sin(-t) * 1.5
            simulator.set_robot_pose(1, (x1, y1), theta=-t / 2, in_meters=True)

            simulator.draw_scene()
            clock.tick(30)
            t += 0.05
    finally:
        simulator.close()


if __name__ == "__main__":
    demo()
