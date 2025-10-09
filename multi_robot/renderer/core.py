"""High level pygame renderer used by the multi robot simulator."""
from __future__ import annotations

from collections import deque
from typing import List, Optional

import pygame

from . import control, drawing, formations, motion, sensors, utils
from .environment import SimCoreEnvironment, default_config_path
from .state import Color, Position, RobotState


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
        config_path = config_path or default_config_path()

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
        self.obstacle_color_rgb = utils.to_rgb(self.env.OBSTACLE_COLOR)
        self.lidar_line_color = utils.to_rgb(self.env.LIDAR_LINE_COLOR)
        self.lidar_filtered_points_color = (64, 224, 208)
        self._lidar_detection_surface = pygame.Surface(
            (self.env.SCREEN_W, self.env.SCREEN_H), pygame.SRCALPHA
        )

        self.robot_states: List[RobotState] = []
        for i in range(num_robots):
            name = f"robot_{i}"
            color = utils.color_from_palette(i)
            robot = RobotState(
                name=name,
                position=(self.env.SCREEN_CENTER_X, self.env.SCREEN_CENTER_Y),
                theta=0.0,
                color=color,
                path=deque(maxlen=self.path_history or None),
            )
            robot.sensor = sensors.create_sensor(self)
            self.robot_states.append(robot)

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
        """Convert from screen coordinates back to meters."""

        x_m = (px - self.env.SCREEN_CENTER_X) / self.env.PIXELS_PER_METER
        y_m = (self.env.SCREEN_CENTER_Y - py) / self.env.PIXELS_PER_METER
        return x_m, y_m

    # ------------------------------------------------------------------
    # Robot management
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

        color = color or utils.color_from_palette(len(self.robot_states))
        position = position or (
            self.env.SCREEN_CENTER_X,
            self.env.SCREEN_CENTER_Y,
        )
        robot = RobotState(
            name=name,
            color=color,
            position=position,
            theta=theta,
            path=deque(maxlen=self.path_history or None),
        )
        robot.sensor = sensors.create_sensor(self)
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

        motion.update_robot_motion(self, robot, (float(position[0]), float(position[1])))
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
        formations.recenter_robots(self)

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
            sensors.update_lidar(self, robot)
            if robot.path:
                drawing.draw_path(self, robot)
            if robot.target is not None:
                drawing.draw_target(self, robot)
            drawing.draw_robot(self, robot)
            if self.show_labels:
                drawing.draw_label(self, robot)
        drawing.draw_parallel_formation(self)
        drawing.draw_hud(self)
        self.env.screen.blit(self.env.robot_and_lidar_layer, (0, 0))

        self.display_surface.fill(self.env.BACKGROUND_COLOR)
        self.display_surface.blit(self.simulation_surface, (0, 0))
        drawing.draw_info_panel(self)

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
                control.handle_keydown(self, event)

        if self.manual_control and self.robot_states:
            control.operate_moving(self)

    def close(self) -> None:
        """Shut down pygame resources."""

        pygame.quit()


def demo() -> None:
    """Simple interactive demo used for manual testing."""

    simulator = MultiRobotSimulator(
        num_robots=2, render_mode="human", show_labels=False
    )
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


__all__ = ["MultiRobotSimulator", "demo"]
