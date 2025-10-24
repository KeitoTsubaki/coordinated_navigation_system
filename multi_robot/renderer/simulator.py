"""Pygame based renderer for a multi-robot simulator."""

from __future__ import annotations

import argparse
import math
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame
from scipy import ndimage

try:  # pragma: no cover - optional dependency
    from pygame._sdl2.video import Renderer as SDLRenderer, Texture as SDLTexture, Window as SDLWindow
except ImportError:  # pragma: no cover - SDL2 extension may be unavailable during import time
    SDLRenderer = None  # type: ignore[assignment]
    SDLTexture = None  # type: ignore[assignment]
    SDLWindow = None  # type: ignore[assignment]

from .env import LiDARSensor, SimCoreEnvironment, utils_module
# from .formation import FormationManager
# from .formation_previous_research import FormationManager
from .formation_sidebyside_1 import FormationManager
from .info_panel import InfoPanelRenderer
from .policy import PolicyController
from .types import Color, FormationPlan, Position, RobotState
from .utils import brighten_color, color_from_palette, to_rgb, with_alpha


class MultiRobotSimulator:
    """Utility that renders multiple differential drive robots using pygame."""

    def __init__(
        self,
        num_robots: Optional[int] = None,
        *,
        render_mode: str = "human",
        config_path: Optional[str] = None,
        path_history: int = 256,
        show_labels: bool = True,
        manual_control: bool = True,
        policy_model_path: Optional[str] = "multi_robot/models/model_ep3700.zip", #"multi_robot/models/model_ep11200.zip",
        show_info_panel: bool = True,
        formation_type: Optional[str] = None,
        formation_spacing_m: Optional[float] = None,
    ) -> None:
        if config_path is None:
            import importlib.resources

            config_path = str(
                importlib.resources.files("simulation_core")
                .joinpath("config", "simulator_config.yaml")
            )

        self._policy_controller: Optional[PolicyController] = (
            PolicyController(self, policy_model_path)
            if policy_model_path is not None
            else None
        )
        self.policy_control_active = False

        self.env = SimCoreEnvironment(render_mode=render_mode, config_path=config_path)
        self.default_formation_robot_count = (
            self.env.DEFAULT_FORMATION_ROBOT_COUNT
        )
        self.default_formation_type = self.env.DEFAULT_FORMATION_TYPE
        self.default_formation_spacing = self.env.DEFAULT_FORMATION_SPACING_PIXELS
        self.default_formation_spacing_m = self.env.DEFAULT_FORMATION_SPACING_M
        self.formation_centroid_sampling_interval = (
            self.env.FORMATION_CENTROID_SAMPLING_INTERVAL
        )
        self.formation_centroid_look_back_steps = self.env.FORMATION_LOOK_BACK_STEPS
        self.formation_centroid_angular_tolerance_deg = (
            self.env.FORMATION_ANGULAR_TOLERANCE_DEG
        )
        self.formation_centroid_angular_tolerance_rad = (
            self.env.FORMATION_ANGULAR_TOLERANCE_RAD
        )
        self.formation_centroid_min_displacement_m = (
            self.env.FORMATION_MIN_DISPLACEMENT_M
        )
        self.formation_centroid_min_displacement_pixels = (
            self.env.FORMATION_MIN_DISPLACEMENT_PIXELS
        )
        if formation_type is not None:
            self.default_formation_type = formation_type
        if formation_spacing_m is not None:
            if formation_spacing_m <= 0:
                raise ValueError("formation_spacing_m must be positive")
            self.default_formation_spacing_m = formation_spacing_m
            self.default_formation_spacing = (
                formation_spacing_m * self.env.PIXELS_PER_METER
            )
        resolved_robot_count = (
            num_robots if num_robots is not None else self.default_formation_robot_count
        )
        if num_robots is not None:
            self.default_formation_robot_count = num_robots
        self.path_history = max(0, path_history)
        self.show_labels = show_labels
        self.running = True
        self.manual_control = manual_control
        self._manual_control_enabled = manual_control
        self.selected_robot_index = 0
        self.manual_step_pixels = 5
        self.manual_rotation_step = 0.1
        self.show_debug_text = True

        self._clock = pygame.time.Clock()
        self._fps = 0.0

        self.robot_radius_pixels = (
            self.env.ROBOT_RADIUS_M * 0.8 * self.env.PIXELS_PER_METER
        )
        self.robot_frame_radius = self.env.ROBOT_RADIUS_M * self.env.PIXELS_PER_METER
        self.obstacle_color_rgb = to_rgb(self.env.OBSTACLE_COLOR)
        self.lidar_line_color = to_rgb(self.env.LIDAR_LINE_COLOR)
        self.lidar_filtered_points_color: Color = (64, 224, 208)
        self._lidar_detection_surface = pygame.Surface(
            (self.env.SCREEN_W, self.env.SCREEN_H), pygame.SRCALPHA
        )

        self._brake_font: Optional[pygame.font.Font] = None

        self._obstacle_mask: Optional[np.ndarray] = None
        self._obstacle_distance_map: Optional[np.ndarray] = None

        self.policy_model_path = policy_model_path
        self.formation_vertical_constraint_enabled = True

        self.robot_states: List[RobotState] = []
        for i in range(resolved_robot_count):
            name = f"robot_{i}"
            color = color_from_palette(i)
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

        self.info_panel = InfoPanelRenderer(self)
        self.formation = FormationManager(self)
        self._last_display_size: Optional[Tuple[int, int]] = None

        self.info_panel_window: Optional[Any] = None
        self.info_panel_renderer: Optional[Any] = None
        self.info_panel_texture: Optional[Any] = None
        self._info_panel_window_size: Optional[Tuple[int, int]] = None
        self._info_panel_texture_size: Optional[Tuple[int, int]] = None

        total_width = self.env.SCREEN_W + self.info_panel.width
        if self.env.render_mode == "human":
            pygame.display.set_mode((self.env.SCREEN_W, self.env.SCREEN_H))
            self.display_surface = pygame.display.get_surface()
            self._last_display_size = self.display_surface.get_size()
            self.env.screen = pygame.Surface((self.env.SCREEN_W, self.env.SCREEN_H))
            self.env.map_creator.map = self.env.screen
            self.env.clear_all_layers()
        else:
            self.display_surface = pygame.Surface((total_width, self.env.SCREEN_H))
            self._last_display_size = self.display_surface.get_size()
        self.simulation_surface = self.env.screen

        self.map_descriptions = {
            "trap_shaped": "Trap-shaped course with a narrow passage leading to the target zone.",
            "zigzag": "Zigzag corridors that require repeated tight turns to reach the goal.",
            "random": "Randomly distributed obstacles that create unpredictable navigation challenges.",
            "empty": "Open field without obstacles for free navigation and debugging.",
        }

        self.set_map(self.env.map_type)

        if not show_info_panel:
            self.set_info_panel_visible(False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def fps(self) -> float:
        return self._fps

    @property
    def policy_controller(self) -> Optional[PolicyController]:
        return self._policy_controller

    # ------------------------------------------------------------------
    # Coordinate conversion helpers
    # ------------------------------------------------------------------
    def meters_to_pixels(self, x_m: float, y_m: float) -> Position:
        px = self.env.SCREEN_CENTER_X + x_m * self.env.PIXELS_PER_METER
        py = self.env.SCREEN_CENTER_Y - y_m * self.env.PIXELS_PER_METER
        return px, py

    def pixels_to_meters(self, px: float, py: float) -> Position:
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
        color = color or color_from_palette(len(self.robot_states))
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

    def load_policy_model(self, model_path: str) -> None:
        self.policy_model_path = model_path
        self._policy_controller = PolicyController(self, model_path)
        self.policy_control_active = False
        self.manual_control = self._manual_control_enabled

    def set_robot_pose(
        self,
        robot_index: int,
        position: Position,
        theta: float,
        *,
        record_path: bool = True,
        in_meters: bool = False,
    ) -> None:
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
        target_theta: Optional[float] = None,
    ) -> None:
        robot = self.robot_states[robot_index]
        if target is not None and in_meters:
            target = self.meters_to_pixels(*target)
        robot.target = target
        if target is None:
            robot.target_theta = None
        else:
            if target_theta is None:
                target_theta = self._infer_target_heading(robot.position, target)
            robot.target_theta = target_theta
        self.formation.register_direct_target(robot_index, target, robot.target_theta)
        if target is None:
            self.formation.clear()

    def clear_paths(self) -> None:
        for robot in self.robot_states:
            robot.path.clear()

    def set_map(self, map_type: str) -> None:
        self._invalidate_obstacle_cache()
        self.env.reset(map_type)
        self.clear_paths()
        self._recenter_robots()
        self.formation.clear()

    # ------------------------------------------------------------------
    # Rendering utilities
    # ------------------------------------------------------------------
    def draw_scene(self) -> pygame.Surface:
        self.simulation_surface = self.env.screen
        if self.env.render_mode == "human":
            self._clock.tick(60)
        else:
            self._clock.tick()
        self._fps = self._clock.get_fps()

        self.env.clear_robot_and_lidar_layer()
        for robot in self.robot_states:
            self._update_lidar(robot)
            if robot.path and robot.path_enabled:
                self._draw_path(robot)
            if robot.target is not None:
                self._draw_target(robot)
            self._draw_robot(robot)
            if self.show_labels and self.show_debug_text:
                self._draw_label(robot)
        self.formation.update_visualisation()
        self.formation.draw()
        self.formation.draw_parallel_overlay()
        self._draw_hud()
        self.env.screen.blit(self.env.robot_and_lidar_layer, (0, 0))

        self.display_surface.fill(self.env.BACKGROUND_COLOR)
        self.display_surface.blit(self.simulation_surface, (0, 0))
        if self.info_panel.visible:
            self.info_panel.draw()
        else:
            self.present_info_panel_surface(None, 0, self.env.SCREEN_H)

        if self.env.render_mode == "human":
            pygame.display.flip()

        return self.display_surface

    def draw_rgb_array(self):
        import numpy as np

        surface = self.draw_scene()
        array = pygame.surfarray.array3d(surface)
        return np.transpose(array, (1, 0, 2))

    # ------------------------------------------------------------------
    # Info panel window management
    # ------------------------------------------------------------------
    def present_info_panel_surface(
        self,
        surface: Optional[pygame.Surface],
        scroll_offset: int,
        view_height: int,
    ) -> None:
        if self.env.render_mode == "human":
            if surface is None or self.info_panel.width <= 0:
                self._hide_info_panel_window()
                return

            self._ensure_info_panel_window()
            if self.info_panel_renderer is None:
                return

            width, height = surface.get_size()
            if width == 0 or height == 0:
                self.info_panel_renderer.clear()
                self.info_panel_renderer.present()
                return

            if (
                self.info_panel_texture is None
                or self._info_panel_texture_size != (width, height)
            ):
                if self.info_panel_texture is not None and hasattr(
                    self.info_panel_texture, "destroy"
                ):
                    self.info_panel_texture.destroy()
                if SDLTexture is None:
                    raise RuntimeError(
                        "pygame._sdl2.video.Texture is required for multi-window rendering"
                    )
                self.info_panel_texture = SDLTexture.from_surface(
                    self.info_panel_renderer, surface
                )
                self._info_panel_texture_size = (width, height)
            else:
                assert self.info_panel_texture is not None
                self.info_panel_texture.update(surface)

            self.info_panel_renderer.clear()
            src_rect = pygame.Rect(0, scroll_offset, width, view_height)
            dst_rect = pygame.Rect(0, 0, width, view_height)
            # ``Renderer.copy`` is not universally available across pygame versions,
            # whereas ``Renderer.blit`` is. Use ``blit`` with an explicit destination
            # rectangle so the info panel contents are cropped and positioned
            # correctly even when the window is taller than the view.
            self.info_panel_renderer.blit(
                self.info_panel_texture, target=dst_rect, area=src_rect
            )
            self.info_panel_renderer.present()

            if self.info_panel_window is not None and hasattr(
                self.info_panel_window, "show"
            ):
                self.info_panel_window.show()
        else:
            if surface is None or self.info_panel.width <= 0:
                return
            dest_rect = pygame.Rect(
                self.env.SCREEN_W, 0, surface.get_width(), view_height
            )
            src_rect = pygame.Rect(0, scroll_offset, surface.get_width(), view_height)
            self.display_surface.blit(surface, dest_rect, src_rect)

    def _ensure_info_panel_window(self) -> None:
        if self.info_panel_window is None:
            if SDLWindow is None or SDLRenderer is None:
                raise RuntimeError(
                    "pygame._sdl2.video multi-window support is required but unavailable"
                )
            window_size = (
                max(1, self.info_panel.width or self.info_panel.min_width),
                self.env.SCREEN_H,
            )
            self.info_panel_window = SDLWindow("Simulation Info", size=window_size)
            self.info_panel_renderer = SDLRenderer(
                self.info_panel_window, accelerated=1, vsync=0
            )
            self._info_panel_window_size = window_size
            if hasattr(self.info_panel_renderer, "logical_size"):
                self.info_panel_renderer.logical_size = window_size

        self._resize_info_panel_window()

    def _resize_info_panel_window(self) -> None:
        if self.info_panel_window is None:
            return
        target_size = (max(1, self.info_panel.width), self.env.SCREEN_H)
        if self._info_panel_window_size != target_size:
            self.info_panel_window.size = target_size
            self._info_panel_window_size = target_size
            if self.info_panel_renderer is not None and hasattr(
                self.info_panel_renderer, "logical_size"
            ):
                self.info_panel_renderer.logical_size = target_size
            if self.info_panel_texture is not None and hasattr(
                self.info_panel_texture, "destroy"
            ):
                self.info_panel_texture.destroy()
                self.info_panel_texture = None
                self._info_panel_texture_size = None

    def _hide_info_panel_window(self) -> None:
        if self.info_panel_window is None:
            return
        if hasattr(self.info_panel_window, "hide"):
            self.info_panel_window.hide()
        if self.info_panel_renderer is not None:
            self.info_panel_renderer.clear()
            self.info_panel_renderer.present()


    def process_events(self) -> None:
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
                elif event.key == pygame.K_f:
                    enabled = self.formation.toggle_follow_formation()
                    if enabled:
                        print("Formation following enabled: robots will pursue formation targets.")
                    else:
                        print("Formation following disabled: robots will pursue direct goals.")
                elif event.key == pygame.K_v:
                    enabled = self.formation.toggle_vertical_constraint()
                    state_text = "enabled" if enabled else "disabled"
                    action_text = "respect" if enabled else "ignore"
                    print(
                        "Vertical formation heading constraint"
                        f" {state_text}: formation search will {action_text} perpendicular targets."
                    )
                elif event.key == pygame.K_d:
                    self.show_debug_text = not self.show_debug_text
                elif event.key == pygame.K_l:
                    robot = self.robot_states[self.selected_robot_index]
                    robot.lidar_enabled = not robot.lidar_enabled
                elif event.key == pygame.K_p:
                    robot = self.robot_states[self.selected_robot_index]
                    robot.path_enabled = not robot.path_enabled
                elif event.key == pygame.K_i:
                    self.toggle_info_panel()
                elif event.key == pygame.K_m:
                    controller = self._policy_controller
                    if controller is None and self.policy_model_path:
                        controller = PolicyController(self, self.policy_model_path)
                        self._policy_controller = controller

                    if controller is not None:
                        previous_state = controller.active
                        new_state = controller.toggle()
                        if new_state != previous_state:
                            self.policy_control_active = new_state
                            if new_state:
                                self.manual_control = False
                            else:
                                self.manual_control = self._manual_control_enabled
                    else:
                        print(
                            "Policy control is unavailable. Call load_policy_model() to configure a model."
                        )
            elif event.type == pygame.MOUSEWHEEL:
                mouse_x, _ = pygame.mouse.get_pos()
                if mouse_x >= self.env.SCREEN_W:
                    self.info_panel.scroll_panel(-event.y * self.info_panel.scroll_speed)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in (4, 5):
                    direction = -1 if event.button == 4 else 1
                    mouse_x, _ = pygame.mouse.get_pos()
                    if mouse_x >= self.env.SCREEN_W:
                        self.info_panel.scroll_panel(direction * self.info_panel.scroll_speed)

        if self.manual_control and self.robot_states:
            self._operate_moving()

        if self._policy_controller is not None:
            self._policy_controller.step()
            self.policy_control_active = self._policy_controller.active

    def close(self) -> None:
        if self.info_panel_texture is not None and hasattr(
            self.info_panel_texture, "destroy"
        ):
            self.info_panel_texture.destroy()
            self.info_panel_texture = None
        if self.info_panel_renderer is not None and hasattr(
            self.info_panel_renderer, "destroy"
        ):
            self.info_panel_renderer.destroy()
            self.info_panel_renderer = None
        if self.info_panel_window is not None and hasattr(
            self.info_panel_window, "destroy"
        ):
            self.info_panel_window.destroy()
            self.info_panel_window = None
        self._info_panel_window_size = None
        self._info_panel_texture_size = None
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
        head_position = self._compute_orientation_endpoint(
            position, robot.theta, self.robot_frame_radius
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
        if (
            robot is self.robot_states[self.selected_robot_index]
            and not self.policy_control_active
        ):
            pygame.draw.circle(
                self.env.robot_and_lidar_layer,
                (0, 255, 0),
                (int(position[0]), int(position[1])),
                int(self.robot_frame_radius + 6),
                width=2,
            )

        self._draw_velocity_arrow(robot)
        self._draw_angular_velocity_arc(robot)

        self._draw_speed_bar(robot)
        if robot.braking:
            self._draw_brake_overlay(robot)

    def _draw_velocity_arrow(self, robot: RobotState) -> None:
        v_e = getattr(robot, "executable_velocity", 0.0)
        if abs(v_e) < 1e-3:
            return

        pixels_per_meter = self.env.PIXELS_PER_METER
        length_pixels = v_e * pixels_per_meter
        if abs(length_pixels) < 1.0:
            return

        surface = self.env.robot_and_lidar_layer
        start_x, start_y = robot.position
        direction_x = -math.sin(robot.theta)
        direction_y = -math.cos(robot.theta)
        end_x = start_x + direction_x * length_pixels
        end_y = start_y + direction_y * length_pixels

        color = (255, 215, 0)
        width = 4
        start_pos = (int(round(start_x)), int(round(start_y)))
        end_pos = (int(round(end_x)), int(round(end_y)))
        pygame.draw.line(surface, color, start_pos, end_pos, width=width)

        angle = math.atan2(end_y - start_y, end_x - start_x)
        head_length = max(6.0, min(12.0, abs(length_pixels) * 0.25))
        head_angle = math.pi / 8

        left_x = end_x - head_length * math.cos(angle - head_angle)
        left_y = end_y - head_length * math.sin(angle - head_angle)
        right_x = end_x - head_length * math.cos(angle + head_angle)
        right_y = end_y - head_length * math.sin(angle + head_angle)

        pygame.draw.line(
            surface,
            color,
            end_pos,
            (int(round(left_x)), int(round(left_y))),
            width=width,
        )
        pygame.draw.line(
            surface,
            color,
            end_pos,
            (int(round(right_x)), int(round(right_y))),
            width=width,
        )

    def _draw_angular_velocity_arc(self, robot: RobotState) -> None:
        omega_e = getattr(robot, "executable_angular_velocity", 0.0)
        if abs(omega_e) < 1e-3:
            return

        arc_angle = min(abs(omega_e), math.tau)
        if arc_angle < 1e-3:
            return

        center_x, center_y = robot.position
        radius = self.robot_frame_radius + 10.0
        direction_x = -math.sin(robot.theta)
        direction_y = -math.cos(robot.theta)
        forward_angle = math.atan2(direction_y, direction_x)
        # Positive angular velocity should appear as a counter-clockwise rotation
        # around the robot. The screen space Y axis is inverted (down is
        # positive), so we flip the sign when mapping from the angular velocity
        # to the direction of the arc sweep.
        sign = -1.0 if omega_e >= 0.0 else 1.0
        segments = max(4, int(math.ceil(arc_angle / (math.pi / 24.0))))

        points: list[tuple[float, float]] = []
        for i in range(segments + 1):
            t = i / segments
            angle = forward_angle + sign * t * arc_angle
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((x, y))

        if len(points) < 2:
            return

        surface = self.env.robot_and_lidar_layer
        color = (100, 149, 237)
        width = 4
        point_list = [(int(round(x)), int(round(y))) for x, y in points]
        pygame.draw.lines(surface, color, False, point_list, width)

        end_x, end_y = points[-1]
        prev_x, prev_y = points[-2]
        angle = math.atan2(end_y - prev_y, end_x - prev_x)
        head_length = 10.0
        head_angle = math.pi / 8
        left_x = end_x - head_length * math.cos(angle - head_angle)
        left_y = end_y - head_length * math.sin(angle - head_angle)
        right_x = end_x - head_length * math.cos(angle + head_angle)
        right_y = end_y - head_length * math.sin(angle + head_angle)

        pygame.draw.line(
            surface,
            color,
            (int(round(end_x)), int(round(end_y))),
            (int(round(left_x)), int(round(left_y))),
            width=width,
        )
        pygame.draw.line(
            surface,
            color,
            (int(round(end_x)), int(round(end_y))),
            (int(round(right_x)), int(round(right_y))),
            width=width,
        )

    def _draw_speed_bar(self, robot: RobotState) -> None:
        v_max = getattr(self.env, "V_MAX", None)
        if v_max is None or v_max <= 0:
            return

        surface = self.env.robot_and_lidar_layer
        velocity = max(0.0, float(robot.velocity or 0.0))
        ratio = min(velocity / v_max, 1.0)

        bar_width = max(20, int(round(self.robot_frame_radius * 2)))
        bar_height = max(6, int(round(self.robot_radius_pixels * 0.4)))
        bar_offset = self.robot_frame_radius + bar_height + 6

        bar_left = robot.position[0] - bar_width / 2
        bar_top = robot.position[1] - bar_offset
        bar_rect = pygame.Rect(
            int(round(bar_left)),
            int(round(bar_top)),
            bar_width,
            bar_height,
        )

        pygame.draw.rect(surface, (60, 60, 60), bar_rect)

        fill_width = int(round((bar_rect.width - 2) * ratio))
        if fill_width > 0:
            fill_rect = pygame.Rect(
                bar_rect.left + 1,
                bar_rect.top + 1,
                fill_width,
                max(1, bar_rect.height - 2),
            )
            pygame.draw.rect(surface, (0, 200, 100), fill_rect)

        pygame.draw.rect(surface, (20, 20, 20), bar_rect, width=1)

        tick_color = (200, 200, 200)
        pygame.draw.line(
            surface,
            tick_color,
            (bar_rect.left, bar_rect.top - 1),
            (bar_rect.left, bar_rect.bottom + 1),
            width=1,
        )
        pygame.draw.line(
            surface,
            tick_color,
            (bar_rect.right - 1, bar_rect.top - 1),
            (bar_rect.right - 1, bar_rect.bottom + 1),
            width=1,
        )

    def _draw_brake_overlay(self, robot: RobotState) -> None:
        if self._brake_font is None:
            font_size = max(24, int(round(self.robot_radius_pixels*2)))
            self._brake_font = pygame.font.SysFont("arial", font_size, bold=True)

        font = self._brake_font
        text_surface = font.render("B", True, (255, 255, 255))
        shadow_surface = font.render("B", True, (0, 0, 0))
        center = (int(robot.position[0]), int(robot.position[1]))

        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shadow_rect = shadow_surface.get_rect(
                center=(center[0] + dx, center[1] + dy)
            )
            self.env.robot_and_lidar_layer.blit(shadow_surface, shadow_rect)

        text_rect = text_surface.get_rect(center=center)
        self.env.robot_and_lidar_layer.blit(text_surface, text_rect)

    def _draw_target(self, robot: RobotState) -> None:
        assert robot.target is not None
        color = brighten_color(robot.color)
        pygame.draw.circle(
            self.env.robot_and_lidar_layer,
            color,
            (int(robot.target[0]), int(robot.target[1])),
            max(1, int(round(self.robot_radius_pixels))),
            width=2,
        )
        if robot.target_theta is not None:
            start_x, start_y = robot.target
            end_x, end_y = self._compute_orientation_endpoint(
                robot.target, robot.target_theta, self.robot_frame_radius
            )
            pygame.draw.line(
                self.env.robot_and_lidar_layer,
                color,
                (int(round(start_x)), int(round(start_y))),
                (int(round(end_x)), int(round(end_y))),
                width=2,
            )

    def _draw_path(self, robot: RobotState) -> None:
        points = list(robot.path)
        for (start_pos, _), (end_pos, _) in zip(points[:-1], points[1:]):
            pygame.draw.line(
                self.env.robot_and_lidar_layer,
                with_alpha(robot.color, 0.6),
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
        hud_lines = [
            "Controls: TAB/0-9 select, W move forward, Q/E rotate, C cycle map, SPACE reset, L toggle LiDAR, P toggle path, I toggle info panel, D toggle debug text, F toggle formation follow, V toggle vertical constraint",
        ]
        for index, text in enumerate(hud_lines):
            label = font.render(text, True, (0, 0, 0))
            self.env.robot_and_lidar_layer.blit(label, (10, 10 + index * 20))

    def toggle_info_panel(self) -> None:
        self.info_panel.toggle()

    def set_info_panel_visible(self, visible: bool) -> None:
        self.info_panel.set_visible(visible)

    def resize_display_surface(self, new_info_panel_width: int) -> None:
        self.info_panel.update_width(new_info_panel_width)
        if self.env.render_mode == "human":
            if self.info_panel.visible and self.info_panel.width > 0:
                self._ensure_info_panel_window()
                self._resize_info_panel_window()
            else:
                self._hide_info_panel_window()
            self._last_display_size = (
                self.env.SCREEN_W,
                self.env.SCREEN_H,
            )
            return

        total_width = self.env.SCREEN_W + self.info_panel.width
        new_size = (total_width, self.env.SCREEN_H)

        needs_new_surface = (
            self._last_display_size != new_size
            or self.display_surface.get_size() != new_size
        )
        if needs_new_surface:
            self.display_surface = pygame.Surface(new_size)

        self._last_display_size = new_size

        self.display_surface.fill(self.env.BACKGROUND_COLOR)
        self.display_surface.blit(self.simulation_surface, (0, 0))

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

        spacing = self.default_formation_spacing
        robot_positions = self.formation.generate_randomized_positions(
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
            robot.braking = False

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

        spacing = self.default_formation_spacing
        target_positions = self.formation.generate_randomized_positions(
            target_spawn_area, len(self.robot_states), spacing
        )

        if not target_positions:
            return

        for index, (robot, target) in enumerate(zip(self.robot_states, target_positions)):
            heading = self._infer_target_heading(robot.position, target)
            robot.target = target
            robot.target_theta = heading
            self.formation.register_direct_target(index, target, heading)
        self.formation.clear()

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

    # ------------------------------------------------------------------
    # Perception driven helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_orientation_endpoint(
        origin: Position, theta: float, length: float
    ) -> Position:
        return (
            float(origin[0] - length * math.sin(theta)),
            float(origin[1] - length * math.cos(theta)),
        )

    def _infer_target_heading(self, start: Position, target: Position) -> float:
        dx = float(target[0]) - float(start[0])
        dy = float(target[1]) - float(start[1])
        norm = math.hypot(dx, dy)
        if norm <= 0.0:
            return 0.0

        vx = dx / norm
        vy = dy / norm
        heading = math.atan2(-vx, -vy)
        return self._normalize_angle(heading)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize *angle* to the range [-pi, pi]."""

        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _robot_relative_angle(self, robot: RobotState, point: Position) -> float:
        """Return the LiDAR-relative angle from *robot* to *point*.

        The renderer uses a screen coordinate system where ``y`` increases
        downward.  This helper mirrors the transformations applied by the
        LiDAR sensor so that formation planning can align world-space
        queries with per-robot measurements.
        """

        dx = point[0] - robot.position[0]
        dy = point[1] - robot.position[1]
        if dx == 0.0 and dy == 0.0:
            return 0.0

        global_angle = math.atan2(-dx, -dy)
        relative_angle = robot.theta - math.pi / 2 - global_angle
        return self._normalize_angle(relative_angle)

    def _robot_query_distance(self, robot: RobotState, point: Position) -> float:
        """Return the Euclidean distance in meters from *robot* to *point*."""

        dx = point[0] - robot.position[0]
        dy = point[1] - robot.position[1]
        return math.hypot(dx, dy) / self.env.PIXELS_PER_METER

    def _robot_observation_samples(
        self, robot: RobotState
    ) -> List[Tuple[float, float]]:
        """Collect LiDAR measurements relevant for visibility estimation."""

        if not robot.sensor_data:
            return []

        return [
            (float(distance), float(angle))
            for distance, angle in robot.sensor_data
        ]

    def _robot_observed_distance(
        self, robot: RobotState, point: Position, *, default: float = 0.0
    ) -> float:
        """Return the distance (in meters) measured along the ray to *point*.

        The estimate is derived solely from the robot's own LiDAR samples,
        ensuring that formation planning only reasons about obstacles that the
        robot has actually perceived.  When no point cloud is available we
        fall back to *default*, which should represent the most conservative
        assumption (typically ``0``).
        """

        samples = self._robot_observation_samples(robot)
        if not samples:
            return default

        target_angle = self._robot_relative_angle(robot, point)
        best_distance: Optional[float] = None
        best_diff = math.inf
        for distance, sample_angle in samples:
            diff = abs(self._normalize_angle(target_angle - sample_angle))
            if diff < best_diff - 1e-9:
                best_diff = diff
                best_distance = distance
            elif abs(diff - best_diff) <= 1e-9 and best_distance is not None:
                best_distance = min(best_distance, distance)

        if best_distance is None:
            return default
        return float(best_distance)

    def _robot_clearance(self, robot: RobotState, point: Position) -> float:
        """Estimate free space beyond *point* using only *robot* observations."""

        query_distance = self._robot_query_distance(robot, point)
        observed_distance = self._robot_observed_distance(robot, point, default=0.0)
        if observed_distance <= query_distance:
            return 0.0
        return observed_distance - query_distance

    def _robot_line_blocked(self, robot: RobotState, point: Position) -> bool:
        """Return ``True`` if *robot* perceives an obstacle before *point*.

        Unknown space is treated as blocked so that optimisation remains
        conservative when LiDAR data is unavailable.
        """

        query_distance = self._robot_query_distance(robot, point)
        if query_distance <= 0.0:
            return False

        observed_distance = self._robot_observed_distance(robot, point, default=0.0)
        if observed_distance <= 0.0:
            return True

        tolerance = 1e-3
        return observed_distance + tolerance < query_distance

    # ------------------------------------------------------------------
    # Debug helpers that consult the ground-truth map.
    # ------------------------------------------------------------------

    def _is_point_obstacle_map(self, point: Position) -> bool:
        x = int(round(point[0]))
        y = int(round(point[1]))
        if not (0 <= x < self.env.SCREEN_W and 0 <= y < self.env.SCREEN_H):
            return True
        mask = self._get_obstacle_mask()
        return bool(mask[y, x])

    def _distance_to_nearest_obstacle_map(self, point: Position) -> float:
        x = int(round(point[0]))
        y = int(round(point[1]))
        if not (0 <= x < self.env.SCREEN_W and 0 <= y < self.env.SCREEN_H):
            return 0.0
        distance_map = self._get_obstacle_distance_map()
        return float(distance_map[y, x])

    def _is_line_blocked_map(self, start: Position, end: Position) -> bool:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.hypot(dx, dy)
        steps = max(int(distance), 1)
        for step in range(1, steps + 1):
            t = step / steps
            x = start[0] + dx * t
            y = start[1] + dy * t
            if self._is_point_obstacle_map((x, y)):
                return True
        return False

    def optimize_formation(
        self,
        formation_type: Optional[str] = None,
        *,
        weights: Optional[Sequence[float]] = None,
        ego_index: Optional[int] = None,
        spacing: Optional[float] = None,
        apply: bool = True,
    ) -> FormationPlan:
        return self.formation.optimize(
            formation_type,
            weights=weights,
            ego_index=ego_index,
            spacing=spacing,
            apply=apply,
        )

    def _cycle_map(self) -> None:
        next_map_type = utils_module.switch_map_type(
            self.env.map_type, self.env.np_random, switch_mode="cycle"
        )
        if next_map_type is not None:
            self.set_map(next_map_type)

    def _operate_moving(self) -> None:
        robot = self.robot_states[self.selected_robot_index]
        keys = pygame.key.get_pressed()
        moved = False
        translated = False

        robot.braking = False
        robot.executable_velocity = 0.0
        robot.executable_angular_velocity = 0.0

        if keys[pygame.K_w]:
            move_success = self._attempt_move(robot, -self.manual_step_pixels)
            moved |= move_success
            translated |= move_success
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
                    color = self.lidar_filtered_points_color

                obstacle_angle = robot.theta - math.pi / 2 - relative_angle
                point = utils_module.AD2pos(
                    distance,
                    obstacle_angle,
                    robot.position,
                    self.env.PIXELS_PER_METER,
                )
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

                if angle_diff <= angle_half_width * buffer and distance_min * (2 - buffer) <= distance <= distance_max * buffer:
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


def _run_event_loop(simulator: "MultiRobotSimulator", fps_limit: int) -> None:
    clock = pygame.time.Clock()
    try:
        while simulator.running:
            simulator.process_events()
            simulator.draw_scene()
            clock.tick(fps_limit)
    finally:
        simulator.close()


def demo() -> None:
    simulator = MultiRobotSimulator(num_robots=2, render_mode="human", show_labels=False)
    simulator.set_robot_target(0, simulator.meters_to_pixels(1.0, 1.0))
    simulator.set_robot_target(1, simulator.meters_to_pixels(-1.0, -1.0))
    _run_event_loop(simulator, fps_limit=30)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Interactive multi-robot simulator with formation control."
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        help="Path to the simulator configuration YAML file.",
    )
    parser.add_argument(
        "--robots",
        type=int,
        dest="robots",
        help="Number of robots to spawn. Defaults to the configuration value.",
    )
    parser.add_argument(
        "--formation-type",
        dest="formation_type",
        help="Override the default formation label (e.g. line, triangle, square).",
    )
    parser.add_argument(
        "--formation-spacing",
        dest="formation_spacing",
        type=float,
        help="Override the default spacing between robots in metres.",
    )
    parser.add_argument(
        "--render-mode",
        dest="render_mode",
        choices=("human", "rgb_array"),
        default="human",
        help="Rendering mode used by pygame (default: human).",
    )
    parser.add_argument(
        "--fps",
        dest="fps_limit",
        type=int,
        default=30,
        help="Limit the draw loop to the specified frames per second (default: 30).",
    )
    args = parser.parse_args(argv)

    if args.robots is not None and args.robots <= 0:
        parser.error("--robots must be a positive integer")
    if args.formation_spacing is not None and args.formation_spacing <= 0:
        parser.error("--formation-spacing must be positive")
    if args.fps_limit <= 0:
        parser.error("--fps must be positive")

    simulator = MultiRobotSimulator(
        num_robots=args.robots,
        render_mode=args.render_mode,
        config_path=args.config_path,
        formation_type=args.formation_type,
        formation_spacing_m=args.formation_spacing,
    )
    _run_event_loop(simulator, fps_limit=args.fps_limit)


if __name__ == "__main__":
    main()

