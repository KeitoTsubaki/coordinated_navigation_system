"""Pygame based renderer for a multi-robot simulator."""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame
from scipy import ndimage

from .env import LiDARSensor, SimCoreEnvironment, utils_module
from .formation import FormationManager
from .info_panel import InfoPanelRenderer
from .policy import PolicyController
from .types import Color, FormationPlan, Position, RobotState
from .utils import brighten_color, color_from_palette, to_rgb, with_alpha


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
        policy_model_path: Optional[str] = "multi_robot/model_ep11200.zip",
        show_info_panel: bool = True,
    ) -> None:
        if config_path is None:
            import importlib.resources

            config_path = str(
                importlib.resources.files("simulation_core")
                .joinpath("config", "simulator_config.yaml")
            )

        self.env = SimCoreEnvironment(render_mode=render_mode, config_path=config_path)
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

        self._obstacle_mask: Optional[np.ndarray] = None
        self._obstacle_distance_map: Optional[np.ndarray] = None

        self.policy_model_path = policy_model_path
        self._policy_controller: Optional[PolicyController] = (
            PolicyController(self, policy_model_path)
            if policy_model_path is not None
            else None
        )
        self.policy_control_active = False

        self.robot_states: List[RobotState] = []
        for i in range(num_robots):
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

        total_width = self.env.SCREEN_W + self.info_panel.width
        if self.env.render_mode == "human":
            pygame.display.set_mode((total_width, self.env.SCREEN_H))
            self.display_surface = pygame.display.get_surface()
            self.env.screen = pygame.Surface((self.env.SCREEN_W, self.env.SCREEN_H))
            self.env.map_creator.map = self.env.screen
            self.env.clear_all_layers()
        else:
            self.display_surface = pygame.Surface((total_width, self.env.SCREEN_H))
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
    ) -> None:
        robot = self.robot_states[robot_index]
        if target is not None and in_meters:
            target = self.meters_to_pixels(*target)
        robot.target = target
        self.formation.register_direct_target(robot_index, target)
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

        if self.env.render_mode == "human":
            pygame.display.flip()

        return self.display_surface

    def draw_rgb_array(self):
        import numpy as np

        surface = self.draw_scene()
        array = pygame.surfarray.array3d(surface)
        return np.transpose(array, (1, 0, 2))

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
        color = brighten_color(robot.color)
        pygame.draw.circle(
            self.env.robot_and_lidar_layer,
            color,
            (int(robot.target[0]), int(robot.target[1])),
            max(1, int(round(self.robot_radius_pixels))),
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
            "Controls: TAB/0-9 select, W move forward, Q/E rotate, C cycle map, SPACE reset, L toggle LiDAR, P toggle path, I toggle info panel, D toggle debug text",
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
        total_width = self.env.SCREEN_W + self.info_panel.width
        if self.env.render_mode == "human":
            pygame.display.set_mode((total_width, self.env.SCREEN_H))
            self.display_surface = pygame.display.get_surface()
        else:
            self.display_surface = pygame.Surface((total_width, self.env.SCREEN_H))

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

        spacing = self.env.ROBOT_RADIUS_PIXELS * 2.5
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
        target_positions = self.formation.generate_randomized_positions(
            target_spawn_area, len(self.robot_states), spacing
        )

        if not target_positions:
            return

        for index, (robot, target) in enumerate(zip(self.robot_states, target_positions)):
            robot.target = target
            self.formation.register_direct_target(index, target)
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

    def optimize_formation(
        self,
        formation_type: str = "triangle",
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


def demo() -> None:
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

