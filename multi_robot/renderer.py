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
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple, Union

import importlib
import importlib.resources
import importlib.util
import pygame

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment, minimize_scalar
import yaml

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
    path_enabled: bool = True
    last_update_time: Optional[float] = None
    velocity: float = 0.0


@dataclass
class _PolicyRobotControlState:
    """Keeps track of temporal state required by the RL policy controller."""

    lidar_history: Deque[np.ndarray]
    previous_v: float = 0.0
    previous_omega: float = 0.0
    current_v: float = 0.0
    current_omega: float = 0.0


@dataclass
class FormationPlan:
    """Result of the constrained formation optimisation routine."""

    theta: float
    targets: List[Position]
    assignment: List[int]
    cost: float
    per_robot_costs: Dict[int, float]
    formation_type: str
    spacing: float
    center: Position


class PolicyController:
    """Applies a trained policy to control a robot inside the simulator."""

    def __init__(self, simulator: "MultiRobotSimulator", model_path: str) -> None:
        self.simulator = simulator
        self.model_path = model_path
        self.active = False
        self._model = None
        self._model_device = "cuda"
        self._last_error: Optional[str] = None
        self._last_action_timestamp: Optional[float] = None
        self._dt = 0.1  # Matches rl_training.gym_environment.Env.DT
        self._control_state: Dict[int, _PolicyRobotControlState] = {}
        self._lidar_history_len = 1
        self._n_l = 0
        self._n_e = 0
        self._load_policy_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        return self._model is not None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def toggle(self) -> bool:
        """Toggle policy-controlled mode. Returns the new active state."""

        if not self.is_available:
            self._initialise_model()

        if not self.is_available:
            return False

        self.active = not self.active
        if self.active:
            self._control_state.clear()
            self._last_action_timestamp = None
        else:
            self._last_action_timestamp = None
        return self.active

    def step(self) -> None:
        if not self.active or not self.is_available:
            return

        if not self.simulator.robot_states:
            return

        robot_index = self.simulator.selected_robot_index
        if not (0 <= robot_index < len(self.simulator.robot_states)):
            return

        now = pygame.time.get_ticks() / 1000.0 if pygame.get_init() else None
        if now is not None and self._last_action_timestamp is not None:
            if now - self._last_action_timestamp < self._dt:
                return

        robot = self.simulator.robot_states[robot_index]
        control_state = self._control_state.setdefault(
            robot_index,
            _PolicyRobotControlState(lidar_history=deque(maxlen=self._lidar_history_len)),
        )

        self.simulator._update_lidar(robot)
        observation = self._build_observation(robot, control_state)
        if observation is None:
            return

        action, _ = self._model.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 2:
            self._last_error = (
                f"Unexpected action dimension {action.shape}; expected size 2"
            )
            return

        v_e, omega_e = self._action_to_executable_action(action, control_state)
        moved = self._apply_motion(robot, v_e, omega_e)

        control_state.current_v = v_e if moved else 0.0
        control_state.current_omega = omega_e
        control_state.previous_v = control_state.current_v
        control_state.previous_omega = control_state.current_omega

        if self.simulator.path_history != 0 and moved:
            robot.path.append((robot.position, robot.theta))

        self.simulator._update_lidar(robot)

        if now is not None:
            self._last_action_timestamp = now

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_policy_config(self) -> None:
        try:
            config_path = importlib.resources.files("rl_training").joinpath(
                "config", "ddpg_config.yaml"
            )
            with importlib.resources.as_file(config_path) as path:
                with open(path, "r", encoding="utf-8") as handle:
                    config = yaml.safe_load(handle)
        except Exception as exc:  # pragma: no cover - defensive path
            self._last_error = f"Failed to read DDPG config: {exc}"
            return

        lidar_cfg = config.get("lidar", {}) if isinstance(config, dict) else {}
        self._lidar_history_len = int(lidar_cfg.get("lidar_history_len", 1))
        self._n_l = int(lidar_cfg.get("n_l", 0))
        self._n_e = int(lidar_cfg.get("n_e", 0))

        for state in self._control_state.values():
            state.lidar_history = deque(maxlen=self._lidar_history_len)

    def _initialise_model(self) -> None:
        if self._model is not None:
            return

        if not self.model_path:
            self._last_error = "No policy model path has been provided."
            return

        if not os.path.exists(self.model_path):
            self._last_error = f"Policy model not found: {self.model_path}"
            return

        try:
            from stable_baselines3 import DDPG
        except ImportError as exc:  # pragma: no cover - missing optional dependency
            self._last_error = (
                "stable-baselines3 is required to load the policy model."
                f" ({exc})"
            )
            return

        try:
            from rl_training import custom_ddpg_policy  # noqa: F401
            self._model = DDPG.load(self.model_path, device=self._model_device)
            self._model.policy.set_training_mode(False)
            self._last_error = None
        except Exception as exc:  # pragma: no cover - model loading failure
            self._last_error = f"Failed to load policy model: {exc}"
            self._model = None

    def _build_observation(
        self, robot: RobotState, control_state: _PolicyRobotControlState
    ) -> Optional[Dict[str, np.ndarray]]:
        lidar_measurement = self._process_lidar_data(robot.sensor_data)
        control_state.lidar_history.append(lidar_measurement)
        history = list(control_state.lidar_history)
        if not history:
            history = [lidar_measurement]
        if len(history) < self._lidar_history_len:
            padding = [history[0].copy() for _ in range(self._lidar_history_len - len(history))]
            history = padding + history
        history = history[-self._lidar_history_len :]
        obs_lidar = np.array(history, dtype=np.float32)

        distance_to_target, angle_to_target = self._compute_goal_info(robot)
        goal_angle_n = float(np.clip(angle_to_target / math.pi, -1.0, 1.0))
        obs_goal = np.array([distance_to_target, goal_angle_n], dtype=np.float32)

        env = self.simulator.env
        v_n = float(np.clip((control_state.current_v or 0.0) / env.V_MAX, 0.0, 1.0))
        omega_n = float(
            np.clip((control_state.current_omega or 0.0) / env.OMEGA_MAX, -1.0, 1.0)
        )
        obs_velocity = np.array([v_n, omega_n], dtype=np.float32)

        co_distance, co_angle = robot.closest_obstacle
        if co_distance == float("inf"):
            co_distance = env.LIDAR_RANGE_M
        co_dist_n = float(np.clip(co_distance / env.LIDAR_RANGE_M, 0.0, 1.0))
        co_angle_n = float(np.clip(co_angle / math.pi, -1.0, 1.0))
        obs_closest = np.array([co_dist_n, co_angle_n], dtype=np.float32)

        return {
            "lidar": obs_lidar,
            "goal": obs_goal,
            "velocity": obs_velocity,
            "closest_obstacle": obs_closest,
        }

    def _action_to_executable_action(
        self, action: np.ndarray, control_state: _PolicyRobotControlState
    ) -> Tuple[float, float]:
        env = self.simulator.env
        dt = self._dt
        v_n = float((action[0] + 1.0) / 2.0 * env.V_MAX)
        omega_n = float(action[1] * env.OMEGA_MAX)

        v_kmin = max(0.0, control_state.previous_v - env.A_MAX * dt)
        v_kmax = min(env.V_MAX, control_state.previous_v + env.A_MAX * dt)
        omega_kmin = max(-env.OMEGA_MAX, control_state.previous_omega - env.ALPHA_MAX * dt)
        omega_kmax = min(env.OMEGA_MAX, control_state.previous_omega + env.ALPHA_MAX * dt)

        v_e = float(np.clip(v_n, v_kmin, v_kmax))
        omega_e = float(np.clip(omega_n, omega_kmin, omega_kmax))
        return v_e, omega_e

    def _apply_motion(self, robot: RobotState, v_e: float, omega_e: float) -> bool:
        env = self.simulator.env
        v_pixels_per_sec = v_e * env.PIXELS_PER_METER
        dx = -v_pixels_per_sec * math.sin(robot.theta) * self._dt
        dy = -v_pixels_per_sec * math.cos(robot.theta) * self._dt
        next_position = (robot.position[0] + dx, robot.position[1] + dy)

        moved = False
        if not self.simulator._is_collision(next_position, ignore_robot=robot):
            self.simulator._update_robot_motion(robot, next_position)
            moved = True
        else:
            robot.velocity = 0.0

        robot.theta = (robot.theta + omega_e * self._dt + math.pi) % (2 * math.pi) - math.pi
        return moved

    def _process_lidar_data(self, sensor_data: Sequence[Tuple[float, float]]) -> np.ndarray:
        env = self.simulator.env
        max_range = env.LIDAR_RANGE_M
        if not sensor_data:
            return np.full(self._n_l + self._n_e, max_range, dtype=np.float32)

        processed: List[float] = []
        for distance, _ in sensor_data:
            if distance < max_range:
                processed.append(distance / max_range)
            else:
                processed.append(max_range * 100.0 / max_range)

        if len(processed) < self._n_l:
            processed.extend([processed[-1] if processed else max_range] * (self._n_l - len(processed)))

        extended = processed[: self._n_l] + processed[: self._n_e]
        if len(extended) < self._n_l + self._n_e:
            extended.extend(
                [extended[-1] if extended else max_range]
                * (self._n_l + self._n_e - len(extended))
            )

        return np.array(extended, dtype=np.float32)

    def _compute_goal_info(self, robot: RobotState) -> Tuple[float, float]:
        env = self.simulator.env
        if robot.target is None:
            return 0.0, 0.0

        dx_pixels = robot.target[0] - robot.position[0]
        dy_pixels = robot.target[1] - robot.position[1]
        distance_pixels = math.hypot(dx_pixels, dy_pixels)
        dy = -dy_pixels
        distance_m = distance_pixels / env.PIXELS_PER_METER

        global_angle_to_target = math.atan2(dy, dx_pixels)
        robot_global_angle = robot.theta + math.pi / 2
        angle_to_target = global_angle_to_target - robot_global_angle
        angle_to_target = (angle_to_target + math.pi) % (2 * math.pi) - math.pi
        return distance_m, angle_to_target
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
        policy_model_path: Optional[str] = "./model_ep11200.zip",
        show_info_panel: bool = True,
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
        self._manual_control_enabled = manual_control
        self.selected_robot_index = 0
        self.manual_step_pixels = 5
        self.manual_rotation_step = 0.1
        self.info_panel_min_width = 320
        self.info_panel_width = max(self.info_panel_min_width, int(self.env.SCREEN_W * 0.4))
        self._last_info_panel_width = self.info_panel_width
        self.info_panel_visible = True
        self.info_panel_background = (245, 245, 245)
        self.info_panel_text_color = (32, 32, 32)
        self.info_panel_highlight = (220, 235, 250)
        self.info_panel_divider = (200, 200, 200)
        self.info_padding = 16
        self.info_panel_scroll = 0
        self.info_panel_scroll_speed = 40
        self._info_panel_content_height = self.env.SCREEN_H
        self.map_descriptions = {
            "trap_shaped": "Trap-shaped course with a narrow passage leading to the target zone.",
            "zigzag": "Zigzag corridors that require repeated tight turns to reach the goal.",
            "random": "Randomly distributed obstacles that create unpredictable navigation challenges.",
            "empty": "Open field without obstacles for free navigation and debugging.",
        }
        self._clock = pygame.time.Clock()
        self._fps = 0.0
        self._current_formation_plan: Optional[FormationPlan] = None
        self._formation_plan_candidates: Dict[int, FormationPlan] = {}

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
        self.info_panel_surface = pygame.Surface((self.info_panel_width, self.env.SCREEN_H))

        # Create the first map.  By default we use the same map type as the
        # single robot simulator so that existing scenarios work out of the box.
        self.set_map(self.env.map_type)

        if not show_info_panel:
            self.set_info_panel_visible(False)

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

    def load_policy_model(self, model_path: str) -> None:
        """Load or replace the policy model used for autonomous control."""

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
        if target is None:
            self._clear_formation_visualisation()

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
        self._clear_formation_visualisation()

    def _clear_formation_visualisation(self) -> None:
        """Reset cached formation plans used for rendering overlays."""

        self._current_formation_plan = None
        self._formation_plan_candidates.clear()

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
            if robot.path and robot.path_enabled:
                self._draw_path(robot)
            if robot.target is not None:
                self._draw_target(robot)
            self._draw_robot(robot)
            if self.show_labels:
                self._draw_label(robot)
        self._update_formation_visualisation()
        self._draw_formation_plan()
        self._draw_parallel_formation()
        self._draw_hud()
        self.env.screen.blit(self.env.robot_and_lidar_layer, (0, 0))

        self.display_surface.fill(self.env.BACKGROUND_COLOR)
        self.display_surface.blit(self.simulation_surface, (0, 0))
        if self.info_panel_visible:
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
                    self._scroll_info_panel(-event.y * self.info_panel_scroll_speed)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in (4, 5):
                    direction = -1 if event.button == 4 else 1
                    mouse_x, _ = pygame.mouse.get_pos()
                    if mouse_x >= self.env.SCREEN_W:
                        self._scroll_info_panel(direction * self.info_panel_scroll_speed)


        if self.manual_control and self.robot_states:
            self._operate_moving()

        if self._policy_controller is not None:
            self._policy_controller.step()
            self.policy_control_active = self._policy_controller.active

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

    def _draw_formation_plan(self) -> None:
        if not self._formation_plan_candidates:
            self._draw_manual_target_overlay()
            return

        target_radius = int(max(4, self.robot_radius_pixels * 0.4))
        font = pygame.font.SysFont("arial", 14)
        small_font = pygame.font.SysFont("consolas", 12)

        highlight_queue: List[Tuple[int, FormationPlan, Color]] = []

        for ego_index in sorted(self._formation_plan_candidates):
            plan = self._formation_plan_candidates[ego_index]
            if not plan.targets or not self.robot_states:
                continue
            if len(plan.targets) != len(self.robot_states):
                continue

            if 0 <= ego_index < len(self.robot_states):
                base_color = self.robot_states[ego_index].color
            else:
                base_color = (72, 61, 139)

            highlight = plan is self._current_formation_plan
            if highlight:
                highlight_queue.append((ego_index, plan, base_color))
                continue

            self._draw_single_formation_plan(
                plan,
                ego_index=ego_index,
                target_radius=target_radius,
                base_color=base_color,
                highlight=False,
                font=font,
                small_font=small_font,
            )

        for ego_index, plan, base_color in highlight_queue:
            self._draw_single_formation_plan(
                plan,
                ego_index=ego_index,
                target_radius=target_radius,
                base_color=base_color,
                highlight=True,
                font=font,
                small_font=small_font,
            )

    def _draw_single_formation_plan(
        self,
        plan: FormationPlan,
        *,
        ego_index: int,
        target_radius: int,
        base_color: Color,
        highlight: bool,
        font: "pygame.font.Font",
        small_font: "pygame.font.Font",
    ) -> None:
        layer = self.env.robot_and_lidar_layer

        connection_color = base_color if highlight else self._with_alpha(base_color, 0.5)
        target_color = (
            self._brighten_color(base_color)
            if highlight
            else self._with_alpha(self._brighten_color(base_color), 0.5)
        )
        center_color = (
            self._brighten_color(base_color)
            if highlight
            else self._with_alpha(self._brighten_color(base_color), 0.7)
        )
        text_color = (30, 30, 30) if highlight else (90, 90, 90)

        center_x, center_y = plan.center
        orientation_length = max(self.env.ROBOT_RADIUS_PIXELS * 3.0, plan.spacing)
        orientation_dx = math.cos(plan.theta) * orientation_length
        orientation_dy = math.sin(plan.theta) * orientation_length
        orientation_end = (center_x + orientation_dx, center_y + orientation_dy)

        radius = 6 if highlight else 4
        line_width = 3 if highlight else 2
        pygame.draw.circle(layer, center_color, (int(center_x), int(center_y)), radius)
        # pygame.draw.line(
        #     layer,
        #     center_color,
        #     (int(center_x), int(center_y)),
        #     (int(orientation_end[0]), int(orientation_end[1])),
        #     width=line_width,
        # )

        arrow_size = 10.0 if highlight else 8.0
        arrow_angle = math.atan2(orientation_dy, orientation_dx)
        left_angle = arrow_angle + math.pi * 3.0 / 4.0
        right_angle = arrow_angle - math.pi * 3.0 / 4.0
        left_point = (
            orientation_end[0] - arrow_size * math.cos(left_angle),
            orientation_end[1] - arrow_size * math.sin(left_angle),
        )
        right_point = (
            orientation_end[0] - arrow_size * math.cos(right_angle),
            orientation_end[1] - arrow_size * math.sin(right_angle),
        )
        # pygame.draw.polygon(
        #     layer,
        #     center_color,
        #     [
        #         (int(orientation_end[0]), int(orientation_end[1])),
        #         (int(left_point[0]), int(left_point[1])),
        #         (int(right_point[0]), int(right_point[1])),
        #     ],
        # )

        if highlight:
            label_text = f"θ={math.degrees(plan.theta):.1f}°"
        else:
            label_text = f"R{ego_index}: {math.degrees(plan.theta):.1f}°"
        angle_label = font.render(label_text, True, text_color)
        # layer.blit(
        #     angle_label,
        #     (
        #         int(orientation_end[0] - angle_label.get_width() / 2),
        #         int(orientation_end[1] - angle_label.get_height() - 6),
        #     ),
        # )

        for target_index, position in enumerate(plan.targets):
            draw_kwargs = {}
            if highlight:
                draw_kwargs["width"] = 0
            else:
                draw_kwargs["width"] = 1
            pygame.draw.circle(
                layer,
                target_color,
                (int(position[0]), int(position[1])),
                target_radius,
                **draw_kwargs,
            )

            if highlight:
                label = small_font.render(f"T{target_index}", True, text_color)
                # layer.blit(
                #     label,
                #     (
                #         int(position[0] - label.get_width() / 2),
                #         int(position[1] - label.get_height() / 2),
                #     ),
                # )

        if not plan.assignment:
            return

        for robot_index, target_index in enumerate(plan.assignment):
            if not (0 <= target_index < len(plan.targets)):
                continue
            if robot_index >= len(self.robot_states):
                continue

            robot = self.robot_states[robot_index]
            target_position = plan.targets[target_index]
            pygame.draw.line(
                layer,
                connection_color,
                (int(robot.position[0]), int(robot.position[1])),
                (int(target_position[0]), int(target_position[1])),
                width=2 if highlight else 1,
            )

            if not highlight:
                continue

            midpoint = (
                (robot.position[0] + target_position[0]) / 2.0,
                (robot.position[1] + target_position[1]) / 2.0,
            )
            cost_value = plan.per_robot_costs.get(robot_index)
            if cost_value is not None:
                cost_label = small_font.render(f"{cost_value:.2f}", True, text_color)
                # layer.blit(
                #     cost_label,
                #     (
                #         int(midpoint[0] - cost_label.get_width() / 2),
                #         int(midpoint[1] - cost_label.get_height() / 2),
                #     ),
                # )

    def _update_formation_visualisation(self) -> None:
        """Refresh the cached formation plans used for visualisation."""

        if not self.robot_states:
            self._clear_formation_visualisation()
            return

        self._formation_plan_candidates.clear()
        selected_plan: Optional[FormationPlan] = None

        for ego_index in range(len(self.robot_states)):
            try:
                plan = self.optimize_formation(ego_index=ego_index, apply=False)
            except ValueError:
                continue

            self._formation_plan_candidates[ego_index] = plan
            if ego_index == self.selected_robot_index:
                selected_plan = plan

        if not self._formation_plan_candidates:
            self._current_formation_plan = None
            return

        if selected_plan is not None:
            self._current_formation_plan = selected_plan
        else:
            self._current_formation_plan = next(
                iter(self._formation_plan_candidates.values())
            )

    def _draw_manual_target_overlay(self) -> None:
        robots_with_targets = [
            (index, robot)
            for index, robot in enumerate(self.robot_states)
            if robot.target is not None
        ]
        if not robots_with_targets:
            return

        layer = self.env.robot_and_lidar_layer
        connection_color = (72, 61, 139)
        text_color = (30, 30, 30)
        small_font = pygame.font.SysFont("consolas", 12)

        for index, robot in robots_with_targets:
            assert robot.target is not None
            start = (int(robot.position[0]), int(robot.position[1]))
            end = (int(robot.target[0]), int(robot.target[1]))

            pygame.draw.line(layer, connection_color, start, end, width=2)

            midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
            distance_pixels = math.hypot(
                robot.target[0] - robot.position[0],
                robot.target[1] - robot.position[1],
            )
            distance_m = distance_pixels / self.env.PIXELS_PER_METER
            label = small_font.render(
                f"R{index} → target ({distance_m:.2f}m)", True, text_color
            )
            layer.blit(
                label,
                (
                    int(midpoint[0] - label.get_width() / 2),
                    int(midpoint[1] - label.get_height() / 2),
                ),
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
            f"Controls: TAB/0-9 select, W move forward, Q/E rotate, C cycle map, SPACE reset, L toggle LiDAR, P toggle path, I toggle info panel",
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

            # pygame.draw.line(
            #     self.env.robot_and_lidar_layer,
            #     formation_color,
            #     (int(start[0]), int(start[1])),
            #     (int(end[0]), int(end[1])),
            #     width=2,
            # )

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

    def toggle_info_panel(self) -> None:
        """Toggle the visibility of the information panel."""

        self.set_info_panel_visible(not self.info_panel_visible)

    def set_info_panel_visible(self, visible: bool) -> None:
        """Set the visibility of the information panel."""

        if self.info_panel_visible == visible:
            return

        if visible:
            target_width = max(self.info_panel_min_width, self._last_info_panel_width)
            self.info_panel_visible = True
            self._resize_display_surface(target_width)
        else:
            self._last_info_panel_width = max(self.info_panel_min_width, self.info_panel_width)
            self.info_panel_visible = False
            self._resize_display_surface(0)
            self.info_panel_scroll = 0

    def _draw_info_panel(self) -> None:
        if not self.info_panel_visible:
            return

        while True:
            required_inner_width, required_height = self._render_info_panel(resize_only=True)
            target_width = max(
                self.info_panel_min_width,
                required_inner_width + 2 * self.info_padding,
            )
            if target_width == self.info_panel_width:
                break
            self._resize_display_surface(target_width)

        panel_height = max(self.env.SCREEN_H, required_height + self.info_padding)
        if (
            self.info_panel_surface.get_width() != self.info_panel_width
            or self.info_panel_surface.get_height() != panel_height
        ):
            self.info_panel_surface = pygame.Surface((self.info_panel_width, panel_height))

        self._info_panel_content_height = panel_height
        self._render_info_panel(resize_only=False, surface=self.info_panel_surface)

        max_scroll = max(0, panel_height - self.env.SCREEN_H)
        self.info_panel_scroll = max(0, min(self.info_panel_scroll, max_scroll))

        panel_left = self.env.SCREEN_W
        dest_rect = pygame.Rect(panel_left, 0, self.info_panel_width, self.env.SCREEN_H)
        src_rect = pygame.Rect(0, self.info_panel_scroll, self.info_panel_width, self.env.SCREEN_H)
        self.display_surface.blit(self.info_panel_surface, dest_rect, src_rect)

    def _render_info_panel(
        self, *, resize_only: bool, surface: Optional[pygame.Surface] = None
    ) -> Tuple[int, int]:
        if not resize_only and surface is None:
            raise ValueError("surface must be provided when resize_only is False")

        if not resize_only:
            assert surface is not None
            surface.fill(self.info_panel_background)
            pygame.draw.line(
                surface,
                self.info_panel_divider,
                (0, 0),
                (0, surface.get_height()),
                width=2,
            )

        title_font = pygame.font.SysFont("arial", 20, bold=True)
        name_font = pygame.font.SysFont("arial", 18, bold=True)
        value_font = pygame.font.SysFont("consolas", 16)

        x_offset = self.info_padding
        y_offset = self.info_padding
        max_required_inner_width = 0

        def render_line(
            text: str,
            font: pygame.font.Font,
            color: Color,
            *,
            indent: int = 0,
            spacing: int = 4,
        ) -> None:
            nonlocal y_offset, max_required_inner_width
            if not text:
                return
            width, height = font.size(text)
            max_required_inner_width = max(max_required_inner_width, indent + width)
            if not resize_only and surface is not None:
                label = font.render(text, True, color)
                surface.blit(label, (x_offset + indent, y_offset))
            y_offset += height + spacing

        render_line("Simulation Info", title_font, self.info_panel_text_color, spacing=10)

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
        control_mode = "Policy" if self.policy_control_active else "Manual"
        if self._policy_controller is None and self.policy_model_path:
            control_mode += " (pending)"
        general_lines.append(f"Control: {control_mode}")

        if self._policy_controller is not None:
            model_status = "Loaded" if self._policy_controller.is_available else "Not loaded"
            general_lines.append(f"Policy model: {model_status}")
        for line in general_lines:
            render_line(line, value_font, self.info_panel_text_color, spacing=4)

        if self._policy_controller is not None and self._policy_controller.last_error:
            render_line("Policy error:", value_font, self.info_panel_text_color, spacing=2)
            wrap_width = max(10, self.info_panel_width - 2 * self.info_padding - 10)
            for line in self._wrap_text_by_pixels(
                value_font, self._policy_controller.last_error, wrap_width
            ):
                render_line(line, value_font, self.info_panel_text_color, indent=10, spacing=2)

        y_offset += 10

        render_line("Map", title_font, self.info_panel_text_color, spacing=8)

        map_width_m = self.env.SCREEN_W / self.env.PIXELS_PER_METER
        map_height_m = self.env.SCREEN_H / self.env.PIXELS_PER_METER
        map_lines = [
            f"Type: {self.env.map_type}",
            f"Size: {map_width_m:.1f}m x {map_height_m:.1f}m",
        ]
        for line in map_lines:
            render_line(line, value_font, self.info_panel_text_color, spacing=4)

        description = self.map_descriptions.get(self.env.map_type)
        if description:
            render_line("Description:", value_font, self.info_panel_text_color, spacing=4)
            desc_indent = 10
            wrap_width = max(10, self.info_panel_width - 2 * self.info_padding - desc_indent)
            for line in self._wrap_text_by_pixels(value_font, description, wrap_width):
                render_line(
                    line,
                    value_font,
                    self.info_panel_text_color,
                    indent=desc_indent,
                    spacing=2,
                )

            y_offset += 6

        if self._formation_plan_candidates:
            render_line("Formation Plans", title_font, self.info_panel_text_color, spacing=8)

            for ego_index in sorted(self._formation_plan_candidates):
                plan = self._formation_plan_candidates[ego_index]
                if not plan.targets:
                    continue

                robot_name = (
                    self.robot_states[ego_index].name
                    if 0 <= ego_index < len(self.robot_states)
                    else f"Robot {ego_index}"
                )
                spacing_m = plan.spacing / self.env.PIXELS_PER_METER
                center_m = self.pixels_to_meters(*plan.center)
                heading_deg = math.degrees(plan.theta)
                cost_text = f"{plan.cost:.3f}" if plan.cost != float("inf") else "∞"

                highlight = plan is self._current_formation_plan
                indicator = "*" if highlight else "•"
                if highlight and 0 <= ego_index < len(self.robot_states):
                    robot_color = self.robot_states[ego_index].color
                    summary_color = tuple(
                        min(255, int(tc * 0.5 + rc * 0.5))
                        for tc, rc in zip(self.info_panel_text_color, robot_color)
                    )
                else:
                    summary_color = self.info_panel_text_color

                summary = (
                    f"{indicator} {robot_name}: {plan.formation_type.title()}"
                    f" θ={heading_deg:.1f}° cost={cost_text}"
                )
                render_line(summary, value_font, summary_color, spacing=2)

                if highlight:
                    detail_lines = [
                        f"Spacing: {spacing_m:.2f} m",
                        f"Center: ({center_m[0]:.2f}, {center_m[1]:.2f}) m",
                    ]
                    for line in detail_lines:
                        render_line(
                            line,
                            value_font,
                            self.info_panel_text_color,
                            indent=10,
                            spacing=2,
                        )

                    render_line(
                        "Assignments:",
                        value_font,
                        self.info_panel_text_color,
                        indent=10,
                        spacing=2,
                    )

                    for robot_index, target_index in enumerate(plan.assignment):
                        if robot_index >= len(self.robot_states):
                            continue
                        robot_label = self.robot_states[robot_index].name
                        if 0 <= target_index < len(plan.targets):
                            cost_value = plan.per_robot_costs.get(robot_index)
                            if cost_value is not None:
                                text = (
                                    f"{robot_label} → T{target_index}" f" ({cost_value:.2f})"
                                )
                            else:
                                text = f"{robot_label} → T{target_index}"
                        else:
                            text = f"{robot_label} → --"
                        render_line(
                            text,
                            value_font,
                            self.info_panel_text_color,
                            indent=20,
                            spacing=2,
                        )

                y_offset += 8
        elif any(robot.target is not None for robot in self.robot_states):
            render_line("Target Assignments", title_font, self.info_panel_text_color, spacing=8)

            for robot in self.robot_states:
                if robot.target is None:
                    continue

                target_m = self.pixels_to_meters(*robot.target)
                distance_pixels = math.hypot(
                    robot.target[0] - robot.position[0],
                    robot.target[1] - robot.position[1],
                )
                distance_m = distance_pixels / self.env.PIXELS_PER_METER
                text = (
                    f"{robot.name}: ({target_m[0]:.2f}, {target_m[1]:.2f}) m"
                    f" / {distance_m:.2f} m away"
                )
                render_line(
                    text,
                    value_font,
                    self.info_panel_text_color,
                    indent=10,
                    spacing=2,
                )

            y_offset += 12

        render_line("Robots", title_font, self.info_panel_text_color, spacing=8)

        if not self.robot_states:
            render_line("No robots", value_font, self.info_panel_text_color, spacing=4)
            return max_required_inner_width, y_offset

        line_height = value_font.get_height() + 4

        for index, robot in enumerate(self.robot_states):
            robot_name_text = robot.name
            pos_m = self.pixels_to_meters(*robot.position)
            metrics: List[str] = [
                f"Pos: ({pos_m[0]:.2f}, {pos_m[1]:.2f}) m",
                f"Theta: {math.degrees(robot.theta):.1f}°",
                f"Speed: {robot.velocity:.2f} m/s",
            ]

            if robot.target is not None:
                target_m = self.pixels_to_meters(*robot.target)
                distance = math.hypot(target_m[0] - pos_m[0], target_m[1] - pos_m[1])
                metrics.append(f"Target dist: {distance:.2f} m")
            else:
                metrics.append("Target: None")

            if robot.closest_obstacle[0] != float("inf"):
                metrics.append(f"Closest obs: {robot.closest_obstacle[0]:.2f} m")
            else:
                metrics.append("Closest obs: --")

            metrics.append(f"LiDAR: {'On' if robot.lidar_enabled else 'Off'}")
            metrics.append(f"Path: {'On' if robot.path_enabled else 'Off'}")

            name_width, name_height = name_font.size(robot_name_text)
            block_height = name_height + 4 + len(metrics) * line_height + 12
            block_rect = pygame.Rect(
                8,
                y_offset - 6,
                self.info_panel_width - 16,
                block_height,
            )

            text_indent = max(0, (block_rect.x + 10) - x_offset)
            max_required_inner_width = max(max_required_inner_width, text_indent + name_width)
            for metric in metrics:
                metric_width, _ = value_font.size(metric)
                max_required_inner_width = max(
                    max_required_inner_width, text_indent + metric_width
                )

            if not resize_only:
                background_color = (
                    self.info_panel_highlight
                    if index == self.selected_robot_index
                    else self.info_panel_background
                )
                pygame.draw.rect(
                    surface,
                    background_color,
                    block_rect,
                    border_radius=8,
                )
                pygame.draw.rect(
                    surface,
                    self.info_panel_divider,
                    block_rect,
                    width=1,
                    border_radius=8,
                )

                text_x = block_rect.x + 10
                text_y = y_offset
                robot_name_surface = name_font.render(
                    robot_name_text, True, self.info_panel_text_color
                )
                surface.blit(robot_name_surface, (text_x, text_y))
                text_y += robot_name_surface.get_height() + 4

                for metric in metrics:
                    metric_label = value_font.render(metric, True, self.info_panel_text_color)
                    surface.blit(metric_label, (text_x, text_y))
                    text_y += line_height

            y_offset = block_rect.bottom + 10

        return max_required_inner_width, y_offset

    def _resize_display_surface(self, new_info_panel_width: int) -> None:
        self.info_panel_width = new_info_panel_width
        total_width = self.env.SCREEN_W + self.info_panel_width
        if self.env.render_mode == "human":
            pygame.display.set_mode((total_width, self.env.SCREEN_H))
            self.display_surface = pygame.display.get_surface()
        else:
            self.display_surface = pygame.Surface((total_width, self.env.SCREEN_H))

        self.display_surface.fill(self.env.BACKGROUND_COLOR)
        self.display_surface.blit(self.simulation_surface, (0, 0))
        panel_height = max(self.env.SCREEN_H, self._info_panel_content_height)
        self.info_panel_surface = pygame.Surface((self.info_panel_width, panel_height))

    def _scroll_info_panel(self, delta: float) -> None:
        max_scroll = max(0, self._info_panel_content_height - self.env.SCREEN_H)
        self.info_panel_scroll = max(0, min(self.info_panel_scroll + int(delta), max_scroll))

    def _wrap_text_by_pixels(
        self, font: pygame.font.Font, text: str, max_width: int
    ) -> List[str]:
        if max_width <= 0:
            return [text]

        words = text.split()
        if not words:
            return []

        lines: List[str] = []

        first_word = words[0]
        if font.size(first_word)[0] > max_width:
            segments = self._break_word_by_pixels(font, first_word, max_width)
            if segments:
                lines.extend(segments[:-1])
                current = segments[-1]
            else:
                current = first_word
        else:
            current = first_word

        for word in words[1:]:
            candidate = f"{current} {word}"
            if font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                lines.append(current)
                if font.size(word)[0] <= max_width:
                    current = word
                else:
                    oversized = self._break_word_by_pixels(font, word, max_width)
                    if oversized:
                        lines.extend(oversized[:-1])
                        current = oversized[-1]
                    else:
                        current = word

        lines.append(current)
        return lines

    def _break_word_by_pixels(
        self, font: pygame.font.Font, word: str, max_width: int
    ) -> List[str]:
        if max_width <= 0:
            return [word]

        segments: List[str] = []
        current = ""

        for char in word:
            candidate = current + char
            if not current or font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                segments.append(current)
                current = char

        if current:
            segments.append(current)

        return segments

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
        self._clear_formation_visualisation()

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
            formation_type=formation_type,
            spacing=spacing,
            center=center,
        )

    def optimize_formation(
        self,
        formation_type: str = "triangle",
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
            self._clear_formation_visualisation()
            raise ValueError("Unable to find a feasible formation configuration")

        if apply:
            for robot_index, target_index in enumerate(plan.assignment):
                target = plan.targets[target_index]
                self.robot_states[robot_index].target = target
            self._current_formation_plan = plan
            if ego_index is not None:
                self._formation_plan_candidates[ego_index] = plan

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
