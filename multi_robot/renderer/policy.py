"""Implementation of the policy based controller."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import importlib
import importlib.abc
import importlib.resources
from pathlib import Path
import numpy as np
import pygame
import yaml

from typing import TYPE_CHECKING

from .speed_adjustment import adjust_velocity
from .types import RobotState
from .utils import ensure_repo_root_on_path, process_lidar_for_observation

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .simulator import MultiRobotSimulator


@dataclass
class PolicyRobotControlState:
    """Keeps track of temporal state required by the RL policy controller."""

    lidar_history: Deque[np.ndarray]
    previous_v: float = 0.0
    previous_omega: float = 0.0
    current_v: float = 0.0
    current_omega: float = 0.0


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
        self._control_state: Dict[int, PolicyRobotControlState] = {}
        self._completed_robots: set[int] = set()
        self._lidar_history_len = 1
        self._n_l = 0
        self._n_e = 0
        self._load_policy_config()
        self._initialise_model()

    @property
    def is_available(self) -> bool:
        return self._model is not None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def toggle(self) -> bool:
        if not self.is_available:
            self._initialise_model()

        if not self.is_available:
            return False

        self.active = not self.active
        if self.active:
            self._control_state.clear()
            self._completed_robots.clear()
            self._last_action_timestamp = None
        else:
            self._last_action_timestamp = None
            self._completed_robots.clear()
        return self.active

    def step(self) -> None:
        if not self.active or not self.is_available:
            return

        if not self.simulator.robot_states:
            return

        now = pygame.time.get_ticks() / 1000.0 if pygame.get_init() else None
        if now is not None and self._last_action_timestamp is not None:
            if now - self._last_action_timestamp < self._dt:
                return

        robot_states = self.simulator.robot_states

        stale_indices = [
            index for index in self._control_state if index >= len(robot_states)
        ]
        for index in stale_indices:
            del self._control_state[index]
            self._completed_robots.discard(index)

        for robot_index, robot in enumerate(robot_states):
            control_state = self._control_state.setdefault(
                robot_index,
                PolicyRobotControlState(
                    lidar_history=deque(maxlen=self._lidar_history_len)
                ),
            )

            robot.braking = False

            if robot_index in self._completed_robots:
                if self._has_reached_target(robot_index, robot):
                    robot.velocity = 0.0
                    robot.executable_velocity = 0.0
                    robot.executable_angular_velocity = 0.0
                    control_state.current_v = 0.0
                    control_state.current_omega = 0.0
                    continue
                self._completed_robots.discard(robot_index)

            if self._has_reached_target(robot_index, robot):
                self._completed_robots.add(robot_index)
                robot.velocity = 0.0
                robot.executable_velocity = 0.0
                robot.executable_angular_velocity = 0.0
                control_state.current_v = 0.0
                control_state.current_omega = 0.0
                control_state.previous_v = 0.0
                control_state.previous_omega = 0.0
                continue

            self.simulator._update_lidar(robot)
            observation = self._build_observation(robot_index, robot, control_state)
            if observation is None:
                continue

            action, _ = self._model.predict(observation, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.size != 2:
                self._last_error = (
                    f"Unexpected action dimension {action.shape}; expected size 2"
                )
                continue

            env = self.simulator.env
            v_n = float((action[0] + 1.0) / 2.0 * env.V_MAX)
            omega_n = float(action[1] * env.OMEGA_MAX)

            translation_blocked = self._should_block_translation(robot)
            robot.braking = translation_blocked
            if translation_blocked:
                # Zero the nominal forward speed before applying the acceleration limits.
                v_n = 0.0

            v_e, omega_e = self._action_to_executable_action(
                v_n, omega_n, control_state
            )

            robot.executable_velocity = v_e
            robot.executable_angular_velocity = omega_e

            formation_target = self.simulator.formation.get_navigation_target(
                robot_index, fallback=None
            )
            if formation_target is not None:
                d_f, theta_f = self._compute_goal_info_for_target(
                    robot, formation_target
                )
                d_o = d_f * math.sin(theta_f)
                env_params = self.simulator.env
                v_a, omega_a = adjust_velocity(
                    v_e,
                    omega_e,
                    d_f,
                    d_o,
                    theta_f,
                    env_params.ETA_THETA,
                    env_params.ETA_D,
                    env_params.K_D,
                    env_params.OMEGA_FIXED,
                    env_params.T_DF,
                    env_params.V_MAX,
                    env_params.OMEGA_MAX,
                )
            else:
                v_a, omega_a = v_e, omega_e

            moved = self._apply_motion(robot, v_a, omega_a)

            if moved:
                robot.velocity = v_a
            else:
                robot.velocity = 0.0

            control_state.current_v = v_a if moved else 0.0
            control_state.current_omega = omega_a
            control_state.previous_v = control_state.current_v
            control_state.previous_omega = control_state.current_omega

            if self.simulator.path_history != 0 and moved:
                robot.path.append((robot.position, robot.theta))

            self.simulator._update_lidar(robot)

        if now is not None:
            self._last_action_timestamp = now

        if self._completed_robots and len(self._completed_robots) == len(robot_states):
            self._deactivate_policy_control()

    def _deactivate_policy_control(self) -> None:
        if not self.active:
            return

        self.active = False
        self._last_action_timestamp = None
        self._control_state.clear()
        self._completed_robots.clear()
        self.simulator.policy_control_active = False
        self.simulator.manual_control = self.simulator._manual_control_enabled
        for robot in self.simulator.robot_states:
            robot.braking = False
            robot.executable_velocity = 0.0
            robot.executable_angular_velocity = 0.0

    def _load_policy_config(self) -> None:
        config: Optional[Dict[str, object]] = None
        last_exc: Optional[Exception] = None
        package_files: Optional[importlib.abc.Traversable] = None

        try:
            package_files = importlib.resources.files("rl_training")
        except ModuleNotFoundError:
            repo_root = ensure_repo_root_on_path()
            try:
                package_files = importlib.resources.files("rl_training")
            except ModuleNotFoundError:
                fallback = repo_root / "rl_training" / "config" / "ddpg_config.yaml"
                if fallback.exists():
                    try:
                        with open(fallback, "r", encoding="utf-8") as handle:
                            config = yaml.safe_load(handle)
                    except Exception as exc:  # pragma: no cover - defensive path
                        last_exc = exc
                else:
                    last_exc = FileNotFoundError(fallback)

        if config is None and package_files is not None:
            try:
                config_path = package_files.joinpath("config", "ddpg_config.yaml")
                with importlib.resources.as_file(config_path) as path:
                    with open(path, "r", encoding="utf-8") as handle:
                        config = yaml.safe_load(handle)
            except Exception as exc:  # pragma: no cover - defensive path
                last_exc = exc

        if config is None:
            message = "unknown error" if last_exc is None else str(last_exc)
            self._last_error = f"Failed to read DDPG config: {message}"
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

        original_path = Path(self.model_path)
        candidate_paths: List[Path] = []

        if original_path.is_absolute():
            candidate_paths.append(original_path)
        else:
            candidate_paths.append(original_path)
            package_root = Path(__file__).resolve().parents[1]
            candidate_paths.append(package_root / original_path)
            candidate_paths.append(package_root / original_path.name)
            try:
                repo_root = ensure_repo_root_on_path()
            except Exception:  # pragma: no cover - defensive path
                repo_root = None
            if repo_root is not None:
                repo_root = Path(repo_root)
                candidate_paths.append(repo_root / original_path)
                candidate_paths.append(repo_root / original_path.name)

        resolved_path: Optional[Path] = None
        for candidate in candidate_paths:
            if candidate.exists():
                resolved_path = candidate
                break

        if resolved_path is None:
            self._last_error = f"Policy model not found: {self.model_path}"
            return

        self.model_path = str(resolved_path)

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
        except ModuleNotFoundError:
            ensure_repo_root_on_path()
            try:
                from rl_training import custom_ddpg_policy  # type: ignore  # noqa: F401
            except ModuleNotFoundError as exc:  # pragma: no cover - missing dependency
                self._last_error = (
                    "The rl_training package is required to load the policy model."
                    f" ({exc})"
                )
                return

        try:
            self._model = DDPG.load(self.model_path, device=self._model_device)
            self._model.policy.set_training_mode(False)
            self._last_error = None
        except Exception as exc:  # pragma: no cover - model loading failure
            self._last_error = f"Failed to load policy model: {exc}"
            self._model = None

    def _build_observation(
        self,
        robot_index: int,
        robot: RobotState,
        control_state: PolicyRobotControlState,
    ) -> Optional[Dict[str, np.ndarray]]:
        lidar_measurement = self._process_lidar_data(robot)
        control_state.lidar_history.append(lidar_measurement)
        history = list(control_state.lidar_history)
        if not history:
            history = [lidar_measurement]
        if len(history) < self._lidar_history_len:
            padding = [history[0].copy() for _ in range(self._lidar_history_len - len(history))]
            history = padding + history
        history = history[-self._lidar_history_len :]
        obs_lidar = np.array(history, dtype=np.float32)

        distance_to_target, angle_to_target = self._compute_goal_info(robot_index, robot)
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
        self,
        v_n: float,
        omega_n: float,
        control_state: PolicyRobotControlState,
    ) -> Tuple[float, float]:
        env = self.simulator.env
        dt = self._dt

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

    def _should_block_translation(self, robot: RobotState) -> bool:
        robot_radius = self.simulator.robot_frame_radius
        for other in self.simulator.robot_states:
            if other is robot:
                continue
            other_radius = self.simulator.robot_frame_radius
            dx_pixels = other.position[0] - robot.position[0]
            dy_pixels = other.position[1] - robot.position[1]
            distance = math.hypot(dx_pixels, dy_pixels)
            if distance <= other_radius + 1.5 * robot_radius:
                dy_global = -dy_pixels
                global_angle = math.atan2(dy_global, dx_pixels)
                robot_global_angle = robot.theta + math.pi / 2
                angle_diff = (
                    (global_angle - robot_global_angle + math.pi) % (2 * math.pi)
                    - math.pi
                )
                if abs(angle_diff) <= math.radians(20.0):
                    # Respect the requested 1.5× radius buffer when neighbours are ahead.
                    return True
        return False

    def _process_lidar_data(self, robot: RobotState) -> np.ndarray:
        env = self.simulator.env
        sensor_data: Sequence[Tuple[float, float]] = robot.sensor_data

        if sensor_data:
            filtered_data, replaced_measurements = self._filter_robot_reflections(
                robot, sensor_data
            )
            robot.sensor_data = list(filtered_data)
            sensor_data = robot.sensor_data
            robot.filtered_sensor_data = list(replaced_measurements)
        else:
            robot.filtered_sensor_data = []

        return process_lidar_for_observation(
            sensor_data,
            self._n_l,
            self._n_e,
            env.LIDAR_RANGE_M,
        )

    def _filter_robot_reflections(
        self,
        robot: RobotState,
        sensor_data: Sequence[Tuple[float, float]],
    ) -> Tuple[list[Tuple[float, float]], list[Tuple[float, float]]]:
        sensor = robot.sensor
        if sensor is None:
            return list(sensor_data), []

        env = self.simulator.env
        robot_radius_m = env.ROBOT_RADIUS_M
        lidar_range = sensor.LIDAR_RANGE_M
        pixels_per_meter = env.PIXELS_PER_METER
        other_robot_filters: list[Tuple[float, float, float, float]] = []

        for other in self.simulator.robot_states:
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
            return list(sensor_data), []

        filtered_data: list[Tuple[float, float]] = []
        replaced_measurements: list[Tuple[float, float]] = []
        theta = robot.theta
        buffer = 1.1

        for distance, relative_angle in sensor_data:
            global_angle = (theta - math.pi / 2 - relative_angle) % (2 * math.pi)
            replaced = False

            for angle_center, angle_half_width, distance_min, distance_max in other_robot_filters:
                angle_diff = abs(global_angle - angle_center)
                angle_diff = min(angle_diff, 2 * math.pi - angle_diff)

                if (
                    angle_diff <= angle_half_width * buffer
                    and distance_min * (2 - buffer) <= distance <= distance_max * buffer
                ):
                    replaced_measurements.append((distance, relative_angle))
                    filtered_data.append((sensor.out_of_range_value, relative_angle))
                    replaced = True
                    break

            if not replaced:
                filtered_data.append((distance, relative_angle))

        return filtered_data, replaced_measurements

    def _compute_goal_info_for_target(
        self, robot: RobotState, target: Tuple[float, float]
    ) -> Tuple[float, float]:
        env = self.simulator.env
        dx_pixels = target[0] - robot.position[0]
        dy_pixels = target[1] - robot.position[1]
        distance_pixels = math.hypot(dx_pixels, dy_pixels)
        dy = -dy_pixels
        distance_m = distance_pixels / env.PIXELS_PER_METER

        global_angle_to_target = math.atan2(dy, dx_pixels)
        robot_global_angle = robot.theta + math.pi / 2
        angle_to_target = global_angle_to_target - robot_global_angle
        angle_to_target = (angle_to_target + math.pi) % (2 * math.pi) - math.pi
        return distance_m, angle_to_target

    def _compute_goal_info(
        self, robot_index: int, robot: RobotState
    ) -> Tuple[float, float]:
        target = self.simulator.formation.get_navigation_target(
            robot_index, fallback=robot.target
        )
        if target is None:
            return 0.0, 0.0

        return self._compute_goal_info_for_target(robot, target)

    def _has_reached_target(self, robot_index: int, robot: RobotState) -> bool:
        target = self.simulator.formation.get_navigation_target(
            robot_index, fallback=robot.target
        )
        if target is None:
            return False

        dx_pixels = target[0] - robot.position[0]
        dy_pixels = target[1] - robot.position[1]
        distance_pixels = math.hypot(dx_pixels, dy_pixels)
        distance_m = distance_pixels / self.simulator.env.PIXELS_PER_METER
        return distance_m < self.simulator.env.ROBOT_RADIUS_M

