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

# from .speed_adjustment import adjust_velocity
from ..types import RobotState
from ..utils import ensure_repo_root_on_path

# --- 速度調整モジュールのインポート ---
from ..speed_adjustment import adjust_velocity
from ..types import RobotState
from ..utils import ensure_repo_root_on_path

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .simulator_v5_previous import MultiRobotSimulator


@dataclass
class PolicyRobotControlState:
    """Keeps track of temporal state required by the RL policy controller."""
    lidar_history: Deque[np.ndarray]
    previous_v: float = 0.0
    previous_omega: float = 0.0
    current_v: float = 0.0
    current_omega: float = 0.0
    policy_omega_n: float = 0.0

# --- インナーターゲット判定関数 ---
def _determine_target_is_inner(
    robot_position: Tuple[float, float],
    target: Tuple[float, float],
    *,
    formation_center: Optional[Tuple[float, float]] = None,
) -> bool:
    """Return ``True`` when the formation target is no farther from the center."""
    if formation_center is None:
        return False

    cx, cy = formation_center
    rx, ry = robot_position
    tx, ty = target

    target_vec_x = tx - cx
    target_vec_y = ty - cy
    robot_vec_x = rx - cx
    robot_vec_y = ry - cy

    target_dist_sq = target_vec_x * target_vec_x + target_vec_y * target_vec_y
    robot_dist_sq = robot_vec_x * robot_vec_x + robot_vec_y * robot_vec_y

    return target_dist_sq <= robot_dist_sq

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
        self._dt = 0.1
        self._control_state: Dict[int, PolicyRobotControlState] = {}
        self._completed_robots: set[int] = set()
        self._lidar_history_len = 1
        self._n_l = 0
        self._n_e = 0

        # --- サブゴール待機用の変数を追加 ---
        self._waiting_at_subgoal: Dict[int, bool] = {} # ロボット毎の待機フラグ
        self.subgoal_stop_threshold = 0.1   # この距離(m)以内に入ったら停止
        self.subgoal_resume_threshold = 0.3 # この距離(m)以上離れたら再開
        # ----------------------------------------
        
        self.params = {
            "V_MAX": 1.0,
            "OMEGA_MAX": 1.0,
            "A_MAX": 1.0,
            "ALPHA_MAX": 1.0,
            "ETA_THETA": 1.0,
            "ETA_D": 1.0,
            "K_D": 1.0,
            "OMEGA_FIXED": 1.2,
            "T_DF": 0.5
        }
        self._load_local_params()
        
        self._load_policy_config()
        self._initialise_model()

    def _load_local_params(self) -> None:
        try:
            config_path = Path(__file__).resolve().parents[2] / "config" / "policy_params.yaml"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if data:
                        for key in self.params.keys():
                            if key in data:
                                self.params[key] = float(data[key])
                print(f"Loaded local policy params from {config_path}: {self.params}")
            else:
                print(f"Local policy params not found at {config_path}, using defaults: {self.params}")
        except Exception as e:
            print(f"Error loading local policy params: {e}")

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
        formation_manager = self.simulator.formation

        # =================================================================
        # LiDAR更新とフォーメーション更新の順序整理
        # =================================================================
        # 1. 全ロボットのLiDARを更新（生データ取得）
        for robot in robot_states:
            self.simulator._update_lidar(robot)
        
        # 2. フォーメーションターゲットの計算（最新のLiDARデータが必要）
        #    ポリシー制御中は強制的にターゲット計算を回すため、一時的に有効化して呼ぶ
        #    （元々有効ならそのまま、無効なら計算だけさせる意図）
        #    ※ simulator側で follow_formation_targets=True になっていれば
        #       update_visualisation() だけで十分だが、念のためここでも呼ぶ。
        formation_manager.update_visualisation()
        # =================================================================

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

            # 到達判定
            is_final = getattr(robot, 'is_final_target', True)
            if robot_index in self._completed_robots:
                if self._has_reached_target(robot):
                    self._stop_robot(robot, control_state)
                    continue
                self._completed_robots.discard(robot_index)

            if is_final and self._has_reached_target(robot):
                self._completed_robots.add(robot_index)
                self._stop_robot(robot, control_state)
                continue

            # ここで再度 _update_lidar を呼ぶとフィルタ結果が消えるので削除
            # if not lidar_preupdated: self.simulator._update_lidar(robot) <-- 削除
                
            # 観測の構築（内部で _process_lidar_data -> _filter_robot_reflections が呼ばれる）
            observation = self._build_observation(robot_index, robot, control_state)
            if observation is None:
                continue

            action, _ = self._model.predict(observation, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            
            V_MAX = self.params["V_MAX"]
            OMEGA_MAX = self.params["OMEGA_MAX"]

            v_n = float((action[0] + 1.0) / 2.0 * V_MAX)
            omega_n = float(action[1] * OMEGA_MAX)
            
            if OMEGA_MAX > 1e-6:
                control_state.policy_omega_n = float(max(-1.0, min(1.0, omega_n / OMEGA_MAX)))
            else:
                control_state.policy_omega_n = 0.0

            v_e, omega_e = self._action_to_executable_action(v_n, omega_n, control_state)
            robot.executable_velocity = v_e
            robot.executable_angular_velocity = omega_e

            # 速度調整 (Formation Adjustment)
            formation_target = self.simulator.formation.get_formation_target(robot_index)
            
            if formation_target is not None:
                formation_plan = self.simulator.formation.current_plan
                formation_center = (
                    formation_plan.center if formation_plan is not None else None
                )
                target_is_inner = _determine_target_is_inner(
                    robot.position,
                    formation_target,
                    formation_center=formation_center,
                )
                d_f, theta_f = self._compute_goal_info_for_target(robot, formation_target)
                d_o = -d_f * math.sin(theta_f)

                closest_distance, _ = robot.closest_obstacle
                obstacle_is_close = closest_distance <= self.simulator.env.ROBOT_RADIUS_M * 3

                v_a, omega_a = adjust_velocity(
                    v_e,
                    omega_e,
                    d_f,
                    d_o,
                    theta_f,
                    target_is_inner,
                    obstacle_is_close,
                    self.params.get("ETA_THETA"),
                    self.params.get("ETA_D"),
                    self.params.get("K_D"),
                    self.params.get("OMEGA_FIXED"),
                    self.params.get("T_DF"),
                    V_MAX,
                    OMEGA_MAX,
                )
            else:
                v_a, omega_a = v_e, omega_e

            # 衝突回避（Translation Block）
            translation_blocked = self._should_block_translation(robot)
            robot.braking = translation_blocked
            if translation_blocked:
                v_a = 0.0
            
            v_o, omega_o = self._action_to_executable_action(v_a, omega_a, control_state)

            robot.output_velocity = v_o
            robot.output_angular_velocity = omega_o

            moved = self._apply_motion(robot, v_o, omega_o)

            if moved:
                robot.velocity = v_o
            else:
                robot.velocity = 0.0

            control_state.current_v = v_o if moved else 0.0
            control_state.current_omega = omega_o
            control_state.previous_v = control_state.current_v
            control_state.previous_omega = control_state.current_omega

            if self.simulator.path_history != 0 and moved:
                robot.path.append((robot.position, robot.theta))
            
            # LiDAR更新はループ外で行うため、ここでは行わない
            # self.simulator._update_lidar(robot) <-- 削除

        if now is not None:
            self._last_action_timestamp = now

        if self._completed_robots and len(self._completed_robots) == len(robot_states):
            self._deactivate_policy_control()

    def _stop_robot(self, robot, control_state):
        robot.velocity = 0.0
        robot.executable_velocity = 0.0
        robot.executable_angular_velocity = 0.0
        robot.output_velocity = 0.0
        robot.output_angular_velocity = 0.0
        control_state.current_v = 0.0
        control_state.current_omega = 0.0
        control_state.previous_v = 0.0
        control_state.previous_omega = 0.0
        control_state.policy_omega_n = 0.0

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
            self._stop_robot(robot, PolicyRobotControlState(deque()))

    def get_last_policy_angular_velocities(self, normalized: bool = True) -> Dict[int, float]:
        if not self._control_state:
            return {}
        OMEGA_MAX = self.params["OMEGA_MAX"]
        scale = OMEGA_MAX if not normalized else 1.0
        result: Dict[int, float] = {}
        for index, state in self._control_state.items():
            result[index] = float(state.policy_omega_n) * scale
        return result

    def _load_policy_config(self) -> None:
        config: Optional[Dict[str, object]] = None
        try:
            package_files = importlib.resources.files("rl_training")
            config_path = package_files.joinpath("config", "ddpg_config.yaml")
            with importlib.resources.as_file(config_path) as path:
                with open(path, "r", encoding="utf-8") as handle:
                    config = yaml.safe_load(handle)
        except Exception:
            pass
            
        if config is None:
            self._lidar_history_len = 1
            self._n_l = 0
            self._n_e = 0
            return

        lidar_cfg = config.get("lidar", {}) if isinstance(config, dict) else {}
        self._lidar_history_len = int(lidar_cfg.get("lidar_history_len", 1))
        self._n_l = int(lidar_cfg.get("n_l", 0))
        self._n_e = int(lidar_cfg.get("n_e", 0))

    def _initialise_model(self) -> None:
        if self._model is not None:
            return
        if not self.model_path:
            return
        
        try:
            from stable_baselines3 import DDPG
            self._model = DDPG.load(self.model_path, device=self._model_device)
            self._model.policy.set_training_mode(False)
        except Exception as e:
            self._last_error = str(e)

    def _build_observation(self, robot_index: int, robot: RobotState, control_state: PolicyRobotControlState) -> Optional[Dict[str, np.ndarray]]:
        # LiDAR: 学習環境と同様に処理
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

        # Goal: [dist, angle_n, orientation_diff_n] (3次元)
        distance_to_target, angle_to_target = self._compute_goal_info(robot)
        goal_angle_n = float(np.clip(angle_to_target / math.pi, -1.0, 1.0))
        
        # ターゲットとの向きの差分
        target_theta = robot.target_theta if robot.target_theta is not None else 0.0
        angle_diff = target_theta - robot.theta
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        orientation_diff_n = float(np.clip(angle_diff / math.pi, -1.0, 1.0))

        obs_goal = np.array([distance_to_target, goal_angle_n, orientation_diff_n], dtype=np.float32)

        V_MAX = self.params["V_MAX"]
        OMEGA_MAX = self.params["OMEGA_MAX"]

        v_n = float(np.clip((control_state.current_v or 0.0) / V_MAX, 0.0, 1.0))
        omega_n = float(np.clip((control_state.current_omega or 0.0) / OMEGA_MAX, -1.0, 1.0))
        obs_velocity = np.array([v_n, omega_n], dtype=np.float32)

        # Closest Obstacle
        env = self.simulator.env
        co_distance, co_angle = robot.closest_obstacle
        if co_distance == float("inf"):
            co_distance = env.LIDAR_RANGE_M
        co_distance -= env.ROBOT_RADIUS_M
        co_dist_n = float(np.clip(co_distance / env.LIDAR_RANGE_M, 0.0, 1.0))
        co_angle_n = float(np.clip(co_angle / math.pi, -1.0, 1.0))
        obs_closest = np.array([co_dist_n, co_angle_n], dtype=np.float32)

        return {
            "lidar": obs_lidar,
            "goal": obs_goal,
            "velocity": obs_velocity,
            "closest_obstacle": obs_closest,
        }

    # --- LiDARフィルタリング処理 ---
    def _process_lidar_data(self, robot: RobotState) -> np.ndarray:
        sensor_data = robot.sensor_data
        
        # 1. フィルタリング (policy.pyの機能を統合)
        #    他のロボットを検出データから除去する
        if sensor_data:
            filtered_data, replaced_measurements = self._filter_robot_reflections(robot, sensor_data)
            # ロボットの状態も更新（レンダラーでの表示等に反映）
            robot.sensor_data = list(filtered_data)
            robot.filtered_sensor_data = list(replaced_measurements)
            # 処理用データ
            data_to_process = robot.sensor_data
        else:
            robot.filtered_sensor_data = []
            data_to_process = []
        
        # 2. 観測データ形式への変換 (newobsのロジック維持)
        lidar_range = self.simulator.env.LIDAR_RANGE_M
        if not data_to_process:
            return np.full(self._n_l + self._n_e, 100.0, dtype=np.float32)

        processed_lidar = []
        for data in data_to_process:
            dist = data[0]
            if dist < lidar_range:
                distance = dist / lidar_range
            else:
                distance = 100.0
            processed_lidar.append(distance)
        
        # 必要ならパディング（newobsの仕様に従い、足りない分を先頭データで埋める等ではなく、単に拡張している）
        processed_lidar += processed_lidar[: self._n_e]

        return np.array(processed_lidar, dtype=np.float32)

    # --- ロボット反射除去フィルター ---
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
    # -------------------------------------------------------------

    def _action_to_executable_action(self, v_n, omega_n, control_state):
        V_MAX = self.params["V_MAX"]
        OMEGA_MAX = self.params["OMEGA_MAX"]
        A_MAX = self.params["A_MAX"]
        ALPHA_MAX = self.params["ALPHA_MAX"]
        
        dt = self._dt
        v_kmin = max(0.0, control_state.previous_v - A_MAX * dt)
        v_kmax = min(V_MAX, control_state.previous_v + A_MAX * dt)
        omega_kmin = max(-OMEGA_MAX, control_state.previous_omega - ALPHA_MAX * dt)
        omega_kmax = min(OMEGA_MAX, control_state.previous_omega + ALPHA_MAX * dt)
        v_e = float(np.clip(v_n, v_kmin, v_kmax))
        omega_e = float(np.clip(omega_n, omega_kmin, omega_kmax))
        return v_e, omega_e

    def _apply_motion(self, robot, v_e, omega_e):
        env = self.simulator.env
        v_pixels_per_sec = v_e * env.PIXELS_PER_METER
        dx = -v_pixels_per_sec * math.sin(robot.theta) * self._dt
        dy = -v_pixels_per_sec * math.cos(robot.theta) * self._dt
        next_position = (robot.position[0] + dx, robot.position[1] + dy)
        moved = False
        
        # 衝突判定を行い、結果をロボットの状態に記録する
        if not self.simulator._is_collision(next_position, ignore_robot=robot):
            self.simulator._update_robot_motion(robot, next_position)
            moved = True
            robot.is_collided = False # 安全に移動できた
        else:
            robot.velocity = 0.0
            robot.is_collided = True  # 衝突により移動が阻まれたフラグを立てる
            
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
            
            if distance <= other_radius + 2.0 * robot_radius:   # brake 1.5
                dy_global = -dy_pixels
                global_angle = math.atan2(dy_global, dx_pixels)
                robot_global_angle = robot.theta + math.pi / 2
                angle_diff = (
                    (global_angle - robot_global_angle + math.pi) % (2 * math.pi)
                    - math.pi
                )
                if abs(angle_diff) <= math.radians(20.0):   # angle 20
                    return True
        return False
        
    def _compute_goal_info(self, robot: RobotState) -> Tuple[float, float]:
        if robot.target is None:
            return 0.0, 0.0
        return self._compute_goal_info_for_target(robot, robot.target)

    def _compute_goal_info_for_target(self, robot: RobotState, target: Tuple[float, float]) -> Tuple[float, float]:
        env = self.simulator.env
        dx_pixels = target[0] - robot.position[0]
        dy_pixels = target[1] - robot.position[1]
        distance_pixels = math.hypot(dx_pixels, dy_pixels)
        distance_m = distance_pixels / env.PIXELS_PER_METER

        heading_vector = (-math.sin(robot.theta), -math.cos(robot.theta))
        target_vector = (dx_pixels, dy_pixels)
        heading_math = (heading_vector[0], -heading_vector[1])
        target_math = (target_vector[0], -target_vector[1])
        dot = heading_math[0] * target_math[0] + heading_math[1] * target_math[1]
        cross = heading_math[0] * target_math[1] - heading_math[1] * target_math[0]
        if dot == 0.0 and cross == 0.0:
            angle_to_target = 0.0
        else:
            angle_to_target = math.atan2(cross, dot)
        return distance_m, angle_to_target

    def _has_reached_target(self, robot: RobotState) -> bool:
        if robot.target is None:
            return False
        dx_pixels = robot.target[0] - robot.position[0]
        dy_pixels = robot.target[1] - robot.position[1]
        distance_pixels = math.hypot(dx_pixels, dy_pixels)
        distance_m = distance_pixels / self.simulator.env.PIXELS_PER_METER
        return distance_m < self.simulator.env.ROBOT_BODY_RADIUS_M