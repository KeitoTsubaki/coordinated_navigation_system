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
from .types import RobotState
from .utils import ensure_repo_root_on_path

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .simulator_v5 import MultiRobotSimulator


@dataclass
class PolicyRobotControlState:
    """Keeps track of temporal state required by the RL policy controller."""
    lidar_history: Deque[np.ndarray]
    previous_v: float = 0.0
    previous_omega: float = 0.0
    current_v: float = 0.0
    current_omega: float = 0.0
    policy_omega_n: float = 0.0


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

        # --- 【追加】サブゴール待機用の変数を追加 ---
        self._waiting_at_subgoal: Dict[int, bool] = {} # ロボット毎の待機フラグ
        self.subgoal_stop_threshold = 0.1   # この距離(m)以内に入ったら停止
        self.subgoal_resume_threshold = 0.3 # この距離(m)以上離れたら再開
        # ----------------------------------------
        
        self.params = {
            "V_MAX": 1.0,
            "OMEGA_MAX": 1.0,
            "A_MAX": 1.0,
            "ALPHA_MAX": 1.0
        }
        self._load_local_params()
        
        self._load_policy_config()
        self._initialise_model()

    def _load_local_params(self) -> None:
        try:
            config_path = Path(__file__).parent / "policy_params.yaml"
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

        for robot in robot_states:
            self.simulator._update_lidar(robot)
        lidar_preupdated = True

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

            # =================================================================
            # 【修正】最終ゴールの場合のみ、到達判定と停止を行う
            # =================================================================
            is_final = getattr(robot, 'is_final_target', False)

            if robot_index in self._completed_robots:
                # 既に到達済みで、まだゴール付近にいるなら停止継続
                if self._has_reached_target(robot) :
                    self._stop_robot(robot, control_state)
                    continue
                # ゴールから離れた、または新しいゴールが設定された（is_final=False）場合
                self._completed_robots.discard(robot_index)

            if is_final and self._has_reached_target(robot):
                self._completed_robots.add(robot_index)
                self._stop_robot(robot, control_state)
                continue
            # =================================================================

            # =================================================================
            # 【追加】サブゴール(is_final=False)での一時停止ロジック
            # =================================================================
            # if not is_final:
            #     # ターゲット(サブゴール)までの距離を計算
            #     dist_m, _ = self._compute_goal_info(robot)

            #     # 初回アクセスの初期化
            #     if robot_index not in self._waiting_at_subgoal:
            #         self._waiting_at_subgoal[robot_index] = False
                
            #     # 状態遷移判定 (ヒステリシス)
            #     if self._waiting_at_subgoal[robot_index]:
            #         # 待機中: 目標が十分離れたら(resume_threshold超え)、移動再開
            #         if dist_m > self.subgoal_resume_threshold:
            #             self._waiting_at_subgoal[robot_index] = False
            #     else:
            #         # 移動中: 目標に十分近づいたら(stop_threshold未満)、一時停止
            #         if dist_m < self.subgoal_stop_threshold:
            #             self._waiting_at_subgoal[robot_index] = True
                
            #     # 待機状態なら強制停止して次のロボットへ (推論をスキップ)
            #     if self._waiting_at_subgoal[robot_index]:
            #         self._stop_robot(robot, control_state)
            #         continue
            # =================================================================

            if not lidar_preupdated:
                self.simulator._update_lidar(robot)
                
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

            v_a, omega_a = v_e, omega_e
            
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

            self.simulator._update_lidar(robot)

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

    def _process_lidar_data(self, robot: RobotState) -> np.ndarray:
        sensor_data = robot.sensor_data
        lidar_range = self.simulator.env.LIDAR_RANGE_M
        
        # データがない場合は、学習環境に合わせて100.0で埋める
        if not sensor_data:
            return np.full(self._n_l + self._n_e, 100.0, dtype=np.float32)

        processed_lidar = []
        for data in sensor_data:
            dist = data[0]
            if dist < lidar_range:
                distance = dist / lidar_range
            else:
                # レンジ外は 100.0
                distance = 100.0
            processed_lidar.append(distance)
        
        processed_lidar += processed_lidar[: self._n_e]

        return np.array(processed_lidar, dtype=np.float32)

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
        
        # 【修正】衝突判定を行い、結果をロボットの状態に記録する
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
            
            if distance <= other_radius + robot_radius + 0.3 * self.simulator.env.PIXELS_PER_METER:   # brake 0.5m
                dy_global = -dy_pixels
                global_angle = math.atan2(dy_global, dx_pixels)
                robot_global_angle = robot.theta + math.pi / 2
                angle_diff = (
                    (global_angle - robot_global_angle + math.pi) % (2 * math.pi)
                    - math.pi
                )
                if abs(angle_diff) <= math.radians(45.0):   # angle 20
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