"""Policy-backed velocity controller for ROS 2 integration.
(修正版v5: 座標変換ロジックの完全修正)
"""

from __future__ import annotations

import argparse
import importlib.resources
import math
from copy import deepcopy
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Protocol, Tuple

import numpy as np
import yaml
from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, PointCloud2

from .policy_utils_3 import ensure_repo_root_on_path, process_lidar_for_observation


_POLICY_LIDAR_HISTORY_LEN = 4
_POLICY_LIDAR_POINTS = 629
_POLICY_LIDAR_EXTENSION_POINTS = 11

# 学習環境と合わせるためのロボット半径設定
_DEFAULT_ROBOT_RADIUS_M = 0.45 


try:  # pragma: no cover - optional dependency path
    from sensor_msgs_py import point_cloud2
except ImportError:  # pragma: no cover - ROS helper may be absent in some tests
    point_cloud2 = None  # type: ignore[assignment]


@dataclass
class _PolicyTemporalState:
    previous_v: float = 0.0
    previous_omega: float = 0.0
    current_v: float = 0.0
    current_omega: float = 0.0


def quaternion_to_yaw(quaternion: object) -> float:
    x = float(getattr(quaternion, "x", 0.0))
    y = float(getattr(quaternion, "y", 0.0))
    z = float(getattr(quaternion, "z", 0.0))
    w = float(getattr(quaternion, "w", 1.0))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def pointcloud2_to_polar(msg: Optional[PointCloud2]) -> List[Tuple[float, float]]:
    if msg is None or point_cloud2 is None:
        return []
    samples: List[Tuple[float, float]] = []
    for x, y, z in point_cloud2.read_points(
        msg, field_names=("x", "y", "z"), skip_nans=True,
    ):
        xy_distance = math.hypot(x, y)
        angle = math.atan2(y, x)
        samples.append((float(xy_distance), float(angle)))
    return samples


def laser_scan_to_polar(msg: Optional[LaserScan]) -> List[Tuple[float, float]]:
    if msg is None:
        return []
    samples: List[Tuple[float, float]] = []
    angle = float(msg.angle_min)
    for distance in msg.ranges:
        current_angle = angle
        dist_val = float(distance)
        if not math.isfinite(dist_val):
            dist_val = float(msg.range_max)
        elif dist_val < msg.range_min:
            dist_val = float(msg.range_min)
            
        samples.append((dist_val, current_angle))
        angle += float(msg.angle_increment)
    return samples


if TYPE_CHECKING:
    from rclpy.node import Node


class _LoggerProtocol(Protocol):
    def error(self, message: str) -> None: ...
    def info(self, message: str) -> None: ...


class PolicyVelocityController:
    def __init__(
        self,
        *,
        model_path: str,
        step_period: float,
        node: Optional["Node"] = None,
        logger: Optional[_LoggerProtocol] = None,
        device: str = "cuda",
    ) -> None:
        self.model_path = model_path
        self._dt = float(step_period)
        self._node = node
        self._logger = node.get_logger() if node is not None else logger
        self._model_device = device
        self._model = None
        self._last_error: Optional[str] = None
        
        self._lidar_history_len = _POLICY_LIDAR_HISTORY_LEN
        self._n_l = _POLICY_LIDAR_POINTS
        self._n_e = _POLICY_LIDAR_EXTENSION_POINTS
        self._lidar_range_m = 10.0
        self._v_max = 1.0
        self._omega_max = 1.0
        self._a_max = 1.0
        self._alpha_max = 1.0
        self._robot_radius_m = _DEFAULT_ROBOT_RADIUS_M
        
        self._state = _PolicyTemporalState()
        self._lidar_history: Deque[np.ndarray] = deque(maxlen=self._lidar_history_len)
        self._last_robot_pose: Optional[Pose] = None

        self._load_policy_config()
        self._initialise_model()

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def compute_velocity_command(
        self,
        *,
        target_pose: Optional[PoseStamped],
        robot_pose: Optional[PoseStamped],
        odometry: Optional[Odometry],
        pointcloud: Optional[PointCloud2],
        laser_scan: Optional[LaserScan] = None,
    ) -> Optional[Twist]:
        if self._model is None:
            self._initialise_model()
            if self._model is None:
                return None

        observation = self._build_observation(
            target_pose=target_pose,
            robot_pose=robot_pose,
            odometry=odometry,
            pointcloud=pointcloud,
            laser_scan=laser_scan,
        )
        if observation is None:
            return None

        try:
            action, _ = self._model.predict(observation, deterministic=True)
        except Exception as exc:
            self._set_error(f"Policy prediction failed: {exc}")
            return None

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        # Gym Logic: v_cmd = (action[0] + 1) / 2 * self.V_MAX
        v_n = float((action[0] + 1.0) / 2.0 * self._v_max)
        # Gym Logic: omega_cmd = action[1] * self.OMEGA_MAX
        omega_n = float(action[1] * self._omega_max)
        
        v_e, omega_e = self._action_to_executable_action(v_n, omega_n)

        twist = Twist()
        twist.linear.x = v_e
        twist.angular.z = omega_e

        self._state.current_v = v_e
        self._state.current_omega = omega_e
        self._state.previous_v = v_e
        self._state.previous_omega = omega_e
        self._last_error = None
        return twist

    # --- Internal Helpers ---
    def _set_error(self, message: str) -> None:
        self._last_error = message
        if self._logger: self._logger.error(message)

    def _log_info(self, message: str) -> None:
        if self._logger and hasattr(self._logger, "info"): self._logger.info(message)

    def _log_warning(self, message: str) -> None:
        if self._logger and hasattr(self._logger, "warning"): self._logger.warning(message)

    def _load_policy_config(self) -> None:
        limits_cfg: Optional[Dict[str, object]] = None
        try:
            limits_resource = importlib.resources.files("coordinated_navigation").joinpath("config", "policy_simulator_limits.yaml")
        except ModuleNotFoundError: limits_resource = None
        
        if limits_resource:
            try:
                with importlib.resources.as_file(limits_resource) as path:
                    with open(path, "r", encoding="utf-8") as handle: limits_cfg = yaml.safe_load(handle)
            except Exception: pass
            
        if limits_cfg is not None:
            robot_cfg = limits_cfg.get("robot", {})
            lidar_cfg_sim = limits_cfg.get("lidar", {})
            self._v_max = float(robot_cfg.get("v_max", self._v_max))
            self._omega_max = float(robot_cfg.get("omega_max", self._omega_max))
            self._a_max = float(robot_cfg.get("a_max", self._a_max))
            self._alpha_max = float(robot_cfg.get("alpha_max", self._alpha_max))
            self._lidar_range_m = float(lidar_cfg_sim.get("range_m", self._lidar_range_m))
            self._robot_radius_m = float(robot_cfg.get("radius_m", self._robot_radius_m))

    def _initialise_model(self) -> None:
        if self._model is not None or not self.model_path: return
        resolved_path = self._resolve_model_path(self.model_path)
        if not resolved_path: self._set_error(f"Model not found: {self.model_path}"); return
        self.model_path = str(resolved_path)
        
        try: from stable_baselines3 import DDPG
        except ImportError: self._set_error("stable-baselines3 missing"); return
        
        try: from rl_training import custom_ddpg_policy
        except: ensure_repo_root_on_path(); from rl_training import custom_ddpg_policy
        
        try:
            self._model = DDPG.load(self.model_path, device=self._model_device)
            self._model.policy.set_training_mode(False)
            self._log_info(f"Loaded policy from {self.model_path}")
        except Exception as exc: self._set_error(f"Load failed: {exc}"); self._model = None

    def _resolve_model_path(self, path_str: str) -> Optional[Path]:
        path = Path(path_str)
        if path.exists(): return path.resolve()
        return path

    def _remap_lidar_to_gym_domain(self, polar_samples_ros: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        ROSのLiDARデータ(不定長, CCW, Front=0)を
        Gym環境のLiDAR仕様(固定長629点, CW, Front=0)に線形補間する。
        """
        if not polar_samples_ros:
            return []

        ros_dists = []
        gym_mapped_angles = []

        # ★重要修正★
        # ROS(CCW) -> Gym(CW) の変換
        # Gym 0 (Front) = ROS 0 (Front)
        # Gym pi/2 (Right) = ROS -pi/2 (Right)
        # Gym pi (Back) = ROS pi (Back)
        # Gym 3pi/2 (Left) = ROS pi/2 (Left)
        # 式: gym_angle = -ros_angle (mod 2pi)
        
        for dist, ros_angle in polar_samples_ros:
            # 負の角度を正の2pi範囲に正規化して反転
            gym_angle = (-ros_angle) % (2.0 * math.pi)
            ros_dists.append(dist)
            gym_mapped_angles.append(gym_angle)

        # 角度順にソート (np.interpのために必須)
        sorted_pairs = sorted(zip(gym_mapped_angles, ros_dists), key=lambda x: x[0])
        sorted_angles = np.array([p[0] for p in sorted_pairs])
        sorted_dists = np.array([p[1] for p in sorted_pairs])

        # データが空でなければ補間準備
        if len(sorted_angles) > 0:
            # 0と2piの境界をまたぐ補間のために、データを循環させる
            # 末尾に (最初の角度 + 2pi) のデータを追加
            sorted_angles = np.append(sorted_angles, sorted_angles[0] + 2.0 * math.pi)
            sorted_dists = np.append(sorted_dists, sorted_dists[0])
            
            # 先頭に (最後の角度 - 2pi) のデータを追加 (0付近の補間精度向上のため)
            sorted_angles = np.insert(sorted_angles, 0, sorted_angles[-2] - 2.0 * math.pi)
            sorted_dists = np.insert(sorted_dists, 0, sorted_dists[-2])

        # 学習済みモデルが期待する正確な角度グリッドを作成 (0 to 2pi, CW)
        target_gym_angles = np.linspace(0, 2.0 * math.pi, self._n_l, endpoint=False)

        # 線形補間実行
        interpolated_dists = np.interp(target_gym_angles, sorted_angles, sorted_dists)

        return list(zip(interpolated_dists, target_gym_angles))

    def _build_observation(
        self,
        *,
        target_pose: Optional[PoseStamped],
        robot_pose: Optional[PoseStamped],
        odometry: Optional[Odometry],
        pointcloud: Optional[PointCloud2],
        laser_scan: Optional[LaserScan],
    ) -> Optional[Dict[str, np.ndarray]]:
        if target_pose is None: return None

        pose_source = robot_pose.pose if robot_pose else (odometry.pose.pose if odometry else None)
        if pose_source is None: return None

        robot_position = pose_source.position
        target_position = target_pose.pose.position
        
        dx = float(target_position.x - robot_position.x)
        dy = float(target_position.y - robot_position.y)
        distance_to_target = math.hypot(dx, dy)
        
        robot_heading = quaternion_to_yaw(pose_source.orientation)
        target_heading = math.atan2(dy, dx)
        angle_to_target = wrap_to_pi(target_heading - robot_heading)
        goal_angle_n = float(np.clip(angle_to_target / math.pi, -1.0, 1.0))
        
        target_yaw = quaternion_to_yaw(target_pose.pose.orientation)
        orientation_diff = wrap_to_pi(target_yaw - robot_heading)
        orientation_diff_n = float(np.clip(orientation_diff / math.pi, -1.0, 1.0))

        # --- LiDAR処理 ---
        polar_samples_ros: List[Tuple[float, float]]
        if pointcloud:
            polar_samples_ros = pointcloud2_to_polar(pointcloud)
        else:
            polar_samples_ros = laser_scan_to_polar(laser_scan)
        
        polar_samples_gym = self._remap_lidar_to_gym_domain(polar_samples_ros)

        lidar_measurement = process_lidar_for_observation(
            polar_samples_gym,
            self._n_l,
            self._n_e,
            self._lidar_range_m,
        )
        self._lidar_history.append(lidar_measurement)
        
        history = list(self._lidar_history)
        if not history: history = [lidar_measurement]
        if len(history) < self._lidar_history_len:
            padding = [history[0].copy() for _ in range(self._lidar_history_len - len(history))]
            history = padding + history
        obs_lidar = np.array(history[-self._lidar_history_len:], dtype=np.float32)

        if odometry:
            self._state.current_v = float(odometry.twist.twist.linear.x)
            self._state.current_omega = float(odometry.twist.twist.angular.z)
        
        v_n = float(np.clip((self._state.current_v) / self._v_max, 0.0, 1.0))
        omega_n = float(np.clip((self._state.current_omega) / self._omega_max, -1.0, 1.0))
        obs_velocity = np.array([v_n, omega_n], dtype=np.float32)

        # --- Closest Obstacle ---
        closest_distance = float("inf")
        normalized_relative_angle = 0.0

        if polar_samples_gym:
            closest_sample = min(polar_samples_gym, key=lambda sample: sample[0])
            closest_distance = closest_sample[0]
            closest_gym_angle = closest_sample[1]
            
            # Gym sensor.py: atan2(sin, cos). 
            # gym_angleは [0, 2pi]. sin/cos経由で [-pi, pi] に正規化される
            normalized_relative_angle = math.atan2(
                math.sin(closest_gym_angle), math.cos(closest_gym_angle)
            )

        if not math.isfinite(closest_distance):
            closest_distance = self._lidar_range_m

        # ロボット半径を引く
        dist_surface = closest_distance - self._robot_radius_m
        
        co_dist_n = float(np.clip(dist_surface / self._lidar_range_m, 0.0, 1.0))
        co_angle_n = float(np.clip(normalized_relative_angle / math.pi, -1.0, 1.0))
        obs_closest = np.array([co_dist_n, co_angle_n], dtype=np.float32)

        obs_goal = np.array([distance_to_target, goal_angle_n, orientation_diff_n], dtype=np.float32)
        
        # Debug Log
        if self._logger:
            self._logger.info(
                f"OBS: Goal[{distance_to_target:.1f}m, {goal_angle_n:.2f}pi] "
                f"Obst[{dist_surface:.1f}m, {co_angle_n:.2f}pi] "
                f"Vel[{v_n:.2f}, {omega_n:.2f}]"
            )

        return {
            "lidar": obs_lidar,
            "goal": obs_goal,
            "velocity": obs_velocity,
            "closest_obstacle": obs_closest,
        }

    def _action_to_executable_action(self, v_n: float, omega_n: float) -> Tuple[float, float]:
        v_e = float(np.clip(v_n, 0.0, self._v_max))
        omega_e = float(np.clip(omega_n, -self._omega_max, self._omega_max))
        return v_e, omega_e

def main() -> None: pass 

__all__ = ["PolicyVelocityController", "pointcloud2_to_polar", "laser_scan_to_polar", "quaternion_to_yaw", "wrap_to_pi"]