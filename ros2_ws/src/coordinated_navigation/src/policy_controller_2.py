"""Policy-backed velocity controller for ROS 2 integration.

This module adapts the renderer's :mod:`multi_robot.renderer.policy` controller
logic so that a pre-trained DDPG policy can be executed inside a ROS 2 node.
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

from .policy_utils import ensure_repo_root_on_path, process_lidar_for_observation


_POLICY_LIDAR_HISTORY_LEN = 4
"""Number of LiDAR scans stacked when generating observations."""

_POLICY_LIDAR_POINTS = 629
"""Number of raw LiDAR samples expected by the trained policy."""

_POLICY_LIDAR_EXTENSION_POINTS = 11
"""Number of extension points appended to each LiDAR scan."""


try:  # pragma: no cover - optional dependency path
    from sensor_msgs_py import point_cloud2
except ImportError:  # pragma: no cover - ROS helper may be absent in some tests
    point_cloud2 = None  # type: ignore[assignment]


@dataclass
class _PolicyTemporalState:
    """Tracks temporal values required to enforce actuator limits."""

    previous_v: float = 0.0
    previous_omega: float = 0.0
    current_v: float = 0.0
    current_omega: float = 0.0


def quaternion_to_yaw(quaternion: object) -> float:
    """Return the yaw component of a quaternion-like object."""

    x = float(getattr(quaternion, "x", 0.0))
    y = float(getattr(quaternion, "y", 0.0))
    z = float(getattr(quaternion, "z", 0.0))
    w = float(getattr(quaternion, "w", 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle: float) -> float:
    """Normalise ``angle`` into ``[-pi, pi]``."""

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def pointcloud2_to_polar(msg: Optional[PointCloud2]) -> List[Tuple[float, float]]:
    """Convert a :class:`sensor_msgs.msg.PointCloud2` into polar LiDAR samples."""

    if msg is None or point_cloud2 is None:
        return []

    samples: List[Tuple[float, float]] = []
    for x, y, z in point_cloud2.read_points(  # type: ignore[attr-defined]
        msg,
        field_names=("x", "y", "z"),
        skip_nans=True,
    ):
        xy_distance = math.hypot(x, y)
        angle = math.atan2(y, x)
        # convert angle to 2D sim frame
        angle = wrap_to_pi(angle - math.pi / 2.0)
        samples.append((float(xy_distance), float(angle)))
    # processed_samples = samples[157:] + samples[:157]

    return samples


def laser_scan_to_polar(msg: Optional[LaserScan]) -> List[Tuple[float, float]]:
    """Convert a :class:`sensor_msgs.msg.LaserScan` into polar LiDAR samples."""

    if msg is None:
        return []

    samples: List[Tuple[float, float]] = []
    angle = float(msg.angle_min)
    for distance in msg.ranges:
        # print(f"distance: {distance}")
        if not math.isfinite(distance):
            # samples.append((float("inf"), angle))
            samples.append((float(msg.range_max), angle))
        elif distance < msg.range_min:
            samples.append((float(msg.range_min), angle))
        else:
            samples.append((float(distance), angle))
        angle += float(msg.angle_increment)
    return samples


if TYPE_CHECKING:
    from rclpy.node import Node


class _LoggerProtocol(Protocol):
    """Minimal protocol describing the logger interface used in this module."""

    def error(self, message: str) -> None:
        ...


class PolicyVelocityController:
    """Load and execute a trained DDPG policy to generate velocity commands."""

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
        self._state = _PolicyTemporalState()
        self._lidar_history: Deque[np.ndarray] = deque(maxlen=self._lidar_history_len)

        self._last_robot_pose: Optional[Pose] = None

        self._load_policy_config()
        self._initialise_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
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
        except Exception as exc:  # pragma: no cover - runtime model failure
            self._set_error(f"Policy prediction failed: {exc}")
            return None

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 2:
            self._set_error(f"Unexpected action shape {action.shape}; expected size 2")
            return None

        v_n = float((action[0] + 1.0) / 2.0 * self._v_max)
        omega_n = float(action[1] * self._omega_max)
        print(f"v_n: {v_n}, omnga_n: {omega_n}")

        v_e, omega_e = self._action_to_executable_action(v_n, omega_n)
        print(f"v_e: {v_e}, omega_e: {omega_e}")

        twist = Twist()
        twist.linear.x = v_e
        twist.angular.z = omega_e

        self._state.current_v = v_e
        self._state.current_omega = omega_e
        self._state.previous_v = v_e
        self._state.previous_omega = omega_e

        self._last_error = None
        return twist

    # ------------------------------------------------------------------
    # Internal helpers mirroring the renderer's policy controller
    # ------------------------------------------------------------------
    def _set_error(self, message: str) -> None:
        self._last_error = message
        if self._logger is not None:
            self._logger.error(message)

    def _log_info(self, message: str) -> None:
        if self._logger is not None and hasattr(self._logger, "info"):
            self._logger.info(message)

    def _log_warning(self, message: str) -> None:
        if self._logger is not None and hasattr(self._logger, "warning"):
            self._logger.warning(message)

    def _load_policy_config(self) -> None:
        limits_cfg: Optional[Dict[str, object]] = None
        last_exc: Optional[Exception] = None

        try:
            limits_resource = importlib.resources.files("coordinated_navigation").joinpath(
                "config", "policy_simulator_limits.yaml"
            )
        except ModuleNotFoundError as exc:
            limits_resource = None
            last_exc = exc

        if limits_resource is not None:
            try:
                with importlib.resources.as_file(limits_resource) as path:
                    with open(path, "r", encoding="utf-8") as handle:
                        limits_cfg = yaml.safe_load(handle)
            except Exception as exc:  # pragma: no cover - defensive path
                last_exc = exc

        if limits_cfg is None:
            local_candidate = (
                Path(__file__).resolve().parent.parent / "config" / "policy_simulator_limits.yaml"
            )
            if local_candidate.exists():
                try:
                    with open(local_candidate, "r", encoding="utf-8") as handle:
                        limits_cfg = yaml.safe_load(handle)
                except Exception as exc:  # pragma: no cover - defensive path
                    last_exc = exc

        if limits_cfg is None:
            try:
                from ament_index_python.packages import get_package_share_directory

                share_directory = Path(get_package_share_directory("coordinated_navigation"))
                share_candidate = share_directory / "config" / "policy_simulator_limits.yaml"
                if share_candidate.exists():
                    with open(share_candidate, "r", encoding="utf-8") as handle:
                        limits_cfg = yaml.safe_load(handle)
            except Exception as exc:  # pragma: no cover - defensive path
                last_exc = exc

        if not isinstance(limits_cfg, dict):
            limits_cfg = None

        if limits_cfg is None:
            message = "unknown error" if last_exc is None else str(last_exc)
            self._log_warning(
                "Unable to read policy simulator limits; using built-in defaults"
            )
            return

        robot_cfg = limits_cfg.get("robot", {})
        lidar_cfg_sim = limits_cfg.get("lidar", {})

        self._v_max = float(robot_cfg.get("v_max", self._v_max))
        self._omega_max = float(robot_cfg.get("omega_max", self._omega_max))
        self._a_max = float(robot_cfg.get("a_max", self._a_max))
        self._alpha_max = float(robot_cfg.get("alpha_max", self._alpha_max))
        self._lidar_range_m = float(lidar_cfg_sim.get("range_m", self._lidar_range_m))

        self._lidar_history_len = _POLICY_LIDAR_HISTORY_LEN
        self._n_l = _POLICY_LIDAR_POINTS
        self._n_e = _POLICY_LIDAR_EXTENSION_POINTS
        self._lidar_history = deque(maxlen=self._lidar_history_len)

    def _initialise_model(self) -> None:
        if self._model is not None:
            return

        if not self.model_path:
            self._set_error("No policy model path has been provided.")
            return

        resolved_path = self._resolve_model_path(self.model_path)
        if resolved_path is None:
            self._set_error(f"Policy model not found: {self.model_path}")
            return

        self.model_path = str(resolved_path)

        try:
            from stable_baselines3 import DDPG
        except ImportError as exc:  # pragma: no cover - missing optional dependency
            self._set_error(
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
                self._set_error(
                    "The rl_training package is required to load the policy model."
                    f" ({exc})"
                )
                return
            except Exception as exc:  # pragma: no cover - unexpected import failure
                self._set_error(
                    "Unexpected error while importing rl_training for the policy model."
                    f" ({exc})"
                )
                return
        except Exception as exc:  # pragma: no cover - unexpected import failure
            self._set_error(
                "Unexpected error while importing rl_training for the policy model."
                f" ({exc})"
            )
            return

        try:
            self._model = DDPG.load(self.model_path, device=self._model_device)
            self._model.policy.set_training_mode(False)
            self._last_error = None
            self._log_info(f"Loaded policy model from {self.model_path}")
        except Exception as exc:  # pragma: no cover - model loading failure
            self._set_error(f"Failed to load policy model: {exc}")
            self._model = None

    def _resolve_model_path(self, path_str: str) -> Optional[Path]:
        path = Path(path_str)
        candidates: List[Path] = []

        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(path)
            package_root = Path(__file__).resolve().parents[2]
            candidates.append(package_root / path)
            candidates.append(package_root / path.name)
            try:
                repo_root_str = ensure_repo_root_on_path()
            except Exception:  # pragma: no cover - defensive path
                repo_root_str = None
            if repo_root_str:
                repo_root = Path(repo_root_str)
                candidates.append(repo_root / path)
                candidates.append(repo_root / path.name)
            try:
                from ament_index_python.packages import get_package_share_directory

                share_dir = Path(get_package_share_directory("coordinated_navigation"))
                candidates.append(share_dir / path)
                if path.name:
                    candidates.append(share_dir / "models" / path.name)
            except Exception:  # pragma: no cover - ament not available
                pass

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    def _build_observation(
        self,
        *,
        target_pose: Optional[PoseStamped],
        robot_pose: Optional[PoseStamped],
        odometry: Optional[Odometry],
        pointcloud: Optional[PointCloud2],
        laser_scan: Optional[LaserScan],
    ) -> Optional[Dict[str, np.ndarray]]:
        if target_pose is None:
            return None

        pose_source: Optional[Pose]
        # if odometry is not None:
        #     pose_source = odometry.pose.pose
        # elif robot_pose is not None:
        #     pose_source = robot_pose.pose
        if robot_pose is not None:
            pose_source = robot_pose.pose
        elif odometry is not None:
            pose_source = odometry.pose.pose
        else:
            return None

        measured_v: Optional[float] = None
        measured_omega: Optional[float] = None

        if odometry is not None:
            measured_v = float(odometry.twist.twist.linear.x)
            measured_omega = float(odometry.twist.twist.angular.z)
        elif self._last_robot_pose is not None:
            dt = self._dt
            if dt > 0.0:
                dx_pose = float(pose_source.position.x - self._last_robot_pose.position.x)
                dy_pose = float(pose_source.position.y - self._last_robot_pose.position.y)
                measured_v = math.hypot(dx_pose, dy_pose) / dt

                yaw_now = quaternion_to_yaw(pose_source.orientation)
                yaw_prev = quaternion_to_yaw(self._last_robot_pose.orientation)
                measured_omega = wrap_to_pi(yaw_now - yaw_prev) / dt

        if measured_v is not None:
            self._state.current_v = measured_v
        if measured_omega is not None:
            self._state.current_omega = measured_omega

        self._last_robot_pose = deepcopy(pose_source)

        robot_position = pose_source.position
        robot_orientation = pose_source.orientation

        target_position = target_pose.pose.position
        target_orientation = target_pose.pose.orientation

        dx = float(target_position.x - robot_position.x)
        dy = float(target_position.y - robot_position.y)
        distance_to_target = math.hypot(dx, dy)
        # print("distance to target: ", distance_to_target)

        robot_heading = quaternion_to_yaw(robot_orientation)
        # print("robot heading: ", robot_heading)
        target_heading = math.atan2(dy, dx)
        angle_to_target = wrap_to_pi(target_heading - robot_heading)
        goal_angle_n = float(np.clip(angle_to_target / math.pi, -1.0, 1.0))

        target_yaw = quaternion_to_yaw(target_orientation)
        orientation_diff = wrap_to_pi(target_yaw - robot_heading)
        orientation_diff_n = float(np.clip(orientation_diff / math.pi, -1.0, 1.0))

        polar_samples: List[Tuple[float, float]]
        if pointcloud is not None:
            polar_samples = pointcloud2_to_polar(pointcloud)
        else:
            polar_samples = laser_scan_to_polar(laser_scan)
        lidar_measurement = process_lidar_for_observation(
            polar_samples,
            self._n_l,
            self._n_e,
            self._lidar_range_m,
        )
        self._lidar_history.append(lidar_measurement)

        history: List[np.ndarray] = list(self._lidar_history)
        if not history:
            history = [lidar_measurement]
        if len(history) < self._lidar_history_len:
            padding = [history[0].copy() for _ in range(self._lidar_history_len - len(history))]
            history = padding + history
        history = history[-self._lidar_history_len :]
        obs_lidar = np.array(history, dtype=np.float32)
        print(f"history shape: {obs_lidar.shape}")

        # if odometry is not None:
        #     self._state.current_v = float(odometry.twist.twist.linear.x)
        #     self._state.current_omega = float(odometry.twist.twist.angular.z)

        v_n = float(np.clip((self._state.current_v or 0.0) / self._v_max, 0.0, 1.0))
        omega_n = float(
            np.clip((self._state.current_omega or 0.0) / self._omega_max, -1.0, 1.0)
        )
        obs_velocity = np.array([v_n, omega_n], dtype=np.float32)

        if polar_samples:
            closest_distance, closest_angle = min(polar_samples, key=lambda sample: sample[0])
        else:
            print("CLOSEST OBSTACLE DISTANCE IS SET AS INF")
            closest_distance, closest_angle = float("inf"), 0.0

        if not math.isfinite(closest_distance):
            closest_distance = self._lidar_range_m

        co_dist_n = float(np.clip(closest_distance / self._lidar_range_m, 0.0, 1.0))
        co_angle_n = float(np.clip(closest_angle / math.pi, -1.0, 1.0))
        obs_closest = np.array([co_dist_n, co_angle_n], dtype=np.float32)

        obs_goal = np.array(
            [distance_to_target, goal_angle_n, orientation_diff_n], dtype=np.float32
        )
        print(f"obs_goal: {obs_goal}")
        print(f"velocity: {obs_velocity}")
        print(f"closest obstacle: {obs_closest}")
        print()

        return {
            "lidar": obs_lidar,
            "goal": obs_goal,
            "velocity": obs_velocity,
            "closest_obstacle": obs_closest,
        }

    def _action_to_executable_action(self, v_n: float, omega_n: float) -> Tuple[float, float]:
        dt = self._dt

        v_kmin = max(0.0, self._state.previous_v - self._a_max * dt)
        v_kmax = min(self._v_max, self._state.previous_v + self._a_max * dt)
        omega_kmin = max(-self._omega_max, self._state.previous_omega - self._alpha_max * dt)
        omega_kmax = min(self._omega_max, self._state.previous_omega + self._alpha_max * dt)

        v_e = float(np.clip(v_n, v_kmin, v_kmax))
        omega_e = float(np.clip(omega_n, omega_kmin, omega_kmax))
        return v_e, omega_e


def main() -> None:  # pragma: no cover - convenience CLI
    parser = argparse.ArgumentParser(description="Inspect a trained policy model")
    parser.add_argument("model", help="Path to the trained policy model")
    parser.add_argument(
        "--step-period",
        type=float,
        default=0.1,
        help="Controller step period in seconds",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device identifier forwarded to stable-baselines3",
    )
    args = parser.parse_args()

    controller = PolicyVelocityController(
        model_path=args.model,
        step_period=args.step_period,
        device=args.device,
    )
    if not controller.is_available:
        raise SystemExit(controller.last_error or "Failed to load model")

    print(f"Successfully loaded policy from {controller.model_path}")


__all__ = [
    "PolicyVelocityController",
    "pointcloud2_to_polar",
    "laser_scan_to_polar",
    "quaternion_to_yaw",
    "wrap_to_pi",
]

