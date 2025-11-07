"""Policy-backed velocity controller for ROS 2 integration.

This module adapts the renderer's :mod:`multi_robot.renderer.policy` controller
logic so that a pre-trained DDPG policy can be executed inside a ROS 2 node.
"""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.resources
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import yaml
from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.logging import Logger
from sensor_msgs.msg import PointCloud2

from .policy_utils import ensure_repo_root_on_path, process_lidar_for_observation


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
        samples.append((float(xy_distance), float(angle)))

    return samples


class PolicyVelocityController:
    """Load and execute a trained DDPG policy to generate velocity commands."""

    def __init__(
        self,
        *,
        model_path: str,
        step_period: float,
        logger: Optional[Logger] = None,
        device: str = "cuda",
    ) -> None:
        self.model_path = model_path
        self._dt = float(step_period)
        self._logger = logger
        self._model_device = device
        self._model = None
        self._last_error: Optional[str] = None
        self._lidar_history_len = 1
        self._n_l = 0
        self._n_e = 0
        self._lidar_range_m = 10.0
        self._v_max = 1.0
        self._omega_max = 1.0
        self._a_max = 1.0
        self._alpha_max = 1.0
        self._state = _PolicyTemporalState()
        self._lidar_history: Deque[np.ndarray] = deque(maxlen=1)

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

    # ------------------------------------------------------------------
    # Internal helpers mirroring the renderer's policy controller
    # ------------------------------------------------------------------
    def _set_error(self, message: str) -> None:
        self._last_error = message
        if self._logger is not None:
            self._logger.error(message)

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
            self._set_error(f"Failed to read DDPG config: {message}")
            return

        lidar_cfg = config.get("lidar", {}) if isinstance(config, dict) else {}
        self._lidar_history_len = int(lidar_cfg.get("lidar_history_len", 1))
        self._n_l = int(lidar_cfg.get("n_l", 0))
        self._n_e = int(lidar_cfg.get("n_e", 0))

        self._lidar_history = deque(maxlen=self._lidar_history_len)

        simulator_cfg: Optional[Dict[str, object]] = None
        last_exc = None
        sim_package_files: Optional[importlib.abc.Traversable] = None

        try:
            sim_package_files = importlib.resources.files("simulation_core")
        except ModuleNotFoundError:
            repo_root = ensure_repo_root_on_path()
            candidate = repo_root / "simulation_core" / "config" / "simulator_config.yaml"
            if candidate.exists():
                try:
                    with open(candidate, "r", encoding="utf-8") as handle:
                        simulator_cfg = yaml.safe_load(handle)
                except Exception as exc:  # pragma: no cover - defensive path
                    last_exc = exc
            else:
                last_exc = FileNotFoundError(candidate)

        if simulator_cfg is None and sim_package_files is not None:
            try:
                simulator_path = sim_package_files.joinpath("config", "simulator_config.yaml")
                with importlib.resources.as_file(simulator_path) as path:
                    with open(path, "r", encoding="utf-8") as handle:
                        simulator_cfg = yaml.safe_load(handle)
            except Exception as exc:  # pragma: no cover - defensive path
                last_exc = exc

        if simulator_cfg is None:
            message = "unknown error" if last_exc is None else str(last_exc)
            self._set_error(f"Failed to read simulator config: {message}")
            return

        robot_cfg = simulator_cfg.get("robot", {})
        lidar_cfg_sim = simulator_cfg.get("lidar", {})

        self._v_max = float(robot_cfg.get("v_max", self._v_max))
        self._omega_max = float(robot_cfg.get("omega_max", self._omega_max))
        self._a_max = float(robot_cfg.get("a_max", self._a_max))
        self._alpha_max = float(robot_cfg.get("alpha_max", self._alpha_max))
        self._lidar_range_m = float(lidar_cfg_sim.get("range_m", self._lidar_range_m))

    def _initialise_model(self) -> None:
        if self._model is not None:
            return

        if not self.model_path:
            self._set_error("No policy model path has been provided.")
            return

        original_path = Path(self.model_path)
        candidate_paths: List[Path] = []

        if original_path.is_absolute():
            candidate_paths.append(original_path)
        else:
            candidate_paths.append(original_path)
            package_root = Path(__file__).resolve().parents[2]
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

        try:
            self._model = DDPG.load(self.model_path, device=self._model_device)
            self._model.policy.set_training_mode(False)
            self._last_error = None
        except Exception as exc:  # pragma: no cover - model loading failure
            self._set_error(f"Failed to load policy model: {exc}")
            self._model = None

    def _build_observation(
        self,
        *,
        target_pose: Optional[PoseStamped],
        robot_pose: Optional[PoseStamped],
        odometry: Optional[Odometry],
        pointcloud: Optional[PointCloud2],
    ) -> Optional[Dict[str, np.ndarray]]:
        if target_pose is None or pointcloud is None:
            return None

        pose_source: Optional[Pose]
        if odometry is not None:
            pose_source = odometry.pose.pose
        elif robot_pose is not None:
            pose_source = robot_pose.pose
        else:
            return None

        robot_position = pose_source.position
        robot_orientation = pose_source.orientation

        target_position = target_pose.pose.position

        dx = float(target_position.x - robot_position.x)
        dy = float(target_position.y - robot_position.y)
        distance_to_target = math.hypot(dx, dy)

        robot_heading = quaternion_to_yaw(robot_orientation)
        target_heading = math.atan2(dy, dx)
        angle_to_target = wrap_to_pi(target_heading - robot_heading)
        goal_angle_n = float(np.clip(angle_to_target / math.pi, -1.0, 1.0))

        polar_samples = pointcloud2_to_polar(pointcloud)
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

        if odometry is not None:
            self._state.current_v = float(odometry.twist.twist.linear.x)
            self._state.current_omega = float(odometry.twist.twist.angular.z)

        v_n = float(np.clip((self._state.current_v or 0.0) / self._v_max, 0.0, 1.0))
        omega_n = float(
            np.clip((self._state.current_omega or 0.0) / self._omega_max, -1.0, 1.0)
        )
        obs_velocity = np.array([v_n, omega_n], dtype=np.float32)

        if polar_samples:
            closest_distance, closest_angle = min(polar_samples, key=lambda sample: sample[0])
        else:
            closest_distance, closest_angle = float("inf"), 0.0

        if not math.isfinite(closest_distance):
            closest_distance = self._lidar_range_m

        co_dist_n = float(np.clip(closest_distance / self._lidar_range_m, 0.0, 1.0))
        co_angle_n = float(np.clip(closest_angle / math.pi, -1.0, 1.0))
        obs_closest = np.array([co_dist_n, co_angle_n], dtype=np.float32)

        obs_goal = np.array([distance_to_target, goal_angle_n], dtype=np.float32)

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
    "quaternion_to_yaw",
    "wrap_to_pi",
]

