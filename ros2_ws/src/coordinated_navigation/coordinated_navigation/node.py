"""Primary ROS 2 node for coordinating navigation of a single robot."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional, Sequence, Type, cast

import rclpy
from rclpy.any_msg import AnyMsg
from rclpy.node import Node
from rclpy.qos import (
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2

from .policy_controller import PolicyVelocityController


@dataclass
class FormationState:
    """Relevant details extracted from a `Formation` message."""

    formation_id: str
    theta: float
    cost: float
    header: Optional[object] = None


class CoordinatedNavigationNode(Node):
    """ROS 2 node that wires up navigation interfaces for a single robot instance."""

    def __init__(
        self,
        node_name: str = "coordinated_navigation",
        *,
        robot_namespace: Optional[str] = None,
        robot_id: Optional[str] = None,
        formation_msg_type: Optional[Type[object]] = None,
        timer_period: float = 0.1,
    ) -> None:
        super().__init__(node_name)

        self.declare_parameter("robot_namespace", "")
        self.declare_parameter("robot_id", "1")
        self.declare_parameter("formation_msg_module", "coordinated_navigation_interfaces.msg")
        self.declare_parameter("formation_msg_name", "Formation")
        self.declare_parameter("policy.model_path", "")
        self.declare_parameter("policy.step_period", timer_period)
        self.declare_parameter("policy.device", "cpu")

        self._robot_namespace = self._normalise_robot_namespace(robot_namespace, robot_id)
        self._formation_msg_type = self._resolve_formation_msg_type(formation_msg_type)

        policy_model_path = self._parameter_string("policy.model_path")
        policy_step_period = self._parameter_float("policy.step_period", timer_period)
        policy_device = self._parameter_string("policy.device") or "cpu"

        self._policy_controller = PolicyVelocityController(
            model_path=policy_model_path,
            step_period=policy_step_period,
            logger=self.get_logger(),
            device=policy_device,
        )

        self.latest_target_pose: Optional[PoseStamped] = None
        self.latest_robot_pose: Optional[PoseStamped] = None
        self.latest_formation: Optional[FormationState] = None
        self.latest_odometry: Optional[Odometry] = None
        self.latest_pointcloud: Optional[PointCloud2] = None

        qos_reliable = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)
        self._target_pose_subscription = self.create_subscription(
            PoseStamped,
            "/target_pose",
            self._target_pose_callback,
            qos_reliable,
        )

        pose_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)
        formation_qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.RELIABLE)
        odometry_qos = QoSProfile(depth=20, reliability=QoSReliabilityPolicy.RELIABLE)
        cmd_vel_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        self._pose_subscription = self.create_subscription(
            PoseStamped,
            self._topic("pose"),
            self._robot_pose_callback,
            pose_qos,
        )
        self._formation_subscription = self.create_subscription(
            self._formation_msg_type,
            self._topic("formation"),
            self._robot_formation_callback,
            formation_qos,
        )
        self._odometry_subscription = self.create_subscription(
            Odometry,
            self._topic("odom"),
            self._robot_odometry_callback,
            odometry_qos,
        )
        self._pointcloud_subscription = self.create_subscription(
            PointCloud2,
            self._topic("points"),
            self._robot_pointcloud_callback,
            qos_profile_sensor_data,
        )
        self._cmd_vel_publisher = self.create_publisher(
            Twist,
            self._topic("cmd_vel"),
            cmd_vel_qos,
        )

        self._coordination_timer = self.create_timer(
            max(policy_step_period, 1e-3), self._coordination_timer_callback
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _normalise_robot_namespace(
        self,
        explicit_namespace: Optional[str],
        robot_id_override: Optional[str],
    ) -> str:
        namespace_candidate = explicit_namespace or self._parameter_string("robot_namespace")
        if namespace_candidate:
            return self._format_namespace(namespace_candidate)

        robot_id_value = robot_id_override or self._read_robot_id_parameter()
        if robot_id_value:
            return self._format_namespace(f"/robot_{robot_id_value}")
        return ""

    def _read_robot_id_parameter(self) -> str:
        parameter_value = self.get_parameter("robot_id").get_parameter_value()
        if parameter_value.string_value:
            return parameter_value.string_value
        if parameter_value.integer_value:
            return str(parameter_value.integer_value)
        return "1"

    def _parameter_string(self, name: str) -> str:
        parameter_value = self.get_parameter(name).get_parameter_value()
        return parameter_value.string_value or ""

    def _parameter_float(self, name: str, default: float) -> float:
        parameter_value = self.get_parameter(name).get_parameter_value()
        if parameter_value.type_ == parameter_value.Type.DOUBLE:
            return float(parameter_value.double_value)
        if parameter_value.type_ == parameter_value.Type.INTEGER:
            return float(parameter_value.integer_value)
        return float(default)

    @staticmethod
    def _format_namespace(namespace: str) -> str:
        namespace = namespace.strip()
        if not namespace:
            return ""
        namespace = namespace.rstrip("/")
        if not namespace.startswith("/"):
            namespace = f"/{namespace}"
        return namespace

    def _resolve_formation_msg_type(self, override: Optional[Type[object]]) -> Type[object]:
        if override is not None:
            return override

        module_name = self._parameter_string("formation_msg_module")
        type_name = self._parameter_string("formation_msg_name")
        if not module_name or not type_name:
            self.get_logger().warning(
                "Formation message parameters are unset; defaulting to AnyMsg for subscriptions.",
            )
            return AnyMsg

        try:
            module = importlib.import_module(module_name)
            msg_type = getattr(module, type_name)
        except (ImportError, AttributeError) as exc:
            self.get_logger().warning(
                "Failed to import formation message type '%s.%s': %s. Falling back to AnyMsg.",
                module_name,
                type_name,
                exc,
            )
            return AnyMsg
        return msg_type

    def _topic(self, suffix: str) -> str:
        suffix = suffix.strip()
        if not suffix:
            return self._robot_namespace
        if not suffix.startswith("/"):
            suffix = f"/{suffix}"
        if not self._robot_namespace:
            return suffix
        return f"{self._robot_namespace}{suffix}"

    # ------------------------------------------------------------------
    # Callback implementations
    # ------------------------------------------------------------------
    def _target_pose_callback(self, msg: PoseStamped) -> None:
        self.get_logger().debug("Received new target pose")
        self.latest_target_pose = msg

    def _robot_pose_callback(self, msg: PoseStamped) -> None:
        self.get_logger().debug("Robot pose update received")
        self.latest_robot_pose = msg

    def _robot_formation_callback(self, msg: object) -> None:
        self.get_logger().debug("Formation update received")
        self.latest_formation = self._extract_formation_state(msg)
        if self.latest_formation is None:
            self.get_logger().warning(
                "Formation message missing expected fields; ignoring update of type %s",
                type(msg).__name__,
            )
        else:
            self.get_logger().debug(
                "Tracking formation '%s' (theta=%.3f rad, cost=%.3f)",
                self.latest_formation.formation_id,
                self.latest_formation.theta,
                self.latest_formation.cost,
            )

    def _robot_odometry_callback(self, msg: Odometry) -> None:
        self.get_logger().debug("Odometry update received")
        self.latest_odometry = msg

    def _robot_pointcloud_callback(self, msg: PointCloud2) -> None:
        self.get_logger().debug("Pointcloud update received")
        self.latest_pointcloud = msg

    def _coordination_timer_callback(self) -> None:
        if self._policy_controller is None:
            self.get_logger().debug("Policy controller is not initialised")
            return

        required_missing = []
        if self.latest_target_pose is None:
            required_missing.append("target pose")
        if self.latest_robot_pose is None:
            required_missing.append("robot pose")
        if self.latest_pointcloud is None:
            required_missing.append("pointcloud")

        if required_missing:
            self.get_logger().debug(
                "Coordination timer waiting for %s",
                ", ".join(required_missing),
            )
            return

        twist = self._policy_controller.compute_velocity_command(
            target_pose=self.latest_target_pose,
            robot_pose=self.latest_robot_pose,
            odometry=self.latest_odometry,
            pointcloud=self.latest_pointcloud,
        )

        if twist is None:
            if self._policy_controller.last_error:
                self.get_logger().warning(
                    "Policy evaluation failed: %s", self._policy_controller.last_error
                )
            else:
                self.get_logger().debug("Policy controller produced no command")
            return

        self.get_logger().debug(
            "Publishing policy cmd_vel (v=%.3f m/s, omega=%.3f rad/s)",
            twist.linear.x,
            twist.angular.z,
        )
        self._cmd_vel_publisher.publish(twist)

    def _extract_formation_state(self, msg: object) -> Optional[FormationState]:
        """Convert a raw ROS message into a `FormationState` container."""

        if msg is None:
            return None

        required_fields = ("formation_id", "formation_theta", "optimization_cost")
        missing_fields = [name for name in required_fields if not hasattr(msg, name)]
        if missing_fields:
            self.get_logger().debug(
                "Formation message is missing fields %s", ", ".join(missing_fields)
            )
            return None

        formation_id = cast(str, getattr(msg, "formation_id"))
        formation_theta = cast(float, getattr(msg, "formation_theta"))
        optimization_cost = cast(float, getattr(msg, "optimization_cost"))
        header = getattr(msg, "header", None)

        return FormationState(
            formation_id=formation_id,
            theta=float(formation_theta),
            cost=float(optimization_cost),
            header=header,
        )


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = CoordinatedNavigationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


__all__ = ["CoordinatedNavigationNode", "main"]
