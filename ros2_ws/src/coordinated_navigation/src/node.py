#!/usr/bin/env python3

"""Primary ROS 2 node for coordinating navigation of a single robot."""

from __future__ import annotations

import asyncio
import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Type, cast

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.qos import (
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from nav2_msgs.action import NavigateToPose

from coordinated_navigation_interfaces.msg import Formation as FormationMsg

try:
    from .policy_controller import (
        PolicyVelocityController,
        quaternion_to_yaw,
        wrap_to_pi,
    )
except:
    from coordinated_navigation.policy_controller import (
        PolicyVelocityController,
        quaternion_to_yaw,
        wrap_to_pi,
    )

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

        default_policy_model = self._default_policy_model_path()

        self.declare_parameter("robot_namespace", "")
        self.declare_parameter("robot_id", "1")
        self.declare_parameter("formation_msg_module", "coordinated_navigation_interfaces.msg")
        self.declare_parameter("formation_msg_name", "Formation")
        self.declare_parameter("policy.model_path", default_policy_model)
        self.declare_parameter("policy.step_period", timer_period)
        self.declare_parameter("policy.device", "cpu")
        self.declare_parameter("navigation.distance_tolerance", 0.3)
        self.declare_parameter("navigation.yaw_tolerance", 0.5)
        self.declare_parameter("navigation.feedback_period", 0.5)

        self._robot_namespace = self._normalise_robot_namespace(robot_namespace, robot_id)
        self._formation_msg_type = self._resolve_formation_msg_type(formation_msg_type)

        policy_model_path = self._parameter_string("policy.model_path")
        if not policy_model_path and default_policy_model:
            policy_model_path = default_policy_model
            self.get_logger().info(
                f"Using packaged DDPG policy model at {policy_model_path}"
            )
        elif not policy_model_path:
            self.get_logger().warn(
                "No policy model path has been configured; the controller will "
                "remain idle until one is provided."
            )
        policy_step_period = self._parameter_float("policy.step_period", timer_period)
        policy_device = self._parameter_string("policy.device") or "cpu"
        self._goal_distance_tolerance = self._parameter_float(
            "navigation.distance_tolerance", 0.3
        )
        self._goal_yaw_tolerance = self._parameter_float("navigation.yaw_tolerance", 0.5)
        self._action_feedback_period = max(
            self._parameter_float("navigation.feedback_period", 0.5), 0.05
        )

        self._policy_controller = PolicyVelocityController(
            model_path=policy_model_path,
            step_period=policy_step_period,
            node=self,
            device=policy_device,
        )

        self.latest_target_pose: Optional[PoseStamped] = None
        self.latest_robot_pose: Optional[PoseStamped] = None
        self.latest_formation: Optional[FormationState] = None
        self.latest_odometry: Optional[Odometry] = None
        self.latest_scan: Optional[LaserScan] = None
        self._active_goal_handle = None

        self._navigate_action_server = ActionServer(
            self,
            NavigateToPose,
            self._topic("navigate_to_pose"),
            execute_callback=self._execute_navigation,
            goal_callback=self._navigate_goal_callback,
            cancel_callback=self._navigate_cancel_callback,
        )

        pose_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)
        formation_qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.RELIABLE)
        odometry_qos = QoSProfile(depth=20, reliability=QoSReliabilityPolicy.RELIABLE)
        cmd_vel_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        self._amcl_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self._amcl_pose_callback,
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
        self._scan_subscription = self.create_subscription(
            LaserScan,
            self._topic("scan"),
            self._robot_scan_callback,
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

    def _default_policy_model_path(self) -> str:
        """Return the packaged default policy model if it is available."""

        share_candidate: Optional[Path] = None
        try:
            from ament_index_python.packages import get_package_share_directory

            share_dir = Path(get_package_share_directory("coordinated_navigation"))
            share_candidate = share_dir / "models" / "model_ep3700.zip"
        except Exception:  # pragma: no cover - environment-specific lookup failure
            share_candidate = None

        if share_candidate is not None and share_candidate.is_file():
            return str(share_candidate)

        source_candidate = Path(__file__).resolve().parents[1] / "models" / "model_ep3700.zip"
        if source_candidate.is_file():
            return str(source_candidate)

        return ""


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

        module_name = (
            self._parameter_string("formation_msg_module")
            or "coordinated_navigation_interfaces.msg"
        )
        type_name = self._parameter_string("formation_msg_name") or "Formation"

        if (
            module_name == "coordinated_navigation_interfaces.msg"
            and type_name == "Formation"
        ):
            return cast(Type[object], FormationMsg)

        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError(
                "Unable to import formation message module '{module}'. Ensure the package is installed and the node's parameters are correct.".format(
                    module=module_name,
                )
            ) from exc

        try:
            msg_type = getattr(module, type_name)
        except AttributeError as exc:
            raise RuntimeError(
                "Module '{module}' does not define a formation message named '{name}'. Verify the formation_msg_name parameter.".format(
                    module=module_name,
                    name=type_name,
                )
            ) from exc

        return cast(Type[object], msg_type)

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
    def _navigate_goal_callback(self, goal_request: NavigateToPose.Goal) -> GoalResponse:
        if self._active_goal_handle is not None and self._active_goal_handle.is_active:
            self.get_logger().warning(
                "Rejecting NavigateToPose goal while another goal is active"
            )
            return GoalResponse.REJECT

        self.get_logger().info(
            "NavigateToPose goal received (x=%.3f, y=%.3f)",
            goal_request.pose.pose.position.x,
            goal_request.pose.pose.position.y,
        )
        return GoalResponse.ACCEPT

    def _navigate_cancel_callback(self, goal_handle) -> CancelResponse:
        self.get_logger().info("NavigateToPose cancellation requested")
        return CancelResponse.ACCEPT

    async def _execute_navigation(self, goal_handle):
        goal_pose = goal_handle.request.pose
        self.get_logger().info(
            "NavigateToPose goal accepted (x=%.3f, y=%.3f)",
            goal_pose.pose.position.x,
            goal_pose.pose.position.y,
        )

        start_time = self.get_clock().now()
        self.latest_target_pose = goal_pose
        self._active_goal_handle = goal_handle

        try:
            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    self.get_logger().info("NavigateToPose goal cancelled")
                    goal_handle.canceled()
                    self.latest_target_pose = None
                    self._stop_robot()
                    result = self._populate_navigation_result(start_time, success=False)
                    return result

                if self._has_reached_goal(goal_pose):
                    self.get_logger().info("NavigateToPose goal reached")
                    goal_handle.succeed()
                    self.latest_target_pose = None
                    self._stop_robot()
                    result = self._populate_navigation_result(start_time, success=True)
                    return result

                self._publish_navigation_feedback(goal_handle, start_time)
                await asyncio.sleep(self._action_feedback_period)
        finally:
            if self._active_goal_handle is goal_handle:
                self._active_goal_handle = None

    def _amcl_pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self.get_logger().debug("AMCL pose update received")
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        self.latest_robot_pose = pose

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

    def _robot_scan_callback(self, msg: LaserScan) -> None:
        self.get_logger().debug("Laser scan update received")
        self.latest_scan = msg

    def _coordination_timer_callback(self) -> None:
        if self._policy_controller is None:
            self.get_logger().debug("Policy controller is not initialised")
            return

        required_missing = []
        if self.latest_target_pose is None:
            required_missing.append("target pose")
        if self.latest_robot_pose is None:
            required_missing.append("robot pose")
        if self.latest_scan is None:
            required_missing.append("laser scan")

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
            pointcloud=None,
            laser_scan=self.latest_scan,
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

    def _publish_navigation_feedback(self, goal_handle, start_time) -> None:
        if self.latest_robot_pose is None:
            return

        feedback = NavigateToPose.Feedback()
        if hasattr(feedback, "current_pose"):
            current_pose = PoseStamped()
            current_pose.header = self.latest_robot_pose.header
            current_pose.pose = self.latest_robot_pose.pose
            feedback.current_pose = current_pose

        duration = self.get_clock().now() - start_time
        if hasattr(feedback, "navigation_time"):
            navigation_time = getattr(feedback, "navigation_time")
            seconds = duration.nanoseconds / 1e9
            if hasattr(navigation_time, "sec") and hasattr(navigation_time, "nanosec"):
                feedback.navigation_time = duration.to_msg()
            else:
                feedback.navigation_time = float(seconds)
        if hasattr(feedback, "number_of_recoveries"):
            feedback.number_of_recoveries = 0

        goal_handle.publish_feedback(feedback)

    def _populate_navigation_result(self, start_time, *, success: bool) -> NavigateToPose.Result:
        duration = self.get_clock().now() - start_time
        result = NavigateToPose.Result()
        if hasattr(result, "total_time"):
            result.total_time = duration.to_msg()
        if hasattr(result, "success"):
            result.success = bool(success)
        if hasattr(result, "result_code"):
            if success:
                result_code = getattr(NavigateToPose.Result, "RESULT_SUCCEEDED", 0)
            else:
                result_code = getattr(NavigateToPose.Result, "RESULT_CANCELED", 0)
            result.result_code = int(result_code)
        return result

    def _stop_robot(self) -> None:
        twist = Twist()
        self._cmd_vel_publisher.publish(twist)

    def _has_reached_goal(self, goal_pose: PoseStamped) -> bool:
        if self.latest_robot_pose is None:
            return False

        robot_position = self.latest_robot_pose.pose.position
        goal_position = goal_pose.pose.position
        dx = float(goal_position.x - robot_position.x)
        dy = float(goal_position.y - robot_position.y)
        distance = math.hypot(dx, dy)
        if distance > self._goal_distance_tolerance:
            return False

        robot_yaw = quaternion_to_yaw(self.latest_robot_pose.pose.orientation)
        goal_yaw = quaternion_to_yaw(goal_pose.pose.orientation)
        yaw_error = wrap_to_pi(goal_yaw - robot_yaw)
        return abs(yaw_error) <= self._goal_yaw_tolerance

    def destroy_node(self) -> bool:
        if hasattr(self, "_navigate_action_server") and self._navigate_action_server is not None:
            self._navigate_action_server.destroy()
        return super().destroy_node()

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
