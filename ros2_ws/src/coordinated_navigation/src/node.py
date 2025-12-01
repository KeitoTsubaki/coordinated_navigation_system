#!/usr/bin/env python3

"""Primary ROS 2 node for coordinating navigation of a single robot."""

from __future__ import annotations

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
from rclpy.parameter import ParameterType
from rclpy.task import Future
from rclpy.time import Time
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Quaternion, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

from nav2_msgs.action import NavigateToPose

from coordinated_navigation_interfaces.msg import Formation as FormationMsg

from coordinated_navigation.policy_controller import (
    PolicyVelocityController,
    quaternion_to_yaw,
    wrap_to_pi,
)

# from coordinated_navigation.policy_controller_3 import (
#     PolicyVelocityController,
#     quaternion_to_yaw,
#     wrap_to_pi,
# )

# try:
#     # from .policy_controller import (
#     #     PolicyVelocityController,
#     #     quaternion_to_yaw,
#     #     wrap_to_pi,
#     # )
#     # from .policy_controller_2 import (
#     #     PolicyVelocityController,
#     #     quaternion_to_yaw,
#     #     wrap_to_pi,
#     # )
#     from .policy_controller_3 import (
#         PolicyVelocityController,
#         quaternion_to_yaw,
#         wrap_to_pi,
#     )
# except:
    # from coordinated_navigation.policy_controller import (
    #     PolicyVelocityController,
    #     quaternion_to_yaw,
    #     wrap_to_pi,
    # )
    # from coordinated_navigation.policy_controller_2 import (
    #     PolicyVelocityController,
    #     quaternion_to_yaw,
    #     wrap_to_pi,
    # )

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
        timer_period: float = 0.05,
    ) -> None:
        super().__init__(node_name)

        default_policy_model = self._default_policy_model_path()

        self.declare_parameter("robot_namespace", "robot")
        self.declare_parameter("robot_id", "1")
        self.declare_parameter("formation_msg_module", "coordinated_navigation_interfaces.msg")
        self.declare_parameter("formation_msg_name", "Formation")
        self.declare_parameter("policy.model_path", default_policy_model)
        self.declare_parameter("policy.step_period", timer_period)
        self.declare_parameter("policy.device", "cpu")
        self.declare_parameter("navigation.distance_tolerance", 1)
        self.declare_parameter("navigation.yaw_tolerance", 1.5)
        self.declare_parameter("navigation.feedback_period", 0.1)

        # self._robot_namespace = self._normalise_robot_namespace(robot_namespace, robot_id)
        self._robot_namespace = robot_namespace
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

        self._callback_group = ReentrantCallbackGroup()

        self.latest_target_pose: Optional[PoseStamped] = None
        self.latest_robot_pose: Optional[PoseStamped] = None
        self._pose_velocity_odometry: Optional[Odometry] = None
        self._last_pose_stamp: Optional[Time] = None
        self.latest_formation: Optional[FormationState] = None
        self.latest_odometry: Optional[Odometry] = None
        self.latest_scan: Optional[LaserScan] = None
        self._active_goal_handle = None

        self._navigate_action_server = ActionServer(
            self,
            NavigateToPose,
            # self._topic("navigate_to_pose"),
            "navigate_to_pose",
            execute_callback=self._execute_navigation,
            goal_callback=self._navigate_goal_callback,
            cancel_callback=self._navigate_cancel_callback,
            callback_group=self._callback_group
        )

        pose_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE)
        formation_qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.RELIABLE)
        odometry_qos = QoSProfile(depth=20, reliability=QoSReliabilityPolicy.RELIABLE)
        cmd_vel_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        self._amcl_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            # self._topic("amcl_pose"),
            "pose",
            self._pose_callback,
            pose_qos,
        )
        self._formation_subscription = self.create_subscription(
            self._formation_msg_type,
            # self._topic("formation"),
            "formation",
            self._robot_formation_callback,
            formation_qos,
        )
        self._odometry_subscription = self.create_subscription(
            Odometry,
            # self._topic("odom"),
            "odom",
            self._robot_odometry_callback,
            odometry_qos,
        )
        self._scan_subscription = self.create_subscription(
            LaserScan,
            # self._topic("scan"),
            "scan",
            self._robot_scan_callback,
            qos_profile_sensor_data,
        )
        self._cmd_vel_publisher = self.create_publisher(
            Twist,
            # self._topic("cmd_vel"),
            "cmd_vel",
            cmd_vel_qos,
        )
        self._marker_publisher = self.create_publisher(
            MarkerArray,
            # self._topic("navigation_markers"),
            "navigation_markers",
            10
        )

        self.latest_cmd_vel: Optional[Twist] = None

        self._coordination_timer = self.create_timer(
            max(policy_step_period, 1e-3), 
            self._coordination_timer_callback,
            callback_group=self._callback_group
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
        parameter_type = getattr(
            parameter_value, "type", getattr(parameter_value, "type_", None)
        )

        if parameter_type == ParameterType.PARAMETER_DOUBLE:
            return float(parameter_value.double_value)
        if parameter_type == ParameterType.PARAMETER_INTEGER:
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

        goal_position = goal_request.pose.pose.position
        self.get_logger().info(
            f"NavigateToPose goal received (x={goal_position.x:.3f}, y={goal_position.y:.3f})"
        )
        return GoalResponse.ACCEPT

    def _navigate_cancel_callback(self, goal_handle) -> CancelResponse:
        self.get_logger().info("NavigateToPose cancellation requested")
        return CancelResponse.ACCEPT

    async def _execute_navigation(self, goal_handle):
        goal_pose = goal_handle.request.pose
        goal_position = goal_pose.pose.position
        self.get_logger().info(
            f"NavigateToPose goal accepted (x={goal_position.x:.3f}, y={goal_position.y:.3f})"
        )

        start_time = self.get_clock().now()
        self.latest_target_pose = goal_pose
        self._active_goal_handle = goal_handle

        self._navigation_success = False

        try:
            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    self.get_logger().info("NavigateToPose goal cancelled")
                    self.latest_target_pose = None
                    self._stop_robot()
                    goal_handle.canceled()
                    result = self._populate_navigation_result(start_time, success=False)
                    return result

                if self.latest_target_pose is None:
                    break

                self._publish_navigation_feedback(goal_handle, start_time)
                await self._wait_for(self._action_feedback_period)

            if self._navigation_success:
                self.get_logger().info("NavigateToPose goal reached")
                goal_handle.succeed()
                return self._populate_navigation_result(start_time, success=True)
            else:
                goal_handle.abort()
                return self._populate_navigation_result(start_time, success=False)

        finally:
            if self._active_goal_handle is goal_handle:
                self._active_goal_handle = None

    async def _wait_for(self, duration: float) -> None:
        """Suspend execution of an async callback without relying on asyncio."""

        if duration <= 0:
            return

        completion = Future()

        def timer_callback() -> None:
            if not completion.done():
                completion.set_result(True)

        timer = self.create_timer(duration, timer_callback)
        try:
            await completion
        finally:
            timer.cancel()

    def _pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self.get_logger().debug("AMCL pose update received")
        previous_pose = self.latest_robot_pose
        previous_stamp = self._last_pose_stamp
        current_stamp = Time.from_msg(msg.header.stamp)

        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        self.latest_robot_pose = pose

        if previous_pose is not None and previous_stamp is not None:
            dt = (current_stamp - previous_stamp).nanoseconds / 1e9
            if dt > 0.0:
                dx = float(pose.pose.position.x - previous_pose.pose.position.x)
                dy = float(pose.pose.position.y - previous_pose.pose.position.y)
                linear_speed = math.hypot(dx, dy) / dt

                yaw_now = quaternion_to_yaw(pose.pose.orientation)
                yaw_prev = quaternion_to_yaw(previous_pose.pose.orientation)
                angular_speed = wrap_to_pi(yaw_now - yaw_prev) / dt

                odom = Odometry()
                odom.header = msg.header
                odom.twist.twist.linear.x = float(linear_speed)
                odom.twist.twist.angular.z = float(angular_speed)
                self._pose_velocity_odometry = odom

        self._last_pose_stamp = current_stamp

    def _robot_formation_callback(self, msg: object) -> None:
        self.get_logger().debug("Formation update received")
        self.latest_formation = self._extract_formation_state(msg)
        if self.latest_formation is None:
            self.get_logger().warning(
                (
                    "Formation message missing expected fields; ignoring update of type "
                    f"{type(msg).__name__}"
                )
            )
        else:
            self.get_logger().debug(
                (
                    f"Tracking formation '{self.latest_formation.formation_id}' "
                    f"(theta={self.latest_formation.theta:.3f} rad, "
                    f"cost={self.latest_formation.cost:.3f})"
                )
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
            missing_fields = ", ".join(required_missing)
            self.get_logger().debug(
                f"Coordination timer waiting for {missing_fields}"
            )
            return

        self._publish_visualization_markers()

        twist = self._policy_controller.compute_velocity_command(
            target_pose=self.latest_target_pose,
            robot_pose=self.latest_robot_pose,
            # odometry=self.latest_odometry,
            odometry=self._pose_velocity_odometry,
            pointcloud=None,
            laser_scan=self.latest_scan,
        )

        if twist is None:
            if self._policy_controller.last_error:
                self.get_logger().warning(
                    f"Policy evaluation failed: {self._policy_controller.last_error}"
                )
            else:
                self.get_logger().debug("Policy controller produced no command")
            return

        self.get_logger().debug(
            (
                f"Publishing policy cmd_vel (v={twist.linear.x:.3f} m/s, "
                f"omega={twist.angular.z:.3f} rad/s)"
            )
        )
        self._cmd_vel_publisher.publish(twist)
        self.latest_cmd_vel = twist

        if self._has_reached_goal(self.latest_target_pose):
            self._stop_robot()
            self._navigation_success = True
            self.latest_target_pose = None


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
        self.latest_cmd_vel = twist

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

    def _publish_visualization_markers(self) -> None:
        if self.latest_robot_pose is None or self.latest_scan is None:
            return

        frame_id = self.latest_robot_pose.header.frame_id or "map"
        stamp = self.get_clock().now().to_msg()
        robot_position = self.latest_robot_pose.pose.position
        robot_yaw = quaternion_to_yaw(self.latest_robot_pose.pose.orientation)

        markers: list[Marker] = []
        if self.latest_target_pose is not None:
            markers.append(
                self._build_arrow_marker(
                    marker_id=0,
                    frame_id=frame_id,
                    stamp=stamp,
                    start_position=robot_position,
                    yaw=self._goal_heading(),
                    length=self._goal_distance(),
                    color=(0.0, 1.0, 0.0, 0.8),
                )
            )

        obstacle_marker = self._closest_obstacle_marker(
            robot_position=robot_position,
            robot_yaw=robot_yaw,
            frame_id=frame_id,
            stamp=stamp,
        )
        if obstacle_marker is not None:
            markers.append(obstacle_marker)

        if self.latest_cmd_vel is not None:
            markers.append(
                self._build_cmd_vel_text_marker(
                    marker_id=2,
                    frame_id=frame_id,
                    stamp=stamp,
                    robot_position=robot_position,
                    twist=self.latest_cmd_vel,
                )
            )

        if not markers:
            return

        marker_array = MarkerArray(markers=markers)
        self._marker_publisher.publish(marker_array)

    def _goal_distance(self) -> float:
        if self.latest_target_pose is None or self.latest_robot_pose is None:
            return 0.0

        goal_position = self.latest_target_pose.pose.position
        robot_position = self.latest_robot_pose.pose.position
        dx = float(goal_position.x - robot_position.x)
        dy = float(goal_position.y - robot_position.y)
        return math.hypot(dx, dy)

    def _goal_heading(self) -> float:
        if self.latest_target_pose is None or self.latest_robot_pose is None:
            return 0.0

        goal_position = self.latest_target_pose.pose.position
        robot_position = self.latest_robot_pose.pose.position
        dx = float(goal_position.x - robot_position.x)
        dy = float(goal_position.y - robot_position.y)
        return math.atan2(dy, dx)

    def _closest_obstacle_marker(
        self,
        *,
        robot_position,
        robot_yaw: float,
        frame_id: str,
        stamp,
    ) -> Optional[Marker]:
        if self.latest_scan is None:
            return None

        scan = self.latest_scan
        min_distance = None
        min_angle = None
        for index, distance in enumerate(scan.ranges):
            if not math.isfinite(distance):
                continue
            if min_distance is None or distance < min_distance:
                min_distance = distance
                min_angle = scan.angle_min + index * scan.angle_increment

        if min_distance is None or min_angle is None:
            return None

        obstacle_heading = robot_yaw + min_angle

        return self._build_arrow_marker(
            marker_id=1,
            frame_id=frame_id,
            stamp=stamp,
            start_position=robot_position,
            yaw=obstacle_heading,
            length=min_distance,
            color=(1.0, 0.0, 0.0, 0.8),
        )

    @staticmethod
    def _build_cmd_vel_text_marker(
        *,
        marker_id: int,
        frame_id: str,
        stamp,
        robot_position,
        twist: Twist,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "navigation_cmd_vel"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position = Point(
            x=robot_position.x,
            y=robot_position.y - 2.0,
            z=robot_position.z + 0.5,
        )
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.3
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.9
        marker.text = (
            f"cmd_vel\n"
            f"lin.x: {twist.linear.x:.2f} m/s\n"
            f"ang.z: {twist.angular.z:.2f} rad/s"
        )
        return marker

    @staticmethod
    def _build_arrow_marker(
        *,
        marker_id: int,
        frame_id: str,
        stamp,
        start_position,
        yaw: float,
        length: float,
        color: tuple[float, float, float, float],
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "navigation_vectors"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position = start_position
        marker.pose.orientation = CoordinatedNavigationNode._yaw_to_quaternion(yaw)
        marker.scale.x = max(length, 0.01)
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        return marker

    @staticmethod
    def _yaw_to_quaternion(yaw: float) -> Quaternion:
        half_yaw = yaw * 0.5
        qz = math.sin(half_yaw)
        qw = math.cos(half_yaw)

        return Quaternion(x=0.0, y=0.0, z=qz, w=qw)

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
            missing_fields_str = ", ".join(missing_fields)
            self.get_logger().debug(
                f"Formation message is missing fields {missing_fields_str}"
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

    executer = MultiThreadedExecutor()
    executer.add_node(node)
    try:
        executer.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


__all__ = ["CoordinatedNavigationNode", "main"]

if __name__ == "__main__":
    main()