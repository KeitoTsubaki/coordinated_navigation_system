#!/usr/bin/env python3

"""Primary ROS 2 node for coordinating navigation of a single robot."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from nav_msgs.msg import Path as NavPath
from typing import Optional, Sequence, Type, cast

import numpy as np
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

from tf2_ros import Buffer, TransformListener, TransformException
from geometry_msgs.msg import TransformStamped
import tf2_geometry_msgs # これも必要です


# from coordinated_navigation.policy_controller import (
#     PolicyVelocityController,
#     quaternion_to_yaw,
#     wrap_to_pi,
# )

from coordinated_navigation.policy_controller_3 import (
    PolicyVelocityController,
    quaternion_to_yaw,
    wrap_to_pi,
)

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

try:
    from path_msgs.srv import GetPath
except ImportError:
    raise RuntimeError("path_msgs package not found. Please build 'ros2_vwastar' first.")



class CoordinatedNavigationNode(Node):
    """ROS 2 node that wires up navigation interfaces for a single robot instance."""

    def __init__(
        self,
        node_name: str = "coordinated_navigation",
        *,
        robot_namespace: Optional[str] = None,
        robot_id: Optional[str] = None,
        timer_period: float = 0.1,
    ) -> None:
        super().__init__(node_name)

        default_policy_model = self._default_policy_model_path()

        self.declare_parameter("robot_namespace", "robot")
        self.declare_parameter("robot_id", "1")
        self.declare_parameter("partner_name", "whill2")
        self.declare_parameter("role", "leader")  # 'leader' or 'follower'
        self.robot_id = self.get_parameter("robot_id").value
        self.role = self.get_parameter("role").value
        self.partner_name = self.get_parameter("partner_name").value
        
        self.declare_parameter("policy.model_path", default_policy_model)
        self.declare_parameter("policy.step_period", timer_period)
        self.declare_parameter("policy.device", "cpu")
        self.declare_parameter("navigation.distance_tolerance", 1)
        self.declare_parameter("navigation.yaw_tolerance", 1.5)
        self.declare_parameter("navigation.feedback_period", 0.1)


        # self._robot_namespace = self._normalise_robot_namespace(robot_namespace, robot_id)
        # パラメータから正しく名前空間を読み込む
        if robot_namespace is None:
             self._robot_namespace = self.get_parameter("robot_namespace").value
        else:
             self._robot_namespace = robot_namespace

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
            "navigation.distance_tolerance", 1.0
        )
        self._goal_yaw_tolerance = self._parameter_float("navigation.yaw_tolerance", 3.14)
        self._action_feedback_period = max(
            self._parameter_float("navigation.feedback_period", 0.5), 0.05
        )

        self._policy_controller = PolicyVelocityController(
            model_path=policy_model_path,
            step_period=policy_step_period,
            node=self,
            device=policy_device,
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._callback_group = ReentrantCallbackGroup()

        self.latest_target_pose: Optional[PoseStamped] = None
        self.latest_robot_pose: Optional[PoseStamped] = None
        self._pose_velocity_odometry: Optional[Odometry] = None
        self._last_pose_stamp: Optional[Time] = None
        self.latest_odometry: Optional[Odometry] = None
        self.latest_scan: Optional[LaserScan] = None

        self._prev_clean_scan: Optional[LaserScan] = None

        # パス追従用変数
        self._current_global_path_msg: Optional[Path] = None # nav_msgs/Path
        self._lookahead_distance = 2.0 # [m] サブゴールまでの距離
        self._subgoal_pose: Optional[PoseStamped] = None

        # self._active_goal_handle = None
        # self._navigate_action_server = ActionServer(
        #     self,
        #     NavigateToPose,
        #     # self._topic("navigate_to_pose"),
        #     "navigate_to_pose",
        #     execute_callback=self._execute_navigation,
        #     goal_callback=self._navigate_goal_callback,
        #     cancel_callback=self._navigate_cancel_callback,
        #     callback_group=self._callback_group
        # )
        self._planner_client = self.create_client(
            GetPath,
            '/vwastar/get_path',
            callback_group=self._callback_group
        )

        pose_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE)
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
        self._path_publisher = self.create_publisher(NavPath, "following_path", 10)

        self.latest_cmd_vel: Optional[Twist] = None

        self._coordination_timer = self.create_timer(
            max(policy_step_period, 1e-3), 
            self._coordination_timer_callback,
            callback_group=self._callback_group
        )

        # 隊形パラメータ
        self.declare_parameter("navigation.lookahead_distance", 1.0)
        self.declare_parameter("formation.width", 1.5)          
        self.declare_parameter("formation.offset_mode", "auto") 

        self._lookahead_distance = self.get_parameter("navigation.lookahead_distance").value
        self._formation_width = self.get_parameter("formation.width").value
        
        # IDに基づいてオフセットを決定
        self._lateral_offset = 0.0
        mode = self._parameter_string("formation.offset_mode")
        robot_id = self._read_robot_id_parameter()
        
        if mode == "auto":
            if robot_id == "1":
                self._lateral_offset = self._formation_width / 2.0  # Robot 1: Left (+)
            elif robot_id == "2":
                self._lateral_offset = -self._formation_width / 2.0 # Robot 2: Right (-)
        elif mode == "left":
            self._lateral_offset = self._formation_width / 2.0
        elif mode == "right":
            self._lateral_offset = -self._formation_width / 2.0

        # ▼▼▼ 追加: 設定確認用ログ ▼▼▼
        self.get_logger().info(f"--- Robot Config: ID={robot_id} ---")
        self.get_logger().info(f"    Formation Width: {self._formation_width}m")
        self.get_logger().info(f"    My Offset: {self._lateral_offset}m")
        self.get_logger().info(f"-----------------------------------")

        # 変数の初期化
        self.latest_global_goal: Optional[PoseStamped] = None
        self.latest_subgoal: Optional[PoseStamped] = None
        self.latest_path: Optional[NavPath] = None

        self._active_goal_handle = None

        # ▼▼▼ 修正: トピック名をVWA*の出力に合わせる ▼▼▼
        # launchファイルでのリマッピング漏れを防ぐため、デフォルト値を合わせます
        # self._path_subscription = self.create_subscription(
        #     NavPath, 
        #     "/vwastar/nav_path",  # 変更: "plan" -> "/vwastar/nav_path"
        #     self._path_callback, 
        #     10
        # )

        # ==========================================================
        # ROLE SPECIFIC SETUP
        # ==========================================================
        if self.role == "leader":
            self.get_logger().info(f"Initialized as LEADER (ID={self.robot_id})")
            
            # Leader: Global Planner & Action Server
            self._planner_client = self.create_client(GetPath, '/vwastar/get_path', callback_group=self._callback_group)
            self._navigate_action_server = ActionServer(
                self, NavigateToPose, "navigate_to_pose",
                execute_callback=self._execute_leader_navigation,
                goal_callback=self._navigate_goal_callback,      # 修正: _navigate_goal_callback に変更
                cancel_callback=self._navigate_cancel_callback,  # 修正: _navigate_cancel_callback に変更
                callback_group=self._callback_group
            )

            
            
            # Leader: Publish commands to Follower
            # Topic: /<partner_name>/leader_command_pose
            topic_name = f"/{self.partner_name}/leader_command_pose"
            self._follower_cmd_pub = self.create_publisher(PoseStamped, topic_name, 10)
            
            # Leader: Subscribe to Global Path
            self._path_subscription = self.create_subscription(NavPath, "/vwastar/nav_path", self._path_callback, 10)

        else: # follower
            self.get_logger().info(f"Initialized as FOLLOWER (ID={self.robot_id})")
            
            # Follower: Listen to Leader's commands
            topic_name = f"leader_command_pose" # relative, so it becomes /whill2/leader_command_pose
            self.create_subscription(PoseStamped, topic_name, self._follower_command_callback, 10)

    # ==========================================================
    # Leader Logic: Compute Formation Targets
    # ==========================================================
    # ==========================================================
    # Leader Logic: Compute Formation Targets
    # ==========================================================
    def _calc_formation_targets(self, robot_pose: PoseStamped, path: NavPath):
        """
        Leader calculates targets for SELF and PARTNER based on the path.
        Assumes Leader is Left/Front, Follower is Right/Back (or vice versa).
        """
        if not path.poses: return None, None

        # 1. Find point on path for Leader (Lookahead)
        rx, ry = robot_pose.pose.position.x, robot_pose.pose.position.y
        
        # Find closest index
        closest_idx = 0
        min_dist = float('inf')
        for i, p in enumerate(path.poses):
            d = math.hypot(p.pose.position.x - rx, p.pose.position.y - ry)
            if d < min_dist:
                min_dist = d
                closest_idx = i
        
        # Find lookahead index
        target_idx = len(path.poses) - 1
        accum = 0.0
        for i in range(closest_idx, len(path.poses)-1):
            d = math.hypot(path.poses[i+1].pose.position.x - rx,
                           path.poses[i+1].pose.position.y - ry)
            if d > self._lookahead_distance:
                target_idx = i
                break
        
        center_pt = path.poses[target_idx]
        yaw = quaternion_to_yaw(center_pt.pose.orientation)
        
        # 2. Check Formation ID
        formation_id = int(center_pt.pose.position.z)
        is_narrow = (formation_id == 1) # 1 = Narrow, 0 = Wide
        
        # 3. Calculate Offsets & Generate Poses
        # Config: Leader (ID1) = Left/Front (+), Follower (ID2) = Right/Back (-)
        half_width = self._formation_width / 2.0

        def make_target(offset, is_longitudinal):
            t = PoseStamped()
            t.header = center_pt.header
            
            if is_longitudinal:
                # --- Longitudinal (Narrow/Column) ---
                # 進行方向 (cos, sin) に沿ってオフセット
                ox = offset * math.cos(yaw)
                oy = offset * math.sin(yaw)
            else:
                # --- Lateral (Wide/Line) ---
                # 進行方向に対して垂直 (-sin, cos) にオフセット
                ox = -offset * math.sin(yaw)
                oy =  offset * math.cos(yaw)
            
            t.pose.position.x = center_pt.pose.position.x + ox
            t.pose.position.y = center_pt.pose.position.y + oy
            t.pose.position.z = center_pt.pose.position.z
            t.pose.orientation = center_pt.pose.orientation
            return t

        if is_narrow:
            # --- Column Formation (縦列) ---
            # 進行方向に展開 (Longitudinal)
            # Leader: 前 (+), Follower: 後 (-)
            target_leader = make_target(half_width, is_longitudinal=True)
            target_follower = make_target(-half_width, is_longitudinal=True)
            
        else:
            # --- Line Formation (横並び) ---
            # 横方向に展開 (Lateral)
            # Leader: 左 (+), Follower: 右 (-)
            target_leader = make_target(half_width, is_longitudinal=False)
            target_follower = make_target(-half_width, is_longitudinal=False)

        return target_leader, target_follower



    # ==========================================================
    # Action Server (Leader Only)
    # ==========================================================
    async def _execute_leader_navigation(self, goal_handle):
        self.get_logger().info("Leader received Navigation Goal...")
        goal_pose = goal_handle.request.pose
        self.latest_target_pose = goal_pose

        if self.latest_robot_pose is None:
            self.latest_robot_pose = self._get_latest_pose_from_tf()

        self.latest_path = None
        
        # Call Planner
        if not self._planner_client.wait_for_service(3.0):
            goal_handle.abort()
            return NavigateToPose.Result(success=False)
            
        req = GetPath.Request()
        req.init_pose = self.latest_robot_pose
        req.goal_pose = goal_pose
        
        future = self._planner_client.call_async(req)
        result = await future
        
        if not result.path.poses:
            self.get_logger().error("Planning failed")
            goal_handle.abort()
            return NavigateToPose.Result(success=False)
            
        self.latest_path = self._densify_path(result.path, 0.1)
        self._current_global_path_msg = result.path
        self._path_publisher.publish(result.path)
        self.get_logger().info("Path planned. Moving...")
        
        # Keep alive until goal reached
        feedback = NavigateToPose.Feedback()
        success = False
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self._stop_robot()
                self.latest_path = None
                self.latest_global_goal = None
                return NavigateToPose.Result(success=False)
            
            if self.latest_robot_pose:
                dist = math.hypot(
                    goal_pose.pose.position.x - self.latest_robot_pose.pose.position.x,
                    goal_pose.pose.position.y - self.latest_robot_pose.pose.position.y
                )
                if dist < self._goal_distance_tolerance:
                    success = True
                    break
                feedback.current_pose = self.latest_robot_pose
                goal_handle.publish_feedback(feedback)
            
            await self._wait_for(0.5)

        self._stop_robot()
        self.latest_path = None
        self.latest_global_goal = None
        if success:
            goal_handle.succeed()
            return NavigateToPose.Result(success=True)
        else:
            goal_handle.abort()
            return NavigateToPose.Result(success=False)

    # ==========================================================
    # Utilities
    # ==========================================================
    def _densify_path(self, path: NavPath, interval: float) -> NavPath:
        # (Same implementation as before)
        if not path.poses or len(path.poses) < 2: return path
        new_path = NavPath()
        new_path.header = path.header
        new_poses = [path.poses[0]]
        for i in range(len(path.poses) - 1):
            p1, p2 = path.poses[i], path.poses[i+1]
            dx, dy = p2.pose.position.x - p1.pose.position.x, p2.pose.position.y - p1.pose.position.y
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                steps = int(dist / interval)
                for j in range(1, steps + 1):
                    r = j * interval / dist
                    np = PoseStamped()
                    np.header = p1.header
                    np.pose.position.x = p1.pose.position.x + dx * r
                    np.pose.position.y = p1.pose.position.y + dy * r
                    np.pose.position.z = p1.pose.position.z # Keep formation ID
                    np.pose.orientation = p1.pose.orientation
                    new_poses.append(np)
            new_poses.append(p2)
        new_path.poses = new_poses
        return new_path

    # 修正版: タイムアウト設定とエラーログの有効化
    def _get_latest_pose_from_tf(self) -> Optional[PoseStamped]:
        """TFを使って最新のロボット姿勢(map座標系)を取得する"""
        try:
            target_frame = "base_link" 
            if self._robot_namespace:
                 ns = self._robot_namespace.strip('/')
                 if ns:
                     target_frame = f"{ns}/base_link"

            # ▼▼▼ 修正1: タイムアウトを設定 (0.05秒待つ) ▼▼▼
            # これがないと、わずかなタイミングのズレで即失敗します
            timeout = rclpy.duration.Duration(seconds=0.05)

            # 最新の変換を取得 (Time() は最新を意味する)
            transform: TransformStamped = self._tf_buffer.lookup_transform(
                "map",
                target_frame,
                rclpy.time.Time(),
                timeout # ← 引数に追加
            )
            
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation

            self.latest_robot_pose = pose
            
            return pose

        except TransformException as ex:
            # ▼▼▼ 修正2: ログを警告レベル(warn)にして表示する ▼▼▼
            # 頻繁に出る場合は頻度を落とす(throttle)必要がありますが、まずは原因特定のため表示します
            self.get_logger().warn(f"TF Error (map -> {target_frame}): {ex}")
            return None

    def _path_callback(self, msg: NavPath) -> None:
        if not msg.poses:
            return
        # 受信したパスを0.1m刻みに補間して保存
        self.latest_path = self._densify_path(msg, interval=0.1)
    
    # ▼▼▼ 追加: フォロワー用コールバック ▼▼▼
    def _follower_command_callback(self, msg: PoseStamped):
        # リーダーから送られてきた目標位置を自分のサブゴールとして保存
        self.latest_subgoal = msg
        # デバッグログ（必要なら）
        # self.get_logger().debug("Follower received target pose from Leader")
    # ▲▲▲ 追加ここまで ▲▲▲



    def _get_subgoal_from_path(self, robot_pose: PoseStamped, path: NavPath) -> Optional[PoseStamped]:
        if not path.poses:
            return None

        # --- 1. パス上でロボットに最も近い点を探す ---
        rx = robot_pose.pose.position.x
        ry = robot_pose.pose.position.y
        min_dist_sq = float('inf')
        closest_idx = 0
        
        # 全探索
        for i, pose_stamped in enumerate(path.poses):
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y
            d_sq = (px - rx)**2 + (py - ry)**2
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                closest_idx = i

        # --- 2. 【修正】ロボットからの直線距離でターゲットを探す ---
        # これにより、ロボットが動けば即座にターゲットも更新されます
        
        target_point = Point()
        target_idx = len(path.poses) - 1
        formation_id = 0
        found_target = False
        
        # 最も近い点から前方に探索を開始
        for i in range(closest_idx, len(path.poses)):
            p = path.poses[i].pose.position
            # ロボットからの直線距離を計算
            dist_from_robot = math.hypot(p.x - rx, p.y - ry)
            
            # Lookahead距離を超えた最初の点を見つける
            if dist_from_robot > self._lookahead_distance:
                # この点(i)と一つ手前の点(i-1)の間にターゲットがある
                if i == 0:
                    # 始点がいきなり遠い場合（通常ありえないが念のため）
                    target_point = p
                    formation_id = int(p.z)
                    target_idx = 0
                else:
                    prev_p = path.poses[i-1].pose.position
                    
                    dist_prev = math.hypot(prev_p.x - rx, prev_p.y - ry)
                    dist_curr = dist_from_robot
                    
                    # 線形補間: ロボットからの距離に基づいて比率を計算
                    # 厳密な円と直線の交点計算は重いため、区間が短ければこれで十分滑らかになります
                    denom = dist_curr - dist_prev
                    ratio = (self._lookahead_distance - dist_prev) / denom if denom > 1e-6 else 0.0
                    
                    target_point.x = prev_p.x + (p.x - prev_p.x) * ratio
                    target_point.y = prev_p.y + (p.y - prev_p.y) * ratio
                    target_point.z = prev_p.z # IDはセグメントの始点に従う
                    
                    formation_id = int(prev_p.z)
                    target_idx = i # 接線計算用に現在のインデックスを保存
                    
                found_target = True
                break
        
        # 見つからない（パスの残り距離が短い）場合は終点を採用
        if not found_target:
            end_pose = path.poses[-1]
            target_point = end_pose.pose.position
            formation_id = int(target_point.z)
            target_idx = len(path.poses) - 1

        # --- 3. 接線ベクトル (Tangent) の計算 ---
        # ターゲットが存在するセグメントの向きを使用
        tangent_x = 1.0
        tangent_y = 0.0
        
        # target_idx は "Lookahead距離を超えた点" なので、
        # 接線はその点に向かうベクトル (prev -> curr) を使うのが適切
        idx_b = target_idx
        idx_a = max(0, idx_b - 1)
        
        if idx_a < idx_b:
            p_a = path.poses[idx_a].pose.position
            p_b = path.poses[idx_b].pose.position
            dx = p_b.x - p_a.x
            dy = p_b.y - p_a.y
            norm = math.hypot(dx, dy)
            if norm > 1e-6:
                tangent_x = dx / norm
                tangent_y = dy / norm
            else:
                # 点が重なっている場合はOrientationから復元
                yaw = quaternion_to_yaw(path.poses[idx_b].pose.orientation)
                tangent_x = math.cos(yaw)
                tangent_y = math.sin(yaw)
        else:
             yaw = quaternion_to_yaw(path.poses[-1].pose.orientation)
             tangent_x = math.cos(yaw)
             tangent_y = math.sin(yaw)

        # --- 4. フォーメーション展開 (Wide vs Narrow) ---
        dir_x = 0.0
        dir_y = 0.0
        spacing_factor = 1.0 

        if formation_id == 0:
            # === Wide (Line) ===
            # 接線に対して垂直 ([-y, x])
            dir_x = -tangent_y
            dir_y = tangent_x
            spacing_factor = 1.0
        else:
            # === Narrow (Column) ===
            # 接線と同じ方向 (前後)
            dir_x = tangent_x
            dir_y = tangent_y
            spacing_factor = 1.2 
            
        # --- 5. オフセット適用 ---
        my_offset = self._lateral_offset * spacing_factor
        
        ox = dir_x * my_offset
        oy = dir_y * my_offset

        subgoal = PoseStamped()
        subgoal.header = path.header 
        subgoal.header.stamp = self.get_clock().now().to_msg()
        
        # 補間された点にオフセットを加える
        subgoal.pose.position.x = target_point.x + ox
        subgoal.pose.position.y = target_point.y + oy
        subgoal.pose.position.z = float(formation_id)
        
        # 向きはパスの進行方向に向ける
        target_yaw = math.atan2(tangent_y, tangent_x)
        subgoal.pose.orientation = self._yaw_to_quaternion(target_yaw)
        
        return subgoal

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

        # goal_position = goal_request.pose.pose.position
        # self.get_logger().info(
        #     f"NavigateToPose goal received (x={goal_position.x:.3f}, y={goal_position.y:.3f})"
        # )
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
        # self.latest_target_pose = goal_pose
        self.latest_global_goal = goal_pose    # 最終ゴールとして保存
        self.latest_path = None
        self._active_goal_handle = goal_handle

        self._navigation_success = False

        # 1. ロボット位置の確認
        if self.latest_robot_pose is None:
            self.get_logger().error("Robot pose unknown. Cannot plan path.")
            goal_handle.abort()
            return NavigateToPose.Result(success=False)

        # 2. プランナーサービスの呼び出し (VWAstar)
        if not self._planner_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("VWAstar service (/vwastar/get_path) not available.")
            goal_handle.abort()
            return NavigateToPose.Result(success=False)

        req = GetPath.Request()
        req.init_pose = self.latest_robot_pose
        req.goal_pose = goal_pose
        
        try:
            future = self._planner_client.call_async(req)
            # アクションサーバー内で非同期呼び出しを待機
            # (ReentrantCallbackGroupを使用しているためデッドロックしない想定)
            result = await future
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            goal_handle.abort()
            return NavigateToPose.Result(success=False)

        if not result.path.poses:
            self.get_logger().error("VWAstar returned empty path.")
            goal_handle.abort()
            return NavigateToPose.Result(success=False)

        # 3. パスの保存と可視化
        self._current_global_path_msg = result.path
        self._path_publisher.publish(result.path)
        self.get_logger().info(f"Path received with {len(result.path.poses)} points.")

        # ▼▼▼ 追加: 制御ループ用のパス変数を更新する ▼▼▼
        # これが無いと _coordination_timer_callback がパスを認識できません
        self.latest_path = self._densify_path(result.path, interval=0.1) 
        # ▲▲▲ 追加ここまで ▲▲▲

        self.latest_target_pose = goal_pose

        # 4. 制御ループ (ゴール到達判定)
        start_time = self.get_clock().now()
        self._navigation_success = False

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.get_logger().info("NavigateToPose goal cancelled")
                self._stop_robot()
                # 変数をクリアしてタイマーが動かないようにする
                self._current_global_path_msg = None
                self.latest_global_goal = None 
                self.latest_subgoal = None
                self.latest_path = None
                
                goal_handle.canceled()
                result = NavigateToPose.Result(success=False)
                return result

            # ▼▼▼ 修正箇所: ゴール到達時の処理 ▼▼▼
            if self.latest_global_goal is None:
                break

            # 到達判定
            if self._has_reached_goal(self.latest_global_goal):
                self.get_logger().info("NavigateToPose goal reached! Stopping robot.")
                
                # 1. 即座に停止コマンドを送る (速度、角速度を0に)
                self._stop_robot()
                
                # 2. 制御用変数を「即座に」クリアする
                # これをしないと、裏で動いているタイマーコールバックが
                # 「まだゴールがある」と判断して一瞬動いてしまう可能性がある
                self.latest_global_goal = None
                self.latest_subgoal = None
                self.latest_path = None
                self._current_global_path_msg = None
                
                self._navigation_success = True
                
                # 成功としてリターン
                goal_handle.succeed()
                return self._populate_navigation_result(start_time, success=True)
            # ▲▲▲ 修正ここまで ▲▲▲

            # フィードバック送信
            feedback = NavigateToPose.Feedback()
            feedback.current_pose = self.latest_robot_pose
            goal_handle.publish_feedback(feedback)
            
            # 少し待機
            await self._wait_for(self._action_feedback_period)

            # self._publish_navigation_feedback(goal_handle, start_time)
            # await self._wait_for(self._action_feedback_period)

        self._stop_robot()
        self._current_global_path_msg = None
        self._subgoal_pose = None
        self.latest_target_pose = None

        if self._navigation_success:
            self.get_logger().info("NavigateToPose goal reached")
            goal_handle.succeed()
            return NavigateToPose.Result(success=True)
            # return self._populate_navigation_result(start_time, success=True)
        else:
            goal_handle.abort()
            return NavigateToPose.Result(success=False)
            # return self._populate_navigation_result(start_time, success=False)

        # finally:
        #     if self._active_goal_handle is goal_handle:
        #         self._active_goal_handle = None

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


    def _robot_odometry_callback(self, msg: Odometry) -> None:
        self.get_logger().debug("Odometry update received")
        self.latest_odometry = msg

    def _robot_scan_callback(self, msg: LaserScan) -> None:
        self.get_logger().debug("Laser scan update received")
        self.latest_scan = msg

    # ▼▼▼ 修正: 位置関係に基づいて動的にオフセットを決定する ▼▼▼
    def _get_subgoal_from_path(self, robot_pose: PoseStamped, path: NavPath) -> Optional[PoseStamped]:
        if not path.poses:
            return None

        # 1. パス上でロボットに最も近い点を探す
        rx = robot_pose.pose.position.x
        ry = robot_pose.pose.position.y
        min_dist = float('inf')
        closest_idx = 0
        
        for i, pose_stamped in enumerate(path.poses):
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y
            dist = math.hypot(px - rx, py - ry)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # 2. Lookahead距離分だけ先の点をセンターパス上で探す
        center_target = path.poses[-1]
        target_idx = len(path.poses) - 1
        is_final_goal = True
        
        current_dist = 0.0
        for i in range(closest_idx, len(path.poses) - 1):
            p1 = path.poses[i].pose.position
            p2 = path.poses[i+1].pose.position
            segment_len = math.hypot(p2.x - p1.x, p2.y - p1.y)
            current_dist += segment_len
            
            if current_dist >= self._lookahead_distance:
                center_target = path.poses[i+1]
                target_idx = i + 1
                is_final_goal = False
                break

        # 3. 進行方向(yaw)の計算
        yaw = 0.0
        if is_final_goal:
            yaw = quaternion_to_yaw(center_target.pose.orientation)
        else:
            next_idx = target_idx + 1
            if next_idx < len(path.poses):
                p_curr = path.poses[target_idx].pose.position
                p_next = path.poses[next_idx].pose.position
                yaw = math.atan2(p_next.y - p_curr.y, p_next.x - p_curr.x)
            else:
                 yaw = quaternion_to_yaw(center_target.pose.orientation)

        # 4. 【重要】自分の位置がパスの左右どちらにあるか判定 (外積)
        # パス上の点(target_idx)から見たロボットの相対ベクトル
        path_pt = path.poses[target_idx].pose.position
        dx_robot = rx - path_pt.x
        dy_robot = ry - path_pt.y
        
        # パスの進行方向ベクトル
        dx_path = math.cos(yaw)
        dy_path = math.sin(yaw)
        
        # 外積 (2D Cross Product): v1.x * v2.y - v1.y * v2.x
        cross_product = dx_path * dy_robot - dy_path * dx_robot
        
        # 判定結果に基づいてオフセットを決定
        # 外積 > 0: 左側, 外積 < 0: 右側
        dynamic_offset = 0.0
        if cross_product > 0:
            dynamic_offset = self._formation_width / 2.0  # 左へ
        else:
            dynamic_offset = -self._formation_width / 2.0 # 右へ

        self.get_logger().debug(f"Side Check: Cross={cross_product:.2f} -> Offset={dynamic_offset}")

        # 5. オフセット座標の計算
        ox = -dynamic_offset * math.sin(yaw)
        oy =  dynamic_offset * math.cos(yaw)

        subgoal = PoseStamped()
        subgoal.header = center_target.header
        subgoal.pose.position.x = center_target.pose.position.x + ox
        subgoal.pose.position.y = center_target.pose.position.y + oy
        subgoal.pose.position.z = center_target.pose.position.z
        
        subgoal.pose.orientation = self._yaw_to_quaternion(yaw)
        
        return subgoal

    # ==========================================================
    # Safety Logic (Ported from policy2_newobs.py)
    # ==========================================================
    def _should_block_translation(self, scan: LaserScan) -> bool:
        """
        【修正版】
        LiDAR障害物検知の代わりに、相手ロボット（パートナー）の座標をTFで取得し、
        前方にいる場合のみ停止判定を行う。
        """
        try:
            # 1. 自分のフレーム名 (例: whill1/base_link)
            ns = self._robot_namespace.strip('/')
            ego_frame = f"{ns}/base_link" if ns else "base_link"
            
            # 2. 相手のフレーム名 (例: whill2/base_link)
            # launchファイル等で設定された partner_name ("whill2" 等) を使用
            partner_frame = f"{self.partner_name}/base_link"
            
            # 3. TF変換を試みる (相手の座標を自分の座標系で取得)
            # タイムアウトなしで最新の変換(Time())をチェック
            if not self._tf_buffer.can_transform(ego_frame, partner_frame, rclpy.time.Time()):
                return False

            transform = self._tf_buffer.lookup_transform(
                ego_frame,
                partner_frame,
                rclpy.time.Time()
            )
            
            # 4. 相対位置の計算 (x: 前方, y: 左方)
            dx = transform.transform.translation.x
            dy = transform.transform.translation.y
            dist = math.hypot(dx, dy)
            
            # 5. 判定パラメータの設定
            # ロボット半径(0.5m) x 2 + マージン(0.3m) = 1.3m 以内なら危険
            STOP_DIST_THRESH = 1.3 
            STOP_ANGLE_THRESH = math.radians(45.0) # 前方 +/- 45度
            
            if dist < STOP_DIST_THRESH:
                # 距離が近い場合、角度をチェック (atan2で真正面が0)
                angle = math.atan2(dy, dx)
                
                if abs(angle) < STOP_ANGLE_THRESH:
                    # パートナーが前方にいる -> 停止
                    # self.get_logger().info(f"Partner detected ahead: {dist:.2f}m")
                    return True

        except TransformException:
            # TFエラー時は安全のため停止させない（あるいはログを出す）
            pass
            
        return False

    def _coordination_timer_callback(self) -> None:
        if self._policy_controller is None:
            return
        
        # ▼▼▼ 修正: 姿勢情報の取得元を TF 優先に変更 ▼▼▼
        # まずTFから最新の高頻度姿勢を取得しようとする
        current_robot_pose = self._get_latest_pose_from_tf()

        
        # TFが失敗したら、従来のAMCLコールバックの値を使う (フォールバック)
        if current_robot_pose is None:
            current_robot_pose = self.latest_robot_pose

        # それでもPoseがないなら諦める
        if current_robot_pose is None:
            return
        
        # ※これ以降、 self.latest_robot_pose の代わりに current_robot_pose を使います
        # ▲▲▲ 修正ここまで ▲▲▲

        # ▼▼▼ 前回の修正: スキャンデータの保持 ▼▼▼
        current_scan = self.latest_scan
        scan_to_use = None
        if current_scan is not None and len(current_scan.ranges) > 0:
            scan_to_use = current_scan
        
        if scan_to_use is None:
            if self._prev_clean_scan is not None:
                scan_to_use = self._prev_clean_scan
            else:
                return
        # ▲▲▲ 前回の修正ここまで ▲▲▲

        target_to_use = None

        if self.role == "leader":
            # === LEADER のロジック ===
            # パスがあれば、自分と相方の目標を計算する
            if self.latest_path is not None:
                my_target, partner_target = self._calc_formation_targets(current_robot_pose, self.latest_path)
                
                if my_target is not None:
                    self.latest_subgoal = my_target # 自分の目標を更新
                
                if partner_target is not None:
                    # 相方に指令を送る
                    partner_target.header.stamp = self.get_clock().now().to_msg()
                    self._follower_cmd_pub.publish(partner_target)
            
            # リーダーは自分の計算したサブゴールを使う
            target_to_use = self.latest_subgoal
            
            # パスがない場合（ゴール直前など）は最終ゴールを使う
            if target_to_use is None and self.latest_global_goal is not None:
                target_to_use = self.latest_global_goal

        else:
            # === FOLLOWER のロジック ===
            # リーダーから送られてきた latest_subgoal をひたすら追うだけ
            target_to_use = self.latest_subgoal

        if target_to_use is None:
            self._stop_robot()
            return

        self._publish_visualization_markers(target_to_use)

        # スキャンデータのクリーニング
        clean_scan = LaserScan()
        clean_scan.header = scan_to_use.header
        clean_scan.angle_min = scan_to_use.angle_min
        clean_scan.angle_max = scan_to_use.angle_max
        clean_scan.angle_increment = scan_to_use.angle_increment
        clean_scan.time_increment = scan_to_use.time_increment
        clean_scan.scan_time = scan_to_use.scan_time
        clean_scan.range_min = scan_to_use.range_min
        clean_scan.range_max = scan_to_use.range_max
        # ... (前回のコードと同じクリーニング処理) ...
        # 省略... clean_scan.ranges = cleaned_ranges
        # self._prev_clean_scan = clean_scan

        cleaned_ranges = []
        IGNORE_RADIUS = 0.4
        for r in scan_to_use.ranges:
            if math.isnan(r) or r < scan_to_use.range_min or r < IGNORE_RADIUS:
                cleaned_ranges.append(scan_to_use.range_max + 1.0)
            else:
                cleaned_ranges.append(r)
        clean_scan.ranges = cleaned_ranges
        self._prev_clean_scan = clean_scan

        twist = self._policy_controller.compute_velocity_command(
            target_pose=target_to_use,
            robot_pose=current_robot_pose,    # ← ここで最新の姿勢を渡す！
            odometry=self._pose_velocity_odometry,
            pointcloud=None,
            laser_scan=clean_scan,
        )

        if twist is None:
            return
        
        if self._should_block_translation(clean_scan):
            # 前方に障害物がある場合、並進速度(v)を0にする
            # (回転(omega)は許可することで、その場旋回での回避を妨げない)
            twist.linear.x = 0.0
            # self.get_logger().info("Safety Stop Activated")
        # ▲▲▲ 追加ここまで ▲▲▲

        # ▼▼▼ 追加: ゴール接近時の減速ロジック ▼▼▼
        # 最終ゴール (Global Goal) が設定されている場合のみ動作
        if self.latest_subgoal is not None:
            gx = self.latest_subgoal.pose.position.x
            gy = self.latest_subgoal.pose.position.y
            rx = current_robot_pose.pose.position.x
            ry = current_robot_pose.pose.position.y
            
            dist_to_goal = math.hypot(gx - rx, gy - ry)
            
            DECEL_DIST = 1.0  # 減速を開始する距離 [m]
            MIN_VEL = 0.1     # 最低速度 [m/s] (公差内に入るまで完全に止まらないように)

            if dist_to_goal < DECEL_DIST:
                # 距離に応じて速度係数を計算 (距離が近いほど 0 に近づく)
                scale = dist_to_goal * 1.0 / DECEL_DIST# / 2.0
                
                # 現在の速度指令に係数を掛ける
                target_vel = twist.linear.x * scale
                
                # 「計算した減速速度」と「最低速度」のうち大きい方を採用
                # (ただし、障害物回避などで元々遅い場合は、元の速度を超えないように min を取る)
                if twist.linear.x > 0: # 後退中は適用しない
                     twist.linear.x = min(twist.linear.x, max(target_vel, MIN_VEL))
        # ▲▲▲ 追加ここまで ▲▲▲

        # if self.latest_global_goal is not None and self._has_reached_goal(self.latest_global_goal):
        #     twist.linear.x = 0.0
        #     twist.angular.z = 0.0
        #     # self.get_logger().info("Force Stop at Goal") # デバッグ用

        twist = self._apply_acceleration_limit(twist)

        self._cmd_vel_publisher.publish(twist)
        self.latest_cmd_vel = twist

    def _apply_acceleration_limit(self, twist: Twist) -> Twist:
        """
        Apply dynamic window acceleration limits to the command.
        Ensures smooth transitions for velocity changes (gradual braking/accelerating).
        """
        ACC_LIM = 1.0
        ANG_ACC_LIM = 1.0
        dt = 0.1

        # Get previous state (from last command or 0 if starting)
        if self.latest_cmd_vel is None:
            prev_v = 0.0
            prev_omega = 0.0
        else:
            prev_v = self.latest_cmd_vel.linear.x
            prev_omega = self.latest_cmd_vel.angular.z

        target_v = twist.linear.x
        target_omega = twist.angular.z

        # Calculate limits: [v_prev - a*dt, v_prev + a*dt]
        # This allows gradual change in both directions (accel and braking)
        v_min = prev_v - ACC_LIM * dt
        v_max = prev_v + ACC_LIM * dt
        
        # Apply clip
        v_e = float(np.clip(target_v, v_min, v_max))
        
        # Angular limits
        w_min = prev_omega - ANG_ACC_LIM * dt
        w_max = prev_omega + ANG_ACC_LIM * dt
        
        omega_e = float(np.clip(target_omega, w_min, w_max))

        # Create new Twist message
        new_twist = Twist()
        new_twist.linear.x = v_e
        new_twist.angular.z = omega_e
        
        return new_twist

    def _get_subgoal_on_path(self, current_pos, path_msg: Path, lookahead: float) -> Optional[PoseStamped]:
        """
        nav_msgs/Path 上で、現在地から lookahead [m] 先の PoseStamped を返す。
        """
        poses = path_msg.poses
        if not poses:
            return None
            
        cx, cy = current_pos.x, current_pos.y
        
        # 1. パス上で現在地に最も近い点を探す
        min_dist = float('inf')
        closest_idx = 0
        for i, pose_stamped in enumerate(poses):
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y
            d = math.hypot(px - cx, py - cy)
            if d < min_dist:
                min_dist = d
                closest_idx = i
        
        # 2. closest_idx から前方に探索して lookahead 距離離れた点を探す
        # デフォルトはパスの終点
        subgoal = poses[-1]
        
        dist_accum = 0.0
        # パスの終わりまで探索
        for i in range(closest_idx, len(poses) - 1):
            p1 = poses[i].pose.position
            p2 = poses[i+1].pose.position
            
            segment_len = math.hypot(p2.x - p1.x, p2.y - p1.y)
            dist_accum += segment_len
            
            # 現在地点からの直線距離が lookahead を超えたら、その点をサブゴールとする
            # (より正確には累積距離 path distance を使うのが一般的だが、ここでは直線距離で簡易判定)
            dist_from_curr = math.hypot(p2.x - cx, p2.y - cy)

            if dist_from_curr > lookahead:
                subgoal = poses[i+1] # p2
                break
                
        return subgoal


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
        twist.linear.x = 0.0
        twist.angular.z = 0.0
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

    def _publish_visualization_markers(self, target_pose: Optional[PoseStamped]) -> None:
        if self.latest_robot_pose is None or self.latest_scan is None:
            return

        # ロボット座標系の情報
        robot_position = self.latest_robot_pose.pose.position
        frame_id = self.latest_robot_pose.header.frame_id or "map"
        stamp = self.get_clock().now().to_msg()

        markers: list[Marker] = []
        
        if target_pose is not None:
            # 緑の矢印（ゴール方向）
            dx = float(target_pose.pose.position.x - robot_position.x)
            dy = float(target_pose.pose.position.y - robot_position.y)
            yaw = math.atan2(dy, dx)
            length = math.hypot(dx, dy)
            markers.append(self._build_arrow_marker(
                marker_id=0, frame_id=frame_id, stamp=stamp,
                start_position=robot_position, yaw=yaw, length=length, color=(0.0, 1.0, 0.0, 0.8)
            ))
            # 水色の球体（サブゴール位置）
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = "subgoal_point"
            marker.id = 3
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = target_pose.pose.position
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2; marker.scale.y = 0.2; marker.scale.z = 0.2
            marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.color.a = 0.9
            markers.append(marker)

        # 障害物マーカー (引数を修正: robot_position等は不要になりました)
        obstacle_marker = self._closest_obstacle_marker(
            robot_position=None, 
            robot_yaw=0.0, 
            frame_id="", 
            stamp=stamp
        )
        if obstacle_marker is not None:
            markers.append(obstacle_marker)

        # cmd_vel テキスト
        if self.latest_cmd_vel is not None:
            markers.append(self._build_cmd_vel_text_marker(
                marker_id=2, frame_id=frame_id, stamp=stamp,
                robot_position=robot_position, twist=self.latest_cmd_vel,
            ))

        if markers:
            self._marker_publisher.publish(MarkerArray(markers=markers))

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
        robot_position, # 使わない
        robot_yaw: float, # 使わない
        frame_id: str, # 使わない
        stamp,
    ) -> Optional[Marker]:
        if self.latest_scan is None:
            return None

        scan = self.latest_scan
        
        # ▼▼▼ データのクリーニング（表示用） ▼▼▼
        # 無効なデータを除外して最小値を探索
        min_distance = None
        min_angle = None
        
        valid_min_range = scan.range_min if scan.range_min > 0.0 else 0.05
        
        for index, distance in enumerate(scan.ranges):
            # inf/nan/0.0 を無視
            if not math.isfinite(distance) or distance < valid_min_range:
                continue
            
            if min_distance is None or distance < min_distance:
                min_distance = distance
                min_angle = scan.angle_min + index * scan.angle_increment
        
        if min_distance is None or min_angle is None:
            return None

        # ▼▼▼ 修正: センサ座標系(frame_id)で表示することでズレをなくす ▼▼▼
        # ロボットの現在位置や姿勢(robot_yaw)を足し合わせる計算を削除し、
        # センサの "生データ" をそのまま可視化します。
        
        # センサ中心からの相対位置
        return self._build_arrow_marker(
            marker_id=1,
            frame_id=scan.header.frame_id, # ここが重要: mapではなくセンサのフレームを使う
            stamp=stamp,
            start_position=Point(x=0.0, y=0.0, z=0.0), # センサの中心
            yaw=min_angle, # センサ中心からの角度だけでOK
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