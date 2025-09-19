import math
import os

import pygame
import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from simulation_core.environment import SimCoreEnvironment


class Environment(SimCoreEnvironment, Node):

    def __init__(self):
        # Initialize both parent classes
        SimCoreEnvironment.__init__(self)
        Node.__init__(self, "environment_node")

        self.publisher_ = self.create_publisher(PointCloud2, "pointcloud", 10)
        self.get_logger().info("PointCloud Publisher Node has been started.")

        self.subscription = self.create_subscription(
            Twist,
            "cmd_vel",
            self._listener_callback,
            10,
        )
        self.tf_broadcaster = TransformBroadcaster(self)

    def _listener_callback(self, msg: Twist):
        self.get_logger().info(
            f"Received Twist:\n"
            f"  linear:  x={msg.linear.x:.2f}, y={msg.linear.y:.2f}, z={msg.linear.z:.2f}\n"
            f"  angular: x={msg.angular.x:.2f}, y={msg.angular.y:.2f}, z={msg.angular.z:.2f}"
        )
        # This logic is now handled by _operate_moving, which can be adapted
        # to take velocity commands instead of keyboard presses.
        # For now, we'll just update the robot's pose based on the message.
        
        # Convert linear velocity from m/s to pixels/update
        # Let's assume the update rate is similar to manual key presses for now
        step_size = msg.linear.x * 5 # A simple scaling factor

        next_position = (
            self.robot_position[0] + step_size * math.sin(self.robot_theta),
            self.robot_position[1] - step_size * math.cos(self.robot_theta),
        )

        # Boundary and collision checks
        if 0 < next_position[0] < self.screen_w and 0 < next_position[1] < self.screen_h:
            if not self.is_collision(next_position):
                self.robot_position = next_position

        self.robot_theta += msg.angular.z  # Rotate robot clockwise
        if self.robot_theta > 2 * math.pi:
            self.robot_theta -= 2 * math.pi
        elif self.robot_theta < 0:
            self.robot_theta += 2 * math.pi

    def publish_tf(self):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"        # ワールド座標系
        t.child_frame_id = "base_link"    # ロボットのフレーム

        # 2Dの位置を3Dに拡張
        t.transform.translation.x = -(self.robot_position[1] - self.screen_center_y) / self.pixels_per_meter
        t.transform.translation.y = -(self.robot_position[0] - self.screen_center_x) / self.pixels_per_meter
        t.transform.translation.z = 0.0

        # 角度をクォータニオンに変換
        angle = self.robot_theta + math.pi  # 2Dの角度を調整
        qz = math.cos(angle / 2.0)
        qw = math.sin(angle / 2.0)
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(t)
        

def main():
    pygame.init()
    rclpy.init()

    env = Environment()
    
    env.render_obstacle_and_robots()

    print((env.screen_center_x, env.screen_center_y))
    while env.running:
        rclpy.spin_once(env, timeout_sec=0.01)

        env.clear_robot_and_lidar_layer()
        
        env.change_map_by_event()

        env._operate_moving()  # to handle robot movement

        sensor_data = env.lidar_scan()
        env.show_robot_and_lidar(sensor_data)

        if sensor_data:
            point_cloud_msg = convert_to_pointcloud2(
                sensor_data,
                env.robot_theta,
                time_stamp=env.get_clock().now().to_msg(),
                pixels_per_meter=env.pixels_per_meter,
            )
            env.publisher_.publish(point_cloud_msg)
        env.is_collision(env.robot_position)
        
        env.publish_tf()

        pygame.display.update()

    env.destroy_node()
    rclpy.shutdown()
    pygame.quit()
    

def convert_to_pointcloud2(
    sensor_data, robot_theta, time_stamp, pixels_per_meter, frame_id="base_link"
):
    header = Header()
    header.stamp = time_stamp
    header.frame_id = frame_id

    points = []
    # sensor_data: [[(distance, angle(radian), (robot_x, robot_y)], ...]
    # print("sensor_data:", sensor_data)
    for point_data in sensor_data:
        # print(point_data)
        distance = point_data[0]
        angle = point_data[1]
        angle -= robot_theta  # add robot's orientation to angle
        # x_distance = (
        #     point_data[2][0] + distance * math.cos(angle)
        # ) / pixels_per_meter
        # y_distance = (
        #     point_data[2][1] + distance * math.sin(angle)
        # ) / pixels_per_meter
        # robot centoroid
        # x_distance = (distance * math.cos(angle)) / pixels_per_meter
        # y_distance = (distance * math.sin(angle)) / pixels_per_meter
        y_distance = -(distance * math.cos(angle)) / pixels_per_meter
        x_distance = -(distance * math.sin(angle)) / pixels_per_meter
        
        
        # print("x_distance, y_distance: ", x_distance, y_distance)
        # x_distance = point_data[0]
        # y_distance = point_data[1]
        points.append([x_distance, y_distance, 0.0])  # 2Dデータのためz=0
        # points.append([-y_distance, -x_distance, 0.0])  # 2Dデータのためz=0

    # PointCloud2のフィールドを定義
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    # Python listをNumPy arrayに変換
    points_array = np.array(points, dtype=np.float32)

    # NumPy配列をバイトデータに変換
    # tobytes()はPython 3で推奨される方法
    bytes_data = points_array.tobytes()

    # PointCloud2メッセージの作成
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(points)
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 12  # 3 fields * 4 bytes/field
    msg.row_step = 12 * len(points)
    msg.data = bytes_data
    msg.is_dense = True

    return msg


if __name__ == "__main__":
    main()
