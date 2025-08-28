"""
Defines the 2D environment, obstacles, start/target

This class will procedurally generate and manage the state of the 2D world.

- `__init__(self, config)`:
    - Reads parameters from the config file.
    - Generates random obstacle positions and sizes (e.g., as circles or rectangles).
    - Randomly determines and stores the robot's start and target positions.
- `get_obstacles(self)`:
    - Returns a list of obstacle objects for collision checking.
- `get_start_target(self)`:
    - Returns the generated start and target coordinates.
"""

import math
import random
import os

import yaml
import pygame
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

# import sensors
from matplotllib_2d_simulator import sensors


class Environment(Node):

    def __init__(self):
        super().__init__("environment")
        self.window_name = "2D LiDAR Simulation"
        pygame.display.set_caption(self.window_name)

        # Set parameters
        config_path = os.path.join(
            get_package_share_directory("matplotllib_2d_simulator"),
            "config",
            "simulator_config.yaml",
        )
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        self.random_np = np.random.default_rng(0) # random number generator with seed 0

        self.background_color = config["map"]["background_color"]
        self.map_w, self.map_h = config["map"]["size"], config["map"]["size"]

        self.robot_radius_m = config["robot"]["radius_m"]
        self.robot_color = config["robot"]["robot_color"]
        self.target_color = config["robot"]["target_color"]
        self.T_dt_m_trap_shaped = config["robot"]["T_dt_m_trap_shaped"]
        self.T_dt_m_zigzag = config["robot"]["T_dt_m_zigzag"]
        self.T_dt_m_random = config["robot"]["T_dt_m_random"]
        self.T_dt_m_empty = config["robot"]["T_dt_m_empty"]

        self.lidar_range_m = config["lidar"]["range_m"]
        self.lidar_line_color = config["lidar"]["line_color"]
        self.lidar_points_color = config["lidar"]["points_color"]

        self.T_do_m = config["obstacles"]["T_do_m"]
        self.n_obstacles = config["obstacles"]["count"]
        self.min_obstacle_size_m = config["obstacles"]["min_size_m"]
        self.max_obstacle_size_m = config["obstacles"]["max_size_m"]
        self.obstacle_color = config["obstacles"]["color"]

        self.pixels_per_meter = self.map_w / (2 * self.lidar_range_m)

        self.robot_radius_pixels = self.robot_radius_m * self.pixels_per_meter
        self.lidar_range_pixels = self.lidar_range_m * self.pixels_per_meter

        self.map = pygame.display.set_mode((self.map_w, self.map_h))
        self.map.fill(self.background_color)

        # self.pointCloud = (
        #     []
        # )  # array containing coordinates of sensed objects (walls). Purely just (x,y) information.

        self.map_center_x = self.map_w // 2
        self.map_center_y = self.map_h // 2

        self.obstacle_layer = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
        self.lidar_sensor_layer = pygame.Surface(
            (self.map_w, self.map_h), pygame.SRCALPHA
        )
        self.robot_layer = pygame.Surface((self.map_w, self.map_h))
        # Robot's orientation in radians
        # 0 radians is facing up (north), and increases clockwise
        self.robot_position = (self.map_center_x, self.map_center_y)
        self.target_position = (self.map_center_x, self.map_center_y)
        self.robot_theta = 0.0
        self.target_theta = 0.0

        self.map_type = "empty"  # ["trap_shaped", "zigzag", "random", "empty"]

        """
        Trap-shaped map
        """
        self.s_t = 0  # number of success in the trap-shaped map
        self.k8 = config["simulation"]["k8"]

        """
        Zigzag-shaped map
        """
        self.s_z = 2400  # number of success in the zigzag-shaped map
        self.k9 = config["simulation"]["k9"]
        self.l_passage = 0  # length of the passage in the zigzag-shaped map
        self.divided_space_width = (
            self.map_h // 5
        )  # width of the divided space in the zigzag-shaped map

        self.publisher_ = self.create_publisher(PointCloud2, "pointcloud", 10)
        self.get_logger().info("PointCloud Publisher Node has been started.")

        self.subscription = self.create_subscription(
            Twist,
            "cmd_vel",  # 通常はcmd_velというトピック名
            self._listener_callback,
            10,  # QoS
        )
        self.tf_broadcaster = TransformBroadcaster(self)

    def _listener_callback(self, msg: Twist):
        self.get_logger().info(
            f"Received Twist:\n"
            f"  linear:  x={msg.linear.x:.2f}, y={msg.linear.y:.2f}, z={msg.linear.z:.2f}\n"
            f"  angular: x={msg.angular.x:.2f}, y={msg.angular.y:.2f}, z={msg.angular.z:.2f}"
        )
        speed = 5
        if (
            self.robot_position[1] > self.robot_radius_pixels
            and self.robot_position[1] < self.map_h - self.robot_radius_pixels
        ):
            next_position = (
                self.robot_position[0]
                + msg.linear.x * speed * math.sin(self.robot_theta),
                self.robot_position[1]
                - msg.linear.x * speed * math.cos(self.robot_theta),
            )
            if next_position[1] > 0:
                if _check_surrounding_colors(self, next_position):
                    self.robot_position = next_position  # Move robot left
                    self.reset_window()
        self.robot_theta += msg.angular.z  # Rotate robot clockwise
        if self.robot_theta > 2 * math.pi:
            self.robot_theta -= 2 * math.pi

    def _create_base_map(self):
        """Creates a base map with walls."""
        wall_width = 1
        pygame.draw.polygon(
            self.obstacle_layer,
            self.obstacle_color,
            [
                [0, 0],
                [self.map_w - wall_width, 0],
                [self.map_w - wall_width, self.map_h - wall_width],
                [0, self.map_h - wall_width],
            ],
            width=wall_width,
        )
        font = pygame.font.SysFont("arial", 16)
        text = font.render(
            f"{self.map_w // self.pixels_per_meter}m x {self.map_h // self.pixels_per_meter}m",
            True,
            (0, 0, 0),
        )
        self.obstacle_layer.blit(
            text,
            (
                self.map_w - text.get_width() - 5,
                self.map_h - text.get_height() - 5,
            ),
        )

    def _create_random_map(self):
        """Creates a random map with obstacles"""
        # Draw walls
        self._create_base_map()

        # Define the area for obstacles
        obstacle_area_width = self.map_w // 15 * 13
        obstacle_area_height = self.map_h // 15 * 13
        pygame.draw.rect(
            self.obstacle_layer,
            (250, 250, 250),
            (
                self.map_center_x - obstacle_area_width // 2,
                self.map_center_y - obstacle_area_height // 2,
                obstacle_area_width,
                obstacle_area_height,
            ),
            width=0,
        )

        self.obstacle_pos_list = []  # The list of obstacle positions (cx, cy)
        count = 0
        while len(self.obstacle_pos_list) < self.n_obstacles:
            count += 1
            if count > 100:
                self.obstacle_pos_list = []
                count = 0

            obstacle_cx = self.random_np.integers(
                self.map_center_x - obstacle_area_width // 2,
                self.map_center_x + obstacle_area_width // 2,
            )
            obstacle_cy = self.random_np.integers(
                self.map_center_y - obstacle_area_height // 2,
                self.map_center_y + obstacle_area_height // 2,
            )
            if len(self.obstacle_pos_list) == 0:
                self.obstacle_pos_list.append((obstacle_cx, obstacle_cy))
                # break
            else:
                is_ok = True
                for pos in self.obstacle_pos_list:
                    # Check if the new obstacle is far enough from existing obstacles
                    # T_do_m is the threshold distance between each object in meters
                    # Convert to pixels by multiplying with PIXELS_PER_METER
                    distance = math.sqrt(
                        (obstacle_cx - pos[0]) ** 2 + (obstacle_cy - pos[1]) ** 2
                    )
                    print(f"Distance (m): {distance // self.pixels_per_meter}")
                    if distance < self.T_do_m * self.pixels_per_meter:
                        is_ok = False
                if is_ok:
                    self.obstacle_pos_list.append((obstacle_cx, obstacle_cy))
                    # break
        print(f"Obstacle Position: {obstacle_cx}, {obstacle_cy}")

        for obstacle_cx, obstacle_cy in self.obstacle_pos_list:
            # obstacle_width = self.random_np.integers(60, 120)
            # obstacle_height = self.random_np.integers(60, 120)
            shape_random = self.random_np.integers(0, 1)
            if shape_random == 0:
                # Draw a rectangle obstacle
                rectangular_l_m = 4 + self.random_np.uniform(0, 1)
                rectangular_w_m = 0.5 + self.random_np.uniform(0, 1) * 0.5
                rectangular_l_pixels = int(rectangular_l_m * self.pixels_per_meter)
                rectangular_w_pixels = int(rectangular_w_m * self.pixels_per_meter)
                if self.random_np.integers(0, 1) == 0:
                    object_width = rectangular_w_pixels
                    object_height = rectangular_l_pixels
                else:
                    object_width = rectangular_l_pixels
                    object_height = rectangular_w_pixels
                pygame.draw.rect(
                    self.obstacle_layer,
                    self.obstacle_color,
                    (
                        obstacle_cx - object_width // 2,
                        obstacle_cy - object_height // 2,
                        object_width,
                        object_height,
                    ),
                    width=0,
                )
            elif shape_random == 1:
                # Draw a circular obstacle
                radius_m = 0.5 + self.random_np.uniform(0, 1)
                radius_pixels = int(radius_m * self.pixels_per_meter)
                pygame.draw.circle(
                    self.obstacle_layer,
                    self.obstacle_color,
                    (obstacle_cx, obstacle_cy),
                    radius_pixels,
                    width=0,
                )
            # Draw a center point for the obstacle
            # pygame.draw.circle(
            #     self.obstacle_layer,
            #     (10, 10, 10),
            #     (obstacle_cx, obstacle_cy),
            #     1,
            # )

    def _create_trap_shaped_map(self):
        """Creates a trap-shaped map with obstacles"""
        # Parameters
        p_t = min(self.s_t / self.k8, 1)
        w_wall = 0.2  # Width of the walls in meters
        l_horizontal = []
        for i in range(3):
            l_horizontal.append(1.5 * p_t + 3 + self.random_np.uniform(0, 1))
        l_vertical = []
        for i in range(3):
            l_vertical_left = 6 * p_t + 1 + 4 * self.random_np.uniform(0, 1)
            l_vertical_right = 6 * p_t + 1 + 4 * self.random_np.uniform(0, 1)
            if i == 0:
                if self.random_np.integers(0, 1) == 0:
                    l_vertical_left = self.map_h
            elif i == 2:
                if self.random_np.integers(0, 1) == 0:
                    l_vertical_right = self.map_h
            l_vertical.append([l_vertical_left, l_vertical_right])
        w_passage_0 = -1.6 * p_t + 2.5 + 0.2 * self.random_np.uniform(0, 1)
        w_passage_1 = (0.5 + 0.5 * self.random_np.uniform(0, 1)) * w_passage_0

        distance_btwn_H_shaped_obstacles_left = (
            l_horizontal[1] / 2 + w_passage_0 + w_wall * 2 + l_horizontal[0] / 2
        ) * self.pixels_per_meter
        distance_btwn_H_shaped_obstacles_right = (
            l_horizontal[1] / 2 + w_passage_1 + w_wall * 2 + l_horizontal[2] / 2
        ) * self.pixels_per_meter

        # Define the positions of the H-shaped obstacles
        positions_of_H_shaped_obstacles = [
            (
                self.map_center_x - distance_btwn_H_shaped_obstacles_left,
                self.map_center_y + self.random_np.uniform(-1, 1) * self.pixels_per_meter,
            ),
            (
                self.map_center_x,
                self.map_center_y + self.random_np.uniform(-1, 1) * self.pixels_per_meter,
            ),
            (
                self.map_center_x + distance_btwn_H_shaped_obstacles_right,
                self.map_center_x + self.random_np.uniform(-1, 1) * self.pixels_per_meter,
            ),
        ]

        # Draw walls
        self._create_base_map()

        # Define the area for obstacles
        for i, position in enumerate(positions_of_H_shaped_obstacles):
            H_shaped_obstacle_cx, H_shaped_obstacle_cy = position

            # Draw the horizontal walls
            pygame.draw.rect(
                self.obstacle_layer,
                self.obstacle_color,
                (
                    H_shaped_obstacle_cx
                    - (l_horizontal[i] / 2) * self.pixels_per_meter,
                    H_shaped_obstacle_cy - (w_wall / 2) * self.pixels_per_meter,
                    l_horizontal[i] * self.pixels_per_meter,
                    w_wall * self.pixels_per_meter,
                ),
                width=0,
            )
            # Draw the vertical walls (left side)
            pygame.draw.rect(
                self.obstacle_layer,
                self.obstacle_color,
                (
                    H_shaped_obstacle_cx
                    - (l_horizontal[i] / 2 + w_wall) * self.pixels_per_meter,
                    H_shaped_obstacle_cy
                    - (l_vertical[i][0] / 2) * self.pixels_per_meter,
                    w_wall * self.pixels_per_meter,
                    l_vertical[i][0] * self.pixels_per_meter,
                ),
                width=0,
            )
            # Draw the vertical walls (right side)
            pygame.draw.rect(
                self.obstacle_layer,
                self.obstacle_color,
                (
                    H_shaped_obstacle_cx
                    + (l_horizontal[i] / 2) * self.pixels_per_meter
                    - 1,  # add 1 to avoid gap between horizontal and vertical walls
                    H_shaped_obstacle_cy
                    - (l_vertical[i][1] / 2) * self.pixels_per_meter,
                    w_wall * self.pixels_per_meter,
                    l_vertical[i][1] * self.pixels_per_meter,
                ),
                width=0,
            )

    def _create_zigzag_map(self):
        """Creates a zigzag-shaped map with obstacles"""
        # Parameters
        p_z = min(self.s_z / self.k9, 1)
        w_wall = 0.2
        self.l_passage = 6 * p_z + 8 + 2 * self.random_np.uniform(0, 1)
        w_openings = []
        for _ in range(4):
            w_openings.append(-4.6 * p_z + 6 + 0.4 * self.random_np.uniform(0, 1))
        d_deviations = []
        for _ in range(4):
            d_deviations.append(2.4 * self.random_np.uniform(0, 1) - 1.2)

        self._create_base_map()

        # Draw dotted lines
        for i in range(1, 5):
            pygame.draw.line(
                self.obstacle_layer,
                (240, 240, 240),
                (0, (self.divided_space_width * i)),
                (
                    self.map_w,
                    (self.divided_space_width * i),
                ),
                width=3,
            )

        # Draw side walls
        side_wall_xs = [
            self.map_center_x - (self.l_passage / 2 + w_wall) * self.pixels_per_meter,
            self.map_center_x + (self.l_passage / 2) * self.pixels_per_meter,
        ]
        for x in side_wall_xs:
            pygame.draw.rect(
                self.obstacle_layer,
                self.obstacle_color,
                (
                    x,
                    0,
                    w_wall * self.pixels_per_meter,
                    self.map_h,
                ),
                width=0,
            )

        # Draw horizontal walls
        r = self.random_np.integers(0, 1)
        for i in range(4):
            if i % 2 == r:
                horizontal_wall_xmin = (
                    self.map_center_x
                    - (self.l_passage / 2 - w_openings[i]) * self.pixels_per_meter
                )
            else:
                horizontal_wall_xmin = (
                    self.map_center_x - (self.l_passage / 2) * self.pixels_per_meter
                )
            pygame.draw.rect(
                self.obstacle_layer,
                self.obstacle_color,
                (
                    horizontal_wall_xmin,
                    self.divided_space_width * (i + 1)
                    + d_deviations[i] * self.pixels_per_meter,
                    (self.l_passage - w_openings[i]) * self.pixels_per_meter
                    + 1,  # add 1 to avoid gap between horizontal and side walls
                    w_wall * self.pixels_per_meter,
                ),
                width=0,
            )

    def _create_empty_map(self):
        """Creates an empty map with no obstacles."""
        self._create_base_map()

    # Uses Distance and Angle from data[] in sensors.sensesObstacles() to obtain cartesian coordinates of walls for point cloud
    def _AD2pos(self, distance, angle, robotPosition):
        x = robotPosition[0] + distance * math.cos(angle)
        y = robotPosition[1] + distance * math.sin(angle)
        return (int(x), int(y))  # returns cartesian coordinates of walls.

    def dataStorage(
        self, data
    ):  # data = [Distance, Angle, (x,y)] --> This is the input. Storing data in point cloud.
        # print(len(self.pointCloud))
        self.pointCloud = []  # Reset point cloud for new data
        if data != False:  # if data exists, then start for loop
            for element in data:
                point = self._AD2pos(
                    element[0], element[1]  # , element[2]
                )  # Input: data = [Distance, Angle, (x,y)]. Output: (x,y) coordinate of objects/walls.
                if (
                    point not in self.pointCloud
                ):  # Note: Point cloud should ONLY be the WALLS (x,y). If the wall has not already been sensed, then add to pointcloud.
                    self.pointCloud.append(point)  # [(x1,y1), (x2,y2), (x3,y3)]
            self.last_robot_position = (int(data[0][2][0]), int(data[0][2][1]))

    def show_sensorData(self, data=None):
        if data != False:  # if data exists, then start for loop
            for element in data:
                point = self._AD2pos(
                    element[0], element[1], self.robot_position
                )  # Input: data = [Distance, Angle, (x,y)]. Output: (x,y) coordinate of objects/walls.
                pygame.draw.line(
                    self.lidar_sensor_layer,
                    self.lidar_line_color,
                    (int(self.robot_position[0]), int(self.robot_position[1])),
                    (int(point[0]), int(point[1])),
                    1,
                )
                pygame.draw.circle(
                    self.lidar_sensor_layer,
                    self.lidar_points_color,
                    (int(point[0]), int(point[1])),
                    2,
                )

            robot_frame_radius = self.robot_radius_m * self.pixels_per_meter
            robot_radius = self.robot_radius_m * 0.8 * self.pixels_per_meter

            # Draw target position
            pygame.draw.circle(
                self.lidar_sensor_layer,
                self.target_color,
                self.target_position,
                robot_frame_radius,
                width=2,  # Draw a circle around the robot
            )
            pygame.draw.circle(
                self.lidar_sensor_layer,
                self.target_color,
                self.target_position,
                robot_radius,
            )
            target_head_position = (
                self.target_position[0]
                + robot_frame_radius * math.sin(self.target_theta),
                self.target_position[1]
                - robot_frame_radius * math.cos(self.target_theta),
            )
            pygame.draw.line(
                self.lidar_sensor_layer,
                (255, 0, 0),  # Red color for the target's head
                self.target_position,
                target_head_position,
                2,  # Draw the target's head
            )
            # font = pygame.font.SysFont("arial", 16)
            # text = font.render("Target", True, (0, 0, 0))
            # self.lidar_sensor_layer.blit(
            #     text,
            #     (
            #         self.target_position[0],
            #         self.target_position[1] + text.get_height() // 2,
            #     ),
            # )

            # Draw the robot's position
            pygame.draw.circle(
                self.lidar_sensor_layer,
                self.robot_color,
                self.robot_position,
                robot_frame_radius,
                width=2,  # Draw a circle around the robot
            )
            pygame.draw.circle(
                self.lidar_sensor_layer,
                self.robot_color,
                self.robot_position,
                robot_radius,
            )
            robot_head_position = (
                self.robot_position[0]
                + robot_frame_radius * math.sin(self.robot_theta),
                self.robot_position[1]
                - robot_frame_radius * math.cos(self.robot_theta),
            )
            pygame.draw.line(
                self.lidar_sensor_layer,
                (255, 0, 0),  # Red color for the robot's head
                self.robot_position,
                robot_head_position,
                2,  # Draw the robot's head
            )

            self.map.blit(
                self.lidar_sensor_layer, (0, 0)
            )  # Overlay lidar sensor layer on the map

    def reset_environment(self):
        """Resets the environment by clearing the point cloud and reinitializing the map."""
        self.map.fill(self.background_color)  # Clear the map
        self.obstacle_layer = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
        self.lidar_sensor_layer = pygame.Surface(
            (self.map_w, self.map_h), pygame.SRCALPHA
        )

    def reset_window(self):
        self.map.fill(self.background_color)
        self.map.blit(
            self.obstacle_layer, (0, 0)
        )  # Draw the obstacle layer onto the map
        self.lidar_sensor_layer = pygame.Surface(
            (self.map_w, self.map_h), pygame.SRCALPHA
        )
    
    def publish_tf(self):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"        # ワールド座標系
        t.child_frame_id = "base_link"    # ロボットのフレーム

        # 2Dの位置を3Dに拡張
        t.transform.translation.x = -(self.robot_position[1] - self.map_center_y) / self.pixels_per_meter
        t.transform.translation.y = -(self.robot_position[0] - self.map_center_x) / self.pixels_per_meter
        t.transform.translation.z = 0.0

        # 角度をクォータニオンに変換
        angle = self.robot_theta + math.pi # 2Dの角度を調整
        qz = math.cos(angle / 2.0)
        qw = math.sin(angle / 2.0)
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(t)


    def _hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip("#")  # 先頭の # を削除
        # 6文字 (RRGGBB) でなければエラー
        if len(hex_color) != 6:
            raise ValueError("カラーコードは6桁の16進数で指定してください。")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)


    def _check_surrounding_colors(self, position):
        """Checks the surrounding colors of a position in the environment."""
        surrounding_colors = []
        # Number of pixels to check in each direction (odd number)
        num_surrounding_points = 5
        for i in range(-2, 3):
            for j in range(-2, 3):
                if (
                    int(
                        position[0]
                        + i / (num_surrounding_points - 1) * 2 * self.robot_radius_pixels
                    )
                    >= 0
                    and int(
                        position[1]
                        + j / (num_surrounding_points - 1) * 2 * self.robot_radius_pixels
                    )
                    >= 0
                    and int(
                        position[0]
                        + i / (num_surrounding_points - 1) * 2 * self.robot_radius_pixels
                    )
                    <= self.map_w
                    and int(
                        position[1]
                        + j / (num_surrounding_points - 1) * 2 * self.robot_radius_pixels
                    )
                    <= self.map_h
                ):
                    surrounding_colors.append(
                        self.map.get_at(
                            (
                                int(
                                    position[0]
                                    + i
                                    / (num_surrounding_points - 1)
                                    * 2
                                    * self.robot_radius_pixels
                                ),
                                int(
                                    position[1]
                                    + j
                                    / (num_surrounding_points - 1)
                                    * 2
                                    * self.robot_radius_pixels
                                ),
                            )
                        )
                    )
        is_ok = True
        for color in surrounding_colors:
            if color == self._hex_to_rgb(self.obstacle_color):
                is_ok = False
                return is_ok
        return is_ok


    def _operate_moving(self):
        keys = pygame.key.get_pressed()
        step_size = 5  # Move by 10 pixels per key press
        if keys[pygame.K_w] and self.robot_position[1] > self.robot_radius_pixels:
            next_position = (
                self.robot_position[0] + step_size * math.sin(self.robot_theta),
                self.robot_position[1] - step_size * math.cos(self.robot_theta),
            )
            if next_position[1] > 0:
                if self._check_surrounding_colors(next_position):
                    self.robot_position = next_position  # Move robot left
                    self.reset_window()
        if keys[pygame.K_s] and self.robot_position[1] < self.map_h - self.robot_radius_pixels:
            next_position = (
                self.robot_position[0] - step_size * math.sin(self.robot_theta),
                self.robot_position[1] + step_size * math.cos(self.robot_theta),
            )
            if next_position[1] < self.map_h:
                if self._check_surrounding_colors(next_position):
                    self.robot_position = next_position  # Move robot left
                    self.reset_window()
        if keys[pygame.K_e]:
            self.robot_theta += 0.1  # Rotate robot clockwise
            if self.robot_theta > 2 * math.pi:
                self.robot_theta -= 2 * math.pi
        if keys[pygame.K_q]:
            self.robot_theta -= 0.1  # Rotate robot counterclockwise
            if self.robot_theta < 0:
                self.robot_theta += 2 * math.pi


    def randomize_position(self, spawn_area):
        """Generates a random position within the environment.
        area: [xmin, xmax, ymin, ymax] defines the area where the robot can be placed.
        """
        if spawn_area is None:
            spawn_area_xmin = self.map_center_x - self.map_w // 2 + self.robot_radius_pixels
            spawn_area_xmax = self.map_center_x + self.map_w // 2 - self.robot_radius_pixels
            spawn_area_ymin = self.map_center_y - self.map_h // 2 + self.robot_radius_pixels
            spawn_area_ymax = self.map_center_y + self.map_h // 2 - self.robot_radius_pixels
        else:
            spawn_area_xmin, spawn_area_xmax, spawn_area_ymin, spawn_area_ymax = spawn_area

        while True:
            buffer = 10
            x = self.random_np.integers(
                spawn_area_xmin + buffer,
                spawn_area_xmax - buffer,
            )
            y = self.random_np.integers(
                spawn_area_ymin + buffer,
                spawn_area_ymax - buffer,
            )
            if self._check_surrounding_colors((x, y)):
                return (x, y)  # Return the random position


    def create_start_and_target_position(self, start_spawn_area, target_spawn_area):
        """Creates random start and target positions for the robot."""
        if self.map_type == "trap_shaped":
            T_dt_m = self.T_dt_m_trap_shaped
        elif self.map_type == "zigzag":
            T_dt_m = self.T_dt_m_zigzag
        elif self.map_type == "random":
            T_dt_m = self.T_dt_m_random
        elif self.map_type == "empty":
            T_dt_m = self.T_dt_m_empty
        while True:
            self.robot_position = self.randomize_position(start_spawn_area)
            self.target_position = self.randomize_position(target_spawn_area)
            distance_btwn_robot_and_target = math.sqrt(
                (self.target_position[0] - self.robot_position[0]) ** 2
                + (self.target_position[1] - self.robot_position[1]) ** 2
            )
            if distance_btwn_robot_and_target > T_dt_m * self.pixels_per_meter:
                print(
                    f"Distance between robot and target: {distance_btwn_robot_and_target // self.pixels_per_meter} m"
                )
                break


    def _calculate_spawan_area_in_zigzag_map(self):
        """Calculates the spawn area for the robot in the zigzag map."""
        print("l_passage:", self.l_passage)
        start_spawn_area = [
            self.map_center_x
            - self.l_passage // 2 * self.pixels_per_meter
            + self.robot_radius_pixels,
            self.map_center_x
            + self.l_passage // 2 * self.pixels_per_meter
            - self.robot_radius_pixels,
            self.map_h - int(self.divided_space_width * 0.85) + self.robot_radius_pixels,
            self.map_h - int(self.divided_space_width * 0.15) - self.robot_radius_pixels,
        ]
        target_spawn_area = [
            self.map_center_x
            - self.l_passage // 2 * self.pixels_per_meter
            + self.robot_radius_pixels,
            self.map_center_x
            + self.l_passage // 2 * self.pixels_per_meter
            - self.robot_radius_pixels,
            0 + int(self.divided_space_width * 0.15) + self.robot_radius_pixels,
            0 + int(self.divided_space_width * 0.85) - self.robot_radius_pixels,
        ]
        return start_spawn_area, target_spawn_area


    def initialize_map_parameters(self):
        start_spawn_area = None  # [xmin, xmax, ymin, ymax]
        target_spawn_area = None  # [xmin, xmax, ymin, ymax]

        if self.map_type == "trap_shaped":
            start_spawn_area = [
                self.map_center_x - int(self.map_w * 0.4) + self.robot_radius_pixels,
                self.map_center_x + int(self.map_w * 0.4) - self.robot_radius_pixels,
                self.map_center_y + int(self.map_h * 0.35) + self.robot_radius_pixels,
                self.map_center_y + int(self.map_h * 0.5) - self.robot_radius_pixels,
            ]
            target_spawn_area = [
                self.map_center_x - int(self.map_w * 0.4) + self.robot_radius_pixels,
                self.map_center_x + int(self.map_w * 0.4) - self.robot_radius_pixels,
                self.map_center_y - int(self.map_h * 0.5) + self.robot_radius_pixels,
                self.map_center_y - int(self.map_h * 0.35) - self.robot_radius_pixels,
            ]
        elif self.map_type == "zigzag":
            start_spawn_area, target_spawn_area = self._calculate_spawan_area_in_zigzag_map()
        elif self.map_type == "random":
            start_spawn_area = [
                self.map_center_x - self.map_w // 2 + self.robot_radius_pixels,
                self.map_center_x + self.map_w // 2 - self.robot_radius_pixels,
                self.map_center_y - self.map_h // 2 + self.robot_radius_pixels,
                self.map_center_y + self.map_h // 2 - self.robot_radius_pixels,
            ]
            target_spawn_area = start_spawn_area
        elif self.map_type == "empty":
            start_spawn_area = [
                self.map_center_x - self.map_w // 2 + self.robot_radius_pixels,
                self.map_center_x + self.map_w // 2 - self.robot_radius_pixels,
                self.map_center_y - self.map_h // 2 + self.robot_radius_pixels,
                self.map_center_y + self.map_h // 2 - self.robot_radius_pixels,
            ]
            target_spawn_area = start_spawn_area
        return start_spawn_area, target_spawn_area

    def create_map(self):
        """Creates the environment and initializes the map."""
        if self.map_type == "trap_shaped":
            self._create_trap_shaped_map()
        elif self.map_type == "zigzag":
            self._create_zigzag_map()
        elif self.map_type == "random":
            self._create_random_map()
        elif self.map_type == "empty":
            self._create_empty_map()



def main():
    pygame.init()
    rclpy.init()

    env = Environment()

    map_type_list = ["trap_shaped", "zigzag", "random", "empty"]
    env.map_type = "empty"  # ["trap_shaped", "zigzag", "random", "empty"]
    env.create_map()
    start_spawn_area, target_spawn_area = env.initialize_map_parameters()
    env.reset_window()
    print("Environment created with obstacles.")
    running = True
    env.create_start_and_target_position(start_spawn_area, target_spawn_area)
    # self.robot_position, self.target_position = create_start_and_target_position(
    #     env, start_spawn_area, target_spawn_area
    # )

    print((env.map_center_x, env.map_center_y))
    while running:
        rclpy.spin_once(env, timeout_sec=0.01)

        env.reset_window()  # Reset the environment window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_c:
                    # Change the map type
                    map_type_index = map_type_list.index(env.map_type)
                    env.map_type = map_type_list[
                        (map_type_index + 1) % len(map_type_list)
                    ]
                    print(f"Map type changed to: {env.map_type}")

                    env.reset_environment()
                    env.create_map()
                    start_spawn_area, target_spawn_area = env.initialize_map_parameters()
                    env.create_start_and_target_position(
                        start_spawn_area, target_spawn_area
                    )
                    # self.robot_position, self.target_position = (
                    #     create_start_and_target_position(
                    #         env, start_spawn_area, target_spawn_area
                    #     )
                    # )
                if event.key == pygame.K_SPACE:
                    env.reset_environment()  # Reset the environment
                    env.create_map()
                    start_spawn_area, target_spawn_area = env.initialize_map_parameters()
                    env.create_start_and_target_position(
                        start_spawn_area, target_spawn_area
                    )
                    # self.robot_position, self.target_position = (
                    #     create_start_and_target_position(
                    #         env, start_spawn_area, target_spawn_area
                    #     )
                    # )

        env._operate_moving()  # Handle robot movement
        # self.robot_position = _operate_moving(
        #     self.robot_position, env
        # )  # Handle robot movement

        sensor = sensors.LiDARSensor(
            env.lidar_range_pixels, env.map, uncertainty=(0, 0)
        )
        sensor.position = env.robot_position
        sensor.theta = env.robot_theta
        sensor_data = sensor.sense_obstacles(
            env.map, env.obstacle_color
        )  # Step 3: Sense. Stores sensed data array into sensor_data
        # env.dataStorage(sensor_data)  # Step 4: point cloud data storage
        env.show_sensorData(sensor_data)

        point_cloud_msg = sensor.convert_to_pointcloud2(
            sensor_data,
            time_stamp=env.get_clock().now().to_msg(),
            pixels_per_meter=env.pixels_per_meter,
        )

        # env.get_logger().info(
        #     f"Publishing a point cloud with {len(point_cloud_msg)} points."
        # )
        env.publisher_.publish(point_cloud_msg)
        env.publish_tf()

        # ウィンドウを更新
        pygame.display.flip()

    # 4. 終了処理
    pygame.quit()


if __name__ == "__main__":
    main()
