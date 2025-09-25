# Main simulation environment code
# It uses pygame library to render windows and handle user inputs.

import math
import random
import os

import yaml
import pygame
import numpy as np

from . import sensor, utils
from . import map_creator as mc


class SimCoreEnvironment:

    def __init__(self, render_mode=None, config_path=None):

        # ===== Load config file =====
        # For ROS2 package
        # TODO: Make it work without sourcing the ROS2 workspace
        if config_path is None:
            from ament_index_python.packages import get_package_share_directory

            config_path = os.path.join(
                get_package_share_directory("simulation_core"),
                "config",
                "simulator_config.yaml",
            )
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # ===== import parameters from config file =====
        self.WINDOW_NAME = config["window"]["window_title"]
        self.BACKGROUND_COLOR = config["window"]["background_color"]
        self.SCREEN_W = config["window"]["size"]
        self.SCREEN_H = config["window"]["size"]

        self.ROBOT_RADIUS_M = config["robot"]["radius_m"]
        self.ROBOT_COLOR = config["robot"]["robot_color"]
        self.TARGET_COLOR = config["robot"]["target_color"]
        self.T_DT_M_TRAP_SHAPED = config["robot"]["T_dt_m_trap_shaped"]
        self.T_DT_M_ZIGZAG = config["robot"]["T_dt_m_zigzag"]
        self.T_DT_M_RANDOM = config["robot"]["T_dt_m_random"]
        self.T_DT_M_EMPTY = config["robot"]["T_dt_m_empty"]
        self.V_MAX = config["robot"]["v_max"]
        self.OMEGA_MAX = config["robot"]["omega_max"]
        self.A_MAX = config["robot"]["a_max"]
        self.ALPHA_MAX = config["robot"]["alpha_max"]
        self.MAX_D = config["robot"]["max_d"]
        self.MAX_THETA = config["robot"]["max_theta"]
        self.NUM_CHECK_POINTS = config["robot"]["num_check_points"]

        self.LIDAR_RANGE_M = config["lidar"]["range_m"]
        self.LIDAR_LINE_COLOR = config["lidar"]["line_color"]
        self.LIDAR_POINTS_COLOR = config["lidar"]["points_color"]
        self.LIDAR_RESOLUTION = config["lidar"]["resolution"]

        self.OBSTACLE_COLOR = config["obstacles"]["color"]
        self.T_DO_M = config["obstacles"]["T_do_m"]
        self.map_type = config["obstacles"]["default_map_type"]
        self.n_obstacles = config["obstacles"]["count"]
        self.min_obstacle_size_m = config["obstacles"]["min_size_m"]
        self.max_obstacle_size_m = config["obstacles"]["max_size_m"]
        self.l_passage = config["obstacles"]["l_passage"]

        self.K8 = config["simulation"]["k8"]
        self.K9 = config["simulation"]["k9"]

        # initialize random number generator with seed 0
        self.np_random = np.random.default_rng(0)

        #! Prerequisite: window_w = max lidar range * 2
        self.PIXELS_PER_METER = self.SCREEN_W / (2 * self.LIDAR_RANGE_M)

        # ===== Post processing to convert parameters from meters to pixels =====
        self.ROBOT_RADIUS_PIXELS = self.ROBOT_RADIUS_M * self.PIXELS_PER_METER
        self.LIDAR_RANGE_PIXELS = self.LIDAR_RANGE_M * self.PIXELS_PER_METER

        # ===== Initialize rendering screen =====
        self.render_mode = render_mode
        pygame.init()
        if self.render_mode == "human":
            pygame.display.init()
            pygame.display.set_caption(self.WINDOW_NAME)
            self.screen = pygame.display.set_mode((self.SCREEN_W, self.SCREEN_H))
        else:
            self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))

        self.screen.fill(self.BACKGROUND_COLOR)
        self.SCREEN_CENTER_X = self.SCREEN_W // 2
        self.SCREEN_CENTER_Y = self.SCREEN_H // 2

        self.obstacle_layer = pygame.Surface(
            (self.SCREEN_W, self.SCREEN_H), pygame.SRCALPHA
        )
        self.robot_and_lidar_layer = pygame.Surface(
            (self.SCREEN_W, self.SCREEN_H), pygame.SRCALPHA
        )
        # Robot's orientation in radians
        # 0 radians is facing up (north), and increases clockwise
        self.robot_position = (self.SCREEN_CENTER_X, self.SCREEN_CENTER_Y)
        self.target_position = (self.SCREEN_CENTER_X, self.SCREEN_CENTER_Y)
        self.robot_theta = 0.0
        self.target_theta = 0.0

        # ===== Initialize variables for different map types for RL =====
        # These parameters affect map difficulty
        self.st = 0  # number of success in the trap-shaped map
        self.sz = 0  # number of success in the zigzag-shaped map
        self.pt = 0
        self.pz = 0

        # ===== Initialize LiDAR sensor module =====
        self.sensor = sensor.LiDARSensor(
            Range=self.LIDAR_RANGE_M,
            uncertainty=(0, 0),
            pixels_per_meter=self.PIXELS_PER_METER,
            lidar_resolution=self.LIDAR_RESOLUTION,
            screen_w=self.SCREEN_W,
            screen_h=self.SCREEN_H,
        )

        # ===== Initialize map creator module =====
        self.map_creator = mc.MapCreator(
            map=self.screen,
            map_size=(self.SCREEN_W, self.SCREEN_H),
            obstacle_color=self.OBSTACLE_COLOR,
            n_obstacles=self.n_obstacles,
            pixels_per_meter=self.PIXELS_PER_METER,
            np_random=self.np_random,
            T_do_m=self.T_DO_M,
            l_passage=self.l_passage,
            robot_radius_pixels=self.ROBOT_RADIUS_PIXELS,
        )

        # ===== Other variables =====
        self.running = True  # ROS2で使ってる
        self.closest_obstacle_info = None
        self.sensor_data = None
        self.path = []

    # ===== Visualization functions =====

    def visualize_search_point(self, position, color=(255, 0, 0)):
        # The number of points to check around the robot's circumference.
        # This should match the value in is_collision.
        for i in range(self.NUM_CHECK_POINTS):
            # maybe clockwise
            angle = 2 * math.pi * i / self.NUM_CHECK_POINTS
            # Calculate the position of a point on the robot's circumference
            check_x = int(position[0] + self.ROBOT_RADIUS_PIXELS * math.cos(angle))
            check_y = int(position[1] + self.ROBOT_RADIUS_PIXELS * math.sin(angle))

            # Ensure the check point is within the map boundaries before drawing
            if 0 <= check_x < self.SCREEN_W and 0 <= check_y < self.SCREEN_H:
                pygame.draw.circle(
                    self.robot_and_lidar_layer,
                    color,  # Red color for visualization
                    (check_x, check_y),
                    1,  # radius of the visualized point
                )

    def visualize_robot_theta(self, position, color=(0, 0, 255)):
        font = pygame.font.SysFont("arial", 16)
        theta_text = font.render(
            f"{math.degrees(self.robot_theta):.1f}(deg)", True, color
        )
        self.robot_and_lidar_layer.blit(
            theta_text,
            (
                position[0] - theta_text.get_width() // 2,
                position[1] - self.ROBOT_RADIUS_PIXELS - theta_text.get_height() - 5,
            ),
        )

    def visualize_goal_info(self, position, color=(255, 0, 0)):
        distance_to_target, angle_to_target = self.get_goal_info()
        goal_angle = self.robot_theta + angle_to_target
        font = pygame.font.SysFont("arial", 16)

        dot_x = int(position[0] - self.ROBOT_RADIUS_PIXELS * 2 * math.sin(goal_angle))
        dot_y = int(position[1] - self.ROBOT_RADIUS_PIXELS * 2 * math.cos(goal_angle))

        pygame.draw.circle(
            self.robot_and_lidar_layer,
            color,  # Red color for visualization
            (dot_x, dot_y),
            2,
        )
        angle_to_target_deg = math.degrees(angle_to_target)
        sign = ""
        if angle_to_target_deg > 0:
            sign = "+"
        elif angle_to_target_deg < 0:
            sign = "-"
        distance_text = font.render(
            f"{distance_to_target:.2f}(m), {sign}{angle_to_target_deg:.2f}(deg)",
            True,
            color,
        )
        self.robot_and_lidar_layer.blit(
            distance_text,
            (
                dot_x - distance_text.get_width() - 5,
                dot_y - distance_text.get_height() - 5,
            ),
        )

    def visualize_closest_obstacle_info(self, position, color=(60, 180, 110)):
        distance_to_obstacle, angle_to_obstacle = self.get_closest_obstacle_info()
        font = pygame.font.SysFont("arial", 16)
        # Convert robot-relative angle to a global angle for drawing
        obstacle_angle = self.robot_theta - math.pi / 2 - angle_to_obstacle

        dot_x = int(
            position[0] - self.ROBOT_RADIUS_PIXELS * 1.5 * math.sin(obstacle_angle)
        )
        dot_y = int(
            position[1] - self.ROBOT_RADIUS_PIXELS * 1.5 * math.cos(obstacle_angle)
        )

        pygame.draw.circle(
            self.robot_and_lidar_layer,
            color,  # Red color for visualization
            (dot_x, dot_y),
            2,
        )
        # The angle is already relative to the robot, just convert to degrees for display
        relative_angle = obstacle_angle - self.robot_theta
        normalized_relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi
        angle_to_obstacle_deg = math.degrees(normalized_relative_angle)
        sign = ""
        if angle_to_obstacle_deg > 0:
            sign = "+"
        elif angle_to_obstacle_deg < 0:
            sign = "-"
        distance_text = font.render(
            f"{distance_to_obstacle:.2f}(m), {sign}{abs(angle_to_obstacle_deg):.2f}(deg)",
            True,
            color,
        )
        self.robot_and_lidar_layer.blit(
            distance_text,
            (
                dot_x - distance_text.get_width() - 5,
                dot_y - distance_text.get_height() - 5,
            ),
        )

    def show_robot_and_lidar(self, sensor_data=None):
        if sensor_data != False:  # if sensor_data exists, then start for loop
            for element in sensor_data:
                obstacle_angle = self.robot_theta - math.pi / 2 - element[1]
                point = utils.AD2pos(
                    element[0],
                    obstacle_angle,
                    self.robot_position,
                    self.PIXELS_PER_METER,
                )  # Input: sensor_data = [Distance, Angle]. Output: (x,y) coordinate of objects/walls.
                pygame.draw.line(
                    self.robot_and_lidar_layer,
                    self.LIDAR_LINE_COLOR,
                    (int(self.robot_position[0]), int(self.robot_position[1])),
                    (int(point[0]), int(point[1])),
                    1,
                )
                pygame.draw.circle(
                    self.robot_and_lidar_layer,
                    self.LIDAR_POINTS_COLOR,
                    (int(point[0]), int(point[1])),
                    2,
                )

            robot_frame_radius = self.ROBOT_RADIUS_M * self.PIXELS_PER_METER
            robot_radius = self.ROBOT_RADIUS_M * 0.8 * self.PIXELS_PER_METER

            self.visualize_search_point(self.target_position, self.TARGET_COLOR)
            pygame.draw.circle(
                self.robot_and_lidar_layer,
                self.TARGET_COLOR,
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
                self.robot_and_lidar_layer,
                (255, 0, 0),  # Red color for the target's head
                self.target_position,
                target_head_position,
                2,  # Draw the target's head
            )

            self.visualize_search_point(self.robot_position, self.ROBOT_COLOR)
            self.visualize_goal_info(self.robot_position)
            self.visualize_closest_obstacle_info(self.robot_position)
            # self.visualize_robot_theta(self.robot_position)
            pygame.draw.circle(
                self.robot_and_lidar_layer,
                self.ROBOT_COLOR,
                self.robot_position,
                robot_radius,
            )
            robot_head_position = (
                self.robot_position[0]
                - robot_frame_radius
                * math.sin(self.robot_theta),  # 反時計回りを正とするため符号を反転
                self.robot_position[1]
                - robot_frame_radius * math.cos(self.robot_theta),  # Y軸は不変
            )
            pygame.draw.line(
                self.robot_and_lidar_layer,
                (255, 0, 0),  # Red color for the robot's head
                self.robot_position,
                robot_head_position,
                2,  # Draw the robot's head
            )

            # self.visualize_search_point(self.robot_position)
            # self.visualize_search_point(self.target_position)
            self.screen.blit(
                self.robot_and_lidar_layer, (0, 0)
            )  # Overlay lidar sensor layer on the map

    def render_obstacle_and_robots(self, map_type="empty"):
        screen_center_xy = (self.SCREEN_CENTER_X, self.SCREEN_CENTER_Y)
        screen_size_wh = (self.SCREEN_W, self.SCREEN_H)
        self.map_creator.create_map(
            map_type, self.obstacle_layer, pt=self.pt, pz=self.pz
        )
        start_spawn_area, target_spawn_area = (
            self.map_creator.initialize_map_parameters(
                map_type, screen_center_xy, screen_size_wh, self.ROBOT_RADIUS_PIXELS
            )
        )
        self.create_start_and_target_position(
            map_type, start_spawn_area, target_spawn_area
        )
        # ロボットの向きも初期化する
        self.robot_theta = 0.0
        self.target_theta = 0.0

    def render_robot_and_lidar(self):
        """Draws the robot and the last sensor data to the screen."""
        self.clear_robot_and_lidar_layer()  # Reset the lidar line while remaining the map and obstacles
        self.show_robot_and_lidar(self.sensor_data)

    # ===== Reset functions =====

    def reset(self, map_type="empty"):
        self.map_type = map_type

        self.clear_all_layers()
        self.render_obstacle_and_robots(
            map_type, pt=self.pt, pz=self.pz
        )  # Draw obstacles, robot, and target

    def clear_all_layers(self):
        """Clear the entire layers including main screen, obstacle layer, and lidar sensor layer."""
        self.screen.fill(self.BACKGROUND_COLOR)  # Clear the map
        # Surfaces are created in RGBA mode to see all layers.
        self.obstacle_layer = pygame.Surface(
            (self.SCREEN_W, self.SCREEN_H), pygame.SRCALPHA
        )
        self.robot_and_lidar_layer = pygame.Surface(
            (self.SCREEN_W, self.SCREEN_H), pygame.SRCALPHA
        )

    def clear_robot_and_lidar_layer(self):
        """Clear only the robot and lidar sensor layer while remaining the obstacles to update robot position and lidar scan data."""
        self.screen.fill(self.BACKGROUND_COLOR)
        self.screen.blit(
            self.obstacle_layer, (0, 0)
        )  # Draw the obstacle layer onto the map
        self.robot_and_lidar_layer = pygame.Surface(
            (self.SCREEN_W, self.SCREEN_H), pygame.SRCALPHA
        )

    def randomize_position(self, spawn_area):
        """Generates a random position within the environment.
        area: [xmin, xmax, ymin, ymax] defines the area where the robot can be placed.
        """
        if spawn_area is None:
            spawn_area_xmin = (
                self.SCREEN_CENTER_X - self.SCREEN_W // 2 + self.ROBOT_RADIUS_PIXELS
            )
            spawn_area_xmax = (
                self.SCREEN_CENTER_X + self.SCREEN_W // 2 - self.ROBOT_RADIUS_PIXELS
            )
            spawn_area_ymin = (
                self.SCREEN_CENTER_Y - self.SCREEN_H // 2 + self.ROBOT_RADIUS_PIXELS
            )
            spawn_area_ymax = (
                self.SCREEN_CENTER_Y + self.SCREEN_H // 2 - self.ROBOT_RADIUS_PIXELS
            )
        else:
            spawn_area_xmin, spawn_area_xmax, spawn_area_ymin, spawn_area_ymax = (
                spawn_area
            )

        while True:
            buffer = 10
            x = self.np_random.integers(
                spawn_area_xmin + buffer,
                spawn_area_xmax - buffer,
            )
            y = self.np_random.integers(
                spawn_area_ymin + buffer,
                spawn_area_ymax - buffer,
            )
            if not self.is_collision((x, y)):
                return (x, y)  # Return the random position

    def create_start_and_target_position(
        self, map_type, start_spawn_area, target_spawn_area
    ):
        """Creates random start and target positions for the robot."""
        if map_type == "trap_shaped":
            T_dt_m = self.T_DT_M_TRAP_SHAPED
        elif map_type == "zigzag":
            T_dt_m = self.T_DT_M_ZIGZAG
        elif map_type == "random":
            T_dt_m = self.T_DT_M_RANDOM
        elif map_type == "empty":
            T_dt_m = self.T_DT_M_EMPTY
        while True:
            self.robot_position = self.randomize_position(start_spawn_area)
            self.target_position = self.randomize_position(target_spawn_area)
            distance_btwn_robot_and_target = math.sqrt(
                (self.target_position[0] - self.robot_position[0]) ** 2
                + (self.target_position[1] - self.robot_position[1]) ** 2
            )
            if distance_btwn_robot_and_target > T_dt_m * self.PIXELS_PER_METER:
                # print(
                #     f"Distance between robot and target: {distance_btwn_robot_and_target // self.PIXELS_PER_METER} m"
                # )
                break

    # ===== Detection functions =====

    def is_collision(self, position):
        """Checks the surrounding colors of a position in the environment."""
        for i in range(self.NUM_CHECK_POINTS):
            angle = 2 * math.pi * i / self.NUM_CHECK_POINTS
            # Calculate the position of a point on the robot's circumference
            check_x = int(position[0] + self.ROBOT_RADIUS_PIXELS * math.cos(angle))
            check_y = int(position[1] + self.ROBOT_RADIUS_PIXELS * math.sin(angle))

            # Ensure the check point is within the map boundaries
            if not (0 <= check_x < self.SCREEN_W and 0 <= check_y < self.SCREEN_H):
                # Part of the robot is outside the map, which is a collision
                # print("Collision with wall detected: out of bounds")
                return True

            # Get the color of the pixel at the check point
            try:
                pixel_color = self.obstacle_layer.get_at((check_x, check_y))
                # print("pixel_color:", pixel_color)
            except IndexError:
                # This can happen if the coordinates are somehow still out of bounds
                # print("Collision with wall detected: index error")
                return True

            # Check if the pixel color matches the obstacle color.
            obstacle_color_rgb = utils.hex_to_rgb(self.OBSTACLE_COLOR)
            if (pixel_color[0], pixel_color[1], pixel_color[2]) == obstacle_color_rgb:
                # print("Collision with obstacle detected")
                return True
        return False

        """Check collision by checking if the closest lidar point is in the robot's safe radius."""
        # remain bugs
        # closest_point, _ = self.sensor.get_closest_obstacle_info(self.sensor_data)
        # if closest_point is not float('inf') and closest_point < self.ROBOT_RADIUS_M:
        #     print("collision detected!")
        #     return True
        # else:
        #     return False

    def is_target_reached(self):
        distance_to_target = math.sqrt(
            (self.target_position[0] - self.robot_position[0]) ** 2
            + (self.target_position[1] - self.robot_position[1]) ** 2
        )
        return distance_to_target < self.ROBOT_RADIUS_PIXELS

    # ===== Data functions =====

    def update_sensors(self):
        """Performs a LiDAR scan and updates sensor-related state."""
        self.sensor_data = self.lidar_scan()
        self.closest_obstacle_info = self.sensor.get_closest_obstacle_info(
            self.sensor_data
        )

    def update_path(self, new_position, new_theta):
        self.robot_position = new_position
        self.robot_theta = new_theta

        self.path.append([new_position, new_theta])

    def lidar_scan(self):
        self.sensor.position = self.robot_position
        self.sensor.theta = self.robot_theta
        sensor_data = self.sensor.sense_obstacles(self.screen, self.OBSTACLE_COLOR)
        return sensor_data

    def get_goal_info(self):
        distance_to_target = math.sqrt(
            (self.target_position[0] - self.robot_position[0]) ** 2
            + (self.target_position[1] - self.robot_position[1]) ** 2
        )

        # 1. グローバル座標系でのターゲットへの角度を計算
        #    Pygame座標系(Y軸下向き)なので、yの差はそのまま使用
        #    math.atan2(y, x) はX軸正方向(右)を0 radとし、反時計回りを正とする。
        dx = self.target_position[0] - self.robot_position[0]
        dy = -(
            self.target_position[1] - self.robot_position[1]
        )  # Y軸を数学座標系(上向きが正)に合わせる
        global_angle_to_target = math.atan2(dy, dx)  # 数学座標系での角度
        # print("global_angle_to_target:", global_angle_to_target)

        # 2. ロボットの向きをatan2の座標系に変換
        #    - self.robot_theta の基準: Y軸負方向(上) = 0 rad, 反時計回りが正
        #    - 数学座標系の基準      : X軸正方向(右) = 0 rad, 反時計回りが正
        #    ロボットの向き(上=0)は、数学座標系では +pi/2 rad に相当する。
        #    したがって、ロボットの向きの基準を数学座標系に合わせるには、
        #    現在の角度に +pi/2 を加える必要がある。
        robot_global_angle = self.robot_theta + math.pi / 2

        # 3. ロボットから見たターゲットへの相対角度を計算
        #    両方とも数学座標系に揃えたので、単純な引き算で相対角度が求まる。
        #    結果は反時計回りが正となる。
        angle_to_target = global_angle_to_target - robot_global_angle

        # 4. 角度を-πからπの範囲に正規化
        angle_to_target = (angle_to_target + math.pi) % (2 * math.pi) - math.pi

        distance_to_target_in_m = distance_to_target / self.PIXELS_PER_METER

        return distance_to_target_in_m, angle_to_target

    def get_closest_obstacle_info(self):
        # return closest obstacle distance in meters
        if self.closest_obstacle_info is None:
            return float("inf"), 0.0
        closest_obstacle_info = [
            self.closest_obstacle_info[0],
            self.closest_obstacle_info[1],
        ]
        return closest_obstacle_info

    # ===== Other functions =====

    def _operate_moving(self):
        keys = pygame.key.get_pressed()
        step_size = 5  # Move by 10 pixels per key press
        if keys[pygame.K_w] and self.robot_position[1] > self.ROBOT_RADIUS_PIXELS:
            next_position = (
                self.robot_position[0] - step_size * math.sin(self.robot_theta),
                self.robot_position[1] - step_size * math.cos(self.robot_theta),
            )
            if next_position[1] > 0:
                if not self.is_collision(next_position):
                    self.robot_position = next_position  # Move robot left
                    self.clear_robot_and_lidar_layer()
        if (
            keys[pygame.K_s]
            and self.robot_position[1] < self.SCREEN_H - self.ROBOT_RADIUS_PIXELS
        ):
            next_position = (
                self.robot_position[0] + step_size * math.sin(self.robot_theta),
                self.robot_position[1] + step_size * math.cos(self.robot_theta),
            )
            if next_position[1] < self.SCREEN_H:
                if not self.is_collision(next_position):
                    self.robot_position = next_position  # Move robot left
                    self.clear_robot_and_lidar_layer()
        if keys[pygame.K_e]:
            self.robot_theta -= 0.1  # Rotate robot clockwise (negative omega)
            self.robot_theta = (self.robot_theta + math.pi) % (2 * math.pi) - math.pi
        if keys[pygame.K_q]:
            self.robot_theta += 0.1  # Rotate robot counterclockwise (positive omega)
            self.robot_theta = (self.robot_theta + math.pi) % (2 * math.pi) - math.pi

    def change_map_by_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                if event.key == pygame.K_c:
                    # Change the map type
                    next_map_type = utils.switch_map_type(
                        self.map_type, self.np_random, switch_mode="cycle"
                    )
                    self.map_type = next_map_type
                    self.reset(map_type=next_map_type)
                if event.key == pygame.K_SPACE:
                    self.reset(map_type=self.map_type)

    def count_up_st(self):
        self.st += 1

    def count_up_sz(self):
        self.sz += 1

    def update_pt(self):
        self.pt = min(self.st / self.K8, 1)

    def update_pz(self):
        self.pz = min(self.sz / self.K9, 1)

    def get_pt(self):
        return self.pt

    def get_pz(self):
        return self.pz
