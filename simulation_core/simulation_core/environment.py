# Main simulation environment code

import math
import random
import os

import yaml
import pygame
import numpy as np

from . import sensor, utils
from . import map_creator as mc


class SimCoreEnvironment:

    def __init__(self, config_path=None):
        self.window_name = "2D LiDAR Simulation"
        pygame.display.set_caption(self.window_name)

        # Set parameters
        if config_path is None:
            from ament_index_python.packages import get_package_share_directory
            config_path = os.path.join(
                get_package_share_directory("simulation_core"),
                "config",
                "simulator_config.yaml",
            )
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        self.np_random = np.random.default_rng(0) # random number generator with seed 0

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
        self.r_lidar = config["lidar"]["range_m"]
        self.resolution = config["lidar"]["resolution"]

        self.T_do_m = config["obstacles"]["T_do_m"]
        self.n_obstacles = config["obstacles"]["count"]
        self.min_obstacle_size_m = config["obstacles"]["min_size_m"]
        self.max_obstacle_size_m = config["obstacles"]["max_size_m"]
        self.obstacle_color = config["obstacles"]["color"]
        self.v_max = config["robot"]["v_max"]
        self.omega_max = config["robot"]["omega_max"]
        self.a_max = config["robot"]["a_max"]
        self.alpha_max = config["robot"]["alpha_max"]
        self.max_d = config["robot"]["max_d"]
        self.max_theta = config["robot"]["max_theta"]

        self.pixels_per_meter = self.map_w / (2 * self.lidar_range_m)

        self.robot_radius_pixels = self.robot_radius_m * self.pixels_per_meter
        self.lidar_range_pixels = self.lidar_range_m * self.pixels_per_meter

        self.map = pygame.display.set_mode((self.map_w, self.map_h))
        self.map.fill(self.background_color)

        self.map_center_x = self.map_w // 2
        self.map_center_y = self.map_h // 2

        self.obstacle_layer = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
        self.lidar_sensor_layer = pygame.Surface(
            (self.map_w, self.map_h), pygame.SRCALPHA
        )
        self.robot_layer = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
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
      
        self.sensor = sensor.LiDARSensor(
            self.lidar_range_m, self.map, uncertainty=(0, 0)
        )
        self.sensor.pixels_per_meter = self.pixels_per_meter
        self.sensor.num_scans = self.resolution
        self.running = True
        
        self.map_creator = mc.MapCreator(
            map=self.map,
            map_size=(self.map_w, self.map_h),
            obstacle_color=self.obstacle_color,
            n_obstacles=self.n_obstacles,
            pixels_per_meter=self.pixels_per_meter,
            np_random=self.np_random,
            T_do_m=self.T_do_m,
            s_t=self.s_t,
            k8=self.k8,
            s_z=self.s_z,
            k9=self.k9,
            l_passage=self.l_passage,
            divided_space_width=self.divided_space_width,
            robot_radius_pixels=self.robot_radius_pixels,
            )
        
        self.num_check_points = 16
        self.closest_obstacle_info = None
        self.sensor_data = None

    def visualize_search_point(self, position, color=(255, 0, 0)):
        # The number of points to check around the robot's circumference.
        # This should match the value in is_collision.
        for i in range(self.num_check_points):
            angle = 2 * math.pi * i / self.num_check_points
            # Calculate the position of a point on the robot's circumference
            check_x = int(position[0] + self.robot_radius_pixels * math.cos(angle))
            check_y = int(position[1] + self.robot_radius_pixels * math.sin(angle))

            # Ensure the check point is within the map boundaries before drawing
            if 0 <= check_x < self.map_w and 0 <= check_y < self.map_h:
                pygame.draw.circle(
                    self.lidar_sensor_layer,
                    color,  # Red color for visualization
                    (check_x, check_y),
                    1,  # radius of the visualized point
                )
                    
    def show_robot_and_lidar(self, sensor_data=None):
        if sensor_data != False:  # if sensor_data exists, then start for loop
            for element in sensor_data:
                point = utils.AD2pos(
                    element[0], element[1], self.robot_position, self.pixels_per_meter
                )  # Input: sensor_data = [Distance, Angle]. Output: (x,y) coordinate of objects/walls.
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
            # pygame.draw.circle(
            #     self.lidar_sensor_layer,
            #     self.target_color,
            #     self.target_position,
            #     robot_frame_radius,
            #     width=2,  # Draw a circle around the robot
            # )
            self.visualize_search_point(self.target_position, self.target_color)
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
            # pygame.draw.circle(
            #     self.lidar_sensor_layer,
            #     self.robot_color,
            #     self.robot_position,
            #     robot_frame_radius,
            #     width=2,  # Draw a circle around the robot
            # )
            self.visualize_search_point(self.robot_position, self.robot_color)
            pygame.draw.circle(
                self.lidar_sensor_layer,
                self.robot_color,
                self.robot_position,
                robot_radius,
            )
            robot_head_position = (
                self.robot_position[0]
                - robot_frame_radius * math.sin(self.robot_theta), # 反時計回りを正とするため符号を反転
                self.robot_position[1]
                - robot_frame_radius * math.cos(self.robot_theta), # Y軸は不変
            )
            pygame.draw.line(
                self.lidar_sensor_layer,
                (255, 0, 0),  # Red color for the robot's head
                self.robot_position,
                robot_head_position,
                2,  # Draw the robot's head
            )
            
            # self.visualize_search_point(self.robot_position)
            # self.visualize_search_point(self.target_position)
            self.map.blit(
                self.lidar_sensor_layer, (0, 0)
            )  # Overlay lidar sensor layer on the map

    def clear_environment(self):
        """Resets the environment by clearing the point cloud and reinitializing the map."""
        self.map.fill(self.background_color)  # Clear the map
        self.obstacle_layer = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
        self.lidar_sensor_layer = pygame.Surface(
            (self.map_w, self.map_h), pygame.SRCALPHA
        )

    def clear_lidar_layer(self):
        self.map.fill(self.background_color)
        self.map.blit(
            self.obstacle_layer, (0, 0)
        )  # Draw the obstacle layer onto the map
        self.lidar_sensor_layer = pygame.Surface(
            (self.map_w, self.map_h), pygame.SRCALPHA
        )

    def is_collision(self, position):
        """Checks the surrounding colors of a position in the environment."""
        for i in range(self.num_check_points):
            angle = 2 * math.pi * i / self.num_check_points
            # Calculate the position of a point on the robot's circumference
            check_x = int(position[0] + self.robot_radius_pixels * math.cos(angle))
            check_y = int(position[1] + self.robot_radius_pixels * math.sin(angle))

            # Ensure the check point is within the map boundaries
            if not (0 <= check_x < self.map_w and 0 <= check_y < self.map_h):
                # Part of the robot is outside the map, which is a collision
                return True

            # Get the color of the pixel at the check point
            try:
                pixel_color = self.obstacle_layer.get_at((check_x, check_y))
                # print("pixel_color:", pixel_color)
            except IndexError:
                # This can happen if the coordinates are somehow still out of bounds
                return True

            # Check if the pixel color matches the obstacle color.
            # We check the alpha channel (pixel_color[3]) to see if it's a non-transparent pixel,
            # which indicates an obstacle on the obstacle_layer.
            if pixel_color[3] > 0:
                return True
        return False
    
        """Check collision by checking closest lidar point is in the robot's safe radius."""
        # remain bugs
        # closest_point, _ = self.sensor.get_closest_obstacle_info(self.sensor_data)
        # if closest_point is not float('inf') and closest_point < self.robot_radius_m:
        #     print("collision detected!")
        #     return True
        # else:
        #     return False

    def is_target_reached(self):
        distance_to_target = math.sqrt(
            (self.target_position[0] - self.robot_position[0]) ** 2
            + (self.target_position[1] - self.robot_position[1]) ** 2
        )
        return distance_to_target < self.robot_radius_pixels

    def _operate_moving(self):
        keys = pygame.key.get_pressed()
        step_size = 5  # Move by 10 pixels per key press
        if keys[pygame.K_w] and self.robot_position[1] > self.robot_radius_pixels:
            next_position = (
                self.robot_position[0] - step_size * math.sin(self.robot_theta),
                self.robot_position[1] - step_size * math.cos(self.robot_theta),
            )
            if next_position[1] > 0:
                if not self.is_collision(next_position):
                    self.robot_position = next_position  # Move robot left
                    self.clear_lidar_layer()
        if keys[pygame.K_s] and self.robot_position[1] < self.map_h - self.robot_radius_pixels:
            next_position = (
                self.robot_position[0] + step_size * math.sin(self.robot_theta),
                self.robot_position[1] + step_size * math.cos(self.robot_theta),
            )
            if next_position[1] < self.map_h:
                if not self.is_collision(next_position):
                    self.robot_position = next_position  # Move robot left
                    self.clear_lidar_layer()
        if keys[pygame.K_e]:
            self.robot_theta -= 0.1  # Rotate robot clockwise (negative omega)
            self.robot_theta = (self.robot_theta + math.pi) % (2 * math.pi) - math.pi
        if keys[pygame.K_q]:
            self.robot_theta += 0.1  # Rotate robot counterclockwise (positive omega)
            self.robot_theta = (self.robot_theta + math.pi) % (2 * math.pi) - math.pi


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
            start_spawn_area, target_spawn_area = self.map_creator._calculate_spawan_area_in_zigzag_map()
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

    
    def switch_map_type(self, switch_mode="cycle"):
        # switch_mode: "cycle", "random"
        map_type_list = ["trap_shaped", "zigzag", "random", "empty"]
        
        if switch_mode == "cycle":
            map_type_index = map_type_list.index(self.map_type)
            self.map_type = map_type_list[
                (map_type_index + 1) % len(map_type_list)
            ]
            print(f"Map type changed to: {self.map_type}")
        elif switch_mode == "random":
            self.map_type = self.np_random.choice(map_type_list)
            print(f"Map type changed to: {self.map_type}")

    def render_obstacles(self):
        self.clear_environment()
        self.map_creator.create_map(self.map_type, self.obstacle_layer)
        start_spawn_area, target_spawn_area = self.initialize_map_parameters()
        self.create_start_and_target_position(
            start_spawn_area, target_spawn_area
        )
        # ロボットの向きも初期化する
        self.robot_theta = 0.0
        self.target_theta = 0.0

    def update_sensors(self):
        """Performs a LiDAR scan and updates sensor-related state."""
        self.sensor_data = self.lidar_scan()
        self.closest_obstacle_info = self.sensor.get_closest_obstacle_info(self.sensor_data)

    def render_robot_and_lidar(self):
        """Draws the robot and the last sensor data to the screen."""
        self.clear_lidar_layer()  # Reset the lidar line while remaining the map and obstacles
        self.show_robot_and_lidar(self.sensor_data)
    
    def lidar_scan(self):
        self.sensor.position = self.robot_position
        self.sensor.theta = self.robot_theta
        sensor_data = self.sensor.sense_obstacles(
            self.map, self.obstacle_color
        )
        return sensor_data
        
    
    def change_map_by_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                if event.key == pygame.K_c:
                    # Change the map type
                    self.switch_map_type(switch_mode="cycle")
                    self.render_obstacles()
                if event.key == pygame.K_SPACE:
                    self.render_obstacles()
                    
    def get_goal_info(self):
        distance_to_target = math.sqrt(
            (self.target_position[0] - self.robot_position[0]) ** 2
            + (self.target_position[1] - self.robot_position[1]) ** 2
        )

        # 1. グローバル座標系でのターゲットへの角度を計算
        #    Pygame座標系(Y軸下向き)なので、yの差はそのまま使用
        #    math.atan2(y, x) はX軸正方向(右)を0 radとし、反時計回りを正とする。
        dx = self.target_position[0] - self.robot_position[0]
        dy = -(self.target_position[1] - self.robot_position[1]) # Y軸を数学座標系(上向きが正)に合わせる
        global_angle_to_target = math.atan2(dy, dx) # 数学座標系での角度
        # print("global_angle_to_target:", global_angle_to_target)

        # 2. ロボットの向きをatan2の座標系に変換
        #    - self.robot_theta の基準: Y軸負方向(上) = 0 rad, 反時計回りが正
        #    - 数学座標系の基準      : X軸正方向(右) = 0 rad, 反時計回りが正
        #    ロボットの向き(上=0)は、数学座標系では +pi/2 rad に相当する。
        #    したがって、ロボットの向きの基準を数学座標系に合わせるには、
        #    現在の角度に +pi/2 を加える必要がある。
        robot_global_angle = self.robot_theta + math.pi/2

        # 3. ロボットから見たターゲットへの相対角度を計算
        #    両方とも数学座標系に揃えたので、単純な引き算で相対角度が求まる。
        #    結果は反時計回りが正となる。
        angle_to_target = global_angle_to_target - robot_global_angle

        # 4. 角度を-πからπの範囲に正規化
        angle_to_target = (angle_to_target + math.pi) % (2 * math.pi) - math.pi

        distance_to_target_in_m = distance_to_target / self.pixels_per_meter
        
        return distance_to_target_in_m, angle_to_target

    def get_closest_obstacle_info(self):
        # return closest obstacle distance in meters
        if self.closest_obstacle_info is None:
            return float('inf'), 0.0
        closest_obstacle_info = [self.closest_obstacle_info[0] / self.pixels_per_meter, self.closest_obstacle_info[1]]
        return closest_obstacle_info