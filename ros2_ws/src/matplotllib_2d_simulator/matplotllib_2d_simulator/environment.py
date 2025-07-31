"""
Defines the 2D environment, obstacles, start/goal

This class will procedurally generate and manage the state of the 2D world.

- `__init__(self, config)`:
    - Reads parameters from the config file.
    - Generates random obstacle positions and sizes (e.g., as circles or rectangles).
    - Randomly determines and stores the robot's start and goal positions.
- `get_obstacles(self)`:
    - Returns a list of obstacle objects for collision checking.
- `get_start_goal(self)`:
    - Returns the generated start and goal coordinates.
"""

import math
import pygame
import random
import yaml

import sensors


class buildEnvironment:

    def __init__(self):
        self.window_name = "2D LiDAR Simulation"
        pygame.display.set_caption(self.window_name)

        # Set parameters
        with open("../config/simulator_config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.background_color = config["map"]["background_color"]
        self.map_w, self.map_h = config["map"]["size"], config["map"]["size"]

        self.robot_radius_m = config["robot"]["radius_m"]
        self.robot_color = config["robot"]["color"]

        self.lidar_range_m = config["lidar"]["range_m"]
        self.lidar_line_color = config["lidar"]["line_color"]
        self.lidar_points_color = config["lidar"]["points_color"]

        self.T_do_m = config["obstacles"]["T_do_m"]
        self.n_obstacles = config["obstacles"]["count"]
        self.min_obstacle_size_m = config["obstacles"]["min_size_m"]
        self.max_obstacle_size_m = config["obstacles"]["max_size_m"]
        self.obstacle_color = config["obstacles"]["color"]
        # self.T_dt_m = config['obstacles']['T_dt_m']
        self.pixels_per_meter = self.map_w / (2 * self.lidar_range_m)

        self.robot_radius_pixels = self.robot_radius_m * self.pixels_per_meter
        self.lidar_range_pixels = self.lidar_range_m * self.pixels_per_meter

        self.map = pygame.display.set_mode((self.map_w, self.map_h))
        self.map.fill(self.background_color)

        self.pointCloud = (
            []
        )  # array containing coordinates of sensed objects (walls). Purely just (x,y) information.

        self.map_center_x = self.map_w // 2
        self.map_center_y = self.map_h // 2

        self.obstacle_layer = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
        self.lidar_sensor_layer = pygame.Surface(
            (self.map_w, self.map_h), pygame.SRCALPHA
        )
        self.robot_layer = pygame.Surface((self.map_w, self.map_h))

        # self.nothing_map = self.map.copy()

        # self.random_map = (
        #     self.create_random_map()
        # )  # Random map to visualize the point cloud

        # self.map.blit(self.random_map, (0, 0))

        # self.base_map = self.map.copy()
        # self.point_map = self.map.copy()  # Point map to visualize the point cloud

    # Key Concept: Helper Method that converts raw distance and angle data from sensor.py to cartesian coordinates
    # AD2pos means Angle, Distance to Position?

    # def reset_map(self):
    #     self.random_map = (
    #         self.create_random_map()
    #     )  # Random map to visualize the point cloud

    #     self.map.blit(self.random_map, (0, 0))

    def create_random_map(self):
        """Creates a random map with obstacles"""
        # map = self.nothing_map.copy()

        # self.obstacle_layer.fill((255, 255, 255))  # Fill the map with white background
        # Draw walls
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

            obstacle_cx = random.randint(
                self.map_center_x - obstacle_area_width // 2,
                self.map_center_x + obstacle_area_width // 2,
            )
            obstacle_cy = random.randint(
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
            # obstacle_width = random.randint(60, 120)
            # obstacle_height = random.randint(60, 120)
            shape_random = random.randint(0, 1)
            if shape_random == 0:
                # Draw a rectangle obstacle
                rectangular_l_m = 4 + random.uniform(0, 1)
                rectangular_w_m = 0.5 + random.uniform(0, 1) * 0.5
                rectangular_l_pixels = int(rectangular_l_m * self.pixels_per_meter)
                rectangular_w_pixels = int(rectangular_w_m * self.pixels_per_meter)
                if random.randint(0, 1) == 0:
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
                radius_m = 0.5 + random.uniform(0, 1)
                radius_pixels = int(radius_m * self.pixels_per_meter)
                pygame.draw.circle(
                    self.obstacle_layer,
                    self.obstacle_color,
                    (obstacle_cx, obstacle_cy),
                    radius_pixels,
                    width=0,
                )
            # Draw a center point for the obstacle
            pygame.draw.circle(
                self.obstacle_layer,
                (10, 10, 10),
                (obstacle_cx, obstacle_cy),
                1,
            )

    # Uses Distance and Angle from data[] in sensors.sensesObstacles() to obtain cartesian coordinates of walls for point cloud
    def AD2pos(self, distance, angle, robotPosition):
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
                point = self.AD2pos(
                    element[0], element[1]  # , element[2]
                )  # Input: data = [Distance, Angle, (x,y)]. Output: (x,y) coordinate of objects/walls.
                if (
                    point not in self.pointCloud
                ):  # Note: Point cloud should ONLY be the WALLS (x,y). If the wall has not already been sensed, then add to pointcloud.
                    self.pointCloud.append(point)  # [(x1,y1), (x2,y2), (x3,y3)]
            self.last_robot_position = (int(data[0][2][0]), int(data[0][2][1]))

    def show_sensorData(self, data=None, robot_position=None, goal_position=None):
        if data != False:  # if data exists, then start for loop
            for element in data:
                point = self.AD2pos(
                    element[0], element[1], robot_position
                )  # Input: data = [Distance, Angle, (x,y)]. Output: (x,y) coordinate of objects/walls.
                pygame.draw.line(
                    self.lidar_sensor_layer,
                    self.lidar_line_color,
                    (int(robot_position[0]), int(robot_position[1])),
                    (int(point[0]), int(point[1])),
                    1,
                )
                pygame.draw.circle(
                    self.lidar_sensor_layer,
                    self.lidar_points_color,
                    (int(point[0]), int(point[1])),
                    2,
                )
            # Draw the robot's position
            robot_frame_radius = self.robot_radius_m * self.pixels_per_meter
            robot_radius = self.robot_radius_m * 0.8 * self.pixels_per_meter
            pygame.draw.circle(
                self.lidar_sensor_layer,
                self.robot_color,
                robot_position,
                robot_frame_radius,
                width=2,  # Draw a circle around the robot
            )
            pygame.draw.circle(
                self.lidar_sensor_layer,
                self.robot_color,
                robot_position,
                robot_radius,
            )

            # Draw goal position
            # pygame.draw.circle(
            #     self.lidar_sensor_layer,
            #     (0, 255, 0),
            #     goal_position,
            #     self.robot_radius,
            # )
            env.map.blit(
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
        env.map.fill(env.background_color)
        env.map.blit(env.obstacle_layer, (0, 0))  # Draw the obstacle layer onto the map
        env.lidar_sensor_layer = pygame.Surface((env.map_w, env.map_h), pygame.SRCALPHA)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")  # 先頭の # を削除
    # 6文字 (RRGGBB) でなければエラー
    if len(hex_color) != 6:
        raise ValueError("カラーコードは6桁の16進数で指定してください。")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)


def operate_moving(robot_position, env):
    keys = pygame.key.get_pressed()
    step_size = 10  # Move by 10 pixels per key press
    if keys[pygame.K_a] and robot_position[0] > env.robot_radius_pixels:
        next_position = (
            robot_position[0] - int(env.robot_radius_pixels) - 10,
            robot_position[1],
        )
        if next_position[0] > 0:
            if env.map.get_at(next_position) != hex_to_rgb(env.obstacle_color):
                robot_position = (
                    robot_position[0] - 10,
                    robot_position[1],
                )  # Move robot left
                env.reset_window()
    if keys[pygame.K_d] and robot_position[0] < env.map_w - env.robot_radius_pixels:
        next_position = (
            robot_position[0] + int(env.robot_radius_pixels) + 10,
            robot_position[1],
        )
        if next_position[0] < env.map_w:
            if env.map.get_at(next_position) != hex_to_rgb(env.obstacle_color):
                robot_position = (
                    robot_position[0] + 10,
                    robot_position[1],
                )  # Move robot right
                env.reset_window()
    if keys[pygame.K_w] and robot_position[1] > env.robot_radius_pixels:
        next_position = (
            robot_position[0],
            robot_position[1] - int(env.robot_radius_pixels) - 10,
        )
        if next_position[1] > 0:
            if env.map.get_at(next_position) != hex_to_rgb(env.obstacle_color):
                robot_position = (
                    robot_position[0],
                    robot_position[1] - 10,
                )  # Move robot left
                env.reset_window()
    if keys[pygame.K_s] and robot_position[1] < env.map_h - env.robot_radius_pixels:
        next_position = (
            robot_position[0],
            robot_position[1] + int(env.robot_radius_pixels) + 10,
        )
        if next_position[1] < env.map_h:
            if env.map.get_at(next_position) != hex_to_rgb(env.obstacle_color):
                robot_position = (
                    robot_position[0],
                    robot_position[1] + 10,
                )  # Move robot left
                env.reset_window()
    return robot_position  # Return the updated robot position


def randomize_position(env):
    """Generates a random position within the environment."""
    while True:
        buffer = 10
        x = random.randint(
            env.map_center_x - env.map_w // 2 + env.robot_radius_pixels + buffer,
            env.map_center_x + env.map_w // 2 - env.robot_radius_pixels - buffer,
        )
        y = random.randint(
            env.map_center_y - env.map_h // 2 + env.robot_radius_pixels + buffer,
            env.map_center_y + env.map_h // 2 - env.robot_radius_pixels - buffer,
        )
        surrounding_colors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                surrounding_colors.append(
                    env.map.get_at(
                        (
                            int(x + i * env.robot_radius_pixels),
                            int(y + j * env.robot_radius_pixels),
                        )
                    )
                )
        is_ok = True
        for color in surrounding_colors:
            if color == hex_to_rgb(env.obstacle_color):
                is_ok = False
                break
        if is_ok:
            return (x, y)  # Return the random position


if __name__ == "__main__":
    # Example usage
    env = buildEnvironment()
    env.create_random_map()
    print("Environment created with obstacles.")
    running = True
    robot_position = randomize_position(env)
    # robot_position = (env.map_center_x, env.map_center_y)
    print((env.map_center_x, env.map_center_y))
    while running:

        env.reset_window()  # Reset the environment window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    env.reset_environment()  # Reset the environment
                    env.create_random_map()
                    env.map.blit(env.obstacle_layer, (0, 0))
                    robot_position = randomize_position(env)

        robot_position = operate_moving(robot_position, env)  # Handle robot movement

        laser = sensors.LaserSensor(env.lidar_range_pixels, env.map, uncertainty=(0, 0))
        laser.position = robot_position
        sensor_data = laser.sense_obstacles(
            env.map, env.obstacle_color
        )  # Step 3: Sense. Stores sensed data array into sensor_data
        # env.dataStorage(sensor_data)  # Step 4: point cloud data storage
        env.show_sensorData(sensor_data, robot_position)

        # ウィンドウを更新
        pygame.display.flip()

    # 4. 終了処理
    pygame.quit()
