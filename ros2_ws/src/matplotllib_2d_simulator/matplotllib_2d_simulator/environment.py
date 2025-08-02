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

    def create_base_map(self):
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

    def create_random_map(self):
        """Creates a random map with obstacles"""
        # Draw walls
        self.create_base_map()

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
            # pygame.draw.circle(
            #     self.obstacle_layer,
            #     (10, 10, 10),
            #     (obstacle_cx, obstacle_cy),
            #     1,
            # )

    def create_trap_shaped_map(self):
        """Creates a trap-shaped map with obstacles"""
        # Parameters
        p_t = min(self.s_t / self.k8, 1)
        w_wall = 0.2  # Width of the walls in meters
        l_horizontal = []
        for i in range(3):
            l_horizontal.append(1.5 * p_t + 3 + random.uniform(0, 1))
        l_vertical = []
        for i in range(3):
            l_vertical_left = 6 * p_t + 1 + 4 * random.uniform(0, 1)
            l_vertical_right = 6 * p_t + 1 + 4 * random.uniform(0, 1)
            if i == 0:
                if random.randint(0, 1) == 0:
                    l_vertical_left = env.map_h
            elif i == 2:
                if random.randint(0, 1) == 0:
                    l_vertical_right = env.map_h
            l_vertical.append([l_vertical_left, l_vertical_right])
        w_passage_0 = -1.6 * p_t + 2.5 + 0.2 * random.uniform(0, 1)
        w_passage_1 = (0.5 + 0.5 * random.uniform(0, 1)) * w_passage_0

        distance_btwn_H_shaped_obstacles_left = (
            l_horizontal[1] / 2 + w_passage_0 + w_wall * 2 + l_horizontal[0] / 2
        ) * self.pixels_per_meter
        distance_btwn_H_shaped_obstacles_right = (
            l_horizontal[1] / 2 + w_passage_1 + w_wall * 2 + l_horizontal[2] / 2
        ) * self.pixels_per_meter

        # Define the positions of the H-shaped obstacles
        positions_of_H_shaped_obstacles = [
            (
                env.map_center_x - distance_btwn_H_shaped_obstacles_left,
                env.map_center_y + random.uniform(-1, 1) * self.pixels_per_meter,
            ),
            (
                env.map_center_x,
                env.map_center_y + random.uniform(-1, 1) * self.pixels_per_meter,
            ),
            (
                env.map_center_x + distance_btwn_H_shaped_obstacles_right,
                env.map_center_x + random.uniform(-1, 1) * self.pixels_per_meter,
            ),
        ]

        # Draw walls
        self.create_base_map()

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

    def create_zigzag_map(self):
        """Creates a zigzag-shaped map with obstacles"""
        # Parameters
        p_z = min(self.s_z / self.k9, 1)
        w_wall = 0.2
        self.l_passage = 6 * p_z + 8 + 2 * random.uniform(0, 1)
        w_openings = []
        for _ in range(4):
            w_openings.append(-4.6 * p_z + 6 + 0.4 * random.uniform(0, 1))
        d_deviations = []
        for _ in range(4):
            d_deviations.append(2.4 * random.uniform(0, 1) - 1.2)

        self.create_base_map()

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
        r = random.randint(0, 1)
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

    def create_empty_map(self):
        """Creates an empty map with no obstacles."""
        self.create_base_map()

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

    def show_sensorData(self, data=None, robot_position=None, target_position=None):
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

            robot_frame_radius = self.robot_radius_m * self.pixels_per_meter
            robot_radius = self.robot_radius_m * 0.8 * self.pixels_per_meter

            # Draw target position
            pygame.draw.circle(
                self.lidar_sensor_layer,
                self.target_color,
                target_position,
                robot_frame_radius,
                width=2,  # Draw a circle around the robot
            )
            pygame.draw.circle(
                self.lidar_sensor_layer,
                self.target_color,
                target_position,
                robot_radius,
            )
            # font = pygame.font.SysFont("arial", 16)
            # text = font.render("Target", True, (0, 0, 0))
            # self.lidar_sensor_layer.blit(
            #     text,
            #     (
            #         target_position[0],
            #         target_position[1] + text.get_height() // 2,
            #     ),
            # )

            # Draw the robot's position
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


def check_surrounding_colors(env, position):
    """Checks the surrounding colors of a position in the environment."""
    surrounding_colors = []
    # Number of pixels to check in each direction (odd number)
    num_surrounding_points = 5
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (
                int(
                    position[0]
                    + i / (num_surrounding_points - 1) * 2 * env.robot_radius_pixels
                )
                >= 0
                and int(
                    position[1]
                    + j / (num_surrounding_points - 1) * 2 * env.robot_radius_pixels
                )
                >= 0
                and int(
                    position[0]
                    + i / (num_surrounding_points - 1) * 2 * env.robot_radius_pixels
                )
                <= env.map_w
                and int(
                    position[1]
                    + j / (num_surrounding_points - 1) * 2 * env.robot_radius_pixels
                )
                <= env.map_h
            ):
                surrounding_colors.append(
                    env.map.get_at(
                        (
                            int(
                                position[0]
                                + i
                                / (num_surrounding_points - 1)
                                * 2
                                * env.robot_radius_pixels
                            ),
                            int(
                                position[1]
                                + j
                                / (num_surrounding_points - 1)
                                * 2
                                * env.robot_radius_pixels
                            ),
                        )
                    )
                )
    is_ok = True
    for color in surrounding_colors:
        if color == hex_to_rgb(env.obstacle_color):
            is_ok = False
            return is_ok
    return is_ok


def operate_moving(robot_position, env):
    keys = pygame.key.get_pressed()
    step_size = 5  # Move by 10 pixels per key press
    if keys[pygame.K_a] and robot_position[0] > env.robot_radius_pixels:
        next_position = (
            robot_position[0] - step_size,
            robot_position[1],
        )
        if next_position[0] > 0:
            if check_surrounding_colors(env, next_position):
                robot_position = (
                    robot_position[0] - step_size,
                    robot_position[1],
                )  # Move robot left
                env.reset_window()
    if keys[pygame.K_d] and robot_position[0] < env.map_w - env.robot_radius_pixels:
        next_position = (
            robot_position[0] + step_size,
            robot_position[1],
        )
        if next_position[0] < env.map_w:
            if check_surrounding_colors(env, next_position):
                robot_position = (
                    robot_position[0] + step_size,
                    robot_position[1],
                )  # Move robot right
                env.reset_window()
    if keys[pygame.K_w] and robot_position[1] > env.robot_radius_pixels:
        next_position = (
            robot_position[0],
            robot_position[1] - step_size,
        )
        if next_position[1] > 0:
            if check_surrounding_colors(env, next_position):
                robot_position = (
                    robot_position[0],
                    robot_position[1] - step_size,
                )  # Move robot left
                env.reset_window()
    if keys[pygame.K_s] and robot_position[1] < env.map_h - env.robot_radius_pixels:
        next_position = (
            robot_position[0],
            robot_position[1] + step_size,
        )
        if next_position[1] < env.map_h:
            if check_surrounding_colors(env, next_position):
                robot_position = (
                    robot_position[0],
                    robot_position[1] + step_size,
                )  # Move robot left
                env.reset_window()
    return robot_position  # Return the updated robot position


def randomize_position(env, spawn_area):
    """Generates a random position within the environment.
    area: [xmin, xmax, ymin, ymax] defines the area where the robot can be placed.
    """
    if spawn_area is None:
        spawn_area_xmin = env.map_center_x - env.map_w // 2 + env.robot_radius_pixels
        spawn_area_xmax = env.map_center_x + env.map_w // 2 - env.robot_radius_pixels
        spawn_area_ymin = env.map_center_y - env.map_h // 2 + env.robot_radius_pixels
        spawn_area_ymax = env.map_center_y + env.map_h // 2 - env.robot_radius_pixels
    else:
        spawn_area_xmin, spawn_area_xmax, spawn_area_ymin, spawn_area_ymax = spawn_area

    while True:
        buffer = 10
        x = random.randint(
            spawn_area_xmin + buffer,
            spawn_area_xmax - buffer,
        )
        y = random.randint(
            spawn_area_ymin + buffer,
            spawn_area_ymax - buffer,
        )
        if check_surrounding_colors(env, (x, y)):
            return (x, y)  # Return the random position


def create_start_and_target_position(env, start_spawn_area, target_spawn_area):
    """Creates random start and target positions for the robot."""
    if env.map_type == "trap_shaped":
        T_dt_m = env.T_dt_m_trap_shaped
    elif env.map_type == "zigzag":
        T_dt_m = env.T_dt_m_zigzag
    elif env.map_type == "random":
        T_dt_m = env.T_dt_m_random
    elif env.map_type == "empty":
        T_dt_m = env.T_dt_m_empty
    while True:
        robot_position = randomize_position(env, start_spawn_area)
        target_position = randomize_position(env, target_spawn_area)
        distance_btwn_robot_and_target = math.sqrt(
            (target_position[0] - robot_position[0]) ** 2
            + (target_position[1] - robot_position[1]) ** 2
        )
        if distance_btwn_robot_and_target > T_dt_m * env.pixels_per_meter:
            print(
                f"Distance between robot and target: {distance_btwn_robot_and_target // env.pixels_per_meter} m"
            )
            return robot_position, target_position


def calculate_spawan_area_in_zigzag_map(env):
    """Calculates the spawn area for the robot in the zigzag map."""
    print("l_passage:", env.l_passage)
    start_spawn_area = [
        env.map_center_x
        - env.l_passage // 2 * env.pixels_per_meter
        + env.robot_radius_pixels,
        env.map_center_x
        + env.l_passage // 2 * env.pixels_per_meter
        - env.robot_radius_pixels,
        env.map_h - int(env.divided_space_width * 0.85) + env.robot_radius_pixels,
        env.map_h - int(env.divided_space_width * 0.15) - env.robot_radius_pixels,
    ]
    target_spawn_area = [
        env.map_center_x
        - env.l_passage // 2 * env.pixels_per_meter
        + env.robot_radius_pixels,
        env.map_center_x
        + env.l_passage // 2 * env.pixels_per_meter
        - env.robot_radius_pixels,
        0 + int(env.divided_space_width * 0.15) + env.robot_radius_pixels,
        0 + int(env.divided_space_width * 0.85) - env.robot_radius_pixels,
    ]
    return start_spawn_area, target_spawn_area


def initialize_map_parameters(env):
    start_spawn_area = None  # [xmin, xmax, ymin, ymax]
    target_spawn_area = None  # [xmin, xmax, ymin, ymax]

    if env.map_type == "trap_shaped":
        env.create_trap_shaped_map()
        start_spawn_area = [
            env.map_center_x - int(env.map_w * 0.4) + env.robot_radius_pixels,
            env.map_center_x + int(env.map_w * 0.4) - env.robot_radius_pixels,
            env.map_center_y + int(env.map_h * 0.35) + env.robot_radius_pixels,
            env.map_center_y + int(env.map_h * 0.5) - env.robot_radius_pixels,
        ]
        target_spawn_area = [
            env.map_center_x - int(env.map_w * 0.4) + env.robot_radius_pixels,
            env.map_center_x + int(env.map_w * 0.4) - env.robot_radius_pixels,
            env.map_center_y - int(env.map_h * 0.5) + env.robot_radius_pixels,
            env.map_center_y - int(env.map_h * 0.35) - env.robot_radius_pixels,
        ]
    elif env.map_type == "zigzag":
        env.create_zigzag_map()
        start_spawn_area, target_spawn_area = calculate_spawan_area_in_zigzag_map(env)
    elif env.map_type == "random":
        env.create_random_map()
        start_spawn_area = [
            env.map_center_x - env.map_w // 2 + env.robot_radius_pixels,
            env.map_center_x + env.map_w // 2 - env.robot_radius_pixels,
            env.map_center_y - env.map_h // 2 + env.robot_radius_pixels,
            env.map_center_y + env.map_h // 2 - env.robot_radius_pixels,
        ]
        target_spawn_area = start_spawn_area
    elif env.map_type == "empty":
        env.create_empty_map()
        start_spawn_area = [
            env.map_center_x - env.map_w // 2 + env.robot_radius_pixels,
            env.map_center_x + env.map_w // 2 - env.robot_radius_pixels,
            env.map_center_y - env.map_h // 2 + env.robot_radius_pixels,
            env.map_center_y + env.map_h // 2 - env.robot_radius_pixels,
        ]
        target_spwan_area = start_spawn_area
    return start_spawn_area, target_spawn_area


def create_environment(env, start_spawn_area, target_spawn_area):
    """Creates the environment and initializes the map."""
    if env.map_type == "trap_shaped":
        env.create_trap_shaped_map()
    elif env.map_type == "zigzag":
        env.create_zigzag_map()
        start_spawn_area, target_spawn_area = calculate_spawan_area_in_zigzag_map(env)
    elif env.map_type == "random":
        env.create_random_map()
    elif env.map_type == "empty":
        env.create_empty_map()
    env.map.blit(env.obstacle_layer, (0, 0))
    robot_position, target_position = create_start_and_target_position(
        env, start_spawn_area, target_spawn_area
    )
    return robot_position, target_position


if __name__ == "__main__":
    # Example usage
    pygame.init()
    env = buildEnvironment()

    map_type_list = ["trap_shaped", "zigzag", "random", "empty"]
    env.map_type = "empty"  # ["trap_shaped", "zigzag", "random", "empty"]
    start_spawn_area, target_spawn_area = initialize_map_parameters(env)
    env.reset_window()
    print("Environment created with obstacles.")
    running = True
    robot_position, target_position = create_start_and_target_position(
        env, start_spawn_area, target_spawn_area
    )

    print((env.map_center_x, env.map_center_y))
    while running:

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
                    start_spawn_area, target_spawn_area = initialize_map_parameters(env)
                    robot_position, target_position = create_start_and_target_position(
                        env, start_spawn_area, target_spawn_area
                    )
                if event.key == pygame.K_SPACE:
                    env.reset_environment()  # Reset the environment
                    start_spawn_area, target_spawn_area = initialize_map_parameters(env)
                    robot_position, target_position = create_start_and_target_position(
                        env, start_spawn_area, target_spawn_area
                    )

        robot_position = operate_moving(robot_position, env)  # Handle robot movement

        laser = sensors.LaserSensor(env.lidar_range_pixels, env.map, uncertainty=(0, 0))
        laser.position = robot_position
        sensor_data = laser.sense_obstacles(
            env.map, env.obstacle_color
        )  # Step 3: Sense. Stores sensed data array into sensor_data
        # env.dataStorage(sensor_data)  # Step 4: point cloud data storage
        env.show_sensorData(sensor_data, robot_position, target_position)

        # ウィンドウを更新
        pygame.display.flip()

    # 4. 終了処理
    pygame.quit()
