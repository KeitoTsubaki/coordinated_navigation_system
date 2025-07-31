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


class buildEnvironment:

    def __init__(self):
        self.window_name = "2D LiDAR Simulation"
        pygame.display.set_caption(self.window_name)

        # Set parameters
        with open("../config/simulator_config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.background_color = config["map"]["background_color"]
        self.robot_radius_m = config["robot"]["radius_m"]
        self.lidar_range_m = config["lidar"]["range_m"]
        self.map_w, self.map_h = config["map"]["size"], config["map"]["size"]

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

        # self.pointCloud = (
        #     []
        # )  # array containing coordinates of sensed objects (walls). Purely just (x,y) information.

        self.map_center_x = self.map_w // 2
        self.map_center_y = self.map_h // 2

        # self.lidar_sensor_layer = pygame.Surface((self.map_w, self.map_h))
        # self.robot_layer = pygame.Surface((self.map_w, self.map_h))

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
        self.obstacle_layer = pygame.Surface(
            (self.map_w, self.map_h), flags=pygame.SRCALPHA
        )
        self.obstacle_layer.fill((255, 255, 255))  # Fill the map with white background
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

        obstacle_pos_list = []  # The list of obstacle positions (cx, cy)
        count = 0
        while len(obstacle_pos_list) < self.n_obstacles:
            count += 1
            if count > 100:
                obstacle_pos_list = []
                count = 0

            obstacle_cx = random.randint(
                self.map_center_x - obstacle_area_width // 2,
                self.map_center_x + obstacle_area_width // 2,
            )
            obstacle_cy = random.randint(
                self.map_center_y - obstacle_area_height // 2,
                self.map_center_y + obstacle_area_height // 2,
            )
            if len(obstacle_pos_list) == 0:
                obstacle_pos_list.append((obstacle_cx, obstacle_cy))
                # break
            else:
                is_ok = True
                for pos in obstacle_pos_list:
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
                    obstacle_pos_list.append((obstacle_cx, obstacle_cy))
                    # break
        print(f"Obstacle Position: {obstacle_cx}, {obstacle_cy}")
        
        for obstacle_cx, obstacle_cy in obstacle_pos_list:
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
                (10,10,10),
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
        print(len(self.pointCloud))
        if data != False:  # if data exists, then start for loop
            for element in data:
                point = self.AD2pos(
                    element[0], element[1], element[2]
                )  # Input: data = [Distance, Angle, (x,y)]. Output: (x,y) coordinate of objects/walls.
                if (
                    point not in self.pointCloud
                ):  # Note: Point cloud should ONLY be the WALLS (x,y). If the wall has not already been sensed, then add to pointcloud.
                    self.pointCloud.append(point)  # [(x1,y1), (x2,y2), (x3,y3)]
            self.last_robot_position = (int(data[0][2][0]), int(data[0][2][1]))

    def show_sensorData(self, data=None, robot_position=None, goal_position=None):
        self.infomap = self.base_map.copy()  # Reset infomap to the base map
        if data != False:  # if data exists, then start for loop
            for element in data:
                point = self.AD2pos(
                    element[0], element[1], element[2]
                )  # Input: data = [Distance, Angle, (x,y)]. Output: (x,y) coordinate of objects/walls.
                pygame.draw.line(
                    self.infomap,
                    (135, 206, 250),
                    (int(element[2][0]), int(element[2][1])),
                    (int(point[0]), int(point[1])),
                    1,
                )
            # Draw the robot's position
            pygame.draw.circle(
                self.infomap,
                (0, 0, 255),
                robot_position,
                self.robot_radius,
            )
            # Draw goal position
            # pygame.draw.circle(
            #     self.infomap,
            #     (0, 255, 0),
            #     goal_position,
            #     self.robot_radius,
            # )


if __name__ == "__main__":
    # Example usage
    env = buildEnvironment()
    env.create_random_map()
    env.map.blit(env.obstacle_layer, (0, 0))  # Draw the obstacle layer onto the map
    print("Environment created with obstacles.")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    env.create_random_map()
                    env.map.blit(env.obstacle_layer, (0, 0))


        # ウィンドウを更新
        pygame.display.flip()

    # 4. 終了処理
    pygame.quit()
