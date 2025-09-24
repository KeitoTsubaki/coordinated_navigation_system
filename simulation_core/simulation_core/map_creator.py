import pygame
import math

class MapCreator:

    def __init__(self, map, map_size, obstacle_color, n_obstacles, pixels_per_meter, np_random, T_do_m, s_t, k8, s_z, k9, l_passage, robot_radius_pixels):
        self.map = map
        self.MAP_SIZE = map_size
        self.OBSTACLE_COLOR = obstacle_color
        self.n_obstacles = n_obstacles
        self.PIXELS_PER_METER = pixels_per_meter
        self.np_random = np_random
        self.ROBOT_RADIUS_PIXELS = robot_radius_pixels
        
        self.T_DO_M = T_do_m
        self.s_t = s_t
        self.K8 = k8
        self.s_z = s_z
        self.K9 = k9
        self.l_passage = l_passage
        
        self.MAP_W, self.MAP_H = self.MAP_SIZE
        self.map_center_x = self.MAP_W // 2
        self.map_center_y = self.MAP_H // 2
        self.DIVIDED_SPACE_WIDTH = self.MAP_H // 5 # width of the divided space in the zigzag-shaped map
        self.obstacle_layer = None
        
    def _draw_rect_obstacle(self, xmin, ymin, width, height):
        pygame.draw.rect(
            self.obstacle_layer,
            self.OBSTACLE_COLOR,
            (
                xmin,
                ymin,
                width,
                height,
            ),
            width=0,
        )
    def create_map(self, map_type, obstacle_layer=None):
        self.obstacle_layer = obstacle_layer
        """Creates the environment and initializes the map."""
        # map_size: (map_w, map_h)
        if map_type == "trap_shaped":
            self._create_trap_shaped_map()
        elif map_type == "zigzag":
            self._create_zigzag_map()
        elif map_type == "random":
            self._create_random_map()
        elif map_type == "empty":
            self._create_empty_map()


    def _create_base_map(self):
        """Creates a base map with walls."""
        
        map_w, map_h = self.MAP_SIZE

        wall_width = 1
        pygame.draw.polygon(
            self.obstacle_layer,
            self.OBSTACLE_COLOR,
            [
                [0, 0],
                [map_w - wall_width, 0],
                [map_w - wall_width, map_h - wall_width],
                [0, map_h - wall_width],
            ],
            width=wall_width,
        )
        font = pygame.font.SysFont("arial", 16)
        text = font.render(
            f"{map_w // self.PIXELS_PER_METER}m x {map_h // self.PIXELS_PER_METER}m",
            True,
            (0, 0, 0),
        )
        self.obstacle_layer.blit(
            text,
            (
                map_w - text.get_width() - 5,
                map_h - text.get_height() - 5,
            ),
        )
        
    def _create_random_map(self):
        """Creates a random map with obstacles"""
        # Draw walls
        self._create_base_map()
        
        # Define the area for obstacles
        obstacle_area_width = self.MAP_W // 15 * 13
        obstacle_area_height = self.MAP_H // 15 * 13
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

            obstacle_cx = self.np_random.integers(
                self.map_center_x - obstacle_area_width // 2,
                self.map_center_x + obstacle_area_width // 2,
            )
            obstacle_cy = self.np_random.integers(
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
                    # print(f"Distance (m): {distance // self.PIXELS_PER_METER}")
                    if distance < self.T_DO_M * self.PIXELS_PER_METER:
                        is_ok = False
                if is_ok:
                    obstacle_pos_list.append((obstacle_cx, obstacle_cy))
                    # break
        # print(f"Obstacle Position: {obstacle_cx}, {obstacle_cy}")

        for obstacle_cx, obstacle_cy in obstacle_pos_list:
            shape_random = self.np_random.integers(0, 1, endpoint=True)
            # print(f"shape_random: {shape_random}")
            if shape_random == 0:
                # Draw a rectangle obstacle
                rectangular_l_m = 4 + self.np_random.uniform(0, 1)
                rectangular_w_m = 0.5 + self.np_random.uniform(0, 1) * 0.5
                rectangular_l_pixels = int(rectangular_l_m * self.PIXELS_PER_METER)
                rectangular_w_pixels = int(rectangular_w_m * self.PIXELS_PER_METER)
                if self.np_random.integers(0, 1, endpoint=True) == 0:
                    object_width = rectangular_w_pixels
                    object_height = rectangular_l_pixels
                else:
                    object_width = rectangular_l_pixels
                    object_height = rectangular_w_pixels
                
                self._draw_rect_obstacle(
                    obstacle_cx - object_width // 2,
                    obstacle_cy - object_height // 2,
                    object_width,
                    object_height,
                )
            elif shape_random == 1:
                # Draw a circular obstacle
                radius_m = 0.5 + self.np_random.uniform(0, 1)
                radius_pixels = int(radius_m * self.PIXELS_PER_METER)
                pygame.draw.circle(
                    self.obstacle_layer,
                    self.OBSTACLE_COLOR,
                    (obstacle_cx, obstacle_cy),
                    radius_pixels,
                    width=0,
                )
            # Draw a center point for the obstacle
            # pygame.draw.circle(
            #     obstacle_layer,
            #     (10, 10, 10),
            #     (obstacle_cx, obstacle_cy),
            #     1,
            # )
        # map.blit(
        #     obstacle_layer, (0, 0)
        # )  # Draw the obstacle layer onto the map

    def _create_trap_shaped_map(self):
        """Creates a trap-shaped map with obstacles"""
        
        # Parameters
        p_t = min(self.s_t / self.K8, 1)
        w_wall = 0.2  # Width of the walls in meters
        l_horizontal = []
        for i in range(3):
            l_horizontal.append(1.5 * p_t + 3 + self.np_random.uniform(0, 1))
        l_vertical = []
        for i in range(3):
            l_vertical_left = 6 * p_t + 1 + 4 * self.np_random.uniform(0, 1)
            l_vertical_right = 6 * p_t + 1 + 4 * self.np_random.uniform(0, 1)
            if i == 0:
                if self.np_random.integers(0, 1, endpoint=True) == 0:
                    l_vertical_left = self.MAP_H
            elif i == 2:
                if self.np_random.integers(0, 1, endpoint=True) == 0:
                    l_vertical_right = self.MAP_H
            l_vertical.append([l_vertical_left, l_vertical_right])
        w_passage_0 = -1.6 * p_t + 2.5 + 0.2 * self.np_random.uniform(0, 1)
        w_passage_1 = (0.5 + 0.5 * self.np_random.uniform(0, 1)) * w_passage_0

        distance_btwn_H_shaped_obstacles_left = (
            l_horizontal[1] / 2 + w_passage_0 + w_wall * 2 + l_horizontal[0] / 2
        ) * self.PIXELS_PER_METER
        distance_btwn_H_shaped_obstacles_right = (
            l_horizontal[1] / 2 + w_passage_1 + w_wall * 2 + l_horizontal[2] / 2
        ) * self.PIXELS_PER_METER

        # Define the positions of the H-shaped obstacles
        positions_of_H_shaped_obstacles = [
            (
                self.map_center_x - distance_btwn_H_shaped_obstacles_left,
                self.map_center_y + self.np_random.uniform(-1, 1) * self.PIXELS_PER_METER,
            ),
            (
                self.map_center_x,
                self.map_center_y + self.np_random.uniform(-1, 1) * self.PIXELS_PER_METER,
            ),
            (
                self.map_center_x + distance_btwn_H_shaped_obstacles_right,
                self.map_center_x + self.np_random.uniform(-1, 1) * self.PIXELS_PER_METER,
            ),
        ]

        # Draw walls
        self._create_base_map()

        # Define the area for obstacles
        for i, position in enumerate(positions_of_H_shaped_obstacles):
            H_shaped_obstacle_cx, H_shaped_obstacle_cy = position

            # Draw the horizontal walls
            self._draw_rect_obstacle(
                H_shaped_obstacle_cx
                - (l_horizontal[i] / 2) * self.PIXELS_PER_METER,
                H_shaped_obstacle_cy - (w_wall / 2) * self.PIXELS_PER_METER,
                l_horizontal[i] * self.PIXELS_PER_METER,
                w_wall * self.PIXELS_PER_METER,
            )
            # Draw the vertical walls (left side)
            self._draw_rect_obstacle(
                H_shaped_obstacle_cx
                - (l_horizontal[i] / 2 + w_wall) * self.PIXELS_PER_METER,
                H_shaped_obstacle_cy
                - (l_vertical[i][0] / 2) * self.PIXELS_PER_METER,
                w_wall * self.PIXELS_PER_METER,
                l_vertical[i][0] * self.PIXELS_PER_METER,
            )
            # Draw the vertical walls (right side)
            self._draw_rect_obstacle(
                H_shaped_obstacle_cx
                + (l_horizontal[i] / 2) * self.PIXELS_PER_METER
                - 1,  # add 1 to avoid gap between horizontal and vertical walls
                H_shaped_obstacle_cy
                - (l_vertical[i][1] / 2) * self.PIXELS_PER_METER,
                w_wall * self.PIXELS_PER_METER,
                l_vertical[i][1] * self.PIXELS_PER_METER,
            )
        # map.blit(
        #     obstacle_layer, (0, 0)
        # )  # Draw the obstacle layer onto the map

    def _create_zigzag_map(self):
        """Creates a zigzag-shaped map with obstacles"""
        
        # Parameters
        p_z = min(self.s_z / self.K9, 1)
        w_wall = 0.2
        self.l_passage = 6 * p_z + 8 + 2 * self.np_random.uniform(0, 1)
        w_openings = []
        for _ in range(4):
            w_openings.append(-4.6 * p_z + 6 + 0.4 * self.np_random.uniform(0, 1))
        d_deviations = []
        for _ in range(4):
            d_deviations.append(2.4 * self.np_random.uniform(0, 1) - 1.2)

        self._create_base_map()

        # Draw dotted lines
        for i in range(1, 5):
            pygame.draw.line(
                self.obstacle_layer,
                (240, 240, 240),
                (0, (self.DIVIDED_SPACE_WIDTH * i)),
                (
                    self.MAP_W,
                    (self.DIVIDED_SPACE_WIDTH * i),
                ),
                width=3,
            )

        # Draw side walls
        side_wall_xs = [
            self.map_center_x - (self.l_passage / 2 + w_wall) * self.PIXELS_PER_METER,
            self.map_center_x + (self.l_passage / 2) * self.PIXELS_PER_METER,
        ]
        for x in side_wall_xs:
            self._draw_rect_obstacle(
                x,
                0,
                w_wall * self.PIXELS_PER_METER,
                self.MAP_H,
            )

        # Draw horizontal walls
        r = self.np_random.integers(0, 1, endpoint=True)
        for i in range(4):
            if i % 2 == r:
                horizontal_wall_xmin = (
                    self.map_center_x
                    - (self.l_passage / 2 - w_openings[i]) * self.PIXELS_PER_METER
                )
            else:
                horizontal_wall_xmin = (
                    self.map_center_x - (self.l_passage / 2) * self.PIXELS_PER_METER
                )
            self._draw_rect_obstacle(
                horizontal_wall_xmin,
                self.DIVIDED_SPACE_WIDTH * (i + 1)
                + d_deviations[i] * self.PIXELS_PER_METER,
                (self.l_passage - w_openings[i]) * self.PIXELS_PER_METER
                + 1,  # add 1 to avoid gap between horizontal and side walls
                w_wall * self.PIXELS_PER_METER,
            )
        # self.map.blit(
        #     obstacle_layer, (0, 0)
        # )  # Draw the obstacle layer onto the map

    def _create_empty_map(self):
        """Creates an empty map with no obstacles."""
        self._create_base_map()
    
    def _calculate_spawan_area_in_zigzag_map(self):
        """Calculates the spawn area for the robot in the zigzag map."""
        # print("l_passage:", self.l_passage)
        start_spawn_area = [
            self.map_center_x
            - self.l_passage // 2 * self.PIXELS_PER_METER
            + self.ROBOT_RADIUS_PIXELS,
            self.map_center_x
            + self.l_passage // 2 * self.PIXELS_PER_METER
            - self.ROBOT_RADIUS_PIXELS,
            self.MAP_H - int(self.DIVIDED_SPACE_WIDTH * 0.85) + self.ROBOT_RADIUS_PIXELS,
            self.MAP_H - int(self.DIVIDED_SPACE_WIDTH * 0.15) - self.ROBOT_RADIUS_PIXELS,
        ]
        target_spawn_area = [
            self.map_center_x
            - self.l_passage // 2 * self.PIXELS_PER_METER
            + self.ROBOT_RADIUS_PIXELS,
            self.map_center_x
            + self.l_passage // 2 * self.PIXELS_PER_METER
            - self.ROBOT_RADIUS_PIXELS,
            0 + int(self.DIVIDED_SPACE_WIDTH * 0.15) + self.ROBOT_RADIUS_PIXELS,
            0 + int(self.DIVIDED_SPACE_WIDTH * 0.85) - self.ROBOT_RADIUS_PIXELS,
        ]
        return start_spawn_area, target_spawn_area
    
    def initialize_map_parameters(self, map_type, screen_center_xy, screen_size_wh, robot_radius_pixels):
        start_spawn_area = None  # [xmin, xmax, ymin, ymax]
        target_spawn_area = None  # [xmin, xmax, ymin, ymax]
        screen_center_x, screen_center_y = screen_center_xy
        screen_w, screen_h = screen_size_wh

        if map_type == "trap_shaped":
            start_spawn_area = [
                screen_center_x - int(screen_w * 0.4) + robot_radius_pixels,
                screen_center_x + int(screen_w * 0.4) - robot_radius_pixels,
                screen_center_y + int(screen_h * 0.35) + robot_radius_pixels,
                screen_center_y + int(screen_h * 0.5) - robot_radius_pixels,
            ]
            target_spawn_area = [
                screen_center_x - int(screen_w * 0.4) + robot_radius_pixels,
                screen_center_x + int(screen_w * 0.4) - robot_radius_pixels,
                screen_center_y - int(screen_h * 0.5) + robot_radius_pixels,
                screen_center_y - int(screen_h * 0.35) - robot_radius_pixels,
            ]
        elif map_type == "zigzag":
            start_spawn_area, target_spawn_area = self._calculate_spawan_area_in_zigzag_map()
        elif map_type == "random":
            start_spawn_area = [
                screen_center_x - screen_w // 2 + robot_radius_pixels,
                screen_center_x + screen_w // 2 - robot_radius_pixels,
                screen_center_y - screen_h // 2 + robot_radius_pixels,
                screen_center_y + screen_h // 2 - robot_radius_pixels,
            ]
            target_spawn_area = start_spawn_area
        elif map_type == "empty":
            start_spawn_area = [
                screen_center_x - screen_w // 2 + robot_radius_pixels,
                screen_center_x + screen_w // 2 - robot_radius_pixels,
                screen_center_y - screen_h // 2 + robot_radius_pixels,
                screen_center_y + screen_h // 2 - robot_radius_pixels,
            ]
            target_spawn_area = start_spawn_area
        return start_spawn_area, target_spawn_area