import importlib.resources
import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pygame
import yaml

from . import utils
from .environment import SimCoreEnvironment


class SimCoreSquareEnvironment(SimCoreEnvironment):
    """Simulation environment that models the robot with a rectangular footprint."""

    def __init__(
        self,
        render_mode: Optional[str] = None,
        config_path: Optional[Path] = None,
        min_target_clearance_m: float = 0.0,
    ) -> None:
        if config_path is None:
            config_path = (
                importlib.resources.files("simulation_core")
                .joinpath("config")
                .joinpath("simulator_config_square.yaml")
            )

        config_text = config_path.read_text(encoding="utf-8")
        config = yaml.safe_load(config_text)
        robot_cfg = config["robot"]

        wheel_base_m = float(robot_cfg.get("wheel_base_m", 0.0) or 0.0)
        wheel_track_m = float(robot_cfg.get("wheel_track_m", 0.0) or 0.0)
        wheel_radius_m = float(robot_cfg.get("wheel_radius_m", 0.0) or 0.0)
        self.wheel_width_m = float(robot_cfg.get("wheel_width", 0.0) or 0.0)

        width_candidates = [
            robot_cfg.get("width_m"),
        ]
        height_candidates = [
            robot_cfg.get("height_m"),
        ]

        valid_widths = [float(value) for value in width_candidates if value is not None]
        valid_heights = [float(value) for value in height_candidates if value is not None]

        footprint_width_m = float(max(valid_widths)) if valid_widths else 0.0
        footprint_height_m = float(max(valid_heights)) if valid_heights else 0.0

        super().__init__(
            render_mode=render_mode,
            config_path=config_path,
            min_target_clearance_m=min_target_clearance_m,
        )

        self.ROBOT_WIDTH_M = footprint_width_m
        self.ROBOT_HEIGHT_M = footprint_height_m

        self.ROBOT_HALF_WIDTH_M = self.ROBOT_WIDTH_M / 2.0
        self.ROBOT_HALF_HEIGHT_M = self.ROBOT_HEIGHT_M / 2.0
        self.ROBOT_HALF_WIDTH_PIXELS = self.ROBOT_HALF_WIDTH_M * self.PIXELS_PER_METER
        self.ROBOT_HALF_HEIGHT_PIXELS = (
            self.ROBOT_HALF_HEIGHT_M * self.PIXELS_PER_METER
        )

        self.ROBOT_HALF_DIAGONAL_M = math.hypot(
            self.ROBOT_HALF_WIDTH_M, self.ROBOT_HALF_HEIGHT_M
        )
        self.ROBOT_HALF_DIAGONAL_PIXELS = (
            self.ROBOT_HALF_DIAGONAL_M * self.PIXELS_PER_METER
        )

        self.ROBOT_WIDTH_PIXELS = self.ROBOT_WIDTH_M * self.PIXELS_PER_METER
        self.ROBOT_HEIGHT_PIXELS = self.ROBOT_HEIGHT_M * self.PIXELS_PER_METER
        self.ROBOT_RADIUS_M = self.ROBOT_HALF_DIAGONAL_M
        self.ROBOT_SAFETY_RADIUS_M = self.ROBOT_HALF_DIAGONAL_M
        self.ROBOT_BODY_RADIUS_M = self.ROBOT_HALF_DIAGONAL_M
        self.ROBOT_RADIUS_PIXELS = self.ROBOT_HALF_DIAGONAL_PIXELS
        self.ROBOT_SAFETY_RADIUS_PIXELS = self.ROBOT_HALF_DIAGONAL_PIXELS

        self.map_creator.ROBOT_RADIUS_PIXELS = self.ROBOT_RADIUS_PIXELS

        self.WHEEL_HALF_BASE_PIXELS = (self.WHEEL_BASE_M / 2.0) * self.PIXELS_PER_METER
        self.WHEEL_HALF_TRACK_PIXELS = (
            self.WHEEL_TRACK_M / 2.0
        ) * self.PIXELS_PER_METER
        self.WHEEL_RADIUS_PIXELS = self.WHEEL_RADIUS_M * self.PIXELS_PER_METER
        self.WHEEL_WIDTH_PIXELS = self.wheel_width_m * self.PIXELS_PER_METER

        # --- Adjusted for Rear Wheel Drive ---
        self.REAR_AXLE_TO_FRONT_M = self.ROBOT_HALF_HEIGHT_M * 0.8
        self.REAR_AXLE_TO_FRONT_PIXELS = (
            self.REAR_AXLE_TO_FRONT_M * self.PIXELS_PER_METER
        )
        
        # --- LiDAR Mounting Position ---
        # 車軸（後輪）からのオフセット。正なら前方、負なら後方。
        self.LIDAR_MOUNT_OFFSET_M = -0.15 
        self.LIDAR_MOUNT_OFFSET_PIXELS = self.LIDAR_MOUNT_OFFSET_M * self.PIXELS_PER_METER

        # --- Visualization Variables ---
        self.v = 0.0
        self.omega = 0.0

    def _rotate_vector(self, x: float, y: float, cos_theta: float, sin_theta: float) -> Tuple[float, float]:
        """
        Pygame座標系（Y軸下向き）でのベクトル回転。
        cos_theta, sin_theta はロボットの現在の角度から計算されたもの。
        """
        rotated_x = x * cos_theta + y * sin_theta
        rotated_y = -x * sin_theta + y * cos_theta
        return rotated_x, rotated_y

    def _rectangle_vertices(
        self, position: Tuple[float, float], theta: float
    ) -> List[Tuple[int, int]]:
        body_center = self._axle_to_body_center(position, theta)
        
        half_width = self.ROBOT_HALF_WIDTH_PIXELS
        half_height = self.ROBOT_HALF_HEIGHT_PIXELS
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        local_vertices = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        ]

        vertices: List[Tuple[int, int]] = []
        for dx, dy in local_vertices:
            rx, ry = self._rotate_vector(dx, dy, cos_theta, sin_theta)
            vertices.append((int(body_center[0] + rx), int(body_center[1] + ry)))
        return vertices

    def _axle_to_body_center(
        self, position: Tuple[float, float], theta: float
    ) -> Tuple[float, float]:
        forward_offset = self.REAR_AXLE_TO_FRONT_PIXELS
        if forward_offset == 0:
            return position

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        # Local Y is forward (negative screen Y)
        # Vector (0, -offset) is forward
        rx, ry = self._rotate_vector(0, -forward_offset, cos_theta, sin_theta)
        
        return position[0] + rx, position[1] + ry
        
    def _get_lidar_position(self):
        """
        LiDARの取り付け位置を計算して返す。
        self.LIDAR_MOUNT_OFFSET_PIXELS に基づき、車軸からの相対位置を計算する。
        """
        offset = self.LIDAR_MOUNT_OFFSET_PIXELS
        cos_theta = math.cos(self.robot_theta)
        sin_theta = math.sin(self.robot_theta)
        
        # オフセットが正なら前方(0, -offset)、負なら後方(0, +abs(offset))
        rx, ry = self._rotate_vector(0, -offset, cos_theta, sin_theta)
        
        return (self.robot_position[0] + rx, self.robot_position[1] + ry)

    def _rectangle_perimeter_points(
        self, position: Tuple[float, float], theta: float
    ) -> Iterable[Tuple[int, int]]:
        vertices = self._rectangle_vertices(position, theta)
        samples_per_edge = max(1, self.NUM_CHECK_POINTS // 4)

        for i in range(4):
            start_vertex = vertices[i]
            end_vertex = vertices[(i + 1) % 4]
            for step in range(samples_per_edge + 1):
                t = step / samples_per_edge
                x = int(round(start_vertex[0] * (1 - t) + end_vertex[0] * t))
                y = int(round(start_vertex[1] * (1 - t) + end_vertex[1] * t))
                yield x, y

    def _project_polygon(self, vertices: List[Tuple[int, int]], axis: Tuple[float, float]):
        projections = [vertex[0] * axis[0] + vertex[1] * axis[1] for vertex in vertices]
        return min(projections), max(projections)

    def _rectangles_intersect(
        self, first_vertices: List[Tuple[int, int]], second_vertices: List[Tuple[int, int]]
    ) -> bool:
        axes: List[Tuple[float, float]] = []
        for vertices in (first_vertices, second_vertices):
            for i in range(4):
                start_vertex = vertices[i]
                end_vertex = vertices[(i + 1) % 4]
                edge = (end_vertex[0] - start_vertex[0], end_vertex[1] - start_vertex[1])
                axis = (-edge[1], edge[0])
                axis_length = math.hypot(axis[0], axis[1])
                if axis_length == 0:
                    continue
                axes.append((axis[0] / axis_length, axis[1] / axis_length))

        for axis in axes:
            min_a, max_a = self._project_polygon(first_vertices, axis)
            min_b, max_b = self._project_polygon(second_vertices, axis)
            if max_a < min_b or max_b < min_a:
                return False
        return True

    def visualize_search_point(self, position, theta, color=(255, 0, 0)):
        for check_x, check_y in self._rectangle_perimeter_points(position, theta):
            if 0 <= check_x < self.SCREEN_W and 0 <= check_y < self.SCREEN_H:
                pygame.draw.circle(
                    self.robot_and_lidar_layer,
                    color,
                    (check_x, check_y),
                    1,
                )

    def visualize_velocities(self, position, color=(0, 0, 255)):
        """ロボットの上部に現在の速度と角速度を表示する"""
        if not pygame.font.get_init():
            pygame.font.init()
        font = pygame.font.SysFont("arial", 16)
        
        # テキストのレンダリング
        text_surf = font.render(
            f"v: {self.v:.2f} m/s, w: {self.omega:.2f} rad/s", True, color
        )
        
        # ロボットの上部に配置
        draw_pos = (
            int(position[0] - text_surf.get_width() // 2),
            int(position[1] - self.ROBOT_RADIUS_PIXELS - text_surf.get_height() - 10),
        )
        
        self.robot_and_lidar_layer.blit(text_surf, draw_pos)

    def _draw_rectangle(self, position: Tuple[float, float], theta: float, color):
        vertices = self._rectangle_vertices(position, theta)
        pygame.draw.polygon(self.robot_and_lidar_layer, color, vertices)

    def _draw_wheels(self):
        wheel_radius_pixels = self.WHEEL_WIDTH_PIXELS
        half_track = self.WHEEL_HALF_TRACK_PIXELS
        
        if wheel_radius_pixels <= 0 or half_track == 0:
            return

        cos_theta = math.cos(self.robot_theta)
        sin_theta = math.sin(self.robot_theta)

        if self.WHEEL_BASE_M > 0.01:
             front_axle_offset = self.WHEEL_BASE_M * self.PIXELS_PER_METER
        else:
             front_axle_offset = self.REAR_AXLE_TO_FRONT_PIXELS * 0.9

        local_wheel_centers = [
            (-half_track, 0),
            (half_track, 0),
            (-half_track, -front_axle_offset),
            (half_track, -front_axle_offset),
        ]

        wheel_color = utils.hex_to_rgb("#202020")
        axle_center = self.robot_position

        for offset_x, offset_y in local_wheel_centers:
            rx, ry = self._rotate_vector(offset_x, offset_y, cos_theta, sin_theta)
            wheel_center = (
                axle_center[0] + rx,
                axle_center[1] + ry,
            )

            wheel_len = 6 
            wheel_wid = 3
            local_corners = [
                (-wheel_wid, -wheel_len),
                (wheel_wid, -wheel_len),
                (wheel_wid, wheel_len),
                (-wheel_wid, wheel_len),
            ]

            wheel_vertices: List[Tuple[int, int]] = []
            for corner_x, corner_y in local_corners:
                crx, cry = self._rotate_vector(corner_x, corner_y, cos_theta, sin_theta)
                wheel_vertices.append(
                    (
                        int(wheel_center[0] + crx),
                        int(wheel_center[1] + cry),
                    )
                )

            pygame.draw.polygon(self.robot_and_lidar_layer, wheel_color, wheel_vertices)

    def _draw_arrow(self, start_pos, end_pos, theta, color, size=10):
        """
        線を描画し、終点に矢尻（三角形）を追加して矢印にする。
        """
        pygame.draw.line(self.robot_and_lidar_layer, color, start_pos, end_pos, 2)
        
        angle_left = theta + 2.6 # approx 150 degrees
        angle_right = theta - 2.6
        
        dx_l = - size * math.sin(angle_left)
        dy_l = - size * math.cos(angle_left)
        p_left = (int(end_pos[0] + dx_l), int(end_pos[1] + dy_l))
        
        dx_r = - size * math.sin(angle_right)
        dy_r = - size * math.cos(angle_right)
        p_right = (int(end_pos[0] + dx_r), int(end_pos[1] + dy_r))
        
        pygame.draw.polygon(self.robot_and_lidar_layer, color, [end_pos, p_left, p_right])

    def show_robot_and_lidar(self, sensor_data=None):
        if sensor_data is not False:
            lidar_point_color = utils.hex_to_rgb(self.ROBOT_COLOR)
            lidar_origin = self._get_lidar_position()
            lidar_origin_px = (int(lidar_origin[0]), int(lidar_origin[1]))
            
            lidar_line_color = (150, 150, 150) # Light Grey
            
            for element in sensor_data:
                distance_m = element[0]
                local_angle = element[1]

                global_angle = self.robot_theta + local_angle
                distance_pixels = distance_m * self.PIXELS_PER_METER

                dx = -distance_pixels * math.sin(global_angle)
                dy = -distance_pixels * math.cos(global_angle)

                point_px = (
                    int(lidar_origin[0] + dx),
                    int(lidar_origin[1] + dy),
                )
                
                # --- Draw Ray ---
                # pygame.draw.line(
                #     self.robot_and_lidar_layer,
                #     lidar_line_color,
                #     lidar_origin_px,
                #     point_px,
                #     1
                # )
                
                pygame.draw.circle(
                    self.robot_and_lidar_layer,
                    lidar_point_color,
                    point_px,
                    2,
                )

            frame_extent = self.ROBOT_HALF_DIAGONAL_PIXELS

            # Target drawing
            target_body_center = self._axle_to_body_center(
                self.target_position, self.target_theta
            )

            self.visualize_search_point(
                self.target_position, self.target_theta, self.TARGET_COLOR
            )
            self._draw_rectangle(
                self.target_position, self.target_theta, self.TARGET_COLOR
            )
            target_head_position = self._compute_orientation_endpoint(
                target_body_center, self.target_theta, frame_extent
            )
            
            # --- Target Arrow ---
            self._draw_arrow(
                (int(target_body_center[0]), int(target_body_center[1])),
                (int(target_head_position[0]), int(target_head_position[1])),
                self.target_theta,
                (255, 0, 0)
            )

            self.visualize_path()
            
            # Robot drawing
            robot_body_center = self._axle_to_body_center(
                self.robot_position, self.robot_theta
            )
            
            self.visualize_search_point(
                self.robot_position, self.robot_theta, self.ROBOT_COLOR
            )
            
            self.visualize_goal_info(robot_body_center)
            self.visualize_closest_obstacle_info(robot_body_center)
            
            self._draw_rectangle(self.robot_position, self.robot_theta, self.ROBOT_COLOR)
            self._draw_wheels()
            
            # --- Safety Circle Visualization ---
            pygame.draw.circle(
                self.robot_and_lidar_layer,
                (0, 255, 255),  # Cyan
                (int(robot_body_center[0]), int(robot_body_center[1])),
                int(self.ROBOT_SAFETY_RADIUS_PIXELS),
                1  # width=1 for outline
            )
            # -----------------------------------
            
            # --- LiDAR取り付け位置（オレンジ色） ---
            pygame.draw.circle(
                self.robot_and_lidar_layer,
                (255, 165, 0),  # Orange
                lidar_origin_px,
                5,
            )

            # --- 速度の可視化 ---
            self.visualize_velocities(robot_body_center)
            
            robot_head_position = self._compute_orientation_endpoint(
                robot_body_center, self.robot_theta, frame_extent
            )
            
            # --- Robot Arrow ---
            self._draw_arrow(
                (int(robot_body_center[0]), int(robot_body_center[1])),
                (int(robot_head_position[0]), int(robot_head_position[1])),
                self.robot_theta,
                (255, 0, 0)
            )

            self.screen.blit(self.robot_and_lidar_layer, (0, 0))

    def is_collision(self, position):
        for check_x, check_y in self._rectangle_perimeter_points(
            position, self.robot_theta
        ):
            if not (0 <= check_x < self.SCREEN_W and 0 <= check_y < self.SCREEN_H):
                return True

            try:
                pixel_color = self.obstacle_layer.get_at((check_x, check_y))
            except IndexError:
                return True

            if (pixel_color[0], pixel_color[1], pixel_color[2]) == self._obstacle_color_rgb:
                return True
        return False

    def is_target_reached(self):
        robot_vertices = self._rectangle_vertices(self.robot_position, self.robot_theta)
        target_vertices = self._rectangle_vertices(
            self.target_position, self.target_theta
        )
        return self._rectangles_intersect(robot_vertices, target_vertices)

    def update_path(self, new_position, new_theta):
        self.robot_position = new_position
        self.robot_theta = new_theta
        body_center = self._axle_to_body_center(new_position, new_theta)
        self.path.append([body_center, new_theta])

    def render_obstacle_and_robots(self, map_type="empty"):
        screen_center_xy = (self.SCREEN_CENTER_X, self.SCREEN_CENTER_Y)
        screen_size_wh = (self.SCREEN_W, self.SCREEN_H)
        if map_type == "image" and self.image_map_path is None:
            raise FileNotFoundError(
                "Image map type selected but no image path was configured."
            )
        self.map_creator.create_map(
            map_type,
            self.obstacle_layer,
            pt=self.pt,
            pz=self.pz,
            image_path=self.image_map_path,
        )
        start_spawn_area, target_spawn_area = (
            self.map_creator.initialize_map_parameters(
                map_type, screen_center_xy, screen_size_wh, self.ROBOT_RADIUS_PIXELS
            )
        )
        self.create_start_and_target_position(
            map_type, start_spawn_area, target_spawn_area
        )

        # --- 追加: 高速化用マップキャッシュ ---
        map_pixels = pygame.surfarray.array3d(self.obstacle_layer)
        target_color = self._obstacle_color_rgb
        # 障害物がある場所がTrueの2次元配列を作成
        self.collision_map = (map_pixels == target_color).all(axis=2)

        if not self._robot_heading_initialized:
            self.robot_theta = 0.0
        if not self._target_heading_initialized:
            self.target_theta = 0.0

    def lidar_scan(self):
        self.sensor.position = self._get_lidar_position()
        self.sensor.theta = self.robot_theta
        
        # キャッシュ済みのマップ配列を渡す
        sensor_data = self.sensor.sense_obstacles(self.collision_map)
        return sensor_data