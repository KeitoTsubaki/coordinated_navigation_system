import math
import pygame

from simulation_core.environment import SimCoreEnvironment


class MultiRobotEnvironment(SimCoreEnvironment):
    def __init__(self, num_robots=2, render_mode=None, config_path=None):
        super().__init__(render_mode=render_mode, config_path=config_path)

        self.num_robots = num_robots
        self.robots = []  # Store state of each robot

        # ロボットごとの色を定義
        self.robot_colors = [
            (50, 150, 50),  # Green
            (50, 50, 200),  # Blue
            (150, 50, 150),  # Purple
            # Add more colors if needed
        ]
        if num_robots > len(self.robot_colors):
            # If more robots than colors, generate random colors
            for _ in range(num_robots - len(self.robot_colors)):
                self.robot_colors.append(self.np_random.integers(50, 200, size=3).tolist())

    def reset(self, map_type="empty"):
        """環境をリセットし、複数のロボットを初期位置に配置する"""
        super().reset(map_type)
        self.robots.clear()

        for i in range(self.num_robots):
            # 衝突しないようにロボットの開始位置を生成
            while True:
                position = self.randomize_position(None)
                is_too_close = False
                for other_robot in self.robots:
                    dist = math.hypot(
                        position[0] - other_robot["position"][0],
                        position[1] - other_robot["position"][1],
                    )
                    if dist < self.ROBOT_RADIUS_PIXELS * 2.5:
                        is_too_close = True
                        break
                if not is_too_close:
                    break

            robot_state = {
                "position": position,
                "theta": 0.0,
                "color": self.robot_colors[i % len(self.robot_colors)],
                "sensor_data": None,
            }
            self.robots.append(robot_state)

        # 最初のロボットを 'self.robot_position' に設定して、既存の関数との互換性を保つ
        if self.robots:
            self.robot_position = self.robots[0]["position"]
            self.robot_theta = self.robots[0]["theta"]

    def render_robot_and_lidar(self):
        """すべてのロボットとLiDARスキャンを描画する"""
        self.clear_robot_and_lidar_layer()

        # ターゲットを描画
        self.visualize_search_point(self.target_position, self.TARGET_COLOR)
        pygame.draw.circle(
            self.robot_and_lidar_layer,
            self.TARGET_COLOR,
            self.target_position,
            self.ROBOT_RADIUS_PIXELS * 0.8,
        )

        # 各ロボットを描画
        for robot in self.robots:
            self._show_single_robot(robot)

        self.screen.blit(self.robot_and_lidar_layer, (0, 0))

    def _show_single_robot(self, robot):
        """単一のロボットを描画するヘルパー関数"""
        position = robot["position"]
        theta = robot["theta"]
        color = robot["color"]
        sensor_data = robot["sensor_data"]

        # LiDARスキャン結果の描画
        if sensor_data:
            for dist, angle in sensor_data:
                # センサーの角度をグローバル座標系に変換
                scan_angle = theta - math.pi / 2 - angle
                point_x = position[0] - dist * self.PIXELS_PER_METER * math.sin(scan_angle)
                point_y = position[1] - dist * self.PIXELS_PER_METER * math.cos(scan_angle)

                pygame.draw.line(self.robot_and_lidar_layer, self.LIDAR_LINE_COLOR, position, (point_x, point_y), 1)
                pygame.draw.circle(self.robot_and_lidar_layer, self.LIDAR_POINTS_COLOR, (point_x, point_y), 2)

        # ロボット本体の描画
        self.visualize_search_point(position, color)
        pygame.draw.circle(
            self.robot_and_lidar_layer, color, position, self.ROBOT_RADIUS_PIXELS * 0.8
        )

        # ロボットの向きを示す線
        head_pos = (
            position[0] - self.ROBOT_RADIUS_PIXELS * math.sin(theta),
            position[1] - self.ROBOT_RADIUS_PIXELS * math.cos(theta),
        )
        pygame.draw.line(self.robot_and_lidar_layer, (255, 0, 0), position, head_pos, 2)

    def update_all_sensors(self):
        """すべてのロボットのセンサーデータを更新する"""
        for robot in self.robots:
            self.sensor.position = robot["position"]
            self.sensor.theta = robot["theta"]
            robot["sensor_data"] = self.sensor.sense_obstacles(self.obstacle_layer, self.OBSTACLE_COLOR)
