# Sensor functions

import math
import pygame
import numpy as np


def uncertainty_add(distance, angle, sigma):
    mean = np.array([distance, angle])
    covariance = np.diag(
        sigma**2
    )  # noise of distance and angle measurements are not correlated, hence only diagonal is non-zero. https://www.cuemath.com/algebra/covariance-matrix/
    distance, angle = np.random.multivariate_normal(
        mean, covariance
    )  # gets the actual noisy values from Gaussian distribution
    distance = max(distance, 0)  # max() function so we don't get negative values
    angle = max(angle, 0)
    return [distance, angle]


class LiDARSensor:

    def __init__(
        self, Range, uncertainty, pixels_per_meter, lidar_resolution, screen_w, screen_h
    ):  # uncertainty is an array with 2 values [sigma1,sigma2]
        self.LIDAR_RANGE_M = Range
        # self.ROTATION_SPEED = 0.5  # rotations per second #IS NOT USED THROUGHOUT THIS SCRIPT
        self.sigma = np.array([uncertainty[0], uncertainty[1]])
        self.position = (
            0,
            0,
        )  # <<< self.position should update to cursor's position!!!! BUG NEEDS TO BE FIXED
        self.theta = 0
        self.SCREEN_W, self.SCREEN_H = screen_w, screen_h
        self.NUM_SCANS = lidar_resolution
        self.pixels_per_meter = pixels_per_meter
        self.out_of_range_value = self.LIDAR_RANGE_M  # lidar max range value or 0
        self.sensedObstacles = []

    # Euclidean DISTANCE CALCULATOR
    def distance(self, obstaclePosition):
        px = (obstaclePosition[0] - self.position[0]) ** 2
        py = (obstaclePosition[1] - self.position[1]) ** 2
        distance = math.sqrt(px + py) / self.pixels_per_meter
        return distance

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip("#")  # 先頭の # を削除
        # 6文字 (RRGGBB) でなければエラー
        if len(hex_color) != 6:
            raise ValueError("カラーコードは6桁の16進数で指定してください。")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)

    def sense_obstacles(self, map, obstacle_color="#000000"):
        data = []
        x1, y1 = self.position[0], self.position[1]
        obstacle_color_rgb = self.hex_to_rgb(obstacle_color)

        # ロボットの向き(self.theta)を、scan_angleと同じ数学座標系(X軸正方向=0)に変換する
        # self.thetaは真上(Y軸負方向)が0 radなので、pi/2を引くとX軸正方向が0の基準に揃う
        robot_global_angle = self.theta - math.pi / 2

        # self.theta == self.robot_theta
        # clockwise == negative
        # is it reversed?
        # I guess it may be 0 rad at negative x

        for scan_angle in np.linspace(0, 2 * math.pi, self.NUM_SCANS, endpoint=False):
            # 基準を揃えたロボットの向きに、センサーのローカルなスキャン角度を加算する
            # ロボットの向きからセンサーを回転
            global_angle = robot_global_angle - scan_angle
            # x2, y2 = (
            #     x1 + self.LIDAR_RANGE_M * self.pixels_per_meter * math.cos(global_angle),
            #     y1 + self.LIDAR_RANGE_M * self.pixels_per_meter * math.sin(global_angle),
            # )
            x2, y2 = (
                x1
                - self.LIDAR_RANGE_M * self.pixels_per_meter * math.sin(global_angle),
                y1
                - self.LIDAR_RANGE_M * self.pixels_per_meter * math.cos(global_angle),
            )

            # robot frame
            relative_angle = -global_angle + robot_global_angle
            # = scan angle ?

            intersected = False
            for i in range(0, 100):  # If wall is within laser's path ...
                u = i / 100  # Essentially a percentage
                x = int(x1 + (x2 - x1) * u)
                y = int(y1 + (y2 - y1) * u)

                if not (0 <= x < self.SCREEN_W and 0 <= y < self.SCREEN_H):
                    # 光線がマップの境界外に出た場合、その地点を障害物として扱う
                    distance = self.distance((x, y))
                    # data.append([distance, global_angle])
                    data.append([distance, relative_angle])
                    intersected = True
                    break

                color = map.get_at((x, y))
                if (color[0], color[1], color[2]) == obstacle_color_rgb:
                    # Obstacle detected
                    distance = self.distance((x, y))
                    # output = uncertainty_add(distance, angle, self.sigma)
                    # data.append([distance, global_angle])
                    data.append([distance, relative_angle])
                    intersected = True
                    break

            if not intersected:
                # No obstacle found, record max range
                # 本当ならここは+infにしたいけど、そうすると前処理が増えてしまう。今は余裕がないので、max rangeにしておく
                # data.append([self.out_of_range_value, global_angle])
                data.append([self.out_of_range_value, relative_angle])

        return data if data else False

    def get_closest_obstacle_info(self, sensor_data):
        if not sensor_data:
            return float("inf"), 0.0

        # Find the measurement with the minimum distance
        closest_distance, closest_obstacle_angle = min(sensor_data, key=lambda x: x[0])

        # Normalize the angle to the range [-pi, pi] using atan2
        # This is a robust way to handle angle normalization.
        normalized_relative_angle = math.atan2(
            math.sin(closest_obstacle_angle), math.cos(closest_obstacle_angle)
        )
        return closest_distance, normalized_relative_angle
