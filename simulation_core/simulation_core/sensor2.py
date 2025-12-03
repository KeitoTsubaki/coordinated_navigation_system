# Sensor functions

import math
import pygame
import numpy as np


def uncertainty_add(distance, angle, sigma):
    mean = np.array([distance, angle])
    covariance = np.diag(
        sigma**2
    )  # noise of distance and angle measurements are not correlated
    distance, angle = np.random.multivariate_normal(
        mean, covariance
    )  # gets the actual noisy values from Gaussian distribution
    distance = max(distance, 0)  # max() function so we don't get negative values
    angle = max(angle, 0)
    return [distance, angle]


class LiDARSensor:

    def __init__(
        self, Range, uncertainty, pixels_per_meter, lidar_resolution, screen_w, screen_h
    ):
        self.LIDAR_RANGE_M = Range
        self.sigma = np.array([uncertainty[0], uncertainty[1]])
        self.position = (0, 0)
        self.theta = 0
        self.SCREEN_W, self.SCREEN_H = screen_w, screen_h
        self.NUM_SCANS = lidar_resolution
        self.pixels_per_meter = pixels_per_meter
        self.out_of_range_value = self.LIDAR_RANGE_M
        self.sensedObstacles = []

        # Pre-calculate scan angles
        self._scan_angles = np.linspace(
            -math.pi, math.pi, self.NUM_SCANS, endpoint=False
        )

    # ... distance, hex_to_rgb メソッドは削除または無視でOK ...

    def sense_obstacles(self, map_array):
        """
        map_array: 障害物がある場所が True (または 1) となっている 2D Numpy配列 (W, H)
                   pygame.surfarray.array3d などを加工したものを受け取る
        """
        data = []
        x1, y1 = self.position[0], self.position[1]
        
        # Max range in pixels
        max_range_px = int(self.LIDAR_RANGE_M * self.pixels_per_meter)

        # 配列参照の境界チェック用
        W, H = self.SCREEN_W, self.SCREEN_H

        for angle in self._scan_angles:
            global_angle = self.theta + angle

            dx = -math.sin(global_angle)
            dy = -math.cos(global_angle)

            # --- 高速化のポイント: Step Sizeを少し大きくする ---
            # 精度と速度のトレードオフ。2 -> 4-5 にするだけで2倍速くなります。
            step_size = 4  
            
            intersected = False
            measured_distance = self.out_of_range_value

            # ループ回数を事前に計算
            num_steps = max_range_px // step_size
            
            for i in range(num_steps):
                step = i * step_size
                
                # intキャストはコストがかかるので、必要な時だけ
                cx = int(x1 + dx * step)
                cy = int(y1 + dy * step)

                # 1. Check bounds (Numpy indexing would crash otherwise)
                if not (0 <= cx < W and 0 <= cy < H):
                    measured_distance = step / self.pixels_per_meter
                    intersected = True
                    break

                # 2. Check obstacle using Numpy Array Access (爆速)
                # map_array[x, y] が True なら障害物
                if map_array[cx, cy]:
                    measured_distance = step / self.pixels_per_meter
                    intersected = True
                    break

            if intersected:
                data.append([measured_distance, angle])
            else:
                data.append([self.out_of_range_value, angle])

        return data

    def get_closest_obstacle_info(self, sensor_data):
        if not sensor_data:
            return float("inf"), 0.0

        closest_element = min(sensor_data, key=lambda x: x[0])
        closest_distance = closest_element[0]
        closest_angle = closest_element[1]

        normalized_angle = (closest_angle + math.pi) % (2 * math.pi) - math.pi

        return closest_distance, normalized_angle