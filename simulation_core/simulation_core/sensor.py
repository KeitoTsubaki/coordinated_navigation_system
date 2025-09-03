# Sensor functions

import math
import pygame
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


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


class LiDARSensor():

    def __init__(
        self, Range, map, uncertainty
    ):  # uncertainty is an array with 2 values [sigma1,sigma2]
        self.Range = Range
        self.speed = 0.5  # rotations per second #IS NOT USED THROUGHOUT THIS SCRIPT
        self.sigma = np.array([uncertainty[0], uncertainty[1]])
        self.position = (
            0,
            0,
        )  # <<< self.position should update to cursor's position!!!! BUG NEEDS TO BE FIXED
        self.theta = 0
        # self.map = map
        self.W, self.H = pygame.display.get_surface().get_size()
        self.sensedObstacles = []

    # Euclidean DISTANCE CALCULATOR
    def distance(self, obstaclePosition):
        px = (obstaclePosition[0] - self.position[0]) ** 2
        py = (obstaclePosition[1] - self.position[1]) ** 2
        return math.sqrt(px + py)

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
        data = []  # stores distance and angle of ONLY WALLS from bot's position
        x1, y1 = self.position[0], self.position[1]
        obstacle_color_rgb = self.hex_to_rgb(obstacle_color)  # convert hex color to RGB
        for angle in np.linspace(
            0, 2 * math.pi, 180, False
        ):  # scan from 0 to 2pi, 6 degree intervals (Resolution). For every angle, check if there is a wall
            x2, y2 = (
                x1 + self.Range * math.cos(angle),
                y1 + self.Range * math.sin(angle),
            )  # coordinate of end of line segment

            
            for i in range(0, 100):  # If wall is within laser's path ...
                u = i / 100  # Essentially a percentage
                x = int(x1 + (x2 - x1) * u)
                y = int(y1 + (y2 - y1) * u)

                """
                x = int(x2*u + x1*(1-u))
                y = int(y2*u + y1*(1-u))
                """

                if (
                    0 < x < self.W and 0 < y < self.H
                ):  # if within the window/map. Reference point is pixel (0,0).
                    color = map.get_at(
                        (x, y)
                    )  # extract RGB value on map at every point of iteration within laser's path and do the quick check below ...
                    if (
                        color[0],
                        color[1],
                        color[2],
                    ) == obstacle_color_rgb:  # if color is black, aka the walls, calculate this distance from the robot
                        distance = self.distance((x, y))  # References method above
                        # print("distance: ", distance)
                        # output = uncertainty_add(distance,angle, self.sigma) #add uncertainty to measurements
                        output = [distance, angle]  # output is [Distance, Angle]
                        output.append(
                            self.position
                        )  # add robot's position to list. Robot is always at (0,0) position though... ??? BUG
                        data.append(
                            output
                        )  # [Distance, Angle, (x,y)] --> This is the return
                        break
                else:
                    distance = self.distance((x, y))  # References method above
                    # print("distance: ", distance)
                    # output = uncertainty_add(distance,angle, self.sigma) #add uncertainty to measurements
                    output = [distance, angle]  # output is [Distance, Angle]
                    output.append(
                        self.position
                    )  # add robot's position to list. Robot is always at (0,0) position though... ??? BUG
                    data.append(
                        output
                    )  # [Distance, Angle, (x,y)] --> This is the return
                    break

        # when sensor completes a full turn, return the data to be drawn in the map ...which is the responsiblity of the buildEnvironment class in the env.py file
        if len(data) > 0:
            return data  # [Distance, Angle, (x,y)] --> This is the return
        else:
            return False

    def convert_to_pointcloud2(
        self, sensor_data, time_stamp, pixels_per_meter, frame_id="base_link"
    ):
        header = Header()
        header.stamp = time_stamp
        header.frame_id = frame_id

        points = []
        # sensor_data: [[(distance, angle(radian), (robot_x, robot_y)], ...]
        # print("sensor_data:", sensor_data)
        for point_data in sensor_data:
            # print(point_data)
            distance = point_data[0]
            angle = point_data[1]
            angle -= self.theta  # add robot's orientation to angle
            # x_distance = (
            #     point_data[2][0] + distance * math.cos(angle)
            # ) / pixels_per_meter
            # y_distance = (
            #     point_data[2][1] + distance * math.sin(angle)
            # ) / pixels_per_meter
            # robot centoroid
            # x_distance = (distance * math.cos(angle)) / pixels_per_meter
            # y_distance = (distance * math.sin(angle)) / pixels_per_meter
            y_distance = -(distance * math.cos(angle)) / pixels_per_meter
            x_distance = -(distance * math.sin(angle)) / pixels_per_meter
            
            
            # print("x_distance, y_distance: ", x_distance, y_distance)
            # x_distance = point_data[0]
            # y_distance = point_data[1]
            points.append([x_distance, y_distance, 0.0])  # 2Dデータのためz=0
            # points.append([-y_distance, -x_distance, 0.0])  # 2Dデータのためz=0

        # PointCloud2のフィールドを定義
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # Python listをNumPy arrayに変換
        points_array = np.array(points, dtype=np.float32)

        # NumPy配列をバイトデータに変換
        # tobytes()はPython 3で推奨される方法
        bytes_data = points_array.tobytes()

        # PointCloud2メッセージの作成
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 12  # 3 fields * 4 bytes/field
        msg.row_step = 12 * len(points)
        msg.data = bytes_data
        msg.is_dense = True

        return msg
    
    def get_closest_obstacle_distance(self, sensor_data):
        if sensor_data:
            distances = [data[0] for data in sensor_data]
            return min(distances)
        else:
            return float('inf')
