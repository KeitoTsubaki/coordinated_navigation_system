# Utility functions
import math

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")  # 先頭の # を削除
    # 6文字 (RRGGBB) でなければエラー
    if len(hex_color) != 6:
        raise ValueError("カラーコードは6桁の16進数で指定してください。")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)

# Uses Distance and Angle from data[] in sensors.sensesObstacles() to obtain cartesian coordinates of walls for point cloud
def AD2pos(distance, angle, robotPosition, pixels_per_meter):
    x = robotPosition[0] + distance * pixels_per_meter * math.cos(angle)
    y = robotPosition[1] + distance * pixels_per_meter * math.sin(angle)
    return (int(x), int(y))  # returns cartesian coordinates of walls.