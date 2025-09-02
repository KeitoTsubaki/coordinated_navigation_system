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
import random
import os

import yaml
import pygame
import numpy as np

from simulation_core.environment import SimCoreEnvironment


class Environment(SimCoreEnvironment):
    pass


def main():
    pygame.init()

    env = Environment()
    
    env.render_window()

    print((env.map_center_x, env.map_center_y))
    while env.running:

        env.clear_lidar_layer()  # Reset the lidar line while remaining the map and obstacles

        env.change_map_by_event()

        env._operate_moving()  # to handle robot movement

        sensor_data = env.lidar_scan()
        env.show_sensorData(sensor_data)

        pygame.display.update() # ウィンドウを更新

    # 4. 終了処理
    pygame.quit()


if __name__ == "__main__":
    main()
