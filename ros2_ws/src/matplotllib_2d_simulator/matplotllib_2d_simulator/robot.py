"""
Defines the robot's state, physics, and sensor simulation

This class will manage the robot's state and simulate its sensors. It does not handle ROS2
communication directly.

- `__init__(self, config)`:
    - Initializes the robot's pose (x, y, theta) and velocity.
    - Stores LiDAR parameters from the config.
- `update_pose(self, linear_vel, angular_vel, dt)`:
    - Updates the robot's position and orientation based on the velocity commands and the time delta
        (dt). This is basic kinematics.
- `simulate_lidar(self, environment)`:
    - This is a key method. From the robot's current pose, it will:
    - Cast a series of rays (e.g., 360 rays for a 360-degree LiDAR) within the environment.
    - For each ray, calculate the intersection point with the nearest obstacle.
    - Return a list of these intersection points (the point cloud).
"""

