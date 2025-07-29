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