"""
Main ROS2 node, handles simulation loop and visualization

This is the core of the application. It will be a ROS2 node that ties everything together.

- `__init__(self)`:
    - Initializes rclpy and creates a ROS2 node.
    - Loads the simulator_config.yaml.
    - Instantiates the Environment and Robot classes.
    - ROS2 Interface:
        - Creates a Publisher for the /cloud topic (sensor_msgs/PointCloud2).
        - Creates a Subscriber for the /cmd_vel topic (geometry_msgs/Twist) with a callback
            cmd_vel_callback.
    - Matplotlib Setup:
        - Creates a Matplotlib figure and axes for visualization. plt.ion() will be used for
            non-blocking plotting.
    - Keyboard Control:
        - Connects the Matplotlib key_press_event to a handler function on_key_press.
    - Main Loop:
        - Creates a ROS2 Timer that calls a simulation_step function at a regular interval (e.g.,
            30Hz).

- `simulation_step(self)`:
    - Called by the timer. This is the heartbeat of the simulator.
    - Updates the robot's pose using the latest velocity command and the timer's period.
    - Calls robot.simulate_lidar() to get the latest sensor data.
    - Converts the simulated point cloud data into a sensor_msgs/PointCloud2 message. The ros2_numpy
        or sensor_msgs_py.point_cloud2 library is very helpful here.
    - Publishes the PointCloud2 message.
    - Updates the Matplotlib plot with the new robot position and LiDAR scan.

- `cmd_vel_callback(self, msg)`:
    - Stores the received Twist message's linear and angular velocities to be used in the next
        simulation_step.

- `on_key_press(self, event)`:
    - Checks if the pressed key is an arrow key.
    - Creates a Twist message with the corresponding linear/angular velocity.
    - Publishes the Twist message to /cmd_vel.

"""