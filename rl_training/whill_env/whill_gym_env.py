import gymnasium as gym
import rclpy
from rclpy.node import Node

class WhillEnv(gym.Env):
    def __init__(self, config):
        super(WhillEnv, self).__init__()
        
        self.observation_space = None
        self.action_space = None
        
        # Initialize ROS 2 node
        rclpy.init(args=None)
        self.node = Node('whill_env_node')
        
        self.lidar_sub = self.node.create_subscription(
            msg_type='sensor_msgs/msg/LaserScan',
            topic='/whill/lidar',
            callback=self.lidar_callback,
            qos_profile=10
        )
        self.cmd_vel_pub = self.node.create_publisher(
            msg_type='geometry_msgs/msg/Twist',
            topic='/whill/cmd_vel',
            qos_profile=10
        )
        