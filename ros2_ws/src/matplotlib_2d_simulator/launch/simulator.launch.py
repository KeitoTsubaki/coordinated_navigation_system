from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='matplotlib_2d_simulator',
            executable='simulator_node',
            name='simulator_node',
            output='screen',
        ),
    ])
