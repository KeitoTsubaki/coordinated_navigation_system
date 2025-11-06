from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_nav2 = get_package_share_directory('nav2_bringup')
    pkg_gazebo = get_package_share_directory('gazebo_ros')

    world_path = os.path.join(pkg_gazebo, 'worlds', 'empty.world')
    robot_path = '/path/to/your_robot.urdf'

    return LaunchDescription([
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-file', robot_path, '-entity', 'my_robot'],
            output='screen'
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_nav2, 'launch', 'bringup_launch.py')),
            launch_arguments={
                'use_sim_time': 'true',
                'map': '/path/to/map.yaml',
                'params_file': '/path/to/nav2_params.yaml'
            }.items(),
        ),
    ])
