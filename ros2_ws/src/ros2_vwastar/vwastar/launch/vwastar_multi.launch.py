import os
from ament_index_python import get_package_share_directory
from launch_ros.actions import Node 
import launch
from launch import LaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


# Share directories
pkg_ros2_vwastar = get_package_share_directory('ros2_vwastar')
pkg_vwastar = get_package_share_directory('vwastar')

def setup(context, *args, **kwargs):

    use_sim_time = LaunchConfiguration('use_sim_time')
    rviz_config_file = LaunchConfiguration('rviz_config_file')
    rviz_frame = LaunchConfiguration('rviz_frame')

    start_rviz_cmd = Node(
        package='rviz2',
        executable='rviz2',
        arguments=[
            '-d', rviz_config_file,
            '-f', rviz_frame,
        ],
        parameters=[{
                'use_sim_time': use_sim_time,
            }],
        output={'both': 'log'}
    )

    vwastar_algorithm = Node(
        package='vwastar',
        executable='vwastar',
        name='vwastar_node',
        output='screen',
        parameters=[os.path.join(pkg_vwastar, 'config', 'params.yaml')]
    )

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link'],
        output='screen'
    )

    nodes_to_start = []
    nodes_to_start.append(start_rviz_cmd)
    nodes_to_start.append(vwastar_algorithm)
    nodes_to_start.append(static_tf)


    return nodes_to_start

def generate_launch_description():

    return LaunchDescription([

        launch.actions.DeclareLaunchArgument(
            'use_sim_time',
            default_value='True'
        ),
        launch.actions.DeclareLaunchArgument(
            name='rviz_config_file', 
            default_value=os.path.join(
                pkg_vwastar,
                'rviz',
                'config.rviz'
            ),
            description='Rviz config file path'
        ),
        launch.actions.DeclareLaunchArgument(
            name='map_file', 
            default_value=os.path.join(
                pkg_ros2_vwastar,
                # 'maps/empty_world',
                # 'scenario_1.yaml'
                'maps/room2_map',
                'room2_map.yaml'
            ),
            description='Map file path'
        ),
        launch.actions.DeclareLaunchArgument(
            'rviz_frame',
            default_value='map',
            description='Fixed frame'
        ),

        OpaqueFunction(function=setup)
])