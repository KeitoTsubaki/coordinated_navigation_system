from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    planner = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'expected_planner_frequency': 1.0,
            'planner_plugins': ['GridBased'],
            'GridBased.plugin': 'nav2_smac_planner/SmacPlannerHybrid',
            'GridBased.tolerance': 0.5,
            'GridBased.smooth_path': True,
            'GridBased.downsample_costmap': False,
            'GridBased.allow_unknown': True,
            'GridBased.motion_model_for_search': 'DUBIN',
            'GridBased.angle_quantization_bins': 72,
            'GridBased.minimum_turning_radius': 0.4,
            'GridBased.cost_travel_multiplier': 2.0,
            'GridBased.lookup_table_size': 20.0,
        }]
    )

    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': '/home/aisl/coordinated_navigation_system/ros2_ws/src/maps/map1/map1.yaml'}]
    )

    lifecycle_mgr = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': ['map_server', 'planner_server']
        }]
    )

    return LaunchDescription([map_server, planner, lifecycle_mgr])
