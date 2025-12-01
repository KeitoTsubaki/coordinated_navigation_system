from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, LogInfo, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration
import os
import xacro

# ===== Keyboard control =====================================================================
# ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/cmd_vel
# ============================================================================================

# ===== How to save maps ======================================================
# [Save Map]        -> save .pgm and .yaml for nav2 and viewer
# [Serialize Map]   -> save .posegraph and .data for slam_toolbox localization
# =============================================================================

def generate_launch_description():

    ld = LaunchDescription()

    # Get package directories
    pkg_gazebo = get_package_share_directory('gazebo_ros')
    pkg_coordination = get_package_share_directory('coordinated_navigation')
    pkg_slam_toolbox = get_package_share_directory('slam_toolbox')
    
    # URDF / Xacro
    xacro_file = os.path.join(pkg_coordination, 'description', 'whill.urdf.xacro')
    urdf = xacro.process_file(xacro_file)

    # World
    # world_file_path = os.path.join(pkg_coordination, 'worlds', 'test.world')
    world_file_path = os.path.join(pkg_coordination, 'worlds', 'empty_with_obstacles.world')

    rviz_config_file = os.path.join(pkg_coordination, 'rviz', 'slam_toolbox.rviz')


    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    param_file = LaunchConfiguration(
        'params_file',
        default=os.path.join(
            pkg_coordination,
            'params',
            'whill_nav2_params.yaml',
            )
        )
    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace',
        default_value='',   # TODO: bug occurs when namespace is exist
        description='Top-level namespace'
    )

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=param_file,
        description='Full path to param file to load'
    )
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation/Gazebo clock'
    )

    slam_param_file = LaunchConfiguration('slam_params_file')
    declare_slam_params_file_cmd = DeclareLaunchArgument(
        'slam_params_file',
        default_value=os.path.join(
            pkg_coordination,
            'params',
            'slam_toolbox_slam.yaml'
        ),
        description='Full path to SLAM Toolbox param file to load'
    )


    # Node definition

    # Include the robot_state_publisher to publish transforms
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': urdf.toxml()},
            {'use_sim_time': use_sim_time},
            ]
    )

    # Include the Gazebo launch file, provided by the gazebo_ros package
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            pkg_gazebo, 'launch', 'gazebo.launch.py')]
        ),
        launch_arguments={
            'world': world_file_path,
            'use_sim_time': 'true',    
        }.items()
    )

    start_slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            pkg_slam_toolbox, 'launch', 'online_async_launch.py')]
        ),
        launch_arguments={
            'use_sim_time': 'true',
            'slam_params_file': slam_param_file,
        }.items()
    )

    room_sdf = os.path.join(pkg_coordination, 'worlds', 'room1', 'model.sdf')
    spawn_room = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=[
            '-file', room_sdf,
            '-entity', 'room',
            '-x', '0.0', '-y', '0.0', '-z', '0.0'
        ],
    )

    spawn_whill = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=[
            '-topic', 'robot_description',   # The the robot description is published by the rsp node on the /robot_description topic
            '-entity', 'whill',
            '-x', '-5.0', '-y', '-5.0', '-z', '0.0'
            ],        # The name of the entity to spawn (doesn't matter if you only have one robot)
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )


    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_slam_params_file_cmd)


    ld.add_action(gazebo)
    ld.add_action(spawn_room)
    ld.add_action(rsp)
    ld.add_action(spawn_whill)

    ld.add_action(start_slam_toolbox)

    ld.add_action(rviz)


    return ld
