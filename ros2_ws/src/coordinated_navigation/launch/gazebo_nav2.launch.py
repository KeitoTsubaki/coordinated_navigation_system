from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, LogInfo, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration
import os
import xacro

def generate_launch_description():
    pkg_nav2 = get_package_share_directory('nav2_bringup')
    pkg_gazebo = get_package_share_directory('gazebo_ros')
    pkg_coordination = get_package_share_directory('coordinated_navigation')
    
    xacro_file = os.path.join(pkg_coordination, 'description', 'whill.urdf.xacro')
    urdf = xacro.process_file(xacro_file)
    
    set_env_library = SetEnvironmentVariable(
        name='LD_LIBRARY_PATH',
        value=os.environ.get('LD_LIBRARY_PATH', '') + ':/opt/ros/humble/lib'
    )

    set_env_ament = SetEnvironmentVariable(
        name='AMENT_PREFIX_PATH',
        value=os.environ.get('AMENT_PREFIX_PATH', '') + ':/opt/ros/humble'
    )

    world_file_path = os.path.join(pkg_coordination, 'worlds', 'test.world')
    robot_path = '/path/to/your_robot.urdf'

    use_sim_time = LaunchConfiguration(
        'use_sim_time',
        default='true'
    )
    map_dir = LaunchConfiguration(
        'map',
        default=os.path.join(
            get_package_share_directory('coordinated_navigation'),
            'worlds',
            'room1_map',
            'room1_map.yaml',
        )
    )

    # Params file
    param_dir = LaunchConfiguration(
            'params_file',
            default=os.path.join(
                pkg_coordination,
                'params',
                'whill_nav2_params.yaml',
                )
            )

    # Rviz config file
    rviz_config_dir = os.path.join(
        get_package_share_directory('coordinated_navigation'),
        'rviz',
        'tb3_navigation2.rviz'
    )



    # Include the robot_state_publisher to publish transforms
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': urdf.toxml()}]
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

    return LaunchDescription([
        set_env_library,
        set_env_ament,
        DeclareLaunchArgument(
          'map',
          default_value=map_dir,
          description='Full path to map file to load'
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=param_dir,
            description='Full path to param file to load'
        ),
        DeclareLaunchArgument(
          'use_sim_time',
          default_value='true',
          description='Use simulation/Gazebo clock'
        ),
        rsp,
        gazebo,
        spawn_room,
        spawn_whill,
        LogInfo(msg=  ["map_dir is ", map_dir]),
        # This action executes other launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_nav2, 'launch', 'bringup_launch.py')
            ),
            launch_arguments={
                'map': map_dir,
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'params_file': str(os.path.join(pkg_coordination, 'params', 'whill_nav2_params.yaml')),
            }.items(),
        ),
        LogInfo(msg=  ["rviz_config_dir is ", rviz_config_dir]),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_dir],
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }],
            output='screen'
        ),
    ])
