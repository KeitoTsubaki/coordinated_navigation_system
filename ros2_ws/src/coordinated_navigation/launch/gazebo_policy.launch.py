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

def generate_launch_description():
    # Get package directories
    pkg_gazebo = get_package_share_directory('gazebo_ros')
    pkg_coordination = get_package_share_directory('coordinated_navigation')
    
    # URDF / Xacro
    xacro_file = os.path.join(pkg_coordination, 'description', 'whill.urdf.xacro')
    urdf = xacro.process_file(xacro_file)

    # World
    # world_file_path = os.path.join(pkg_coordination, 'worlds', 'test.world')
    world_file_path = os.path.join(pkg_coordination, 'worlds', 'empty_with_obstacles.world')

    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    namespace = LaunchConfiguration('namespace')
    policy_model_path = LaunchConfiguration('policy_model_path')
    rviz_navigate_to_pose_action = LaunchConfiguration('rviz_navigate_to_pose_action')

    map_dir = LaunchConfiguration(
        'map',
        default=os.path.join(
            pkg_coordination,
            'worlds',
            'room1_map',
            'room1_map.yaml',
        )
    )

    param_dir = LaunchConfiguration(
        'params_file',
        default=os.path.join(
            pkg_coordination,
            'params',
            'whill_nav2_params.yaml',
            )
        )
    
    # YAML Rewrite setting
    param_substitutions = {
        'use_sim_time': use_sim_time,
        'yaml_filename': map_dir
    }
    configured_params = ParameterFile(
        RewrittenYaml(
            source_file=param_dir,
            root_key=namespace,
            param_rewrites=param_substitutions,
            convert_types=True),
        allow_substs=True
    )

    # Rviz config file
    rviz_config_dir = os.path.join(
        pkg_coordination,
        'rviz',
        'whill_navigation2.rviz'
    )


    # Node definition

    # Include the robot_state_publisher to publish transforms
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace=namespace,
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
        namespace=namespace,
        output='screen',
        arguments=[
            '-topic', 'robot_description',   # The the robot description is published by the rsp node on the /robot_description topic
            '-entity', 'whill',
            '-robot_namespace', namespace,
            '-x', '-5.0', '-y', '-5.0', '-z', '0.0'
            ],        # The name of the entity to spawn (doesn't matter if you only have one robot)
    )

    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        namespace=namespace,
        name='map_server',
        output='screen',
        parameters=[configured_params],
    )

    # AMCL (Localization)
    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        namespace=namespace,
        name='amcl',
        output='screen',
        parameters=[configured_params],
    )

    # Coordinated navigation node
    coordinated_nav = Node(
        package='coordinated_navigation',
        executable='coordinated_navigation_node',
        name='coordinated_navigation',
        output='screen',
        parameters=[
            {
                'use_sim_time': use_sim_time,
                'robot_namespace': namespace,
                'policy.model_path': policy_model_path,
            }
        ],
    )

    # Lifecycle Manager for map_server and amcl
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        namespace=namespace,
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': ['map_server', 'amcl']
        }]
    )

    # Rviz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_dir],
        namespace=namespace,
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'namespace',
            default_value='',   # TODO: bug occurs when namespace is exist
            description='Top-level namespace'
        ),
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
        DeclareLaunchArgument(
            'policy_model_path',
            default_value=os.path.join(
                pkg_coordination,
                'models',
                'model_ep3700.zip'

                # 'model_ep11200.zip'
                # 'model_ep10800.zip'

                # 'model_ep16500.zip' # best model but observation adjustment is needed
            ),
            description='Absolute path to the trained policy model zip file.'
        ),
        rsp,
        gazebo,
        spawn_room,
        spawn_whill,
        map_server,
        amcl,
        lifecycle_manager,
        coordinated_nav,
        rviz,
    ])
