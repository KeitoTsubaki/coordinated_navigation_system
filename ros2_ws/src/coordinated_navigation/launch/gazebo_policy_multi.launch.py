from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, LogInfo, SetEnvironmentVariable, ExecuteProcess, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml, ReplaceString
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessExit
import os
import xacro

def generate_launch_description():
    ld = LaunchDescription()

    # Names and poses of the robots
    robots = [
        {'name': 'whill1', 'x_pose': '-5.0', 'y_pose': '-5.0', 'z_pose': '0.0'},
        {'name': 'whill2', 'x_pose': '5.0', 'y_pose': '-5.0', 'z_pose': '0.0'},
        # ...
    ]

    # Get package directories
    pkg_gazebo = get_package_share_directory('gazebo_ros')
    pkg_coordination = get_package_share_directory('coordinated_navigation')
    
    # URDF / Xacro
    xacro_file = os.path.join(pkg_coordination, 'description', 'whill.urdf.xacro')
    

    # World
    # world_file_path = os.path.join(pkg_coordination, 'worlds', 'test.world')
    world_file_path = os.path.join(pkg_coordination, 'worlds', 'empty_with_obstacles.world')

    room_sdf = os.path.join(pkg_coordination,'worlds','room1','model.sdf')

    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation/Gazebo clock'
    )
    
    policy_model_path = LaunchConfiguration('policy_model_path')
    declare_policy_model_path = DeclareLaunchArgument(
        'policy_model_path',
        default_value=os.path.join(
            pkg_coordination,
            'models',
            'model_ep3700.zip'
            # 'model_ep11200.zip'
            # 'model_ep10800.zip'
            # 'model_ep16500.zip'
        ),
        description='Absolute path to the trained policy model zip file.'
    )
    
    map_file = LaunchConfiguration(
        'map',
        default=os.path.join(
            pkg_coordination,
            'worlds',
            'room1_map',
            'room1_map.yaml',
        )
    )
    declare_map_file = DeclareLaunchArgument(
        'map',
        default_value=map_file,
        description='Full path to map file to load'
    )

    map_dir = LaunchConfiguration(
        'map_dir',
        default=os.path.join(
            pkg_coordination,
            'worlds',
            'room1_localization',
            'room1'
        ),
    )
    declare_map_dir = DeclareLaunchArgument(
        'map_dir',
        default_value=map_dir,
        description='Full path to map file to load'
    )


    # param_file = LaunchConfiguration(
    #     'params_file',
    #     default=os.path.join(
    #         pkg_coordination,
    #         'params',
    #         'multi_whill_nav2_params.yaml',
    #     )
    # )
    # declare_param_file = DeclareLaunchArgument(
    #     'params_file',
    #     default_value=param_file,
    #     description='Full path to param file to load'
    # )

    # SLAM Toolbox param file
    slam_param_fle = LaunchConfiguration(
        'slam_params_file',
        default=os.path.join(
            pkg_coordination,
            'params',
            'slam_toolbox_localization.yaml'
        )
    )
    declare_slam_params_file = DeclareLaunchArgument(
        'slam_params_file',
        default_value=slam_param_fle,
        description='Full path to SLAM Toolbox param file to load'
    )

    # Rviz config file
    rviz_config = LaunchConfiguration(
        'rviz_config',
        default=os.path.join(
            pkg_coordination,
            'rviz',
            'multi_whill_navigation2.rviz'
        )
    )
    declare_rviz_config = DeclareLaunchArgument(
        'rviz_config',
        default_value=rviz_config,
        description='Full path to rviz config file to load'
    )

    
    
    ld.add_action(declare_use_sim_time)
    # ld.add_action(declare_namespace)
    ld.add_action(declare_policy_model_path)
    ld.add_action(declare_map_file)
    # ld.add_action(declare_param_file)
    ld.add_action(declare_map_dir)
    ld.add_action(declare_slam_params_file)
    ld.add_action(declare_rviz_config)

    
    # ===== Node definition =====

    

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

    ld.add_action(gazebo)
    ld.add_action(spawn_room)

    
    

    # remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    last_action = None
    # ===== Spawn =====
    for robot in robots:
        prefix = robot['name'] + '/'
        namespace =  '/' + robot['name'] 
        # for i in range(10):
        #     print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

        urdf = xacro.process_file(
            xacro_file,
            mappings={
                'prefix': prefix,
            }
        )

        # Include the robot_state_publisher to publish transforms
        rsp = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace=namespace,
            output='screen',
            parameters=[
                {'robot_description': urdf.toxml()},
                {'use_sim_time': use_sim_time},
                ],
        )

        spawn_whill = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            namespace=namespace,
            arguments=[
                '-topic', 'robot_description',   # The the robot description is published by the rsp node on the /robot_description topic
                '-entity', robot['name'],
                '-robot_namespace', namespace,
                '-x', robot['x_pose'], '-y', robot['y_pose'], '-z', robot['z_pose'],
                '-unpause',
            ],
            output='screen',
        )

        # YAML Rewrite setting
        # param_substitutions = {
        #     'use_sim_time': use_sim_time,
        #     'yaml_filename': map_file
        # }
        # namespaced_param_file = ReplaceString(
        #     source_file=param_file,
        #     replacements={'<robot_namespace>': namespace}
        # )
        # configured_params = ParameterFile(
        #     RewrittenYaml(
        #         source_file=param_file,
        #         root_key=namespace,
        #         param_rewrites=param_substitutions,
        #         convert_types=True),
        #     allow_substs=True
        # )
        

        # AMCL (Localization)
        # amcl = Node(
        #     package='nav2_amcl',
        #     executable='amcl',
        #     namespace=namespace,
        #     name='amcl',
        #     output='screen',
        #     parameters=[configured_params],
        #     remappings=remappings,
        # )

        

        # # # Lifecycle Manager for map_server and amcl
        
        # amcl_lifecycle = Node(
        #     package='nav2_lifecycle_manager',
        #     executable='lifecycle_manager',
        #     name='lifecycle_manager_amcl',
        #     namespace=namespace,
        #     output='screen',
        #     parameters=[
        #         {'use_sim_time': use_sim_time},
        #         {'autostart': True},
        #         {'node_names': ['amcl']}
        #     ]
        # )

        # ===== SLAM Toolbox Localization =====
        start_async_slam_toolbox_node = Node(
            parameters=[
                slam_param_fle,
                {'use_sim_time': use_sim_time},
                {'map_file_name': map_dir},
                {'map_start_pose': [float(robot['x_pose']), float(robot['y_pose']), float(robot['z_pose'])]},
                {'odom_frame': robot['name'] + '/odom'},
                {'base_frame': robot['name'] + '/base_footprint'}
            ],
            namespace=namespace,
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            remappings=[
                # ('/tf', 'tf'),
                # ('/tf_static', 'tf_static'),
                ('/scan', 'scan'),
                # ('/map', 'map')
            ],
            output='screen'
        )

        if last_action is None:
            ld.add_action(rsp)
            ld.add_action(spawn_whill)
            
            # ld.add_action(amcl)
            # ld.add_action(amcl_lifecycle)
            ld.add_action(start_async_slam_toolbox_node)
        else: 
            spawn_whill_event = RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=last_action,
                    # on_exit=[rsp, spawn_whill, amcl, amcl_lifecycle],
                    on_exit=[rsp, spawn_whill, start_async_slam_toolbox_node],
                    # on_exit=[rsp, spawn_whill],
                )
            )
            ld.add_action(spawn_whill_event)

        last_action = spawn_whill


    # remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    # map_server = Node(
    #     package='nav2_map_server',
    #     executable='map_server',
    #     name='map_server',
    #     output='screen',
    #     parameters=[
    #         {'yaml_filename': map_file},
    #     ],
    #     # remappings=remappings,
    # )
    # map_server_lifecycle = Node(
    #     package='nav2_lifecycle_manager',
    #     executable='lifecycle_manager',
    #     name='lifecycle_manager_map_server',
    #     output='screen',
    #     parameters=[
    #         {'use_sim_time': use_sim_time},
    #         {'autostart': True},
    #         {'node_names': ['map_server']}
    #     ]
    # )

    # ld.add_action(map_server)
    # ld.add_action(map_server_lifecycle)

    for robot in robots:
        namespace =  '/' + robot['name'] 

        # Set initial pose from spawn setting
        # message = '{header: {frame_id: map}, ' + \
        #     'pose: {pose: {position: {x: ' + robot['x_pose']+ \
        #     ', y: ' + robot['y_pose'] + \
        #     ', z: ' + robot['z_pose'] + \
        #     '}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}},}}'
        # initial_pose_pub_cmd = ExecuteProcess(
        #     cmd=['ros2', 'topic', 'pub', '-t', '1', '--qos-reliability', 'reliable',
        #         namespace + '/initialpose', 
        #         'geometry_msgs/msg/PoseWithCovarianceStamped', message],
        #     output='screen'
        # )
        namespaced_rviz_config = ReplaceString(
            source_file=rviz_config,
            replacements={'<robot_namespace>': namespace}
        )

        
        start_rviz = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', namespaced_rviz_config],
            namespace=namespace,
            parameters=[{
                'use_sim_time': use_sim_time,
            }],
            remappings=[
                # ('/tf', 'tf'),
                # ('/tf_static', 'tf_static'),
                # ('/clock', '/clock'),
                ('/goal_pose', 'goal_pose'),
                ('/clicked_point', 'clicked_point'),
                ('/initialpose', 'initialpose'),
                ('/scan', 'scan'),
                ('/robot_description', 'robot_description')
            ],
            output='screen'
        )

        # Coordinated navigation node
        coordinated_nav = Node(
            package='coordinated_navigation',
            executable='coordinated_navigation_node',
            name='coordinated_navigation',
            output='screen',
            namespace=namespace,
            parameters=[
                {
                    'use_sim_time': use_sim_time,
                    'robot_namespace': namespace,
                    'policy.model_path': policy_model_path,
                }
            ],
        )

        # post_spawn_event = RegisterEventHandler(
        #     event_handler=OnProcessExit(
        #         target_action=last_action,
        #         on_exit=[initial_pose_pub_cmd, start_rviz],
        #     )
        # )

        # last_action = initial_pose_pub_cmd

        

        # ld.add_action(post_spawn_event)
        ld.add_action(start_rviz)
        ld.add_action(coordinated_nav)

    return ld
