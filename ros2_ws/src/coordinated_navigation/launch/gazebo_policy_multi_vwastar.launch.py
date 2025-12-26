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
    # Room1
    # robots = [
    #     {'name': 'whill1', 'x_pose': '-6.0', 'y_pose': '3.0', 'z_pose': '0.1'},
    #     {'name': 'whill2', 'x_pose': '-6.0', 'y_pose': '2.0', 'z_pose': '0.1'},
    #     # ...
    # ]
    # Room2
    # robots = [
    #     {'name': 'whill1', 'x_pose': '-0.75', 'y_pose': '-7.0', 'z_pose': '0.1', 'head_yaw': '-1.4'},
    #     {'name': 'whill2', 'x_pose': '0.75', 'y_pose': '-7.0', 'z_pose': '0.1', 'head_yaw': '1.4'},
    #     # ...
    # ]
    # whill_yaw = '1.5708'
    # robots = [
    #     {'name': 'whill1', 'x_pose': '-7.0', 'y_pose': '-5.0', 'z_pose': '0.1', 'head_yaw': '-1.4'},
    #     {'name': 'whill2', 'x_pose': '-7.0', 'y_pose': '-6.0', 'z_pose': '0.1', 'head_yaw': '1.4'},
    #     # ...
    # ]
    # whill_yaw = '0.0'

    # Room3
    robots = [
        {'name': 'whill1', 'x_pose': '6.0', 'y_pose': '-13.0', 'z_pose': '0.1', 'head_yaw': '-1.4'},
        {'name': 'whill2', 'x_pose': '7.5', 'y_pose': '-13.0', 'z_pose': '0.1', 'head_yaw': '1.4'},
        # ...
    ]
    whill_yaw = '1.5708'

    # Get package directories
    pkg_gazebo = get_package_share_directory('gazebo_ros')
    pkg_coordination = get_package_share_directory('coordinated_navigation')
    # pkg_vwastar = get_package_share_directory('vwastar') # 必要であれば取得

    # URDF / Xacro
    xacro_file = os.path.join(pkg_coordination, 'description', 'whill.urdf.xacro')
    

    # World
    # world_file_path = os.path.join(pkg_coordination, 'worlds', 'test.world')
    # world_file_path = os.path.join(pkg_coordination, 'worlds', 'empty_with_obstacles.world')
    world_file_path = os.path.join(pkg_coordination, 'worlds', 'empty.world')

    # room_sdf = os.path.join(pkg_coordination,'worlds','room1','model.sdf')
    # room_sdf = os.path.join(pkg_coordination,'worlds','room2','model.sdf')
    # room_sdf = os.path.join(pkg_coordination,'worlds','room3','model.sdf')
    room_sdf = os.path.join(pkg_coordination,'worlds','room4','model.sdf')

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
            # 'model_ep3700.zip'    # 2 0
            # 'model_ep11200.zip'     # 2 0
            # 'model_ep10800.zip'   # 3 2,3
            'model_ep16500.zip',  # 3 2,3
            # 'model_ep3700_2.zip'    # 3 4
        ),
        description='Absolute path to the trained policy model zip file.'
    )
    
    map_file = LaunchConfiguration(
        'map',
        default=os.path.join(
            pkg_coordination,
            'worlds',
            # 'room1_map',
            # 'room1_map.yaml',
            # 'room2_map',
            # 'room2_map.yaml',
            # 'room3_map',
            # 'room3_map.yaml',
            'room4_map',
            'room4_map.yaml',
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
            # 'room1_localization',
            # 'room1'
            # 'room2_localization',
            # 'room2'
            # 'room3_localization',
            # 'room3'
            'room4_localization',
            'room4'
        ),
    )
    declare_map_dir = DeclareLaunchArgument(
        'map_dir',
        default_value=map_dir,
        description='Full path to map file to load'
    )

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
            'multi_whill_navigation2_vwastar.rviz'
        )
    )
    declare_rviz_config = DeclareLaunchArgument(
        'rviz_config',
        default_value=rviz_config,
        description='Full path to rviz config file to load'
    )

    
    
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_policy_model_path)
    ld.add_action(declare_map_file)
    ld.add_action(declare_map_dir)
    ld.add_action(declare_slam_params_file)
    ld.add_action(declare_rviz_config)

    
    # ===== Node definition =====

    # Include the Gazebo launch file
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
            '-x', '0.0', '-y', '0.0', '-z', '0.0',
        ],
    )

    ld.add_action(gazebo)
    ld.add_action(spawn_room)

    # ===== Global Map Server & Planner =====
    # VWAstarノードがグローバルマップ(/map)を購読するために必要

    # 1. Map Server (Global)
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[
            {'yaml_filename': map_file},
            {'use_sim_time': use_sim_time},
        ],
    )

    map_server_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map_server',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'autostart': True},
            {'node_names': ['map_server']}
        ]
    )

    ld.add_action(map_server)
    ld.add_action(map_server_lifecycle)

    # 2. VWAstar Node (Global Planner Service)
    vwastar_node = Node(
        package='vwastar',
        executable='vwastar',
        name='vwastar_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'InflateRadiusList': [2.0, 1.0], # ロボットサイズに合わせて調整
            'FormationCosts': [0.0, 10.0],
            'rate': 10,
            'ros_service': True
        }]
    )
    ld.add_action(vwastar_node)

    last_action = None
    # ===== Spawn Robots =====
    for robot in robots:
        prefix = robot['name'] + '/'
        namespace =  '/' + robot['name'] 

        urdf = xacro.process_file(
            xacro_file,
            mappings={
                'prefix': prefix,
                'head_yaw': robot.get('head_yaw', '0.0')
            }
        )

        # Robot State Publisher
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
                '-topic', 'robot_description',
                '-entity', robot['name'],
                '-robot_namespace', namespace,
                '-x', robot['x_pose'], '-y', robot['y_pose'], '-z', robot['z_pose'],
                '-Y', whill_yaw,
                '-unpause',
            ],
            output='screen',
        )

        # ===== SLAM Toolbox Localization (Namespace) =====
        start_async_slam_toolbox_node = Node(
            parameters=[
                slam_param_fle,
                {'use_sim_time': use_sim_time},
                {'map_file_name': map_dir},
                {'map_start_pose': [float(robot['x_pose']), float(robot['y_pose']), float(whill_yaw)]},
                {'odom_frame': robot['name'] + '/odom'},
                {'base_frame': robot['name'] + '/base_footprint'}
            ],
            namespace=namespace,
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            remappings=[
                ('/scan', 'scan'),
                # マップは各名前空間内(/whillX/map)で配信される
            ],
            output='screen'
        )

        if last_action is None:
            ld.add_action(rsp)
            ld.add_action(spawn_whill)
            ld.add_action(start_async_slam_toolbox_node)
        else: 
            spawn_whill_event = RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=last_action,
                    on_exit=[rsp, spawn_whill, start_async_slam_toolbox_node],
                )
            )
            ld.add_action(spawn_whill_event)

        last_action = spawn_whill


    # ===== Navigation & Viz per Robot =====
    for robot in robots:
        namespace =  '/' + robot['name'] 

        rid = robot['name'].replace('whill', '')

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
                ('/goal_pose', 'goal_pose'),
                ('/clicked_point', 'clicked_point'),
                ('/initialpose', 'initialpose'),
                ('/scan', 'scan'),
                ('/robot_description', 'robot_description')
            ],
            output='screen'
        )

        # Coordinated navigation node (Updated to use VWAstar service)
        my_role = 'leader' if rid == '1' else 'follower'
        my_partner = 'whill2' if rid == '1' else 'whill1'
        
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
                    'robot_id': rid,                # これで左右が分かれます
                    'formation.width': 1.5,         # ロボット間の距離 (m)
                    'formation.offset_mode': 'auto',
                    'navigation.lookahead_distance': 1.0,
                    'role': my_role,
                    'partner_name': my_partner,
                }
            ],
        )

        ld.add_action(start_rviz)
        ld.add_action(coordinated_nav)

    


    return ld