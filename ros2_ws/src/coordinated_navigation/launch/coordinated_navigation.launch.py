"""Launch description for the coordinated navigation node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    robot_id = LaunchConfiguration("robot_id")
    robot_namespace = LaunchConfiguration("robot_namespace")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_id",
                default_value="1",
                description="Identifier appended to the default /robot_<id> namespace.",
            ),
            DeclareLaunchArgument(
                "robot_namespace",
                default_value="",
                description=(
                    "Override the namespace used for robot-specific topics. "
                    "If unset the node derives /robot_<robot_id>."
                ),
            ),
            Node(
                package="coordinated_navigation",
                executable="coordinated_navigation_node",
                name="coordinated_navigation",
                output="screen",
                parameters=[
                    {
                        "robot_id": robot_id,
                        "robot_namespace": robot_namespace,
                    }
                ],
            ),
        ]
    )
