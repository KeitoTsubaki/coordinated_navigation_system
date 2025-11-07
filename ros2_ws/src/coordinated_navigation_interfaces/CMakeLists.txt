cmake_minimum_required(VERSION 3.8)
project(coordinated_navigation_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Formation.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

install(
  DIRECTORY msg
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
