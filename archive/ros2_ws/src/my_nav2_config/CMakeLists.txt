cmake_minimum_required(VERSION 3.5)
project(my_nav2_config)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
