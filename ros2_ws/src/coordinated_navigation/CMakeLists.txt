cmake_minimum_required(VERSION 3.8)
project(coordinated_navigation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(coordinated_navigation_interfaces REQUIRED)

set(_policy_pip_requirement "stable-baselines3[extra]>=2.7.0")
set(_policy_check_script "import sys\nfrom importlib import metadata\nfrom packaging.specifiers import SpecifierSet\nspecifier = SpecifierSet('>=2.7.0')\ntry:\n    version = metadata.version('stable-baselines3')\nexcept metadata.PackageNotFoundError:\n    sys.exit(1)\nif version in specifier:\n    sys.exit(0)\nsys.exit(1)\n")
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "${_policy_check_script}"
  RESULT_VARIABLE _policy_requirement_satisfied
)
if(NOT _policy_requirement_satisfied EQUAL 0)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pip install --disable-pip-version-check "${_policy_pip_requirement}"
    RESULT_VARIABLE _pip_install_result
    ERROR_VARIABLE _pip_install_error
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  if(NOT _pip_install_result EQUAL 0)
    message(FATAL_ERROR "Failed to install ${_policy_pip_requirement}: ${_pip_install_error}")
  endif()
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY  config description launch models params resource rviz worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src)

install(
  PROGRAMS src/node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME coordinated_navigation_node
)


ament_package()
