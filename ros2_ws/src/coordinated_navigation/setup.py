from glob import glob
from setuptools import setup

package_name = "coordinated_navigation"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
    ],
    install_requires=[
        "setuptools",
        "rclpy",
        "geometry_msgs",
        "nav_msgs",
        "sensor_msgs",
    ],
    zip_safe=True,
    maintainer="Maintainer",
    maintainer_email="maintainer@example.com",
    description="Coordinated navigation node wiring up multi-robot interfaces.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "coordinated_navigation_node = coordinated_navigation.node:main",
        ],
    },
)
