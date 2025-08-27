from setuptools import find_packages, setup

package_name = "matplotllib_2d_simulator"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["matplotllib_2d_simulator/sensors.py"]),
        ("share/" + package_name + "/config", ["config/simulator_config.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="tsubaki",
    maintainer_email="tsubaki@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "environment = matplotllib_2d_simulator.environment:main",
            "sensors = matplotllib_2d_simulator.sensors:main",
        ],
    },
)
