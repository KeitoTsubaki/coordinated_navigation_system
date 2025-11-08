from setuptools import find_packages, setup

package_name = "simulation_core"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/simulator_config.yaml"])
    ],
    include_package_data=True,
    package_data={"simulation_core": ["config/*.yaml"]},
    install_requires=["setuptools", "pygame", "numpy", "pyyaml"],
    zip_safe=True,
    maintainer="tsubaki",
    maintainer_email="tsubaki.fs@tut.ac.jp",
    description="Core simulation logic for the coordinated navigation system.",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["environment = simulation_core.environment:main"],
    },
)