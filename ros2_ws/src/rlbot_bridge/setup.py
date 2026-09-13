from glob import glob

from setuptools import setup

setup(
    name="rlbot_bridge",
    version="0.1.0",
    packages=["rlbot_bridge"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/rlbot_bridge"]),
        ("share/rlbot_bridge", ["package.xml"]),
        ("share/rlbot_bridge/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    entry_points={"console_scripts": [
        "simulation = rlbot_bridge.simulation:main",
        "save_map = rlbot_bridge.map_session:main",
        "navigate = rlbot_bridge.navigate:main",
    ]},
)
