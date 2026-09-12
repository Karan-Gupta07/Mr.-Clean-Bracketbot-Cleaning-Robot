"""Launch the MuJoCo ROS 2 bridge and SLAM Toolbox mapping or localization."""

import math
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown, matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def shutdown_if_running(event, context):
    return [] if context.is_shutdown else [EmitEvent(event=Shutdown(reason="mapping process exited"))]


def setup(context):
    mode = LaunchConfiguration("mode").perform(context)
    map_file = LaunchConfiguration("map_file").perform(context)
    pose = [float(LaunchConfiguration(name).perform(context)) for name in ("x", "y", "yaw")]
    if not all(math.isfinite(v) for v in pose):
        raise ValueError("initial map pose must be finite")
    if mode == "localization" and not map_file:
        raise ValueError("localization requires map_file:=/absolute/path/to/map (without extension)")
    if map_file:
        map_file = str(Path(map_file).expanduser().resolve())
        for suffix in (".posegraph", ".data"):
            path = Path(map_file + suffix)
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing or empty SLAM state: {path}")
    filename = "mapper_params_localization.yaml" if mode == "localization" else "mapper_params_online_async.yaml"
    defaults = Path(get_package_share_directory("slam_toolbox")) / "config" / filename
    params = yaml.safe_load(defaults.read_text())["slam_toolbox"]["ros__parameters"]
    params.update(
        use_sim_time=True, use_lifecycle_manager=False, mode=mode,
        odom_frame="odom", base_frame="base_footprint", map_frame="map", scan_topic="/scan",
        scan_queue_size=1, resolution=0.05, min_laser_range=0.05, max_laser_range=10.0,
        map_update_interval=1.0, minimum_time_interval=0.1,
        minimum_travel_distance=0.05, minimum_travel_heading=0.05,
        check_min_dist_and_heading_precisely=True, use_scan_matching=True,
        do_loop_closing=True, enable_interactive_mode=False, restamp_tf=False,
        use_map_saver=mode == "mapping",
    )
    if map_file:
        params.update(map_file_name=map_file, map_start_pose=pose, map_start_at_dock=False)
    slam = LifecycleNode(
        package="slam_toolbox", name="slam_toolbox", namespace="", output="screen",
        executable="localization_slam_toolbox_node" if mode == "localization" else "async_slam_toolbox_node",
        parameters=[params],
    )
    bridge = Node(
        package="rlbot_bridge", executable="simulation", name="rlbot_bridge", output="screen",
        parameters=[{"use_sim_time": True,
                     "publish_ground_truth": LaunchConfiguration("publish_ground_truth").perform(context) == "true"}],
        additional_env={"PYTHONUNBUFFERED": "1"},
    )
    activate = RegisterEventHandler(OnStateTransition(
        target_lifecycle_node=slam, start_state="configuring", goal_state="inactive",
        entities=[EmitEvent(event=ChangeState(lifecycle_node_matcher=matches_action(slam),
                                             transition_id=Transition.TRANSITION_ACTIVATE))],
    ))
    configure = EmitEvent(event=ChangeState(lifecycle_node_matcher=matches_action(slam),
                                            transition_id=Transition.TRANSITION_CONFIGURE))
    return [activate, slam, bridge, configure,
            RegisterEventHandler(OnProcessExit(target_action=bridge, on_exit=shutdown_if_running)),
            RegisterEventHandler(OnProcessExit(target_action=slam, on_exit=shutdown_if_running))]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="mapping", choices=["mapping", "localization"]),
        DeclareLaunchArgument("map_file", default_value="", description="Saved map stem; mapping resumes it, localization localizes against it"),
        DeclareLaunchArgument("x", default_value="0.0"),
        DeclareLaunchArgument("y", default_value="0.0"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        DeclareLaunchArgument("publish_ground_truth", default_value="false", choices=["true", "false"]),
        OpaqueFunction(function=setup),
    ])
