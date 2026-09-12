"""Ubuntu/ROS 2 integration check: drive a loop, build/save a map, restart and localize.

    .venv/bin/python scripts/check_ros_mapping.py --output out/mapping_check_1

Uses a separate ROS domain, real SLAM Toolbox processes and simulated physics.
Logs and map files are kept for inspection. This does not test Nav2 navigation.
"""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain-id", type=int, default=100 + os.getpid() % 100)
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    if importlib.util.find_spec("rclpy") is None or shutil.which("ros2") is None:
        parser.exit(2, "Blocked: ROS 2 is unavailable. Run this check in sourced ROS 2 Jazzy on Ubuntu with the repo dependencies installed.\n")
    if not 0 <= args.domain_id <= 232 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("require a ROS domain in [0,232] and a positive finite timeout")
    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import OccupancyGrid, Odometry
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformException, TransformListener
    from rlbot_bridge.map_session import save_session

    def yaw(q):
        return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    class Observer(Node):
        def __init__(self):
            super().__init__("mapping_integration_check", parameter_overrides=[Parameter("use_sim_time", value=True)])
            self.odom, self.grid, self.truth = None, None, None
            self.travel = 0.0
            self.cmd = self.create_publisher(Twist, "/cmd_vel", 10)
            self.create_subscription(Odometry, "/odom", self.on_odom, 10)
            self.create_subscription(PoseStamped, "/ground_truth/pose", lambda msg: setattr(self, "truth", msg), 10)
            self.create_subscription(OccupancyGrid, "/map", lambda msg: setattr(self, "grid", msg),
                                     QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            self.buffer = Buffer(node=self)
            self.listener = TransformListener(self.buffer, self)

        def on_odom(self, message):
            if self.odom is not None:
                before, after = self.odom.pose.pose.position, message.pose.pose.position
                self.travel += math.hypot(after.x - before.x, after.y - before.y)
            self.odom = message

        def command(self, speed=0.0, turn=0.0):
            message = Twist()
            message.linear.x, message.angular.z = float(speed), float(turn)
            self.cmd.publish(message)

        def map_pose(self):
            try:
                tf = self.buffer.lookup_transform("map", "base_footprint", Time())
                return np.array([tf.transform.translation.x, tf.transform.translation.y, yaw(tf.transform.rotation)])
            except TransformException:
                return None

        def ready(self):
            if self.odom is None or self.grid is None or self.truth is None or self.map_pose() is None:
                return False
            cells = np.asarray(self.grid.data)
            return np.count_nonzero(cells == 0) > 100 and np.count_nonzero(cells >= 65) > 30

    def stop(process):
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)

    def pump(node, process, deadline):
        if process.poll() is not None:
            raise RuntimeError(f"mapping launch exited with {process.returncode}; inspect logs in {output}")
        if time.monotonic() >= deadline:
            raise RuntimeError(f"mapping check timed out; inspect logs in {output}")
        rclpy.spin_once(node, timeout_sec=0.05)

    rclpy.init(args=[])
    node, process = None, None
    try:
        node = Observer()
        discovery_end = time.monotonic() + 1.0
        while time.monotonic() < discovery_end:
            rclpy.spin_once(node, timeout_sec=0.1)
        if any(name != node.get_name() for name in node.get_node_names()):
            raise RuntimeError("ROS domain already has nodes; choose a different --domain-id")
        alignment, results = None, []
        for mode in ("mapping", "localization"):
            deadline = time.monotonic() + args.timeout
            command = ["ros2", "launch", "rlbot_bridge", "mapping.launch.py",
                       f"mode:={mode}", "publish_ground_truth:=true"]
            if mode == "localization":
                command.append(f"map_file:={output / 'saved' / 'map'}")
            with (output / f"{mode}.log").open("x") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while not node.ready():
                    node.command()
                    pump(node, process, deadline)
                if mode == "mapping":
                    mapped = node.map_pose()
                    actual = node.truth.pose
                    heading = yaw(actual.orientation) - mapped[2]
                    rotation = np.array([[math.cos(heading), -math.sin(heading)],
                                         [math.sin(heading), math.cos(heading)]])
                    alignment = (rotation, np.array([actual.position.x, actual.position.y]) - rotation @ mapped[:2], heading)
                    for goal in ((0.4, 0), (0.4, 0.4), (0, 0.4), (0, 0)):
                        while True:
                            pose = node.odom.pose.pose
                            dx, dy = goal[0] - pose.position.x, goal[1] - pose.position.y
                            distance = math.hypot(dx, dy)
                            if distance < 0.04:
                                break
                            error = math.atan2(dy, dx) - yaw(pose.orientation)
                            error = math.atan2(math.sin(error), math.cos(error))
                            node.command(min(0.1, distance) if abs(error) < 0.2 else 0.0,
                                         max(-0.25, min(0.25, error)))
                            pump(node, process, deadline)
                    if node.travel < 1.2:
                        raise RuntimeError("robot did not traverse the mapping loop")
                target_yaw = 0.0
                while abs(math.atan2(math.sin(yaw(node.odom.pose.pose.orientation) - target_yaw),
                                     math.cos(yaw(node.odom.pose.pose.orientation) - target_yaw))) > 0.05:
                    error = target_yaw - yaw(node.odom.pose.pose.orientation)
                    node.command(0, max(-0.25, min(0.25, math.atan2(math.sin(error), math.cos(error)))))
                    pump(node, process, deadline)
                settle = time.monotonic() + 4.0
                while time.monotonic() < settle:
                    node.command()
                    pump(node, process, deadline)
                mapped = node.map_pose()
                rotation, offset, heading = alignment
                actual = node.truth.pose
                error = np.linalg.norm(rotation @ mapped[:2] + offset - [actual.position.x, actual.position.y])
                yaw_error = math.atan2(math.sin(mapped[2] + heading - yaw(actual.orientation)),
                                       math.cos(mapped[2] + heading - yaw(actual.orientation)))
                if error > 0.1 or abs(yaw_error) > math.radians(5):
                    raise RuntimeError(f"{mode}: pose error {error:.3f} m / {math.degrees(yaw_error):.2f} deg")
                if mode == "mapping":
                    save_session(node, output / "saved")
                results.append({"mode": mode, "position_error_m": float(error),
                                "yaw_error_degrees": math.degrees(yaw_error),
                                "estimated_travel_m": node.travel,
                                "map_width": node.grid.info.width, "map_height": node.grid.info.height,
                                "resolution": node.grid.info.resolution})
                print(f"{mode}: nonempty SLAM map and map->base TF; error {error:.3f} m / {math.degrees(yaw_error):.2f} deg", flush=True)
                stop(process)
                process = None
            node.destroy_node()
            node = Observer() if mode == "mapping" else None
        with (output / "result.json").open("x") as report:
            json.dump({"status": "passed", "ros_distro": os.environ.get("ROS_DISTRO"), "runs": results}, report, indent=2)
        print(f"ROS mapping, loop traversal, save and localization restart passed. Evidence: {output}")
    finally:
        if node is not None:
            node.command()
        stop(process)
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
