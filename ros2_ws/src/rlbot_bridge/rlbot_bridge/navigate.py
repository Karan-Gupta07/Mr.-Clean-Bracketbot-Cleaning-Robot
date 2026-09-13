"""Drive the robot to a table on the SLAM pose.

    ros2 run rlbot_bridge navigate --ros-args -p use_sim_time:=true \
        -p map_yaml:=/artifacts/mapping_check_1/saved/map.yaml -p to:=cubes
    ros2 run rlbot_bridge navigate --ros-args -p use_sim_time:=true \
        -p map_yaml:=... -p to:="1.71 -1.10 0.0"

Runs beside `mapping.launch.py mode:=localization`.  Ten times a second it
looks up map -> base_footprint, hands it to the Navigator, and publishes what
comes back on /cmd_vel.  Exits 0 on arrival, 1 if the navigator gives up.

The map frame is taken to be the room's world frame, because the bridge spawns
the robot at the room's `start` keyframe and the map is built from there.  The
lidar cannot see table tops, so the known ones are unioned into the map.
"""

import math
import os
import sys
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

REPO = Path(os.environ.get("RLBOT_REPO", Path(__file__).resolve().parents[4]))
if not (REPO / "src" / "rlbot" / "robot.py").is_file():
    raise RuntimeError("Set RLBOT_REPO to the RL-BOT checkout, or build this workspace with --symlink-install")
sys.path.insert(0, str(REPO / "src"))

from rlbot.navigate import Navigator                  # noqa: E402
from rlbot.navmap import OccupancyGrid, table_tops    # noqa: E402
from rlbot.planner import goal_for                    # noqa: E402


class NavigateNode(Node):
    def __init__(self):
        super().__init__("rlbot_navigate")
        self.declare_parameter("map_yaml", "")
        self.declare_parameter("to", "")
        map_yaml = self.get_parameter("map_yaml").value
        to = self.get_parameter("to").value.split()
        if not map_yaml or not to:
            raise ValueError("need -p map_yaml:=/path/map.yaml and -p to:=<table | x y yaw>")
        if len(to) not in (1, 3):
            raise ValueError(f"to must be a table name or 'x y yaw', not {' '.join(to)!r}")
        self.goal = goal_for(to[0]) if len(to) == 1 else tuple(float(v) for v in to)
        grid = OccupancyGrid.from_pgm(map_yaml, extra_boxes=table_tops())
        self.navigator = Navigator(grid)
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.timer = self.create_timer(0.1, self.tick)
        self.started = False
        self.finished = None
        self.get_logger().info(f"driving to ({self.goal[0]:.2f}, {self.goal[1]:.2f}, "
                               f"{math.degrees(self.goal[2]):.0f} deg)")

    def pose(self, now: float):
        """The latest map -> base_footprint, or None if there is not a fresh one.

        The age matters: the bridge stops publishing valid scans when the
        chassis tilts, SLAM then stops correcting map -> odom, and the lookup
        keeps succeeding with a frozen transform.  The navigator stamps its
        samples with our clock, not the pose's, so a frozen pose looks like a
        perfectly settled one and it would replan - or declare arrival - on a
        position the robot left some time ago.
        """
        try:
            tf = self.tf.lookup_transform("map", "base_footprint", Time())
        except TransformException as error:
            self.get_logger().warning(f"no map->base_footprint yet: {error}",
                                      throttle_duration_sec=2.0)
            return None
        age = now - (tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9)
        if age > self.navigator.pose_timeout:
            self.get_logger().warning(f"map->base_footprint is {age:.1f} s stale",
                                      throttle_duration_sec=2.0)
            return None
        q = tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        return tf.transform.translation.x, tf.transform.translation.y, yaw

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        pose = self.pose(now)
        if not self.started:
            if pose is None:
                return
            traj = self.navigator.go_to(self.goal, pose, now)
            self.started = True
            if traj is not None:
                self.get_logger().info(f"planned a {traj.duration:.0f} s route from "
                                       f"({pose[0]:.2f}, {pose[1]:.2f}, {math.degrees(pose[2]):.0f} deg)")
        v, omega = self.navigator.step(pose, now)
        message = Twist()
        message.linear.x, message.angular.z = float(v), float(omega)
        self.publisher.publish(message)
        if self.navigator.done:
            self.get_logger().info(f"arrived after {len(self.navigator.replans)} replans")
            self.finished = 0
        elif self.navigator.failed:
            self.get_logger().error(self.navigator.failed)
            self.finished = 1


def main():
    rclpy.init()
    node = None
    code = 1
    try:
        node = NavigateNode()
        while rclpy.ok() and node.finished is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        code = node.finished if node.finished is not None else 1
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.publisher.publish(Twist())
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
