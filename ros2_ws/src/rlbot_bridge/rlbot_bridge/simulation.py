"""MuJoCo -> ROS 2 scans, wheel/gyro odometry and TF; accepts bounded /cmd_vel.

Run in a sourced ROS 2 Jazzy environment with the repo's Python dependencies.
The simulator advances on a wall clock; message stamps and /clock use sim time.
"""

import math
import os
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock as ClockMsg
from sensor_msgs.msg import Imu, JointState, LaserScan
from std_msgs.msg import Bool
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

REPO = Path(os.environ.get("RLBOT_REPO", Path(__file__).resolve().parents[4]))
if not (REPO / "src" / "rlbot" / "robot.py").is_file():
    raise RuntimeError("Set RLBOT_REPO to the RL-BOT checkout, or build this workspace with --symlink-install")
sys.path.insert(0, str(REPO / "src"))

from rlbot import Balancer, Gains
from rlbot.control import DriveController
from rlbot.odometry import WheelImuOdometry, footprint_to_base
from rlbot.robot import ROOM
from rlbot.sensors import Lidar, project_scan


def transform(parent, child, position, quaternion, stamp):
    message = TransformStamped()
    message.header.stamp, message.header.frame_id = stamp, parent
    message.child_frame_id = child
    message.transform.translation.x, message.transform.translation.y, message.transform.translation.z = map(float, position)
    message.transform.rotation.w, message.transform.rotation.x, message.transform.rotation.y, message.transform.rotation.z = map(float, quaternion)
    return message


def scan_message(scan, period, mapping=False):
    message = LaserScan()
    nanoseconds = round(scan.timestamp * 1e9)
    message.header.stamp = TimeMsg(sec=nanoseconds // 1000000000, nanosec=nanoseconds % 1000000000)
    message.header.frame_id = scan.frame_id
    message.angle_min, message.angle_increment = scan.angle_min, scan.angle_increment
    message.angle_max = scan.angle_min + (len(scan.ranges) - 1) * scan.angle_increment
    message.range_min, message.range_max = scan.range_min, scan.range_max
    message.scan_time, message.time_increment = period, 0.0
    ranges = np.where(np.isfinite(scan.ranges), scan.ranges, 0.0) if mapping else scan.ranges
    message.ranges = ranges.astype(float).tolist()
    return message


class SimulationBridge(Node):
    def __init__(self):
        super().__init__("rlbot_bridge")
        self.declare_parameter("scan_hz", 10.0)
        self.declare_parameter("max_tilt_degrees", 2.0)
        self.declare_parameter("publish_ground_truth", False)
        self.declare_parameter("odom_position_stddev", 0.05)
        self.declare_parameter("odom_yaw_stddev", math.radians(5))
        self.declare_parameter("odom_speed_stddev", 0.05)
        self.declare_parameter("odom_yaw_rate_stddev", 0.1)
        self.declare_parameter("max_speed", 0.15)
        self.declare_parameter("max_yaw_rate", 0.3)
        self.declare_parameter("command_timeout", 0.5)
        self.scan_hz = float(self.get_parameter("scan_hz").value)
        self.max_tilt = math.radians(float(self.get_parameter("max_tilt_degrees").value))
        position_stddev = float(self.get_parameter("odom_position_stddev").value)
        yaw_stddev = float(self.get_parameter("odom_yaw_stddev").value)
        speed_stddev = float(self.get_parameter("odom_speed_stddev").value)
        yaw_rate_stddev = float(self.get_parameter("odom_yaw_rate_stddev").value)
        if (not all(math.isfinite(v) and v > 0 for v in
                    (self.scan_hz, position_stddev, yaw_stddev, speed_stddev, yaw_rate_stddev))
                or self.scan_hz > 50 or not 0 < self.max_tilt <= math.radians(10)):
            raise ValueError("require scan_hz <= 50, positive odometry uncertainties and tilt limit in (0,10] degrees")
        self.bot = Balancer(ROOM, chassis="root", keyframe="start")
        model = self.bot.model
        for site in ("imu", "lidar"):
            if model.site(site).bodyid[0] != model.body("root").id:
                raise ValueError(f"{site} must be mounted on root")
        if not np.allclose(model.site("imu").quat, [1, 0, 0, 0]):
            raise ValueError("the gyro must be aligned with the chassis for this estimator")
        self.odometry = WheelImuOdometry.from_model(model)
        self.radius = float(self.odometry.wheel_radii.mean())
        self.axle = (model.body("wheel_left").pos + model.body("wheel_right").pos) / 2
        self.drive = DriveController(self.radius, Gains.for_bracketbot(),
                                     max_speed=float(self.get_parameter("max_speed").value),
                                     max_yaw_rate=float(self.get_parameter("max_yaw_rate").value),
                                     timeout=float(self.get_parameter("command_timeout").value))
        self.lidar = Lidar(model)
        self.wheels = [model.joint(f"wheel_{side}").qposadr[0] for side in ("left", "right")]
        self.joints = [j for j in range(model.njnt) if model.jnt_type[j] in
                       (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
                       and model.body_rootid[model.jnt_bodyid[j]] == model.body("root").id]
        self.estimate = self.odometry.update(0.0, self.bot.data.qpos[self.wheels], self.bot.data.sensor("gyro").data)
        self.pose_covariance = np.diag([position_stddev ** 2, position_stddev ** 2, 1e6, 1e6, 1e6,
                                       yaw_stddev ** 2]).ravel().tolist()
        self.twist_covariance = np.diag([speed_stddev ** 2, 1e6, 1e6, 1e6, 1e6,
                                        yaw_rate_stddev ** 2]).ravel().tolist()
        self.clock_pub = self.create_publisher(ClockMsg, "/clock", 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.imu_pub = self.create_publisher(Imu, "/imu/data", qos_profile_sensor_data)
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", qos_profile_sensor_data)
        self.raw_pub = self.create_publisher(LaserScan, "/scan_raw", qos_profile_sensor_data)
        self.valid_pub = self.create_publisher(Bool, "/scan_valid", 10)
        self.truth_pub = (self.create_publisher(PoseStamped, "/ground_truth/pose", 10)
                          if self.get_parameter("publish_ground_truth").value else None)
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        static = [transform("base_footprint", "lidar_planar", [0, 0, 0], [1, 0, 0, 0], TimeMsg())]
        for name in ("imu", "lidar"):
            static.append(transform("base_link", name, model.site(name).pos, model.site(name).quat, TimeMsg()))
        self.static_tf.sendTransform(static)
        self._scan_index, self._last_valid_scan = 0, None
        self.command_sub = self.create_subscription(Twist, "/cmd_vel", self.command, 10)
        self.timer = self.create_timer(0.02, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info("Publishing estimated odometry and tilt-gated scans; mounting assembly is masked. Waiting for /cmd_vel.")

    def command(self, message):
        now = time.monotonic()
        try:
            unsupported = (message.linear.y, message.linear.z, message.angular.x, message.angular.y)
            if any(not math.isfinite(v) or abs(v) > 1e-12 for v in unsupported):
                raise ValueError("only linear.x and angular.z commands are supported")
            if self._last_valid_scan is None or now - self._last_valid_scan > 0.5:
                self.drive.command(0, 0, now)
                return
            self.drive.command(message.linear.x, message.angular.z, now)
        except ValueError as error:
            self.drive.command(0, 0, now)
            self.get_logger().warning(str(error), throttle_duration_sec=2.0)

    def tick(self):
        now = time.monotonic()
        if self._last_valid_scan is None or now - self._last_valid_scan > 0.5:
            self.drive.command(0, 0, now)
        for _ in range(10):
            self.bot.step(*self.drive(self.bot.state(), now, self.bot.dt))
            mujoco.mj_forward(self.bot.model, self.bot.data)
            if self.bot.has_fallen():
                raise RuntimeError("robot fell; stop and restart the mapping session")
            self.estimate = self.odometry.update(float(self.bot.data.time), self.bot.data.qpos[self.wheels],
                                                 self.bot.data.sensor("gyro").data)
        model, data, estimate = self.bot.model, self.bot.data, self.estimate
        nanoseconds = round(estimate.timestamp * 1e9)
        stamp = TimeMsg(sec=nanoseconds // 1000000000, nanosec=nanoseconds % 1000000000)
        self.clock_pub.publish(ClockMsg(clock=stamp))
        x, y, yaw = estimate.pose
        odom_tf = transform("odom", "base_footprint", [x, y, 0],
                            [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], stamp)
        base_pos, base_quat = footprint_to_base(estimate.roll_pitch, self.axle, self.radius)
        self.tf.sendTransform([odom_tf, transform("base_footprint", "base_link", base_pos, base_quat, stamp)])
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = stamp, "odom", "base_footprint"
        odom.pose.pose.position.x, odom.pose.pose.position.y = float(x), float(y)
        odom.pose.pose.orientation = odom_tf.transform.rotation
        odom.pose.covariance = self.pose_covariance
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = map(float, estimate.twist)
        odom.twist.covariance = self.twist_covariance
        self.odom_pub.publish(odom)
        imu = Imu()
        imu.header.stamp, imu.header.frame_id = stamp, "imu"
        imu.orientation_covariance[0] = -1.0
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = map(float, data.sensor("gyro").data)
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = map(float, data.sensor("accel").data)
        self.imu_pub.publish(imu)
        joints = JointState()
        joints.header.stamp = stamp
        joints.name = [model.joint(j).name for j in self.joints]
        joints.position = data.qpos[model.jnt_qposadr[self.joints]].tolist()
        joints.velocity = data.qvel[model.jnt_dofadr[self.joints]].tolist()
        self.joint_pub.publish(joints)
        if self.truth_pub is not None:
            truth = PoseStamped()
            truth.header.stamp, truth.header.frame_id = stamp, "world"
            centre = (data.body("wheel_left").xpos + data.body("wheel_right").xpos) / 2
            truth.pose.position.x, truth.pose.position.y = map(float, centre[:2])
            rotation = data.body("root").xmat.reshape(3, 3)
            heading = math.atan2(rotation[1, 0], rotation[0, 0])
            truth.pose.orientation.w, truth.pose.orientation.z = math.cos(heading / 2), math.sin(heading / 2)
            self.truth_pub.publish(truth)
        if data.time + 1e-10 >= self._scan_index / self.scan_hz:
            raw = self.lidar.scan(data)
            self.raw_pub.publish(scan_message(raw, 1 / self.scan_hz))
            offset, quaternion = np.zeros(3), np.zeros(4)
            mujoco.mju_rotVecQuat(offset, model.site("lidar").pos, base_quat)
            mujoco.mju_mulQuat(quaternion, base_quat, model.site("lidar").quat)
            projected = project_scan(raw, base_pos + offset, quaternion, max_tilt=self.max_tilt)
            valid = bool(np.isfinite(projected.ranges).sum() >= len(projected.ranges) / 2
                         and np.linalg.norm(data.sensor("gyro").data[:2]) < 0.2)
            if valid:
                self.scan_pub.publish(scan_message(projected, 1 / self.scan_hz, mapping=True))
                self._last_valid_scan = time.monotonic()
            self.valid_pub.publish(Bool(data=valid))
            self._scan_index += 1


def main():
    rclpy.init()
    node = None
    try:
        node = SimulationBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
