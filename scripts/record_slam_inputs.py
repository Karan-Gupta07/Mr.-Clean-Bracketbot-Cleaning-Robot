"""Record LiDAR and wheel/IMU odometry while balancing, without ROS or rendering.

    .venv/bin/python scripts/record_slam_inputs.py --output out/slam_inputs.npz

This is a sensor/odometry recording, not a map. truth_* arrays are evaluation
references only, never estimator inputs. Existing output files are not replaced.
"""

import argparse
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlbot import Balancer, BalanceController, Gains
from rlbot.odometry import WheelImuOdometry
from rlbot.robot import ROOM
from rlbot.sensors import Lidar


def record(seconds=10.0, scan_hz=10.0, beams=360, range_max=10.0, push=0.0):
    if (not np.isfinite([seconds, scan_hz, push]).all() or seconds <= 0 or scan_hz <= 0):
        raise ValueError("seconds and scan_hz must be positive and all arguments finite")
    bot = Balancer(ROOM, chassis="root", keyframe="start")
    if scan_hz > 1 / bot.dt:
        raise ValueError("scan_hz cannot exceed the simulation step rate")
    lidar = Lidar(bot.model, beams=beams, range_max=range_max)
    odometry = WheelImuOdometry.from_model(bot.model)
    controller = BalanceController(Gains.for_bracketbot())
    wheel_qpos = [bot.model.joint(f"wheel_{side}").qposadr[0] for side in ("left", "right")]
    samples = {name: [] for name in ("time", "wheel_angles", "imu_gyro", "imu_accel",
                                     "odom_pose", "odom_twist", "odom_roll_pitch", "truth_pose",
                                     "scan_time", "scan_ranges")}
    scan_index = 0
    steps = math.ceil(seconds / bot.dt)
    for step in range(steps + 1):
        mujoco.mj_forward(bot.model, bot.data)
        state = bot.state()
        if bot.has_fallen(state):
            raise RuntimeError(f"robot fell at {bot.data.time:.3f}s; recording aborted")
        wheels = bot.data.qpos[wheel_qpos].copy()
        gyro = bot.data.sensor("gyro").data.copy()
        estimate = odometry.update(float(bot.data.time), wheels, gyro)
        rotation = bot.data.body("root").xmat.reshape(3, 3)
        centre = (bot.data.body("wheel_left").xpos + bot.data.body("wheel_right").xpos) / 2
        truth = np.array([centre[0], centre[1], math.atan2(rotation[1, 0], rotation[0, 0])])
        for name, value in (("time", estimate.timestamp), ("wheel_angles", wheels),
                            ("imu_gyro", gyro), ("imu_accel", bot.data.sensor("accel").data.copy()),
                            ("odom_pose", estimate.pose), ("odom_twist", estimate.twist),
                            ("odom_roll_pitch", estimate.roll_pitch), ("truth_pose", truth)):
            samples[name].append(value)
        if bot.data.time + 1e-10 >= scan_index / scan_hz:
            scan = lidar.scan(bot.data)
            samples["scan_time"].append(scan.timestamp)
            samples["scan_ranges"].append(scan.ranges)
            scan_index += 1
        if step < steps:
            bot.data.body("root").xfrc_applied[0] = push if seconds / 2 <= bot.data.time < seconds / 2 + 0.05 else 0
            bot.step(*controller(state))

    result = {name: np.asarray(values) for name, values in samples.items()}
    result.update(
        schema_version=np.array(1), scan_angles=lidar.angles.copy(), scan_frame="lidar",
        odom_frame="odom", odom_child_frame="base_footprint", extrinsics_parent_frame="root",
        scan_period=1 / scan_hz, scan_time_increment=0.0,
        range_min=lidar.range_min, range_max=lidar.range_max,
        wheel_radii=odometry.wheel_radii.copy(), track_width=odometry.track_width,
        wheel_signs=odometry.wheel_signs.copy(), gyro_weight=odometry.gyro_weight,
        gyro_bias=odometry.gyro_bias.copy(), mount_masked=True,
        lidar_pos=bot.model.site("lidar").pos.copy(), lidar_quat=bot.model.site("lidar").quat.copy(),
        imu_pos=bot.model.site("imu").pos.copy(), imu_quat=bot.model.site("imu").quat.copy(),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--scan-hz", type=float, default=10)
    parser.add_argument("--beams", type=int, default=360)
    parser.add_argument("--range-max", type=float, default=10)
    parser.add_argument("--push", type=float, default=0, help="forward shove in newtons for 50 ms halfway through")
    parser.add_argument("--output", type=Path, default=Path("out/slam_inputs.npz"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}; choose another path")
    if args.output.suffix != ".npz":
        parser.error("output must have an .npz extension")
    try:
        samples = record(args.seconds, args.scan_hz, args.beams, args.range_max, args.push)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        np.savez_compressed(output, **samples)
    initial = samples["truth_pose"][0]
    yaw = initial[2]
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    actual = (samples["truth_pose"][-1, :2] - initial[:2]) @ rotation
    error = np.linalg.norm(samples["odom_pose"][-1, :2] - actual)
    ranges = samples["scan_ranges"]
    print(f"Saved {args.output}: {len(samples['time'])} odometry samples, {len(ranges)} scans")
    print(f"Returns: {np.isfinite(ranges).sum()} valid, {np.isnan(ranges).sum()} blocked/too close, "
          f"{np.isinf(ranges).sum()} out of range")
    print(f"Final position error against separate simulator truth: {error * 1000:.1f} mm")
    print("No SLAM/map yet. Mounting assembly masked; scans tilt with the chassis and retain floor hits.")


if __name__ == "__main__":
    main()
