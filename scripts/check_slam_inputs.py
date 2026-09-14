"""Run with .venv/bin/python scripts/check_slam_inputs.py; no ROS or graphics needed."""

import math
from pathlib import Path
import subprocess
import sys
import tempfile

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlbot import Balancer, BalanceController, Gains
from rlbot.control import DriveController
from rlbot.odometry import WheelImuOdometry, footprint_to_base
from rlbot.robot import ROOM
from rlbot.sensors import Lidar, project_scan


SCENE = """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <geom name="wall" type="box" size="0.1 3 1" pos="2.1 0 1"/>
    <body name="root" pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <site name="lidar"/>
      <body name="arm" pos="0.4 0 0">
        <joint name="arm_slide" type="slide" axis="0 1 0"/>
        <geom type="sphere" size="0.1"/>
      </body>
    </body>
    <body name="table" pos="0 1 0.5">
      <geom type="box" size="0.1 0.1 0.1" group="2"/>
    </body>
    <body name="object" pos="0 -1 0.5">
      <freejoint/>
      <geom type="sphere" size="0.1" group="3"/>
    </body>
  </worldbody>
</mujoco>"""


def check_lidar():
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rgba = model.geom_rgba.copy()
    lidar = Lidar(model, beams=4)
    scan = lidar.scan(data)
    assert scan.frame_id == "lidar" and scan.timestamp == data.time
    np.testing.assert_allclose(lidar.angles, [-math.pi, -math.pi / 2, 0, math.pi / 2])
    assert np.isinf(scan.ranges[0])
    assert np.isnan(scan.ranges[2]), "moving arm must occlude the wall"
    np.testing.assert_allclose(scan.ranges[[1, 3]], [0.9, 0.9], atol=1e-8)
    np.testing.assert_array_equal(model.geom_rgba, rgba)
    assert np.isnan(Lidar(model, beams=4, mask_mount=False).scan(data).ranges).all()

    data.qpos[model.joint("arm_slide").qposadr[0]] = 1
    data.time = 0.1
    updated = lidar.scan(data)
    assert updated.timestamp == 0.1
    np.testing.assert_allclose(updated.ranges[2], 2, atol=1e-8)
    assert np.isnan(scan.ranges[2]), "scan snapshots must not alias each other"
    assert np.isinf(Lidar(model, beams=4, range_max=1).scan(data).ranges[2])
    assert np.isnan(Lidar(model, beams=4, range_min=1).scan(data).ranges[1])

    angle = math.pi / 4
    data.qpos[3:7] = [math.cos(angle / 2), 0, math.sin(angle / 2), 0]
    np.testing.assert_allclose(lidar.scan(data).ranges[2], 0.5 / math.sin(angle), atol=1e-8)
    data.qpos[3:7] = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]
    np.testing.assert_allclose(lidar.scan(data).ranges[2], 0.9, atol=1e-8)
    fixed = mujoco.MjModel.from_xml_string(SCENE.replace("<freejoint/>", "", 1))
    fixed_data = mujoco.MjData(fixed)
    np.testing.assert_allclose(Lidar(fixed, beams=4).scan(fixed_data).ranges[3], 0.9, atol=1e-8)
    mocap = mujoco.MjModel.from_xml_string(SCENE.replace('name="table"', 'name="table" mocap="true"'))
    mocap_data = mujoco.MjData(mocap)
    mocap_data.mocap_pos[0, 1] = 1.5
    np.testing.assert_allclose(Lidar(mocap, beams=4).scan(mocap_data).ranges[3], 1.4, atol=1e-8)
    for kwargs in ({"beams": 0}, {"beams": 3.5}, {"range_max": float("nan")},
                   {"range_min": 2, "range_max": 1}):
        with np.testing.assert_raises(ValueError):
            Lidar(model, **kwargs)
    print("LiDAR: distances, timestamps, furniture, moving occlusion, tilt and limits OK")


def check_odometry():
    odom = WheelImuOdometry(wheel_radii=(0.1, 0.1), track_width=0.4)
    first = odom.update(0, [0, 0], [0, 0, 0])
    np.testing.assert_array_equal(first.pose, [0, 0, 0])
    np.testing.assert_allclose(odom.update(1, [10, 10], [0, 0, 0]).pose, [1, 0, 0])
    np.testing.assert_allclose(odom.update(2, [0, 0], [0, 0, 0]).pose, [0, 0, 0])
    np.testing.assert_array_equal(first.pose, [0, 0, 0])
    odom.reset()
    odom.update(0, [0, 0], [0, 0, 0.5])
    np.testing.assert_allclose(odom.update(1, [-1, 1], [0, 0, 0.5]).pose, [0, 0, 0.5])

    odom.reset()
    odom.update(0, [0, 0], [0, 0, 1])
    np.testing.assert_allclose(odom.update(math.pi / 2, [3 * math.pi, 5 * math.pi], [0, 0, 1]).pose,
                               [0.8, 0.8, math.pi / 2], atol=1e-10)
    for k in (2, 3, 4):
        loop = odom.update(k * math.pi / 2, [3 * k * math.pi, 5 * k * math.pi], [0, 0, 1])
    np.testing.assert_allclose(loop.pose, [0, 0, 0], atol=1e-10)

    odom.reset()
    odom.update(0, [0, 0], [0, 0.2, 0])
    rocking = odom.update(1, [-0.2, -0.2], [0, 0.2, 0])
    np.testing.assert_allclose(rocking.pose, [0, 0, 0], atol=1e-10)
    np.testing.assert_allclose(rocking.roll_pitch, [0, 0.2], atol=1e-10)
    odom.reset(roll_pitch=(0.1, 0.2))
    gyro = [0, math.sin(0.1) * math.cos(0.2), math.cos(0.1) * math.cos(0.2)]
    gyro[0] = -math.sin(0.2)
    odom.update(0, [0, 0], gyro)
    np.testing.assert_allclose(odom.update(0.1, [-0.2, 0.2], gyro).pose[2], 0.1, atol=1e-6)

    biased = WheelImuOdometry((0.1, 0.1), 0.4, gyro_weight=0.8)
    biased.update(0, [0, 0], [0, 0, 0.1])
    np.testing.assert_allclose(biased.update(1, [0, 0], [0, 0, 0.1]).pose[2], 0.08)
    corrected = WheelImuOdometry((0.1, 0.1), 0.4, wheel_signs=(-1, 1), gyro_bias=(0, 0, 0.1))
    corrected.update(0, [0, 0], [0, 0, 0.1])
    np.testing.assert_allclose(corrected.update(1, [-10, 10], [0, 0, 0.1]).pose, [1, 0, 0])
    for timestamp, wheels, gyro in ((1, [-10, 10], [0, 0, 0]),
                                    (0, [-10, 10], [0, 0, 0]),
                                    (2, [float("nan"), 0], [0, 0, 0]),
                                    (2, [0, 0], [0, float("inf"), 0])):
        with np.testing.assert_raises(ValueError):
            corrected.update(timestamp, wheels, gyro)
    np.testing.assert_allclose(corrected.update(2, [-20, 20], [0, 0, 0.1]).pose, [2, 0, 0])
    for kwargs in ({"wheel_radii": [0, 0.1]}, {"track_width": -1}, {"gyro_weight": float("nan")},
                   {"wheel_signs": [0, 1]}, {"gyro_bias": [0, 0, float("inf")]}):
        with np.testing.assert_raises(ValueError):
            WheelImuOdometry(**({"wheel_radii": (0.1, 0.1), "track_width": 0.4} | kwargs))
    with np.testing.assert_raises(ValueError):
        odom.update(10, [0, 0], [0, 0, 10])
    print("Odometry: straight/reverse, turns, loop, pitch correction, calibration and stale data OK")


def check_room():
    model = mujoco.MjModel.from_xml_path(str(ROOM))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lidar = Lidar(model)
    scan = lidar.scan(data)
    np.testing.assert_allclose(scan.ranges[[0, 90, 180, 270]], [3, 2.25, 3, 2.25], atol=1e-5)
    odom = WheelImuOdometry.from_model(model)
    np.testing.assert_allclose(odom.wheel_radii, [0.0845979, 0.0845979])
    np.testing.assert_allclose(odom.track_width, 0.3222)
    np.testing.assert_array_equal(odom.wheel_signs, [1, 1])
    print("BracketBot room: real assets, cardinal wall distances and wheel calibration OK")


def check_recording():
    from record_slam_inputs import record

    samples = record(seconds=2, scan_hz=7, beams=120)
    assert samples["scan_ranges"].shape == (15, 120)
    assert len(samples["time"]) == 1001
    assert np.all(np.diff(samples["time"]) > 0) and np.all(np.diff(samples["scan_time"]) > 0)
    np.testing.assert_array_equal(samples["time"][np.searchsorted(samples["time"], samples["scan_time"])],
                                   samples["scan_time"])
    assert np.max(np.abs(samples["scan_time"] - np.arange(15) / 7)) <= 0.002 + 1e-10
    np.testing.assert_allclose(samples["odom_pose"][-1, :2],
                               samples["truth_pose"][-1, :2] - samples["truth_pose"][0, :2], atol=0.002)
    samples["truth_pose"][:] = np.nan
    replay = WheelImuOdometry(samples["wheel_radii"], samples["track_width"],
                              gyro_weight=samples["gyro_weight"], gyro_bias=samples["gyro_bias"],
                              wheel_signs=samples["wheel_signs"])
    poses = [replay.update(t, wheels, gyro).pose for t, wheels, gyro in
             zip(samples["time"], samples["wheel_angles"], samples["imu_gyro"])]
    np.testing.assert_array_equal(poses, samples["odom_pose"])
    pushed = record(seconds=6, push=300, beams=120)
    error = np.linalg.norm(pushed["odom_pose"][-1, :2] -
                           (pushed["truth_pose"][-1, :2] - pushed["truth_pose"][0, :2]))
    assert error < 0.05, error
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "inputs.npz"
        command = [sys.executable, str(Path(__file__).with_name("record_slam_inputs.py")),
                   "--seconds", "0.1", "--output", str(path)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        with np.load(path, allow_pickle=False) as loaded:
            assert loaded["scan_ranges"].shape == (2, 360)
            assert loaded["schema_version"] == 1
        before = path.read_bytes()
        rerun = subprocess.run(command, capture_output=True, text=True)
        assert rerun.returncode == 2 and "already exists" in rerun.stderr
        assert path.read_bytes() == before
    print(f"Recording: physics, scan scheduling, truth-free replay and safe archive output OK; shove error {error * 1000:.1f} mm")


def check_mapping_inputs():
    model = mujoco.MjModel.from_xml_path(str(ROOM))
    data = mujoco.MjData(model)
    lidar = Lidar(model)
    axle = (model.body("wheel_left").pos + model.body("wheel_right").pos) / 2
    radius = model.geom("wheel_left_collision").size[0]
    base_pos, base_quat = footprint_to_base([0, 0], axle, radius)
    np.testing.assert_allclose(base_pos, -axle + [0, 0, radius], atol=1e-10)
    for pitch in (0, math.radians(1), math.radians(-1), math.radians(5)):
        data.qpos[3:7] = [math.cos(pitch / 2), 0, math.sin(pitch / 2), 0]
        base_pos, base_quat = footprint_to_base([0, pitch], axle, radius)
        sensor_offset = np.zeros(3)
        mujoco.mju_rotVecQuat(sensor_offset, model.site("lidar").pos, base_quat)
        scan = project_scan(lidar.scan(data), base_pos + sensor_offset, base_quat)
        if abs(pitch) > math.radians(2):
            assert np.isnan(scan.ranges).all(), "excessive tilt must reject a mapping scan"
        else:
            assert np.isfinite(scan.ranges).sum() > 300
            np.testing.assert_allclose(scan.ranges[[0, 90, 180, 270]], [3, 2.25, 3, 2.25], atol=0.01)
        assert scan.frame_id == "lidar_planar"
    raw = lidar.scan(data)
    raw.ranges[:] = np.inf
    assert np.isnan(project_scan(raw, [0, 0, 0.32], [1, 0, 0, 0]).ranges).all()
    raw.ranges[:] = 0.32
    vertical = [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]
    assert np.isnan(project_scan(raw, [0, 0, 0.32], vertical, max_tilt=math.pi / 2).ranges[180])

    from rlbot.grasp import read_state

    bot = Balancer.bracketbot(keyframe="home")
    sensors = tuple(bot.model.sensor(n).adr[0] for n in ("gyro", "vel_left", "vel_right"))
    for heading in (0, math.pi / 2, math.pi, -math.pi / 2):
        spin = np.array([math.cos(heading / 2), 0, 0, math.sin(heading / 2)])
        lean = np.array([math.cos(0.05 / 2), 0, math.sin(0.05 / 2), 0])
        mujoco.mju_mulQuat(bot.data.qpos[3:7], spin, lean)
        mujoco.mj_forward(bot.model, bot.data)
        np.testing.assert_allclose([bot.state().pitch, read_state(bot.model, bot.data, sensors).pitch],
                                   [0.05, 0.05], atol=1e-10)
    bot.reset("home")
    controller = BalanceController(Gains.for_bracketbot())
    for _ in range(1000):
        bot.step(*controller(bot.state(), yaw_rate_ref=0.2))
    mujoco.mj_forward(bot.model, bot.data)
    assert 0.1 < bot.state().yaw_rate < 0.3, bot.state().yaw_rate
    assert not bot.has_fallen()
    for speed in (0.1, -0.1):
        bot.reset("home")
        drive = DriveController(radius, Gains.for_bracketbot())
        for _ in range(2500):
            drive.command(speed, 0, bot.data.time)
            bot.step(*drive(bot.state(), bot.data.time, bot.dt))
        mujoco.mj_forward(bot.model, bot.data)
        assert not bot.has_fallen()
        assert speed * bot.data.qpos[0] > 0.01, bot.data.qpos[0]
        for _ in range(2500):
            bot.step(*drive(bot.state(), bot.data.time, bot.dt))
        mujoco.mj_forward(bot.model, bot.data)
        assert abs(bot.state().forward_speed) < 0.02, bot.state().forward_speed
        assert drive.reference == [0.0, 0.0]
    with np.testing.assert_raises(ValueError):
        drive.command(float("nan"), 0, 0)
    print("Mapping inputs: tilt-gated projection, frame geometry, yaw tracking, forward/reverse and watchdog OK")


def check_driven_loop():
    bot = Balancer(ROOM, chassis="root", keyframe="start")
    odom = WheelImuOdometry.from_model(bot.model)
    radius = float(odom.wheel_radii.mean())
    drive, lidar = DriveController(radius, Gains.for_bracketbot()), Lidar(bot.model)
    wheels = [bot.model.joint(f"wheel_{side}").qposadr[0] for side in ("left", "right")]
    axle = (bot.model.body("wheel_left").pos + bot.model.body("wheel_right").pos) / 2
    estimate = odom.update(0.0, bot.data.qpos[wheels], bot.data.sensor("gyro").data)
    goals = [(0.4, 0), (0.4, 0.4), (0, 0.4), (0, 0)]
    accepted, scans = 0, 0
    for step in range(75000):
        if step % 20 == 0:
            x, y, yaw = estimate.pose
            while goals and math.hypot(goals[0][0] - x, goals[0][1] - y) < 0.04:
                goals.pop(0)
            if not goals:
                break
            dx, dy = goals[0][0] - x, goals[0][1] - y
            error = math.atan2(dy, dx) - yaw
            error = math.atan2(math.sin(error), math.cos(error))
            drive.command(min(0.1, math.hypot(dx, dy)) if abs(error) < 0.2 else 0,
                          max(-0.25, min(0.25, error)), bot.data.time)
        bot.step(*drive(bot.state(), bot.data.time, bot.dt))
        mujoco.mj_forward(bot.model, bot.data)
        assert not bot.has_fallen(), bot.data.time
        estimate = odom.update(float(bot.data.time), bot.data.qpos[wheels], bot.data.sensor("gyro").data)
        if step % 50 == 0:
            pos, quat = footprint_to_base(estimate.roll_pitch, axle, radius)
            offset = np.zeros(3)
            mujoco.mju_rotVecQuat(offset, bot.model.site("lidar").pos, quat)
            scan = project_scan(lidar.scan(bot.data), pos + offset, quat)
            accepted += int(np.isfinite(scan.ranges).sum() >= 180)
            scans += 1
    assert not goals, ("route incomplete", goals, estimate.pose)
    assert accepted / scans > 0.9, (accepted, scans)
    truth = (bot.data.body("wheel_left").xpos + bot.data.body("wheel_right").xpos) / 2
    error = np.linalg.norm(estimate.pose[:2] - truth[:2])
    assert error < 0.1, error
    print(f"Driven loop: completed without falling; {accepted}/{scans} usable scans, odometry error {error * 1000:.1f} mm")


if __name__ == "__main__":
    check_mapping_inputs()
    check_driven_loop()
    check_lidar()
    check_odometry()
    check_room()
    check_recording()
