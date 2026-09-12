"""Local planar wheel/gyro dead reckoning, not SLAM or a covariance estimator.

Inputs are unwrapped wheel angles and body-frame gyro rates in SI units, with
+x forward, +y left, +z up. Gyro integration tracks tilt; body pitch rate is
added to relative wheel rotation to recover ground travel. Translation uses
wheels; yaw blends wheel differential and integrated gyro yaw increments.
No global simulator pose, orientation sensor, or acceleration is consumed.
Initial tilt must be supplied or known upright. Bias, slip and tilt drift remain
limitations; acceleration/gravity fusion and covariance belong to a later step.
"""

from dataclasses import dataclass
import math

import mujoco
import numpy as np


def _rpy(quat):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat)
    return np.array([math.atan2(rotation[7], rotation[8]),
                     math.atan2(-rotation[6], math.hypot(rotation[0], rotation[3])),
                     math.atan2(rotation[3], rotation[0])])


def footprint_to_base(roll_pitch, axle_offset, wheel_radius):
    """Estimate base_link relative to the axle's ground projection, using tilt and geometry."""
    tilt, axle = np.asarray(roll_pitch, dtype=float), np.asarray(axle_offset, dtype=float)
    if (tilt.shape != (2,) or axle.shape != (3,) or not np.isfinite(tilt).all()
            or not np.isfinite(axle).all() or not math.isfinite(wheel_radius) or wheel_radius <= 0):
        raise ValueError("invalid tilt or wheel geometry")
    roll, pitch = tilt / 2
    quat = np.array([math.cos(roll) * math.cos(pitch), math.sin(roll) * math.cos(pitch),
                     math.cos(roll) * math.sin(pitch), -math.sin(roll) * math.sin(pitch)])
    offset = np.zeros(3)
    mujoco.mju_rotVecQuat(offset, axle, quat)
    return np.array([0, 0, wheel_radius]) - offset, quat


@dataclass
class OdometrySample:
    """Axle-centre ground projection: pose=[x,y,yaw], twist=[forward speed,yaw rate].

    Pose is relative to reset's odom frame; roll/pitch are gyro-integrated tilt.
    """

    timestamp: float
    pose: np.ndarray
    twist: np.ndarray
    roll_pitch: np.ndarray


class WheelImuOdometry:
    def __init__(self, wheel_radii, track_width: float, gyro_weight: float = 0.9,
                 wheel_signs=(1, 1), gyro_bias=(0, 0, 0)):
        self.wheel_radii = np.asarray(wheel_radii, dtype=float).copy()
        self.track_width = float(track_width)
        self.gyro_weight = float(gyro_weight)
        self.wheel_signs = np.asarray(wheel_signs, dtype=float).copy()
        self.gyro_bias = np.asarray(gyro_bias, dtype=float).copy()
        if (self.wheel_radii.shape != (2,) or not np.isfinite(self.wheel_radii).all()
                or np.any(self.wheel_radii <= 0) or not math.isfinite(self.track_width)
                or self.track_width <= 0 or not 0 <= self.gyro_weight <= 1
                or self.wheel_signs.shape != (2,) or not np.isin(self.wheel_signs, [-1, 1]).all()
                or self.gyro_bias.shape != (3,) or not np.isfinite(self.gyro_bias).all()):
            raise ValueError("invalid wheel radii, track width, gyro weight, wheel signs or gyro bias")
        self.reset()

    @classmethod
    def from_model(cls, model, **kwargs):
        """Read static BracketBot wheel calibration; never read a simulated pose."""
        root = model.body("root").id
        wheels = [model.body(f"wheel_{side}") for side in ("left", "right")]
        joints = [model.joint(f"wheel_{side}") for side in ("left", "right")]
        for body, joint in zip(wheels, joints):
            if (body.parentid[0] != root or not np.allclose(body.quat, [1, 0, 0, 0])
                    or joint.type[0] != mujoco.mjtJoint.mjJNT_HINGE
                    or not np.allclose(np.abs(joint.axis), [0, 1, 0])):
                raise ValueError("expected wheel hinges on the root's lateral axis")
        return cls([model.geom(f"wheel_{side}_collision").size[0] for side in ("left", "right")],
                   wheels[0].pos[1] - wheels[1].pos[1],
                   wheel_signs=[joint.axis[1] for joint in joints], **kwargs)

    def reset(self, pose=(0, 0, 0), roll_pitch=(0, 0)) -> None:
        pose, tilt = np.asarray(pose, dtype=float), np.asarray(roll_pitch, dtype=float)
        if (pose.shape != (3,) or tilt.shape != (2,) or not np.isfinite(pose).all()
                or not np.isfinite(tilt).all() or np.any(np.abs(tilt) >= math.pi / 2)):
            raise ValueError("initial pose and tilt must be finite; roll/pitch must be within +/-90 degrees")
        self._pose = pose.copy()
        roll, pitch = tilt / 2
        self._attitude = np.array([math.cos(roll) * math.cos(pitch),
                                   math.sin(roll) * math.cos(pitch),
                                   math.cos(roll) * math.sin(pitch),
                                   -math.sin(roll) * math.sin(pitch)])
        self._time = None
        self._wheels = np.zeros(2)
        self._gyro = np.zeros(3)

    def update(self, timestamp: float, wheel_angles, gyro) -> OdometrySample:
        wheels, gyro = np.asarray(wheel_angles, dtype=float), np.asarray(gyro, dtype=float)
        if (not math.isfinite(timestamp) or timestamp < 0 or wheels.shape != (2,)
                or gyro.shape != (3,) or not np.isfinite(wheels).all() or not np.isfinite(gyro).all()):
            raise ValueError("require a finite nonnegative timestamp, two wheel angles and three gyro rates")
        gyro = gyro - self.gyro_bias
        twist = np.zeros(2)
        if self._time is not None:
            dt = timestamp - self._time
            if dt <= 0:
                raise ValueError("timestamps must increase; call reset() after resetting simulation time")
            omega = (self._gyro + gyro) / 2
            if np.linalg.norm(omega) * dt >= math.pi:
                raise ValueError("gyro interval spans >=180 degrees; provide more frequent samples")
            attitude = self._attitude.copy()
            mujoco.mju_quatIntegrate(attitude, omega, dt)
            before, after = _rpy(self._attitude), _rpy(attitude)
            if max(abs(before[1]), abs(after[1])) >= math.radians(80):
                raise ValueError("planar odometry is undefined near vertical pitch")
            travel = self.wheel_radii * (self.wheel_signs * (wheels - self._wheels) + omega[1] * dt)
            distance = float(travel.mean())
            wheel_yaw = (travel[1] - travel[0]) / self.track_width
            gyro_yaw = math.atan2(math.sin(after[2] - before[2]), math.cos(after[2] - before[2]))
            yaw_change = (1 - self.gyro_weight) * wheel_yaw + self.gyro_weight * gyro_yaw
            heading = self._pose[2] + yaw_change / 2
            chord = distance * np.sinc(yaw_change / (2 * math.pi))
            self._pose[:2] += chord * np.array([math.cos(heading), math.sin(heading)])
            self._pose[2] = math.atan2(math.sin(self._pose[2] + yaw_change),
                                       math.cos(self._pose[2] + yaw_change))
            self._attitude = attitude
            twist[:] = distance / dt, yaw_change / dt
        self._time = float(timestamp)
        self._wheels = wheels.copy()
        self._gyro = gyro.copy()
        return OdometrySample(self._time, self._pose.copy(), twist, _rpy(self._attitude)[:2])
