"""What the robot has to navigate on: a planar lidar, and wheel odometry.

Both are what a SLAM front end eats - a scan to match, and a motion prior to
seed the match with.  Neither is ground truth: the odometry drifts the way wheel
odometry does, and the scan is taken from a sensor bolted to a robot that is
never quite upright, so the beams sweep a plane that pitches with it.

The lidar is cast in Python with `mj_ray` rather than built out of MuJoCo
`rangefinder` sensors, for two reasons.  A rangefinder ignores contype and
conaffinity and only skips geoms on its own body, so a beam from the mast axis
hits the robot's own shell at 6 mm; and sensors are evaluated every step, which
for 72 beams against 50 meshes is most of the step time.  Casting by hand lets
the ray filter on geom group - the room is groups 0 and 1, the robot is 2 and
3 - and lets the scan run at a lidar's rate instead of the solver's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

WHEEL_RADIUS = 0.0846      # m, measured off the tyre mesh in scripts/build_mjcf.py
WHEEL_BASE = 0.3222        # m between the two contact patches


class Lidar:
    """A planar scan from the robot's lidar site, against the room only."""

    ENV_GROUPS = (0, 1)     # robot visuals are group 2, robot collision is group 3

    def __init__(self, model, beams: int = 72, max_range: float = 12.0,
                 site: str = "lidar", noise: float = 0.0):
        self.model = model
        self.site = model.site(site).id
        self.max_range = max_range
        self.noise = noise
        self.angles = np.linspace(0, 2 * math.pi, beams, endpoint=False)
        self.mask = np.zeros(6, dtype=np.uint8)
        for group in self.ENV_GROUPS:
            self.mask[group] = 1
        self._local = np.column_stack([np.cos(self.angles), np.sin(self.angles),
                                       np.zeros(beams)])
        self._rng = np.random.default_rng(0)

    def scan(self, data) -> np.ndarray:
        """Range per beam, in metres.  `inf` where the beam hit nothing in range.

        The beams sweep the sensor's own xy plane, so a robot leaning 3 degrees
        forward sweeps a plane tilted 3 degrees - which is the whole difficulty
        of putting a lidar on something that balances.
        """
        origin = data.site_xpos[self.site]
        rot = data.site_xmat[self.site].reshape(3, 3)
        world = self._local @ rot.T

        out = np.empty(len(self.angles))
        geomid = np.zeros(1, dtype=np.int32)
        for i, direction in enumerate(world):
            hit = mujoco.mj_ray(self.model, data, origin, direction,
                                self.mask, 1, -1, geomid)
            out[i] = hit if 0 <= hit <= self.max_range else np.inf

        if self.noise:
            live = np.isfinite(out)
            out[live] += self._rng.normal(0, self.noise, live.sum())
        return out

    def points(self, data, ranges=None) -> np.ndarray:
        """The scan as world-frame points, dropping beams that hit nothing."""
        ranges = self.scan(data) if ranges is None else ranges
        live = np.isfinite(ranges)
        rot = data.site_xmat[self.site].reshape(3, 3)
        world = self._local[live] @ rot.T
        return data.site_xpos[self.site] + world * ranges[live, None]

    def local_points(self, data, ranges=None) -> np.ndarray:
        """The scan in the sensor's own frame - what a scan matcher works in."""
        ranges = self.scan(data) if ranges is None else ranges
        live = np.isfinite(ranges)
        return self._local[live] * ranges[live, None]


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw])


@dataclass
class WheelOdometry:
    """Dead reckoning from the wheel encoders and the gyro.

    Deliberately not the true pose: this is the drifting estimate SLAM exists to
    correct.  On a balancing robot it drifts for an extra reason - the wheels
    also turn to keep the thing upright, so every recovery from a nudge writes
    phantom distance into the odometry.
    """

    radius: float = WHEEL_RADIUS
    base: float = WHEEL_BASE
    pose: Pose = field(default_factory=Pose)

    def reset(self, x=0.0, y=0.0, yaw=0.0) -> Pose:
        self.pose = Pose(x, y, yaw)
        return self.pose

    def update(self, left_rate: float, right_rate: float, dt: float,
               yaw_rate: float | None = None) -> Pose:
        """Integrate one step.  Rates are wheel spin in rad/s."""
        speed = self.radius * (left_rate + right_rate) / 2
        turn = (yaw_rate if yaw_rate is not None
                else self.radius * (right_rate - left_rate) / self.base)
        self.pose.yaw += turn * dt
        self.pose.x += speed * math.cos(self.pose.yaw) * dt
        self.pose.y += speed * math.sin(self.pose.yaw) * dt
        return self.pose


def true_pose(data, body: str = "root") -> Pose:
    """Where the robot actually is - for scoring an estimate, not for using."""
    rot = data.body(body).xmat.reshape(3, 3)
    return Pose(float(data.body(body).xpos[0]), float(data.body(body).xpos[1]),
                math.atan2(rot[1, 0], rot[0, 0]))
