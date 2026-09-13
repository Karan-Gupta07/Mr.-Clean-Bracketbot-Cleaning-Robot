"""Ideal instantaneous LiDAR scans in the moving sensor's +x/+y plane.

Ranges are slant distances in metres: +inf means no return within range_max;
NaN means too close or occluded by the robot. Floor hits remain real returns.
The existing mount is inside the mast, so its rigid mounting assembly is masked
by default. Moving robot links still occlude; no geometry groups are discarded.
This is not a physically validated mount or a levelled 2D SLAM scan.
"""

import copy
from dataclasses import dataclass
import operator

import mujoco
import numpy as np


@dataclass
class LaserScan:
    timestamp: float
    ranges: np.ndarray
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    frame_id: str


def project_scan(scan: LaserScan, position, quaternion, min_height=0.12,
                 max_height=0.52, max_tilt=np.deg2rad(2)) -> LaserScan:
    """Project real returns into base_footprint XY; never invent missing/free rays.

    Position/quaternion express the raw sensor in base_footprint, derived from
    estimated tilt and mount geometry. Reject excessive tilt and floor/overhead
    hits. Empty bins stay unknown. Angular rebinning keeps the nearest return.
    """
    position, quat = np.asarray(position, dtype=float), np.asarray(quaternion, dtype=float)
    if (position.shape != (3,) or quat.shape != (4,) or not np.isfinite(position).all()
            or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1)
            or not np.isfinite([min_height, max_height, max_tilt]).all()
            or not 0 <= min_height < max_height or not 0 <= max_tilt <= np.pi / 2
            or not np.isclose(len(scan.ranges) * scan.angle_increment, 2 * np.pi)):
        raise ValueError("require a calibrated sensor pose, valid height/tilt bounds and a 360-degree scan")
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat)
    rotation = rotation.reshape(3, 3)
    ranges = np.full(len(scan.ranges), np.inf)
    if rotation[2, 2] >= np.cos(max_tilt) - 1e-12:
        valid = np.isfinite(scan.ranges) & (scan.ranges >= scan.range_min) & (scan.ranges <= scan.range_max)
        angles = scan.angle_min + np.flatnonzero(valid) * scan.angle_increment
        points = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(len(angles))))
        points = (points * scan.ranges[valid, None]) @ rotation.T + position
        distances = np.linalg.norm(points[:, :2], axis=1)
        keep = ((points[:, 2] >= min_height) & (points[:, 2] <= max_height)
                & (distances >= scan.range_min) & (distances <= scan.range_max))
        bearings = np.arctan2(points[keep, 1], points[keep, 0])
        bins = np.rint((bearings - scan.angle_min) / scan.angle_increment).astype(int) % len(ranges)
        np.minimum.at(ranges, bins, distances[keep])
    ranges[np.isinf(ranges)] = np.nan
    return LaserScan(scan.timestamp, ranges, scan.angle_min, scan.angle_increment,
                     scan.range_min, scan.range_max, "lidar_planar")


class Lidar:
    def __init__(self, model, beams: int = 360, range_min: float = 0.05,
                 range_max: float = 10.0, site: str = "lidar", robot: str = "root",
                 mask_mount: bool = True):
        try:
            beams = operator.index(beams)
        except TypeError as error:
            raise ValueError("beams must be a positive integer") from error
        if beams < 1 or not np.isfinite([range_min, range_max]).all() or not 0 <= range_min < range_max:
            raise ValueError("require beams > 0 and finite 0 <= range_min < range_max")
        self.site = model.site(site).id
        root = model.body(robot).id
        mount = model.site_bodyid[self.site]
        if root == 0 or model.body_rootid[mount] != root:
            raise ValueError("robot must be a top-level body containing the lidar site")
        self.frame_id = site
        self.range_min, self.range_max = float(range_min), float(range_max)
        self.angle_increment = 2 * np.pi / beams
        self.angles = -np.pi + np.arange(beams) * self.angle_increment
        self._directions = np.column_stack((np.cos(self.angles), np.sin(self.angles), np.zeros(beams)))
        self._robot_geoms = model.body_rootid[model.geom_bodyid] == root
        self._model = copy.copy(model)
        if mask_mount:
            fixed_mount = self._robot_geoms & (model.body_weldid[model.geom_bodyid] == model.body_weldid[mount])
            self._model.geom_rgba[fixed_mount, 3] = 0
        self._data = mujoco.MjData(self._model)

    def scan(self, data) -> LaserScan:
        """Read a paused state, without changing it; all rays share its timestamp.

        A caller using a simulation thread must synchronize access while copying
        the state. This instance owns scratch buffers and is not thread-safe.
        """
        if not np.isfinite(data.time) or data.time < 0 or not np.isfinite(data.qpos).all():
            raise ValueError("scan state must have a finite nonnegative time and finite qpos")
        self._data.qpos[:] = data.qpos
        self._data.mocap_pos[:] = data.mocap_pos
        self._data.mocap_quat[:] = data.mocap_quat
        self._data.time = data.time
        mujoco.mj_kinematics(self._model, self._data)
        directions = self._directions @ self._data.site_xmat[self.site].reshape(3, 3).T
        ids = np.full(len(self.angles), -1, dtype=np.int32)
        distances = np.full(len(self.angles), -1.0)
        mujoco.mj_multiRay(
            self._model, self._data, self._data.site_xpos[self.site], directions.ravel(),
            None, True, -1, ids, distances, None, len(self.angles), self.range_max,
        )
        hit = ids >= 0
        ranges = np.where(hit & (distances <= self.range_max), distances, np.inf)
        blocked = hit & self._robot_geoms[np.maximum(ids, 0)] & (distances <= self.range_max)
        ranges[blocked | (ranges < self.range_min)] = np.nan
        return LaserScan(float(self._data.time), ranges, float(self.angles[0]),
                         self.angle_increment, self.range_min, self.range_max, self.frame_id)
