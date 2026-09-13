"""Execute bounded open-loop phases, checking the actual pose at rest.

Every settled nonarrived phase boundary replans from fresh pose evidence.
Drift or a stale pose stops the schedule; recovery settles rather than resuming
an obsolete clock. A completed turn must actually reach its required heading
before translation can begin. Near the goal, guarded terminal turns replace
exact-anchor micro-detours.

Commands still come only from precomputed speed/yaw-rate schedules. This is
not a per-tick steering controller, and it never touches the sim or ROS.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .navmap import OccupancyGrid
from .planner import Limits, NoPath, Trajectory, plan, profile, terminal_plan, turn_free, turn_plan

_SPATIAL_EPS = 1e-9
_SETTLE_WINDOW = 0.5
_SETTLE_XY = 0.005
_SETTLE_YAW = math.radians(0.5)
_TURN_YAW = math.radians(2)
_LOST_MAX = 30.0                                     # s without any pose before giving up


def _finite_pose(value, name):
    pose = tuple(map(float, value))
    if len(pose) != 3 or not all(math.isfinite(v) for v in pose):
        raise ValueError(f"{name} must be a finite (x, y, yaw) triple")
    return pose


def _finite_time(value):
    now = float(value)
    if not math.isfinite(now) or now < 0:
        raise ValueError("time must be finite and nonnegative")
    return now


class Navigator:
    """Execute rest-to-rest phases with mandatory replanning at settled boundaries.

    `grid` is the raw, uninflated grid: `plan` inflates it for the curve, and
    the turns and dock straights need the raw cells to fit the robot's
    rectangle where its circle would not go.

    `max_replans` bounds every replan, not only drift corrections: each settled
    phase boundary spends one, so the quota has to cover a whole route.
    """

    def __init__(self, grid: OccupancyGrid, limits: Limits | None = None,
                 off_path: float = 0.15,                 # widened from the spec's 0.10 m in sim tuning
                 pose_timeout: float = 0.5, arrive_xy: float = 0.10,
                 arrive_yaw: float = math.radians(5), max_replans: int = 60):
        if limits is None:
            limits = Limits()
        self.grid, self.limits = grid, limits
        self.off_path, self.pose_timeout = off_path, pose_timeout
        self.arrive_xy, self.arrive_yaw = arrive_xy, arrive_yaw
        self.max_replans = max_replans
        self.goal = None
        self.traj: Trajectory | None = None
        self.t0 = 0.0
        self.done = False
        self.failed: str | None = None
        self.replans: list[tuple[float, float]] = []     # (time, metres off schedule)
        self._last_pose = 0.0
        self._last_step = None
        self._hold_since = None
        self._samples = deque()
        self._turn_goal = None
        self._stop_expected = None
        self._settle_min = max(limits.v_max / limits.a_max, limits.omega_max / limits.alpha_max)
        self._settle_max = 8.0

    def go_to(self, goal, pose_now, now: float) -> Trajectory | None:
        """Invalidate prior motion, then plan from a fresh finite pose and goal.

        Missing or invalid inputs latch a failure and return None. Unexpected
        planning errors still propagate, but cannot leave the old schedule live.
        A later valid go_to clears the failure and starts a new schedule and
        clock epoch, even when its timestamp precedes the previous request.
        """
        self.traj = None
        self.goal = None
        self.done, self.failed, self.replans = False, "navigation reset did not complete", []
        self._last_step = None
        self._hold_since, self._turn_goal, self._stop_expected = None, None, None
        self._samples.clear()
        if pose_now is None:
            self.failed = "a fresh pose is required to plan"
            return None
        try:
            goal, pose_now = _finite_pose(goal, "goal"), _finite_pose(pose_now, "pose")
            now = _finite_time(now)
        except (TypeError, ValueError, OverflowError) as error:
            self.failed = f"invalid navigation request: {error}"
            return None
        self.goal = goal
        self._plan(pose_now, now)
        if self.traj is not None:
            self.failed = None
            self._last_step = now
        return self.traj

    def _phase(self):
        return self.traj.phases[0]

    def step(self, pose_now, now: float) -> tuple[float, float]:
        """Read the schedule, or hold zero until fresh evidence permits a new phase.

        None is intentional pose loss with the existing grace period. Other
        poses must be finite triples; time must be finite, nonnegative and
        nondecreasing since go_to. Invalid updates latch zero until a new go_to.
        The schedule stays stashed, and motion off, until this returns: an
        unexpected bug anywhere in the update propagates with motion disabled.
        """
        if self.done or self.failed or self.traj is None:
            return 0.0, 0.0
        traj, self.traj = self.traj, None
        self.failed = "navigation update did not complete"
        try:
            try:
                now = _finite_time(now)
                if self._last_step is not None and now < self._last_step:
                    raise ValueError("time must not move backwards; use go_to to reset the clock")
                if pose_now is not None:
                    pose_now = _finite_pose(pose_now, "pose")
            except (TypeError, ValueError, OverflowError) as error:
                self.failed = f"invalid navigation update: {error}"
                self._samples.clear()
                return 0.0, 0.0
            self._last_step = now
            if pose_now is not None:
                if now - self._last_pose > self.pose_timeout:
                    self._samples.clear()
                self._last_pose = now
            else:
                self._samples.clear()
            if self._hold_since is None:
                kind, end = traj.phases[0]
                elapsed = now - self.t0
                expected = traj.at(min(elapsed, traj.t[end]))[0]
                stale = pose_now is None and now > self._last_pose + self.pose_timeout
                off = math.dist(pose_now[:2], expected) if pose_now is not None else 0.0
                terminal = (pose_now is not None and
                            (self._arrived(pose_now) or (kind in ("curve", "dock")
                             and math.dist(pose_now[:2], self.goal[:2]) <= self.arrive_xy + _SPATIAL_EPS)))
                if not (stale or off > self.off_path + _SPATIAL_EPS or terminal
                        or now >= self.t0 + traj.t[end]):
                    command = traj.at(elapsed)[2:]
                    self.traj, self.failed = traj, None
                    return command
                self._hold_since, self._stop_expected = now, expected
                self._samples.clear()
            self.traj, self.failed = traj, None       # settling may replan, arrive or fail
            self._settle(pose_now, now)
            return 0.0, 0.0
        except BaseException:                         # a bug must not leave the schedule live
            self.traj, self.failed = None, "navigation update did not complete"
            raise

    def _settle(self, pose, now):
        """Require a minimum braking dwell and a continuous fresh, stable pose window.

        A pose that never stabilises re-arms the hold and replans at the
        deadline instead of latching: only the replan quota, or a pose missing
        for _LOST_MAX, ends a route from here.
        """
        stable = False
        if pose is not None:
            if not self._samples or now > self._samples[-1][0]:
                self._samples.append((now, tuple(pose)))
            while len(self._samples) > 1 and self._samples[1][0] <= now - _SETTLE_WINDOW:
                self._samples.popleft()
            if (now >= self._hold_since + self._settle_min
                    and now - self._samples[0][0] >= _SETTLE_WINDOW - 1e-9):
                points = np.array([p for _, p in self._samples])
                yaw = np.unwrap(points[:, 2])
                stable = (np.linalg.norm(np.ptp(points[:, :2], axis=0)) <= _SETTLE_XY
                          and np.ptp(yaw) <= _SETTLE_YAW)
                if stable and all(self._arrived(p) for _, p in self._samples):
                    self.done = True
                    return
        expired = now >= self._hold_since + self._settle_max
        if not stable and not expired:
            return
        if pose is None:                    # missing evidence, not unstable evidence: keep holding
            if now >= self._last_pose + _LOST_MAX:
                self.failed = f"no pose for {_LOST_MAX:.0f} s while holding zero"
            return
        if self._arrived(pose):
            self.done = expired             # a stable window needs every sample inside tolerance
            return
        off = math.dist(pose[:2], self._stop_expected)
        if len(self.replans) >= self.max_replans:
            self.failed = (f"gave up after {self.max_replans} replans, "
                           f"{off * 100:.0f} cm off schedule")
            return
        if expired:                         # the pose never settled: re-arm rather than give up
            self._hold_since = now
            self._samples.clear()
        self.replans.append((now, off))
        self._plan(pose, now)

    def _plan(self, pose_now, now: float) -> None:
        try:
            if (self._turn_goal is not None and
                    abs(math.atan2(math.sin(self._turn_goal - pose_now[2]),
                                   math.cos(self._turn_goal - pose_now[2]))) > _TURN_YAW):
                centre = (self.goal[:2] if math.dist(pose_now[:2], self.goal[:2])
                          <= self.arrive_xy + _SPATIAL_EPS else None)
                path = turn_plan(self.grid, pose_now, self._turn_goal, self.limits, centre, self.arrive_xy)
            else:
                path = terminal_plan(self.grid, pose_now, self.goal, self.limits, self.arrive_xy)
                self._turn_goal = None
                if path is not None:
                    self._turn_goal = path.goal[2]
                else:
                    path = plan(self.grid, pose_now, self.goal, self.limits)
                    if path.exit == 0 and path.turn != 0:
                        heading = path.start[2] + path.turn
                        if not turn_free(self.grid, pose_now, heading, self.limits):
                            self._turn_goal = heading
                            path = turn_plan(self.grid, pose_now, heading, self.limits)
            self.traj = profile(path, self.limits)
        except NoPath as error:
            self.traj, self.failed = None, str(error)
            return
        self.t0 = now
        self._last_pose = now
        self._hold_since = None
        self._samples.clear()
        kind, end = self._phase()
        if kind == "turn":
            self._turn_goal = float(self.traj.yaw[end])

    def _arrived(self, pose) -> bool:
        dx, dy = pose[0] - self.goal[0], pose[1] - self.goal[1]
        dyaw = math.atan2(math.sin(pose[2] - self.goal[2]), math.cos(pose[2] - self.goal[2]))
        return (math.hypot(dx, dy) <= self.arrive_xy + _SPATIAL_EPS
                and abs(dyaw) <= self.arrive_yaw + _SPATIAL_EPS)


def true_pose(data) -> tuple[float, float, float]:
    """The wheel-axle midpoint and the chassis heading, from the sim.

    The same point the ROS bridge calls base_footprint, so the local check
    and the real thing measure arrival the same way. For scoring and for the
    SLAM stand-in only: nothing that runs on the robot may read this.
    """
    axle = (data.body("wheel_left").xpos + data.body("wheel_right").xpos) / 2
    rot = data.body("root").xmat.reshape(3, 3)
    return float(axle[0]), float(axle[1]), math.atan2(rot[1, 0], rot[0, 0])
