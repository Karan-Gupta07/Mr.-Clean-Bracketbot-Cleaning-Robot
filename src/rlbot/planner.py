"""From a pose to a pose across the room: a curved path, and a schedule to drive it.

`plan` finds the route.  A* on the inflated grid gives the corridor; a
line-of-sight shortcut takes the staircase corners out of it; a clamped cubic
spline through what is left gives a curve that leaves tangent to the robot's
heading and arrives tangent to the dock.  Because a dock sits inside the
robot's own inflation radius, the curve is bracketed by straight runs along
the start's and the goal's axes that are checked against the robot's
rectangle rather than its circle, and preceded by a turn in place only when
the corridor leaves more than 45 degrees off the heading.

`profile` turns that into a time-stamped schedule of speed and yaw rate under
the balancer's limits: slow through bends so the yaw rate stays under what it
survives, ramps no steeper than its DriveController will pass.

Nothing here touches the sim.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .navmap import OccupancyGrid
from .room import TABLES

DS = 0.02                          # m between samples of a finished path
TURN_FIRST = math.radians(45)      # heading error above which the robot turns in place before the curve
BACKOFF_MAX = 1.0                  # m to search along a pose's axis for a spot the circle fits


@dataclass
class Limits:
    v_max: float = 0.15        # m/s.       DriveController's max_speed
    omega_max: float = 0.3     # rad/s.     DriveController's max_yaw_rate
    a_max: float = 0.1         # m/s^2.     DriveController's linear_accel
    alpha_max: float = 0.3     # rad/s^2.   DriveController's angular_accel
    radius: float = 0.21       # m, half the 0.42 m driving footprint (navmap.robot_footprint)
    half_depth: float = 0.094  # m, chassis half-extent along the heading
    half_width: float = 0.186  # m, chassis half-extent across it
    margin: float = 0.05       # m of daylight on top of the radius


@dataclass
class Path:
    """A route in four parts, any of which may be empty.

    `exit` metres straight along the start heading (negative is reverse),
    `turn` radians in place, the curve (`xy`, `yaw`, `kappa`, `s`, sampled
    every DS metres of arc), then `dock` metres straight along the goal's
    yaw. `waypoints` are the spline's knots, for drawing.
    """

    start: tuple[float, float, float]
    goal: tuple[float, float, float]
    exit: float
    turn: float
    xy: np.ndarray
    yaw: np.ndarray
    kappa: np.ndarray
    s: np.ndarray
    dock: float
    waypoints: np.ndarray

    @property
    def length(self) -> float:
        """The curve alone, m."""
        return float(self.s[-1]) if len(self.s) else 0.0

    @property
    def min_radius(self) -> float:
        k = float(np.abs(self.kappa).max()) if len(self.kappa) else 0.0
        return math.inf if k == 0 else 1.0 / k


@dataclass
class Trajectory:
    """A schedule, with its (phase name, inclusive final sample index) stops.

    Each listed boundary has zero speed and yaw rate. `profile` always fills
    `phases` in, and the navigator reads the first one to know what it is
    driving, so a schedule built by hand has to fill it in too.
    """

    t: np.ndarray
    xy: np.ndarray
    yaw: np.ndarray
    v: np.ndarray              # m/s along the heading, negative in reverse
    omega: np.ndarray          # rad/s
    phases: tuple[tuple[str, int], ...] = ()

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def at(self, t: float):
        """(xy, yaw, v, omega) at time t, interpolated, clamped to the ends."""
        t = min(max(float(t), 0.0), self.duration)
        xy = np.array([np.interp(t, self.t, self.xy[:, 0]), np.interp(t, self.t, self.xy[:, 1])])
        yaw = math.atan2(np.interp(t, self.t, np.sin(self.yaw)), np.interp(t, self.t, np.cos(self.yaw)))
        return xy, yaw, float(np.interp(t, self.t, self.v)), float(np.interp(t, self.t, self.omega))


class NoPath(RuntimeError):
    """There is no route the planner is willing to drive."""


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


# ---- search --------------------------------------------------------------
def astar(grid: OccupancyGrid, start, goal) -> list[tuple[int, int]]:
    """8-connected A* between two cells. Returns the cells, start to goal.

    Diagonals may not cut a corner: both the cells they pass between have to
    be free, or the robot's circle would clip the corner they cut.
    """
    start, goal = tuple(start), tuple(goal)
    if not grid.inside(*start) or grid.occupied[start]:
        raise NoPath(f"start cell {start} at {grid.world(*start)} is occupied")
    if not grid.inside(*goal) or grid.occupied[goal]:
        raise NoPath(f"goal cell {goal} at {grid.world(*goal)} is occupied")
    moves = [(di, dj, math.hypot(di, dj))
             for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]

    def h(c):
        return math.hypot(c[0] - goal[0], c[1] - goal[1])

    best = {start: 0.0}
    came: dict = {}
    heap = [(h(start), 0.0, start)]
    while heap:
        _, g, cur = heapq.heappop(heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if g > best.get(cur, math.inf):
            continue
        for di, dj, cost in moves:
            nxt = (cur[0] + di, cur[1] + dj)
            if not grid.inside(*nxt) or grid.occupied[nxt]:
                continue
            if di and dj and (grid.occupied[cur[0] + di, cur[1]]
                              or grid.occupied[cur[0], cur[1] + dj]):
                continue
            ng = g + cost
            if ng < best.get(nxt, math.inf):
                best[nxt] = ng
                came[nxt] = cur
                heapq.heappush(heap, (ng + h(nxt), ng, nxt))
    raise NoPath(f"no route on the grid from {grid.world(*start)} to {grid.world(*goal)}")


def shortcut(grid: OccupancyGrid, points) -> list[int]:
    """Indices of the waypoints a straight line cannot see past. Greedy:
    from each kept point, jump to the farthest one still in line of sight."""
    points = np.asarray(points, float)
    keep, i = [0], 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i and not grid.line_free(points[i], points[j]):
            j -= 1
        if j == i:
            raise NoPath(f"no line-of-sight route from waypoint {i}")
        keep.append(j)
        i = j
    return keep


# ---- the curve -----------------------------------------------------------
def _clamped_spline(knots, d0, dn):
    """Coefficients of a clamped cubic spline through `knots` (m, 2), with
    first derivatives d0 and dn at the ends, parametrised by chord length.

    Burden & Faires' clamped spline: solve the tridiagonal system for the
    quadratic coefficients c, then b and d follow. Returns (u, a, b, c, d)
    with one row per segment.
    """
    a = np.asarray(knots, float)
    n = len(a) - 1
    h = np.linalg.norm(np.diff(a, axis=0), axis=1)
    if n < 1 or (h <= 0).any():
        raise ValueError("a spline needs at least two distinct knots")
    u = np.concatenate([[0.0], np.cumsum(h)])
    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros((n + 1, 2))
    A[0, 0], A[0, 1] = 2 * h[0], h[0]
    rhs[0] = 3 * (a[1] - a[0]) / h[0] - 3 * np.asarray(d0, float)
    A[n, n - 1], A[n, n] = h[n - 1], 2 * h[n - 1]
    rhs[n] = 3 * np.asarray(dn, float) - 3 * (a[n] - a[n - 1]) / h[n - 1]
    for i in range(1, n):
        A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        rhs[i] = 3 * (a[i + 1] - a[i]) / h[i] - 3 * (a[i] - a[i - 1]) / h[i - 1]
    c = np.linalg.solve(A, rhs)
    b = (a[1:] - a[:-1]) / h[:, None] - h[:, None] * (c[1:] + 2 * c[:-1]) / 3
    d = (c[1:] - c[:-1]) / (3 * h[:, None])
    return u, a[:-1], b, c[:-1], d


def _check_regular(b, c, d, h):
    """Reject vanishing derivatives anywhere on a cubic, not just at samples.

    On the unit interval its velocity is quadratic. Squared speed has extrema
    at the roots of velocity dot acceleration, a cubic, or at the endpoints.
    Normalise the coefficients so the 1e-6 speed threshold is relative to the
    segment's velocity scale. Testing every root's real part also covers
    repeated real roots that a numerical solver may split into complex pairs.
    """
    v = np.array([b, 2 * h * c, 3 * h ** 2 * d])
    scale = np.linalg.norm(v, axis=1).max()
    if not np.isfinite(scale) or scale == 0:
        raise NoPath("nonregular spline: invalid or vanishing derivative")
    v0, v1, v2 = v / scale
    roots = np.polynomial.polynomial.polyroots(
        [v0 @ v1, v1 @ v1 + 2 * (v0 @ v2), 3 * (v1 @ v2), 2 * (v2 @ v2)])
    t = np.clip(np.r_[0.0, 1.0, roots.real], 0, 1)[:, None]
    if np.linalg.norm(v0 + v1 * t + v2 * t ** 2, axis=1).min() <= 1e-6:
        raise NoPath("nonregular spline: vanishing derivative makes the heading undefined")


def _sample_spline(coef, step):
    """Points, headings, signed curvatures and arc lengths along the spline.

    Start at most `step` apart in the parameter, then bisect only intervals
    needing more detail. A cubic's chord and Bezier control polygon bound its
    arc length; close their relative gap to 1e-4 and limit tangent changes to
    five degrees. This resolves short bends without oversampling straight runs.
    """
    u, a, b, c, d = coef
    p, dp, ddp = [], [], []
    last = len(a) - 1
    for i in range(len(a)):
        h = u[i + 1] - u[i]
        _check_regular(b[i], c[i], d[i], h)
        n = max(2, int(np.ceil(h / step)) + 1)
        t = np.linspace(0, h, n)[:, None]
        for _ in range(24):
            points = a[i] + b[i] * t + c[i] * t ** 2 + d[i] * t ** 3
            velocity = b[i] + 2 * c[i] * t + 3 * d[i] * t ** 2
            dt = np.diff(t, axis=0) / 3
            q1 = points[:-1] + velocity[:-1] * dt
            q2 = points[1:] - velocity[1:] * dt
            chord = np.linalg.norm(np.diff(points, axis=0), axis=1)
            polygon = (np.linalg.norm(q1 - points[:-1], axis=1)
                       + np.linalg.norm(q2 - q1, axis=1)
                       + np.linalg.norm(points[1:] - q2, axis=1))
            speed = np.linalg.norm(velocity, axis=1)
            dot = np.sum(velocity[:-1] * velocity[1:], axis=1)
            refine = ((polygon - chord > 1e-4 * polygon)
                      | (dot < math.cos(math.radians(5)) * speed[:-1] * speed[1:]))
            if not refine.any():
                break
            t = np.sort(np.concatenate([t[:, 0], ((t[:-1, 0] + t[1:, 0]) / 2)[refine]]))[:, None]
        else:
            raise NoPath("spline sampling did not converge")
        if i != last:
            t, points, velocity = t[:-1], points[:-1], velocity[:-1]
        p.append(points)
        dp.append(velocity)
        ddp.append(2 * c[i] + 6 * d[i] * t)
    p, dp, ddp = np.vstack(p), np.vstack(dp), np.vstack(ddp)
    yaw = np.arctan2(dp[:, 1], dp[:, 0])
    speed = np.hypot(dp[:, 0], dp[:, 1])
    kappa = (dp[:, 0] * ddp[:, 1] - dp[:, 1] * ddp[:, 0]) / speed ** 3
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    return p, yaw, kappa, s


def _resample(p, yaw, kappa, s, ds):
    """The same curve at a uniform arc-length spacing, at least three samples."""
    n = max(3, int(np.ceil(s[-1] / ds)) + 1)
    ss = np.linspace(0, s[-1], n)
    x, y = np.interp(ss, s, p[:, 0]), np.interp(ss, s, p[:, 1])
    yw = np.arctan2(np.interp(ss, s, np.sin(yaw)), np.interp(ss, s, np.cos(yaw)))
    return np.column_stack([x, y]), yw, np.interp(ss, s, kappa), ss


def _fit(inflated: OccupancyGrid, corridor, keep, t0, tn):
    """A spline through the kept corridor points that stays on free cells.

    If a sample or segment clips, put back the nearest unused corridor point
    and go again. Check both the dense and resampled polylines, including
    corner contact. Each retry keeps strictly more points, so this terminates.
    """
    keep = list(keep)
    while True:
        knots = corridor[keep]
        p, yaw, kappa, s = _sample_spline(_clamped_spline(knots, t0, tn), inflated.resolution / 2)
        bad = np.flatnonzero(~inflated.free_at(p))
        if len(bad):
            clipped = p[bad[0]]
        else:
            curve = _resample(p, yaw, kappa, s, DS)
            clipped = next(((a + b) / 2 for points in (p, curve[0])
                            for a, b in zip(points, points[1:]) if not inflated.line_free(a, b)), None)
            if clipped is None:
                return (*curve, knots)
        dist = np.linalg.norm(corridor - clipped, axis=1)
        dist[keep] = np.inf
        if not np.isfinite(dist).any():
            raise NoPath("the curve clips an obstacle even through every corridor cell; "
                         "the margin is too small for this map")
        keep = sorted(keep + [int(np.argmin(dist))])


# ---- the whole route -----------------------------------------------------
def _straight_out(grid: OccupancyGrid, inflated: OccupancyGrid, pose, half) -> float:
    """Signed distance along the pose's heading to the nearest point where the
    robot's circle is clear. Backwards first, because that is how you leave
    a table. Every pose on the way has to fit the robot's rectangle on the
    raw grid, so it cannot back through a table leg to get there.

    The whole straight sweep is one rectangle, lengthened along the heading.
    """
    x, y, yaw = pose
    if not grid.rect_free(x, y, yaw, *half):
        raise NoPath(f"the robot does not fit at ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg)")
    if inflated.line_free((x, y), (x, y)):
        return 0.0
    c, s = math.cos(yaw), math.sin(yaw)
    step = grid.resolution / 2
    for sign in (-1, 1):
        for k in range(1, int(BACKOFF_MAX / step) + 1):
            d = sign * k * step
            px, py = x + d * c, y + d * s
            if not grid.rect_free(x + d * c / 2, y + d * s / 2, yaw,
                                  half[0] + abs(d) / 2, half[1]):
                break
            if inflated.line_free((px, py), (px, py)):
                return d
    raise NoPath(f"no room within {BACKOFF_MAX} m along the axis of "
                 f"({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg)")


def plan(grid: OccupancyGrid, start, goal, limits: Limits | None = None) -> Path:
    """The route from `start` to `goal`, both (x, y, yaw) in the grid's frame."""
    if limits is None:
        limits = Limits()
    inflated = grid.inflate(limits.radius + limits.margin)
    start, goal = tuple(map(float, start)), tuple(map(float, goal))
    sx, sy, syaw = start
    gx, gy, gyaw = goal
    half = (limits.half_depth + limits.margin, limits.half_width + limits.margin)
    if start == goal and grid.rect_free(sx, sy, syaw, *half):
        exit_len = d_goal = 0.0
    else:
        exit_len = _straight_out(grid, inflated, start, half)
        d_goal = _straight_out(grid, inflated, goal, half)
    a = np.array([sx + exit_len * math.cos(syaw), sy + exit_len * math.sin(syaw)])
    b = np.array([gx + d_goal * math.cos(gyaw), gy + d_goal * math.sin(gyaw)])
    turn = 0.0
    curve = None
    if not np.array_equal(a, b):
        cells = astar(inflated, inflated.cell(*a), inflated.cell(*b))
        corridor = np.array([tuple(a)] + [inflated.world(i, j) for i, j in cells[1:-1]] + [tuple(b)])
        keep = shortcut(inflated, corridor)
        first = corridor[keep[1]] - a
        heading = math.atan2(first[1], first[0])
        if abs(_wrap(heading - syaw)) > TURN_FIRST:
            turn = _wrap(heading - syaw)
            t0 = first / np.linalg.norm(first)
        else:
            t0 = np.array([math.cos(syaw), math.sin(syaw)])
        tn = np.array([math.cos(gyaw), math.sin(gyaw)])
        curve = _fit(inflated, corridor, keep, t0, tn)
    elif abs(_wrap(gyaw - syaw)) > math.radians(1):
        turn = _wrap(gyaw - syaw)
    if curve is None:
        xy, yaw, kappa, s = np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0)
        knots = np.vstack([a, b])
    else:
        xy, yaw, kappa, s, knots = curve
    return Path(start, goal, exit_len, turn, xy, yaw, kappa, s, -d_goal, knots)


def turn_free(grid: OccupancyGrid, pose, heading: float, limits: Limits) -> bool:
    """Certify the whole shortest rotation of the padded rectangle.

    Extra padding bounds each corner's motion between the nearest angular
    samples, including rotations whose endpoints fit but intermediate poses do not.
    """
    x, y, yaw = pose
    angle = _wrap(heading - yaw)
    half = (limits.half_depth + limits.margin, limits.half_width + limits.margin)
    steps = max(1, int(math.ceil(abs(angle) / DTH)))
    guard = math.hypot(*half) * abs(angle) / (2 * steps)
    return all(grid.rect_free(x, y, yaw + a, half[0] + guard, half[1] + guard)
               for a in np.linspace(0.0, angle, steps + 1))


def turn_plan(grid: OccupancyGrid, start, heading: float, limits: Limits,
              centre=None, tolerance: float = BACKOFF_MAX) -> Path:
    """A guarded turn, or a guarded straight escape followed by that turn.

    An optional tolerance disk constrains the entire escape, not just the
    rotation anchor. Search past the first circle-free point: its padded
    rectangle need not be safe to rotate.
    """
    start = tuple(map(float, start))
    x, y, yaw = start
    half = (limits.half_depth + limits.margin, limits.half_width + limits.margin)
    if not grid.rect_free(*start, *half):
        raise NoPath(f"the robot does not fit at ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg)")
    step = grid.resolution / 2
    candidates = [0.0] + [sign * k * step for sign in (-1, 1)
                         for k in range(1, int(BACKOFF_MAX / step) + 1)]
    for distance in candidates:
        px, py = x + distance * math.cos(yaw), y + distance * math.sin(yaw)
        if centre is not None and (math.dist((x, y), centre) > tolerance + 1e-9
                                   or math.dist((px, py), centre) > tolerance + 1e-9):
            continue
        if not grid.rect_free((x + px) / 2, (y + py) / 2, yaw,
                              half[0] + abs(distance) / 2, half[1]):
            continue
        if turn_free(grid, (px, py, yaw), heading, limits):
            return Path(start, (px, py, heading), distance, _wrap(heading - yaw),
                        np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0), 0.0,
                        np.array([[px, py], [px, py]]))
    raise NoPath("no collision-free turn or straight escape within the allowed area")


def terminal_plan(grid: OccupancyGrid, start, goal, limits: Limits, arrive_xy: float) -> Path | None:
    """Capture a nearby goal disk, then finish without an exact-anchor detour.

    Inside the disk, prefer a fixed-XY turn. Near its rim a straight inward
    move can leave room for balance drift during the turn. Just outside it,
    aim at the actual goal rather than a tiny curve back to a docking anchor.
    All straight sweeps and rotations are checked with the padded rectangle;
    generic `plan` remains exact-pose and arrival acceptance is unchanged.
    """
    start = tuple(map(float, start))
    x, y, yaw = start
    distance = math.dist(start[:2], goal[:2])
    if distance > 2 * arrive_xy + 1e-9:
        return None
    half = (limits.half_depth + limits.margin, limits.half_width + limits.margin)
    reserve = 0.8 * arrive_xy
    if distance > reserve:
        projection = (goal[0] - x) * math.cos(yaw) + (goal[1] - y) * math.sin(yaw)
        steps = max(1, int(math.ceil(max(0.0, projection) / (grid.resolution / 2))))
        for along in np.linspace(max(0.0, projection), 0.0, steps + 1):
            px, py = x + along * math.cos(yaw), y + along * math.sin(yaw)
            if (along > 0 and math.dist((px, py), goal[:2]) <= reserve
                    and grid.rect_free((x + px) / 2, (y + py) / 2, yaw,
                                       half[0] + along / 2, half[1])
                    and turn_free(grid, (px, py, yaw), goal[2], limits)):
                return Path(start, (px, py, goal[2]), float(along), _wrap(goal[2] - yaw),
                            np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0), 0.0,
                            np.array([[px, py], [px, py]]))
    if distance <= arrive_xy + 1e-9:
        return turn_plan(grid, start, goal[2], limits, goal[:2], arrive_xy)
    heading = math.atan2(goal[1] - y, goal[0] - x)
    approach = distance - reserve
    if (abs(_wrap(heading - yaw)) > math.radians(2)
            and grid.rect_free(x + approach * math.cos(heading) / 2,
                               y + approach * math.sin(heading) / 2, heading,
                               half[0] + approach / 2, half[1])):
        return turn_plan(grid, start, heading, limits, start[:2], arrive_xy)
    return None


# ---- the schedule --------------------------------------------------------
DTH = 0.02      # rad between samples of a turn in place


def _ramp(n: int, ds: float, cap, accel: float):
    """Speeds at n samples ds apart: under `cap`, zero at both ends, and never
    changing faster than `accel` per unit time. Forward pass accelerates,
    backward pass leaves room to stop. Returns (v, t).
    Requires n >= 3, ds > 0, and positive caps and acceleration."""
    cap = np.broadcast_to(np.asarray(cap, float), n).copy()
    v = np.zeros(n)
    for i in range(1, n):
        v[i] = min(cap[i], math.sqrt(v[i - 1] ** 2 + 2 * accel * ds))
    v[-1] = 0.0
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * accel * ds))
    avg = (v[1:] + v[:-1]) / 2
    return v, np.concatenate([[0.0], np.cumsum(ds / avg)])


def _straight(p0, yaw: float, length: float, limits: Limits) -> Trajectory:
    n = max(3, int(np.ceil(abs(length) / DS)) + 1)
    ds = abs(length) / (n - 1)
    v, t = _ramp(n, ds, limits.v_max, limits.a_max)
    sign = 1.0 if length >= 0 else -1.0
    along = sign * np.linspace(0, abs(length), n)
    xy = np.asarray(p0, float) + np.outer(along, [math.cos(yaw), math.sin(yaw)])
    return Trajectory(t, xy, np.full(n, yaw), sign * v, np.zeros(n))


def _turn(p, yaw0: float, dyaw: float, limits: Limits) -> Trajectory:
    n = max(3, int(np.ceil(abs(dyaw) / DTH)) + 1)
    ds = abs(dyaw) / (n - 1)
    w, t = _ramp(n, ds, limits.omega_max, limits.alpha_max)
    sign = 1.0 if dyaw >= 0 else -1.0
    yaw = yaw0 + sign * np.linspace(0, abs(dyaw), n)
    return Trajectory(t, np.tile(np.asarray(p, float), (n, 1)), yaw, np.zeros(n), sign * w)


def _curve(path: Path, limits: Limits) -> Trajectory:
    """Slow through bends so omega = v * kappa stays under omega_max, then
    trim the speed cap where d omega / dt still exceeds alpha_max.

    The cap starts at the hard limits; acceleration ramps and trims may make
    the curve slower, but the profile never stops on it.
    Up to ten rounds of 10 % trims. At an acceleration-limited bend entrance,
    d omega / dt = a * kappa, so a residual can remain; the DriveController's
    ramp absorbs it and the next replan corrects.
    """
    s, kappa = path.s, path.kappa
    n, ds = len(s), float(s[1] - s[0])
    cap = np.minimum(limits.v_max, limits.omega_max / np.maximum(np.abs(kappa), 1e-9))
    for _ in range(10):
        v, t = _ramp(n, ds, cap, limits.a_max)
        omega = v * kappa
        alpha = np.abs(np.diff(omega)) / np.diff(t)
        hot = np.flatnonzero(alpha > limits.alpha_max)
        if not len(hot):
            break
        cap[np.unique(np.concatenate([hot, hot + 1]))] *= 0.9
    v, t = _ramp(n, ds, cap, limits.a_max)
    return Trajectory(t, path.xy.copy(), path.yaw.copy(), v, v * kappa)


def _join(parts: list[Trajectory]) -> Trajectory:
    """Concatenate, dropping each part's first sample: it repeats the last of
    the part before, at rest in the same place."""
    t, xy, yaw, v, w = [parts[0].t], [parts[0].xy], [parts[0].yaw], [parts[0].v], [parts[0].omega]
    for p in parts[1:]:
        t.append(p.t[1:] + t[-1][-1])
        xy.append(p.xy[1:])
        yaw.append(p.yaw[1:])
        v.append(p.v[1:])
        w.append(p.omega[1:])
    return Trajectory(np.concatenate(t), np.vstack(xy), np.concatenate(yaw), np.concatenate(v), np.concatenate(w))


def profile(path: Path, limits: Limits | None = None) -> Trajectory:
    """Time-stamp the route: straight out, turn, curve, straight in. The
    robot is at rest between parts and never stops along the curve."""
    if limits is None:
        limits = Limits()
    sx, sy, syaw = path.start
    gyaw = path.goal[2]
    parts, kinds = [], []
    px, py = sx + path.exit * math.cos(syaw), sy + path.exit * math.sin(syaw)
    if path.exit != 0:
        parts.append(_straight((sx, sy), syaw, path.exit, limits))
        kinds.append("exit")
    if path.turn != 0:
        parts.append(_turn((px, py), syaw, path.turn, limits))
        kinds.append("turn")
    if len(path.xy) >= 2:
        parts.append(_curve(path, limits))
        kinds.append("curve")
        px, py = path.xy[-1]
    if path.dock != 0:
        parts.append(_straight((px, py), gyaw, path.dock, limits))
        kinds.append("dock")
    if not parts:
        return Trajectory(np.zeros(1), np.array([[sx, sy]]), np.array([syaw]), np.zeros(1), np.zeros(1),
                          (("hold", 0),))
    traj = _join(parts)
    ends = np.cumsum([len(part.t) - 1 for part in parts])
    traj.phases = tuple((kind, int(end)) for kind, end in zip(kinds, ends))
    return traj


def goal_for(name: str):
    """A table's docking pose (x, y, yaw) by short name: 'ball', 'cubes', 'ware'."""
    for table in TABLES:
        if table.name.split('_')[1] == name:
            return table.dock
    raise KeyError(f"no table called {name!r}; one of " + ', '.join(t.name.split('_')[1] for t in TABLES))
