# Path planning and navigation: design

Date: 2026-09-12
Branch: `path-planning-controls`
Status: approved design, not yet implemented

## What this is for

The robot has to drive from wherever it is to a spot in front of a table, so
the arms can reach what is on it. The command comes from the VLA/VLM layer as
either a table name ("cubes") or a pose (x, y, yaw). This design covers
everything between that command and the wheel-speed and yaw-rate references
the balance controller already accepts:

1. a **path planner** that finds a smooth, curved route on an occupancy grid;
2. a **motion planner** that turns the route into a time-stamped schedule of
   speed and yaw rate the balancing robot can actually follow;
3. a **navigator** that plays the schedule open loop and replans from the
   localised pose when the robot drifts off it.

It does not cover SLAM, perception, fine docking off a perceived table edge,
or the arm handoff after arrival. Those are separate work.

## Decisions and why

| Decision | Choice | Why |
| --- | --- | --- |
| Goal input | A pose. `goal_for(name)` resolves a table name through `Table.dock` in `room.py`. | Covers both what the VLM might hand us. The name lookup is one line on top of the pose planner. |
| Map and pose source | SLAM Toolbox output (Rababb's work). Stubbed with the known room and the sim's true pose until it lands. | Decided by the team. The stub has the same shape as the real thing so nothing changes when it is swapped. |
| Stack | numpy in this repo. No ROS imports in our code. | Matches the repo (mujoco + numpy, macOS). Nav2 was considered and dropped: with no Nav2 controller in use, running Nav2 only to plan means waiting on the ROS bridge, the map server and Docker before a single path exists. |
| Curved paths | A* on the inflated grid, then line-of-sight shortcut, then a clamped cubic spline. | A* alone is blocky. Shortcutting removes the staircase corners; the spline makes what is left smooth and tangent to the robot's heading at the start and the dock yaw at the end. Hybrid A* was considered and deferred: more code, only needed if spline smoothing keeps clipping obstacles. |
| Path following | No tracking loop. The motion planner emits a feasible (v, omega)(t) schedule and the navigator plays it open loop into the balance controller. | The user's call: path planner plus motion planner, execute the schedule, no per-step feedback on cross-track error. |
| Drift correction | Replan from the localised pose every 2 s, or sooner if the robot is more than 10 cm off the schedule. | Open loop on a balancing robot drifts: wheel slip, and every balance recovery writes phantom distance. A slow outer replan corrects it without a tight loop. |
| Branch base | `main`. Wait for PR #5 to merge. | The user's call. Everything that touches the controller (`speed_ref`, the yaw sign fix) waits; the planner and motion planner are pure geometry and can be built now. |

## Architecture

```
goal pose ──► planner.plan(grid, start, goal) ──► Path (curved polyline + heading + curvature)
                                                    │
                                          planner.profile(path, limits)
                                                    │
                                                    ▼
                                        Trajectory (t, x, y, yaw, v, omega)
                                                    │
                          navigate.Navigator.step(pose_now, dt) ──► (speed_ref, yaw_rate_ref)
                                                    │                        │
                                     replan if off schedule > 10 cm     BalanceController
                                     or every 2 s                             │
                                                                          wheel torques
```

Three new modules in `src/rlbot/`, two new scripts.

### `src/rlbot/navmap.py`: the occupancy grid

```python
@dataclass
class OccupancyGrid:
    occupied: np.ndarray      # bool, shape (nx, ny); True where the robot cannot be
    resolution: float         # m per cell
    origin: tuple[float, float]   # world (x, y) of cell (0, 0)

    @classmethod
    def from_room(cls, resolution=0.05) -> "OccupancyGrid":
        """The known room from room.py: walls, pillar, divider, tables.
        Anything reaching into the robot's height band blocks, which is why
        table tops block even though the robot could drive under the legs."""

    @classmethod
    def from_pgm(cls, yaml_path, extra_boxes=()) -> "OccupancyGrid":
        """A map as SLAM Toolbox's map_saver writes it: a PGM image plus a
        YAML with resolution, origin, and the occupied / free thresholds.
        `extra_boxes` are (x, y, yaw, half_x, half_y) rectangles to mark
        occupied on top of it - the table tops the lidar cannot see."""

    def inflate(self, radius: float) -> "OccupancyGrid":
        """Grow every occupied cell by the robot's radius plus a margin, so
        the planner can treat the robot as a point."""

    def free(self, x: float, y: float) -> bool
    def line_free(self, p, q) -> bool
        """True if the straight segment p -> q stays in free cells.
        Sampled at half the cell size, which cannot skip a cell."""

    def cell(self, x, y) -> tuple[int, int]
    def world(self, i, j) -> tuple[float, float]
```

`from_room` is the occupancy code that already lives in `scripts/build_room.py`
(`occupancy`, `robot_footprint`, `GRID`, `MARGIN`), moved into the package.
`build_room.py` then imports it, so the clearance check and the planner agree
on what is free. This is the one existing file the work touches.

### `src/rlbot/planner.py`: path and motion planning

```python
@dataclass
class Limits:
    v_max: float = 0.25        # m/s.  3 rad/s of wheel speed, the drive script's DRIVE
    omega_max: float = 0.6     # rad/s.  The drive script's TURN; 1.2 tips it over
    a_max: float = 0.17        # m/s^2.  The drive script's SLEW of 2 rad/s^2 x wheel radius
    v_min: float = 0.05        # m/s floor on tight curves, so it never parks mid-arc
    radius: float = 0.21       # m, half the 0.42 m driving footprint (navmap.robot_footprint)
    margin: float = 0.05       # m of daylight on top of the radius

@dataclass
class Path:
    xy: np.ndarray             # (n, 2) dense points along the curve
    yaw: np.ndarray            # (n,) heading, tangent to the curve
    kappa: np.ndarray          # (n,) signed curvature
    s: np.ndarray              # (n,) arc length from the start

    @property
    def length(self) -> float
    @property
    def min_radius(self) -> float

@dataclass
class Trajectory:
    t: np.ndarray              # (n,) seconds from the start
    xy: np.ndarray
    yaw: np.ndarray
    v: np.ndarray              # m/s along the path
    omega: np.ndarray          # rad/s, v * kappa

    @property
    def duration(self) -> float
    def at(self, t: float) -> tuple[np.ndarray, float, float, float]
        """(xy, yaw, v, omega) at time t, linearly interpolated, clamped to the ends."""

def plan(grid: OccupancyGrid, start, goal, limits: Limits) -> Path
def profile(path: Path, limits: Limits) -> Trajectory
def goal_for(name: str):
    """A table's docking pose by short name: 'ball', 'cubes', 'ware'."""
```

Poses are plain `(x, y, yaw)` tuples throughout. That is what `Table.dock`
already returns, and what `true_pose(data).as_array()` gives once PR #5 is
in. The planner does not import `sensing.Pose`, so it builds on `main` today.

`plan` does four things in order:

1. **A\*** on `grid.inflate(limits.radius + limits.margin)`, 8-connected,
   Euclidean heuristic, from the start cell to the goal cell. Fails with a
   clear message if either cell is occupied or no route exists.
2. **Shortcut.** Walk the A* cell path and drop every waypoint that
   `line_free` can see past. This removes the staircase corners.
3. **Spline.** A clamped cubic spline through the remaining waypoints,
   parametrised by chord length, with the end tangents fixed: the robot's
   heading at the start, the goal yaw at the end. So the robot arcs out of its
   current heading and arrives already pointing at the table, with no turn in
   place at either end.
4. **Check.** Sample the spline at half the cell size and test every sample
   against the inflated grid. If any sample is occupied, put back the A*
   waypoint nearest that sample and go again from step 3. If that still fails
   with every A* waypoint kept, fall back to the A* polyline itself with a
   small fillet at each corner. This always terminates, because each retry
   keeps strictly more waypoints.

`profile` turns the path into a schedule:

1. Resample the path at a fixed arc-length step (2 cm).
2. Cap the speed at each sample: `v_cap = min(v_max, omega_max / |kappa|)`,
   floored at `v_min`. Tight curves are driven slowly so the yaw rate never
   exceeds what the balancer survives.
3. Forward pass from `v = 0` at the start and backward pass to `v = 0` at the
   goal, each limited by `a_max`. The result is a trapezoid where the path
   is straight and a slower hump around each bend.
4. Integrate `dt = ds / v` to get timestamps. `omega = v * kappa`.

### `src/rlbot/navigate.py`: playing the schedule

```python
class Navigator:
    def __init__(self, grid: OccupancyGrid, limits: Limits = Limits(),
                 replan_every: float = 2.0, off_path: float = 0.10,
                 pose_timeout: float = 0.5)

    def go_to(self, goal, pose_now) -> Trajectory
        """Plan from where the robot is, start the clock."""

    def step(self, pose_now, dt: float) -> tuple[float, float]
        """(speed_ref in wheel rad/s, yaw_rate_ref in rad/s) for this step.

        Advances the clock, reads (v, omega) off the schedule, converts v to
        wheel rad/s through WHEEL_RADIUS.  Replans from pose_now if it is
        more than `off_path` from where the schedule says the robot should
        be, or if `replan_every` seconds have passed.  `pose_now` is an
        (x, y, yaw) tuple, or None when no estimate is available; after
        `pose_timeout` seconds of None it returns (0, 0) and holds."""

    @property
    def done(self) -> bool
        """Within 10 cm and 5 degrees of the goal, and the schedule has ended."""
```

The navigator never touches the sim or the controller. Whoever runs the loop
passes it a pose and hands its output to `BalanceController`. That keeps it
identical whether the pose comes from `true_pose(data)` today or from SLAM
later.

## Handoff with SLAM

Our code takes two things. Both are stubbed until Rababb's work lands.

| Input | From SLAM | Stub now |
| --- | --- | --- |
| Map | PGM + YAML as `map_saver` writes them | `OccupancyGrid.from_room()` |
| Pose (x, y, yaw) in the map frame, at 1 Hz or better | SLAM Toolbox localisation | `true_pose(data)` from PR #5's `sensing.py` |

Two things to raise with Rababb now, before their design hardens:

- **Where the pose crosses over.** At demo time the pose lives in ROS inside
  Docker and the navigator is numpy. Either their bridge publishes the pose
  to us, or the navigator runs as a small `rclpy` node inside their
  container. Not ours to decide alone; decide when their bridge exists.
- **The lidar cannot see table tops.** It sits at 0.32 m. A lidar-built map
  shows four legs per table and open floor between them; the robot is 1.7 m
  tall and the tops are at 0.70 m. Our inflated grid would happily route it
  into a table. The fix is on their side (raise the lidar, or add depth), or
  we union the known table rectangles from `room.py` into the SLAM grid as an
  interim. `from_pgm` takes an optional list of extra rectangles for that.

## Scripts and checks

Non-trivial logic leaves one runnable check behind. Two scripts, each is the
check for what it exercises.

### `scripts/plan_path.py` (buildable now, no PR #5 needed)

Plans every route between the start pose and the three docks, in both
directions: 9 routes. For each it prints the ASCII map with the curve drawn
on it, the path length, the schedule duration, the minimum turning radius,
and the closest approach to any obstacle. It asserts, and exits non-zero if
any fails:

- a path was found;
- every sample of the curve is at least `margin` from the nearest obstacle;
- the schedule never exceeds `v_max`, `omega_max`, or `a_max`;
- the curve ends within 1 cm and 1 degree of the goal pose;
- the curve starts tangent to the start heading.

### `scripts/navigate.py` (after PR #5 merges)

Drives the robot to a table in the sim with the balance controller running,
the navigator replanning from `true_pose` as the SLAM stub. Headless by
default; `--view` opens the viewer (`mjpython`). For each of the 9 routes it
asserts:

- the robot arrives within 10 cm and 5 degrees of the dock pose;
- it never falls (pitch stays under 45 degrees);
- the chassis never touches furniture or walls (checked off `data.contact`).

It reports how many replans each route needed and how far the robot was off
schedule at each one, which is the number that says whether open loop plus
replan is enough or a tracking loop is needed after all.

## Error handling

- **Goal or start inside an obstacle, or unreachable.** `plan` raises with
  the ASCII map and which cell is blocked. The navigator does not drive.
- **Spline clips an obstacle.** Retry with more waypoints, then fall back to
  the filleted A* polyline. Never returns a path that fails the check.
- **Curvature tighter than the robot can drive at speed.** The profile slows
  to `v_min` there. A two-wheeler can turn tight; it just has to do it slowly.
- **Pose stops arriving.** After `pose_timeout` the navigator returns zero
  references and the balancer holds the robot upright where it is.
- **Replan fails mid-route.** Zero references, hold, report. Do not keep
  playing a schedule from a position the robot is no longer at.

## Build order

1. `navmap.py`, lifted from `build_room.py`; `build_room.py` imports it.
2. `planner.py`: `plan`, then `profile`, then `goal_for`.
3. `scripts/plan_path.py`, and get its 9 routes passing.
4. Wait for PR #5.
5. `navigate.py` and `scripts/navigate.py`.
6. README: a "Navigation" entry in what works, the two scripts in how to run
   and the folder layout.

Steps 1 to 3 are one PR. Steps 5 and 6 are a second.

## Out of scope

- Docker and ROS 2 setup. That is Rababb's stack; our code does not import it.
- Fine docking off a perceived table edge. The path ends at `Table.dock`;
  anything tighter than 10 cm / 5 degrees is perception's job later.
- The arm handoff after arrival.
- The VLM itself. Only its entry point: a pose, or a name through `goal_for`.
- A tracking controller. If `scripts/navigate.py` shows open loop plus
  replan cannot reach 10 cm, that is the next design, not this one.
