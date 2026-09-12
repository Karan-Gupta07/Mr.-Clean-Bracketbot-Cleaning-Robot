# Path planning and navigation: design

Date: 2026-09-12
Branch: `path-planning-controls`, based on `feat/slam-cv-camera-inspection`
Status: approved design, not yet implemented

## What this is for

The robot has to drive from wherever it is to a spot in front of a table, so
the arms can reach what is on it. The command comes from the VLA/VLM layer as
either a table name ("cubes") or a pose (x, y, yaw). This design covers
everything between that command and the `(speed, yaw_rate)` command the
existing `DriveController` already accepts:

1. a **path planner** that finds a smooth, curved route on an occupancy grid;
2. a **motion planner** that turns the route into a time-stamped schedule of
   speed and yaw rate the balancing robot can actually follow;
3. a **navigator** that plays the schedule open loop and replans from the
   localised pose when the robot drifts off it;
4. a **ROS node** that feeds the navigator the SLAM pose and publishes its
   output on `/cmd_vel`, so the same code drives the robot in the Docker
   stack that Rababb's SLAM work runs in.

It does not cover SLAM, perception, fine docking off a perceived table edge,
or the arm handoff after arrival. Those are separate work.

## What the SLAM branch already provides

This branch sits on `feat/slam-cv-camera-inspection` (commit 35b104f), which
settles several things the design leans on:

| Provided | Where | What it means for us |
| --- | --- | --- |
| `DriveController` | `src/rlbot/control.py` | Takes `command(speed m/s, yaw_rate rad/s, stamp)`, clamps to 0.15 m/s and 0.3 rad/s, ramps at 0.1 m/s² and 0.3 rad/s², zeros after 0.5 s without a command, and feeds the balancer. The navigator emits `(speed, yaw_rate)` and calls it. No wheel-speed conversion, no slewing in our code. |
| `/cmd_vel` | `ros2_ws/src/rlbot_bridge/rlbot_bridge/simulation.py` | The bridge subscribes to `geometry_msgs/Twist` and routes it into the same `DriveController`. Our ROS node publishes there. |
| Saved map | `ros2 run rlbot_bridge save_map <dir>` | Writes `map.yaml` and `map.pgm` in the `nav2_map_server` format, plus the pose graph. `OccupancyGrid.from_pgm` reads the YAML and PGM. |
| Localised pose | TF `map -> odom -> base_footprint` | SLAM Toolbox owns `map -> odom`, the bridge owns `odom -> base_footprint`. There is no pose topic; the ROS node looks the transform up with tf2. |
| Map frame | The bridge always spawns the robot at the `start` keyframe, world origin, yaw 0. | The map frame coincides with the room's world frame up to SLAM error, so `Table.dock` poses from `room.py` can be used as goals directly. If mapping ever starts elsewhere this assumption breaks; the ROS node takes a `map_offset` parameter for that day. |
| `base_footprint` | The wheel-axle midpoint projected to the ground | Our pose convention everywhere. The sim stub uses the same point, not the `root` body (4 mm apart at a 3° lean, but matched anyway). |

## Decisions and why

| Decision | Choice | Why |
| --- | --- | --- |
| Goal input | A pose. `goal_for(name)` resolves a table name through `Table.dock` in `room.py`. | Covers both what the VLM might hand us. The name lookup is one line on top of the pose planner. |
| Map and pose source | SLAM Toolbox output. Stubbed with the known room and the sim's true pose for the local check. | Decided by the team. The stub has the same shape as the real thing so nothing changes when it is swapped. |
| Stack | numpy in `src/rlbot/`. The only ROS code is one thin node in Rababb's existing package. | Matches the repo (mujoco + numpy, macOS). Nav2 was considered and dropped: with no Nav2 controller in use, running Nav2 only to plan means a second Docker stack for one A* call. |
| Curved paths | A* on the inflated grid, then line-of-sight shortcut, then a clamped cubic spline. | A* alone is blocky. Shortcutting removes the staircase corners; the spline makes what is left smooth and tangent to the robot's heading at the start and the dock yaw at the end. Hybrid A* deferred: more code, only needed if spline smoothing keeps clipping obstacles. |
| Path following | No tracking loop. The motion planner emits a feasible `(speed, yaw_rate)(t)` schedule and the navigator plays it open loop. | The user's call: path planner plus motion planner, execute the schedule, no per-step feedback on cross-track error. |
| Drift correction | Replan from the localised pose every 2 s, or sooner if the robot is more than 10 cm off the schedule. | Open loop on a balancing robot drifts: wheel slip, and every balance recovery writes phantom distance. A slow outer replan corrects it without a tight loop. |
| Speed limits | Match `DriveController`'s defaults: 0.15 m/s, 0.3 rad/s, 0.1 m/s², 0.3 rad/s². | Anything faster gets clipped by its clamps and ramps, and then the schedule's timing is wrong. Raising them is a change to Rababb's controller, made together if the sim check says it is safe. |

## Architecture

```
goal pose ──► planner.plan(grid, start, goal) ──► Path (curved polyline + heading + curvature)
                                                    │
                                          planner.profile(path, limits)
                                                    │
                                                    ▼
                                        Trajectory (t, x, y, yaw, v, omega)
                                                    │
                          navigate.Navigator.step(pose_now, now) ──► (speed, yaw_rate)
                                                    │                        │
                                     replan if off schedule > 10 cm    DriveController.command
                                     or every 2 s                       (sim) or /cmd_vel (ROS)
                                                                              │
                                                                     BalanceController ──► wheels
```

Three new modules in `src/rlbot/`, two scripts, one ROS node.

### `src/rlbot/navmap.py`: the occupancy grid

```python
@dataclass
class OccupancyGrid:
    occupied: np.ndarray          # bool, shape (nx, ny); True where the robot cannot be
    resolution: float             # m per cell
    origin: tuple[float, float]   # world (x, y) of cell (0, 0)

    @classmethod
    def from_room(cls, resolution=0.05) -> "OccupancyGrid":
        """The known room from room.py: walls, pillar, divider, tables.
        Anything reaching into the robot's height band blocks, which is why
        table tops block even though the robot could drive under the legs."""

    @classmethod
    def from_pgm(cls, yaml_path, extra_boxes=()) -> "OccupancyGrid":
        """A map as SLAM Toolbox's save_map writes it: a PGM image plus a
        YAML with resolution, origin, and the occupied / free thresholds.
        Unknown cells count as occupied.  `extra_boxes` are
        (x, y, yaw, half_x, half_y) rectangles to mark occupied on top - the
        table tops the lidar cannot see."""

    def inflate(self, radius: float) -> "OccupancyGrid":
        """Grow every occupied cell by the robot's radius plus a margin, so
        the planner can treat the robot as a point."""

    def free(self, x: float, y: float) -> bool
    def line_free(self, p, q) -> bool
        """True if the straight segment p -> q stays in free cells.
        Sampled at half the cell size, which cannot skip a cell."""

    def cell(self, x, y) -> tuple[int, int]
    def world(self, i, j) -> tuple[float, float]
    def ascii(self, path=None) -> str
        """The map as text, with a path drawn on it if given."""

def robot_footprint() -> tuple[float, float, np.ndarray]
    """(radius, height, half-extents) read off the robot's collision boxes."""

def table_tops() -> list[tuple[float, float, float, float, float]]
    """The three table tops as (x, y, yaw, half_x, half_y), for from_pgm."""
```

`from_room` and `robot_footprint` are the occupancy code that already lives
in `scripts/build_room.py` (`occupancy`, `robot_footprint`, `GRID`,
`MARGIN`), moved into the package. `build_room.py` then imports them, so the
clearance check and the planner agree on what is free. This is the one
existing file the work touches.

### `src/rlbot/planner.py`: path and motion planning

```python
@dataclass
class Limits:
    v_max: float = 0.15        # m/s.       DriveController's max_speed
    omega_max: float = 0.3     # rad/s.     DriveController's max_yaw_rate
    a_max: float = 0.1         # m/s^2.     DriveController's linear_accel
    alpha_max: float = 0.3     # rad/s^2.   DriveController's angular_accel
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
already returns.

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
5. Check `|d omega / dt| <= alpha_max`. Where it is exceeded, scale `v_cap`
   down there by 10 % and repeat from step 3, at most ten times. The
   `DriveController` ramp is the backstop if a residual remains, at the cost
   of the schedule running a little late until the next replan.

### `src/rlbot/navigate.py`: playing the schedule

```python
class Navigator:
    def __init__(self, grid: OccupancyGrid, limits: Limits = Limits(),
                 replan_every: float = 2.0, off_path: float = 0.10,
                 pose_timeout: float = 0.5,
                 arrive_xy: float = 0.10, arrive_yaw: float = math.radians(5))

    def go_to(self, goal, pose_now, now: float) -> Trajectory
        """Plan from where the robot is, start the clock at `now`."""

    def step(self, pose_now, now: float) -> tuple[float, float]
        """(speed m/s, yaw_rate rad/s) to command right now.

        Reads (v, omega) off the schedule at `now`.  Replans from pose_now
        if it is more than `off_path` from where the schedule says the robot
        should be, or if `replan_every` seconds have passed since the last
        plan.  `pose_now` is an (x, y, yaw) tuple, or None when no estimate
        is available; after `pose_timeout` seconds of None it returns (0, 0)
        and holds.  Returns (0, 0) once done."""

    @property
    def done(self) -> bool
        """Within arrive_xy and arrive_yaw of the goal, and the schedule has ended."""

    @property
    def replans(self) -> list[tuple[float, float]]
        """(time, distance off schedule) at each replan, for the check script."""

def true_pose(data) -> tuple[float, float, float]
    """The wheel-axle midpoint and the chassis heading, from the sim.  The
    same point the bridge calls base_footprint.  For the local check only."""
```

The navigator never touches the sim, the controller, or ROS. The caller
passes it a pose and time and hands its output to `DriveController.command`
or to a `Twist` publisher. That keeps it identical in both.

### `ros2_ws/src/rlbot_bridge/rlbot_bridge/navigate.py`: the ROS node

```
ros2 run rlbot_bridge navigate --to cubes
ros2 run rlbot_bridge navigate --to 1.7 -1.1 0.0
```

A `rclpy` node in Rababb's package (a new `navigate` entry in its
`setup.py`). It loads the map from the `map_yaml` parameter with
`OccupancyGrid.from_pgm(map_yaml, extra_boxes=table_tops())`, resolves
`--to` through `goal_for` or as a raw pose, and at 10 Hz:

1. looks up `map -> base_footprint` with tf2, passing None to the navigator
   if the lookup fails;
2. calls `Navigator.step(pose, now)` with `now` from the ROS clock, which is
   sim time because the bridge publishes `/clock`;
3. publishes the result as a `Twist` on `/cmd_vel`;
4. exits 0 when `done`, non-zero with the reason if the navigator gives up.

It runs beside `mapping.launch.py mode:=localization`, in the same
container. It is the only file in this design that imports ROS.

## Handoff with SLAM

| Input | From SLAM | Stub for the local check |
| --- | --- | --- |
| Map | `out/<run>/map.yaml` + `map.pgm` from `save_map`, with `table_tops()` unioned in | `OccupancyGrid.from_room()` |
| Pose (x, y, yaw) in the map frame, 10 Hz | tf2 lookup of `map -> base_footprint` | `true_pose(data)` |

Three things to raise with Rababb now:

- **The lidar cannot see table tops.** Their README says so. Our
  `from_pgm` unions the known table rectangles in as an interim. The real
  fix is on their side: raise the lidar or add depth.
- **The tilt gate can stall driving.** The bridge rejects scans when the
  chassis tilts more than 2°, and zeros `/cmd_vel` after 0.5 s without a
  valid scan. Driving means leaning: 0.15 m/s steady is about 1°, but
  accelerating can cross 2°, at which point the robot is stopped mid-path,
  replans, accelerates, and is stopped again. Our low `a_max` reduces the
  lean. If the sim check shows the gate tripping, the ask is to zero the
  command on localisation loss, not on one tilted scan.
- **Two branches edit the same files.** PR #5 (`gripper-and-sensing`) and
  this branch both change `control.py` (`speed_ref` versus
  `wheel_speed_ref`) and both add a lidar module (`sensing.py` versus
  `sensors.py`). Whoever merges second gets a conflict. Not ours, but the
  team should know before it lands.

## Scripts and checks

Non-trivial logic leaves one runnable check behind. Two scripts, each is the
check for what it exercises.

### `scripts/plan_path.py`

Plans every route between the start pose and the three docks, in both
directions: 9 routes. For each it prints the ASCII map with the curve drawn
on it, the path length, the schedule duration, the minimum turning radius,
and the closest approach to any obstacle. It asserts, and exits non-zero if
any fails:

- a path was found;
- every sample of the curve is at least `margin` from the nearest obstacle;
- the schedule never exceeds `v_max`, `omega_max`, `a_max` or `alpha_max`;
- the curve ends within 1 cm and 1 degree of the goal pose;
- the curve starts tangent to the start heading.

### `scripts/navigate.py`

Drives the robot to a table in the sim through `DriveController`, with the
navigator replanning from `true_pose` as the SLAM stub. Headless by
default; `--view` opens the viewer (`mjpython`). For each of the 9 routes it
asserts:

- the robot arrives within 10 cm and 5 degrees of the dock pose;
- it never falls (pitch stays under 45 degrees);
- the chassis never touches furniture or walls (checked off `data.contact`);
- the chassis never tilts past 2° for longer than 0.5 s, which is the
  bridge's gate. Reported, not asserted, on the first pass: it tells us
  whether the tilt-gate concern above is real.

It reports how many replans each route needed and how far the robot was off
schedule at each one, which is the number that says whether open loop plus
replan is enough or a tracking loop is needed after all.

### Docker

The ROS node is exercised inside the image Rababb's Dockerfile builds, on the
`colima-rlbot` context per their README, against a map saved by their
`check_ros_mapping.py`. That is a manual run, not an automated check, in
this design.

## Error handling

- **Goal or start inside an obstacle, or unreachable.** `plan` raises with
  the ASCII map and which cell is blocked. The navigator does not drive.
- **Spline clips an obstacle.** Retry with more waypoints, then fall back to
  the filleted A* polyline. Never returns a path that fails the check.
- **Curvature tighter than the robot can drive at speed.** The profile slows
  to `v_min` there. A two-wheeler can turn tight; it just has to do it slowly.
- **Pose stops arriving.** After `pose_timeout` the navigator returns zero
  and the balancer holds the robot upright where it is.
- **Replan fails mid-route.** Zero, hold, report. Do not keep playing a
  schedule from a position the robot is no longer at.
- **ROS node cannot look up the transform.** Passes None; the navigator's
  timeout handles it. Logs once, throttled.

## Build order

1. `navmap.py`, lifted from `build_room.py`; `build_room.py` imports it.
2. `planner.py`: `plan`, then `profile`, then `goal_for`.
3. `scripts/plan_path.py`, and get its 9 routes passing.
4. `navigate.py` and `scripts/navigate.py`, and get its 9 routes passing.
5. README: a "Navigation" entry in what works, the two scripts in how to
   run and the folder layout.
6. The ROS node, its `setup.py` entry, and a README paragraph beside
   Rababb's Docker section.

Steps 1 to 5 are one PR, numpy only, runnable on the Mac. Step 6 is a
second PR, tested by hand in Docker.

## Out of scope

- SLAM, the bridge, the Dockerfile. Rababb's; we call into them.
- Fine docking off a perceived table edge. The path ends at `Table.dock`;
  anything tighter than 10 cm / 5 degrees is perception's job later.
- The arm handoff after arrival.
- The VLM itself. Only its entry point: a pose, or a name through `goal_for`.
- A tracking controller. If `scripts/navigate.py` shows open loop plus
  replan cannot reach 10 cm, that is the next design, not this one.
- Raising the speed limits. Done together with Rababb if the sim check says
  the balancer has headroom.

## Amendments during planning

1. **Docks are inside the inflation.** `Table.dock` puts the mast axis 0.24 m
   from the table edge; the planner inflates by 0.26 m. So a route has up to
   four segments: a straight run out of the start along its heading (in
   reverse when the free space is behind), a turn in place if the corridor
   leaves more than 45° off the heading, the curve, and a straight run along
   the goal's axis into the goal. The straight segments are checked against
   the robot's oriented rectangle on the raw grid, the curve against its
   circle on the inflated grid. The robot stops between segments; it does
   not stop anywhere along the curve.
2. **Angular acceleration is reported, not asserted.** `omega = v * kappa`,
   so at the start of a bend `d omega / dt = a * kappa`, which exceeds
   0.3 rad/s² whenever the bend is tighter than 3 m at full acceleration.
   The profile scales the speed cap down at those samples, up to ten times;
   whatever residual is left, the `DriveController` ramp absorbs, and the
   next replan corrects. `plan_path.py` prints the peak.
3. **No `map_offset` parameter on the ROS node.** The bridge always spawns
   at the `start` keyframe, so the map frame is the world frame. The
   parameter can be added the day someone maps from somewhere else.
