# RL-BOT

A two-wheeled robot that cleans a room. It is built and tested in simulation.

The robot is the **BracketBot**: a self-balancing base, a tall mast, and two
7-joint arms with grippers. It runs inside **MuJoCo**. One prompt drives the
whole job: an agent plans, the robot drives to a table, and a task-specific
controller does the manipulation.

To see it without installing anything, watch [`demo/tour.mov`](demo/tour.mov):
the whole job in one continuous simulation, 7 min 48 s. [`demo/`](demo/) also
holds a close-up of ACT, the scripted pick-and-place baseline, and the
fly-brain point-goal pilot.

## Contents

1. [Status](#status)
2. [Quick start](#quick-start)
3. [System overview](#system-overview)
4. [Simulation](#simulation)
5. [Path finding](#path-finding)
6. [Agent](#agent)
7. [ACT](#act)
8. [Fly brain](#fly-brain)
9. [Verification](#verification)
10. [Folder layout](#folder-layout)
11. [Known issues and limits](#known-issues-and-limits)

## Status

| Part | State |
| --- | --- |
| Simulation | Done. The robot loads, stands, balances, and survives a 300 N shove. |
| Path finding | Done in simulation. A* plus curved trajectories. All nine routes arrive within 10 cm and 5 degrees. ROS 2 SLAM mapping and localization pass in Docker. |
| Agent | Done. Claude Fable 5.1 calls three top-level tools. A keyword planner runs the same tools with no API key. |
| Cubes table (Fable skills) | Works. 4 of 4 cubes into the crate. |
| Pick table (fly brain) | Works on the live demo. Imitation-only policy, no PPO. 4 of 10 on its fixed-base test seeds. |
| Ball table (ACT) | Runs, misses. 2 of 14 random layouts on the fixed-base sim. 0 of 10 on the shipped layout. |

Every result above is a simulation result. No hardware has been tested.

## Quick start

You need macOS (Apple Silicon works) and Python 3.10 or newer.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # MuJoCo and NumPy
pip install -r requirements-agent.txt    # Anthropic SDK, for the Fable agent
pip install -r requirements-rl.txt       # torch, gymnasium, SB3, for ACT and the fly brain
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`.

Run the full tour headless. The robot tidies the cubes, drives on, picks the
blue cube, drives to the ball table, and tries the ball. It takes a few
minutes.

```bash
.venv/bin/python scripts/demo.py --planner sweep "clean the cubes, then pick up the blue cube, then go to the ACT table"
```

Watch it. Use `mjpython`, not `python`, for any window on macOS.

```bash
.venv/bin/mjpython scripts/demo.py --view --speed 2 --planner sweep "go to the ACT table"
```

Let Claude decide the calls. This needs an API key.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/python scripts/demo.py "tidy the cubes table, then go to the ACT station"
```

With no prompt, `demo.py` opens a `> ` loop. The robot stays where the last
prompt left it. The exit code is 0 only when every tool call succeeds.

Record a run instead of watching it. `--video` renders over the robot's
shoulder at 30 fps and pipes to `ffmpeg`; it runs headless.

```bash
.venv/bin/python scripts/demo.py --planner sweep --video out/tour.mov "clean the cubes, then pick up the blue cube, then go to the ACT table"
```

## System overview

`scripts/demo.py` runs everything in **one** MuJoCo model. A prompt becomes
tool calls from a top-level agent. The agent has three tools.

```
prompt
  │
  ▼
top-level agent (Claude Fable 5.1, or keyword routing with --planner sweep)
  │
  ├── go_to(station)     A* route, curved trajectory, balancing drive, park
  ├── manipulate()       the controller bolted to the table the robot is parked at
  └── finished(summary)  stop
```

`manipulate` dispatches on the station:

| Station | Objects | Controller | Code |
| --- | --- | --- | --- |
| `cubes` | four colored cubes and a crate | Fable skills agent, nested inside the top-level agent | `scripts/agent.py`, `src/rlbot/skills.py` |
| `pick` | one blue cube and a rectangle | Fly-brain policy | `src/rlbot/live_arm.py`, `src/rlbot/connectome.py` |
| `ball` | a red ball and a box | ACT | `src/rlbot/act.py` |

There is no fallback controller. When a table's controller is unavailable, the
tool result says so and the agent decides what to do next.

On arrival the robot does not jump to a keyframe. A pre-declared weld between
the chassis and the world is switched on (`data.eq_active`). The base holds
still like a parking brake. The solver impedance ratio is raised to 200 for
the pinch. Both are switched back before the next drive. The code is
`src/rlbot/live.py`.

Each table's controller was tuned on different contact pads. The live room
carries one pad per blade and rewrites it on arrival (`LiveSim.use_pads`).

## Simulation

### The robot model

The URDF is a shape export from Onshape, not a physics model.
`scripts/build_mjcf.py` converts it to MuJoCo and fixes what is broken. The
original URDF in `models/bracketbot/` is never edited.

| Fix | Detail |
| --- | --- |
| Wheels | They were welded. They now have hinge joints. |
| Wheel size | The URDF has no wheel radius. The script measures it from the tyre mesh: 0.0846 m. |
| Floor contact | There were no collision shapes. Each wheel now has one. |
| Mass | The file said 0.29 kg. Mass is now computed from mesh volumes and scaled to 12.0 kg. This total is a placeholder. |
| Joint strength | Every joint had a 10 N limit. Limits are now sized from the gravity load. |
| Gripper motor | The second finger had its own motor. It now follows the first through a mimic constraint. |
| Gripper contact | MuJoCo collides a mesh as its convex hull. Hulled, the hooked fingers fill their own gap. Each blade now carries a flat pad traced from its real inner face. |
| Head camera | It pointed at the horizon. It now pitches 62 degrees down at a docked table. |
| Finger ringing | `armature="0.005"` on the blade joints stops the mimic finger from oscillating. |

| Quantity | Value |
| --- | --- |
| Joints | 26 = 6 floating base + 2 wheels + 18 in the arms and grippers |
| Motors | 2 wheel motors (±8 N·m) + 16 arm servos |
| Sensors | gyro, accelerometer, orientation on the `imu` site; wheel angles and speeds; a ray-cast lidar at the `lidar` site; a head camera and one camera per wrist. See [How the sensors are simulated](#how-the-sensors-are-simulated). |
| Gripper | holds objects 40 to 60 mm across. Smooth balls are not held. |
| Balance gains | `kp_pitch=80, kd_pitch=15, kp_speed=0.010` in `src/rlbot/control.py` |

The balancer is a hand-tuned PD controller (`src/rlbot/control.py`). It runs
at the 500 Hz physics rate. Pitch is read independently of heading
(`src/rlbot/robot.py`). Two bugs were fixed here: the yaw feedback had the
wrong sign, and pitch measurement depended on yaw.

### The room

`scripts/build_room.py` writes `models/room.xml` (no robot) and
`models/room_scene.xml` (with the robot). The room is 6.0 x 4.5 m with four
walls, a pillar, a divider, and three tables. `src/rlbot/room.py` is the
single source for what is on each table and where.

| Table | Position | Objects |
| --- | --- | --- |
| `table_ball` | (2.25, -1.10) | red ball, crate |
| `table_cubes` | (-0.20, -1.95) | four cubes, 56 to 58 mm, and a crate |
| `table_pick` | (-2.25, 0.90) | one 48 mm blue cube and a rectangular destination marker |

Cube sizes and positions are measured, not chosen. Below 56 mm the pads reach
the table before the cube. Within 0.12 m of the centreline nothing is
pickable, so that is where the crate sits.

The build checks clearance. The robot's footprint is 42 cm across and 1.61 m
tall. 16.2 of the 27 m² of floor is standable, and each docking pose leaves
14.6 cm of clearance.

### Arm control

`src/rlbot/arm.py` solves inverse kinematics for each 7-joint arm.
`src/rlbot/grasp.py` holds the motions a pick is made of: approach from
above, close until stall, lift. `scripts/validate_ik.py` reaches every object
in the room from cold.

## Path finding

### How the sensors are simulated

The robot has no real sensors yet. Every input below is computed from the
MuJoCo state, at a real sensor's rate, with a real sensor's failure modes.
The simulator's true pose is read in exactly one place, `true_pose()` in
`src/rlbot/sensing.py`, and only to score an estimate.

**Wheel encoders.** `scripts/build_mjcf.py` gives each wheel a hinge joint.
The encoder reading is that joint's angle, `data.qpos[wheel_left]` and
`data.qpos[wheel_right]`, in radians and unwrapped. Two `jointvel` sensors,
`vel_left` and `vel_right`, give wheel speed for the balancer. The URDF has
no wheel radius. The builder measures it from the tyre mesh (0.0846 m), and
odometry reads it back from the compiled wheel geom
(`WheelImuOdometry.from_model`), never from a constant.

**IMU.** The builder puts an `imu` site on the chassis 0.20 m up. Three
MuJoCo sensors attach to it: `gyro` (angular rate), `accel` (linear
acceleration with gravity), and `orient` (a frame quaternion). Odometry
consumes only the gyro. `orient` exists for diagnostics and is never fed to
an estimator.

**Odometry.** `WheelImuOdometry` in `src/rlbot/odometry.py` is dead
reckoning, not a filter. Each step it does three things. It integrates the
gyro to track roll and pitch. It adds the pitch rate to the wheel rotation
to recover ground travel. It blends yaw from the wheel differential and the
gyro (`gyro_weight=0.9`). The estimate drifts the way wheel odometry drifts,
plus one extra way. A balancing robot spins its wheels to catch itself, so
every recovery from a nudge writes phantom distance into the estimate. That
drift is what SLAM exists to correct.

**Lidar.** There is no lidar hardware model. Each beam is a ray cast from the
`lidar` site, 0.32 m up the mast, in the site's own xy plane. The ray is cast
with `mujoco.mj_ray` or `mj_multiRay`, not with MuJoCo `rangefinder`
sensors, for two reasons. A rangefinder skips only its own body, so a beam
from the mast axis would hit the robot's shell at 6 mm. And MuJoCo evaluates
sensors every physics step; 72 beams against 50 meshes is most of the step
time. Casting by hand lets the scan run at 10 Hz and filter by geom group.
Two versions exist:

| Class | Beams | Used by | Self-hits | Out of range |
| --- | --- | --- | --- | --- |
| `sensing.Lidar` | 72 | `scripts/room.py`, local checks | filtered by geom group (room is groups 0-1, robot is 2-3) | `+inf` |
| `sensors.Lidar` | 360 | ROS bridge, `record_slam_inputs.py` | mounting assembly masked; moving links still occlude and return `NaN` | `+inf`; below 0.05 m is `NaN` |

The beams follow the chassis. A robot leaning 3 degrees sweeps a plane
tilted 3 degrees, and real floor hits stay in the scan. `project_scan` in
`src/rlbot/sensors.py` re-projects each scan into a level `lidar_planar`
frame using the gyro-estimated tilt and the mount geometry. It drops the
whole scan when the tilt exceeds 2 degrees. It drops single returns outside
the 0.12 to 0.52 m height band. It never fills a missing ray with free
space. Optional Gaussian range noise is available (`noise=`); the recorded
runs use none.

**Cameras.** The head camera is a MuJoCo camera on the chassis at 1.575 m,
pitched 62 degrees down, 58 degree field of view. Each wrist camera sits at
the grip frame and looks along the approach direction, 70 degrees. Frames
are rendered offscreen with `mujoco.Renderer` at 224 px
(`scripts/render_demos.py`, `src/rlbot/act.py`). Demonstrations do not
record frames; they are re-rendered from the saved poses afterwards. The
lidar and odometry never see an image.

**Rates.** Physics runs at 500 Hz. The balancer reads the gyro and wheel
speeds every step. The ROS bridge is
`ros2_ws/src/rlbot_bridge/rlbot_bridge/simulation.py`. It ticks at 50 Hz.

| Topic | Rate |
| --- | --- |
| `/odom`, `/imu/data`, `/joint_states`, TF, `/clock` | 50 Hz |
| `/scan_raw`, `/scan`, `/scan_valid` | 10 Hz |

`scripts/record_slam_inputs.py` records the same signals to one `.npz`
without ROS. It stores wheel angles, gyro, accelerometer, and odometry at
500 Hz, and scans at 10 Hz. `truth_pose` is stored for scoring only.

```bash
.venv/bin/python scripts/record_slam_inputs.py --output out/slam_inputs.npz
.venv/bin/python scripts/record_slam_inputs.py --push 300 --output out/slam_inputs_push.npz
```

### SLAM

SLAM runs in ROS 2 Jazzy with SLAM Toolbox, inside Docker. The bridge package
is `ros2_ws/src/rlbot_bridge/`. It publishes odometry, IMU, TF, and clock at
50 Hz and scans at 10 Hz. SLAM Toolbox owns scan matching, loop closure,
`/map`, and the `map -> odom` transform. It never sees the room's geometry or
the simulator's true pose.

The bridge projects each scan into a fixed `lidar_planar` frame using the
gyro-estimated tilt. It rejects a scan in three cases:

- the tilt exceeds 2 degrees,
- a return falls outside the 0.12 to 0.52 m height band,
- fewer than half the beams are usable.

Build the image and run the acceptance check. Use Docker Desktop or any Linux
Docker host. The original setup used a `colima-rlbot` context. If you use
that, add `--context colima-rlbot` to each `docker` command.

```bash
docker build -t rlbot:jazzy .
docker run --rm -v "$PWD/out:/artifacts" rlbot:jazzy python scripts/check_ros_mapping.py --output /artifacts/mapping_check_1
```

The check launches real ROS nodes and drives a loop. It saves `map.yaml`,
`map.pgm`, `map.posegraph`, and `map.data`. It restarts in localization mode
and compares the SLAM pose to a separately published reference. The last run
measured 1.4 mm / 0.20 degrees during mapping and 0.1 mm / 0.00 degrees after
the localization restart. `result.json` lands in the output directory.

To drive by hand, start mapping and publish a slow command from a second
terminal:

```bash
docker run --rm -it --name rlbot-mapping -v "$PWD/out:/artifacts" rlbot:jazzy
docker exec -it rlbot-mapping /opt/rlbot/docker/entrypoint.sh ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.1}}'
docker exec rlbot-mapping /opt/rlbot/docker/entrypoint.sh ros2 run rlbot_bridge save_map /artifacts/room_1
```

Restart in localization mode and drive to a table on the SLAM pose:

```bash
docker run --rm -it --name rlbot-localize -v "$PWD/out:/artifacts" rlbot:jazzy ros2 launch rlbot_bridge mapping.launch.py mode:=localization map_file:=/artifacts/room_1/map
docker exec -it rlbot-localize /opt/rlbot/docker/entrypoint.sh ros2 run rlbot_bridge navigate --ros-args -p use_sim_time:=true -p map_yaml:=/artifacts/room_1/map.yaml -p to:=cubes
```

`to` takes a table name (`ball`, `cubes`, `pick`) or `"x y yaw"`.

### Occupancy grid and A*

`src/rlbot/navmap.py` builds the grid. It reads a saved PGM/YAML map or
rasterizes the known room from `models/room.xml`. It unions in the table
tops, because the low lidar scan cannot see them. It then inflates every
obstacle by the robot's footprint.

`src/rlbot/planner.py` searches that grid with 8-connected A*. Goals are the
predefined docking poses in `src/rlbot/room.py`. The raw path is shortened by
line-of-sight, fitted with a clamped cubic spline, and turned into a speed and
yaw-rate schedule. Limits: 0.15 m/s, 0.3 rad/s, 0.1 m/s².

### The navigator

`src/rlbot/navigate.py` drives a route in four phases: exit, turn, curve,
dock. Each phase is played open loop from the schedule. Between phases the
robot stops and waits for a steady pose: 0.5 s of poses within 5 mm and half
a degree. It then replans from wherever it actually is. Drifting more than
15 cm off the schedule also triggers a stop and replan. Within 20 cm of the
goal the phases give way to a guarded turn in place. The route ends after 60
replans or 30 s without a pose.

In the continuous demo the pose comes from the simulator, not SLAM
(`src/rlbot/live.py`). The ROS `navigate` node runs the same navigator on the
localized pose.

Results on the nine routes between the start pose and the three tables
(`scripts/navigate.py`):

| Metric | Result |
| --- | --- |
| Arrival | 9 of 9 within 10 cm and 5 degrees |
| Falls | 0 |
| Furniture contacts | 0 |
| Time past the 2-degree tilt gate | 0.00 s on every route |
| Replans per route | 7 to 24 |

The pick table uses a tighter 4 cm / 2 degree tolerance (`LiveSim.ARRIVE_AT`),
because the fly-brain carry fails from 9 cm out.

### Fly-brain navigation pilot

A separate experiment drives the robot to point goals through the FlyWire
graph instead of A*. See [Fly brain](#fly-brain).

## Agent

### Top level

The top-level agent is Claude Fable 5.1 (`MODEL = "claude-fable-5-1"` in
`scripts/agent.py`). It needs `ANTHROPIC_API_KEY`. It receives three tool
schemas (`TOP_TOOLS` in `scripts/demo.py`): `go_to`, `manipulate`, and
`finished`. `--planner sweep` replaces the model with keyword routing over
the same tools: "act" routes to `ball`, "fly" or "blue" to `pick`, "cubes" to
`cubes`. `--budget` caps tool calls per prompt. `--effort` sets the model's
reasoning effort.

### The cubes skills agent

At the cubes table, `manipulate` starts a nested Fable agent
(`scripts/agent.py`). It gets six skills, defined as JSON tool schemas and
implemented in `src/rlbot/skills.py`.

| Skill | What it does |
| --- | --- |
| `look` | Returns the scene as text: where each object is, what each hand holds. No coordinates, no camera, no vision model. |
| `pick(object, arm?)` | Approaches from above, closes, lifts. `arm` is a hint; the code chooses the hand. Retries wrist angles and both hands internally. |
| `place(into?)` | Puts the held object into the crate by default. |
| `home` | Folds the arms to rest. |
| `give_up(object, why)` | Declares an object impossible. This is a first-class action. |
| `finished(summary)` | Ends the episode. |

Eight distinct tool names exist in total: three top-level and six nested,
with `finished` shared.

Four rules come from measured failures:

- Object names are an enum. An unknown name is refused at the tool boundary.
- The code picks the arm, not the model. Published bimanual planners that let
  the model assign arms score near zero.
- Retries live inside `pick`. Models re-sequence well; they do not invent new
  motion strategies.
- The harness owns the loop rules: a per-object failure cap, a repeated-call
  detector, and a step budget.

`--planner sweep` on this level is a fixed policy: pick each object, place
it, home. It tests the skills without a model.

```bash
.venv/bin/python scripts/agent.py --table cubes --planner sweep
.venv/bin/python scripts/agent.py --table cubes --planner fable          # needs ANTHROPIC_API_KEY
.venv/bin/python scripts/agent.py --table cubes --planner sweep --video out/cubes.mp4
```

`agent.py` welds the base at the docking pose and never drives. The idle arm
folds out of the way during a pick with the other hand.

### Recognition and orchestration

`src/rlbot/orchestration.py` is a deterministic selector. It accepts
timestamped recognitions (`colored_cubes`, `red_ball_box`,
`blue_cube_rectangle`), rejects stale or ambiguous inputs, and dispatches
one registered tool. The recognitions are supplied by the operator. There is
no camera detector and no vision-language model in this repository.

```bash
.venv/bin/python scripts/orchestrate.py --station cubes --recognized colored_cubes
.venv/bin/python scripts/orchestrate.py --station cubes --recognized colored_cubes --execute --planner sweep
```

`scripts/live_demo.py` is the older two-window demo: it drives in one
simulation, then opens a separate fixed-base simulation for the arms.
`scripts/demo.py` supersedes it.

## ACT

`src/rlbot/act.py` is our implementation of ACT (Zhao et al. 2023), sized
for a laptop. It is trained on the red-ball table and runs at the `ball`
station.

### Architecture

| Part | Value |
| --- | --- |
| Vision backbone | one ResNet-18, ImageNet-pretrained, shared across cameras, last two layers removed |
| Cameras | 3: `head_cam`, `wrist_right_cam`, `wrist_left_cam`, at 128 px |
| State and action | 16 numbers: both arms' 7 joints plus one gripper blade each |
| Transformer | 4 encoder + 4 decoder layers, hidden 256, 8 heads, feed-forward 1024 |
| CVAE latent | 32 |
| Chunk | 32 commands = 1.6 s at 20 Hz |
| Parameters | about 22 M |

Training loss is L1 on the chunk plus a KL term at weight 10. The optimizer
is AdamW at 1e-4, 1e-5 for the backbone, batch 8. On the MPS backend a step
takes about 0.2 s on an M5.

### Demonstrations

`scripts/teleop.py` puts one arm under the keyboard. The operator commands
where the jaws go, not joints. IK does the rest.

```
W / S   jaws forward / back      SPACE   close / open        1-4   which cube
A / D   jaws left / right        X       swap arms           H     reset the scene
R / F   jaws up / down           ENTER   start / stop        P     print poses
Q / E   wrist turn               C       cancel recording
```

Each episode is scored: is the object in the crate and out of the hand. It is
written to `out/demos/<table>/` as one `.npz` at 20 Hz. Camera frames are
not recorded; `scripts/render_demos.py` renders them afterwards from the
poses.

`scripts/collect_demos.py` drives the same controller from code. Every
episode moves both the object and the crate. Only successes count. The
scripted collector crates the ball about 7 times in 10.

```bash
.venv/bin/mjpython scripts/teleop.py --cube m
.venv/bin/python scripts/collect_demos.py --table ball --episodes 200
.venv/bin/python scripts/render_demos.py out/demos/ball --preview
```

### Training and results

Two runs on the 80-episode ball set, 72 training and 8 held out:

| Checkpoint | Augmentation | Steps | Best val L1 | Into the crate |
| --- | --- | --- | --- | --- |
| `checkpoints/act_ball_run1_noaug.pt` | none | 20 K | 0.0167 | 2 of 14 |
| `checkpoints/act_ball_run2_aug.pt` | random 6 px shift | 17 K | 0.0153 | 1 of 6 |

Rollouts use layouts the policy never saw. Both checkpoints reach the ball
and close on it about four times in ten. Most then lose the ball on the
carry. Flat pads have nothing to bite on a sphere, and the scripted collector
drops a third of its carries too.

```bash
.venv/bin/pip install -r requirements-train.txt
.venv/bin/python scripts/train_act.py --data out/demos/ball --out out/act/ball_aug --shift 6
.venv/bin/mjpython scripts/rollout_act.py --ckpt checkpoints/act_ball_run2_aug.pt --episodes 3 --view --mode open-loop
.venv/bin/python scripts/run_act.py --describe
```

Two lessons:

- Rollouts must start where the demonstrations start, after the collector's
  ready move. From the rest pose the policy swung the arm through the ball.
- The policy closes about half a second early. More data and augmentation
  reduce it.

In the continuous demo, `manipulate` at the ball table runs one closed-loop
episode from the ready pose and returns to it afterwards. The shipped ball
position is one this checkpoint misses.

## Fly brain

The fly-brain controllers route signals through a graph built from measured
fruit-fly neuron connections. This is a connectivity experiment, not a brain
simulation. Activities are artificial rate-like values, not spikes.

### The graph

`scripts/prepare_connectome.py` downloads four public FlyWire FAFB v783
tables (neurons, classification, coordinates, connections; about 58 MB) and
pins their SHA-256 checksums. The source has 139,255 neurons and 2,700,513
directed edges.

The pilot keeps 512 neurons. Selection is by total synapse strength with
interface quotas: one eighth afferent (inputs), one eighth descending
(outputs), the rest central and visual-projection neurons. Edge weights are
synapse counts with a transmitter sign: GABA and glutamate negative, others
positive. Each row is normalized by its absolute incoming sum. The result is
`checkpoints/graph_512.npz` (512 neurons, 8,688 edges, 64 inputs, 64
outputs) with its manifest in `checkpoints/graph_512.json`.

```bash
.venv/bin/python scripts/prepare_connectome.py              # 512-neuron pilot graph
.venv/bin/python scripts/prepare_connectome.py --neurons 0  # the full graph, untrained
.venv/bin/python scripts/benchmark_connectome.py            # full-graph forward pass: ~1.2 s, too slow for 50 ms control
```

### The controller

`ConnectomeFeatures` in `src/rlbot/connectome.py` is a Stable-Baselines3
feature extractor.

1. A linear encoder maps the observation to the 64 input neurons, 4 channels
   each.
2. Four rounds of sparse message passing run over the 512 x 512 matrix. Each
   round: `tanh(0.35 * h + gain * mix(A @ h) + injection + bias)`.
3. The 64 output neurons' activity, 256 values, feeds the SB3 MLP head
   (`pi=[128, 128]`), which produces the action.

The edges are fixed. The encoder, per-neuron gains and biases, channel
mixing, and the MLP head are trained. No hidden state persists between robot
steps.

### Navigation pilot

`src/rlbot/navigation.py` is a Gymnasium env: 12 observation values, 2
actions (speed and yaw-rate requests at 20 Hz), and the PD balancer
underneath. Training (`scripts/train_connectome.py`): 40 scripted teacher
episodes, 800 imitation updates, then 8,192 PPO transitions with
demonstration rehearsal.

| Policy | Held-out goals (seeds 2000–2019) | Falls |
| --- | --- | --- |
| Fly graph, imitation only | 20 of 20 | 0 |
| Fly graph, PPO + rehearsal | 17 of 20 | 0 |
| MLP, imitation only | 18 of 20 | 0 |
| MLP, PPO + rehearsal | 20 of 20 | 0 |

The graph works as a controller. This run does not show a benefit from fly
wiring or from PPO over imitation. Raw numbers are in `docs/results/`. The
full recipe and the recording are in [docs/brain_demo.md](docs/brain_demo.md).

### Arm policy (the pick table)

`src/rlbot/arm_env.py` is the pick-and-place env. The robot base is fixed.
The action is 4 continuous values at 20 Hz: table-relative XYZ motion and
jaw opening. There is no phase controller at inference.

Training (`scripts/train_arm.py`) is teacher-student:

1. A scripted teacher (`Teacher` in `scripts/train_arm.py`) runs seven
   phases: over the cube, down, close, lift, over the goal, down, release.
   It exists only to make training data.
2. `--episodes 20` attempts; the successes (17 on the shipped run) become the
   dataset.
3. Behaviour cloning: 4,000 updates of MSE on the teacher's actions.
4. Optional PPO with demonstration rehearsal (`RehearsalPPO`). `--steps 0`
   skips it.

Single-frame observations failed (0 of 10). The teacher's private wait
counter gives near-identical inputs opposite labels. `--history 16` feeds
16 causal observations (368 values). Two post-training calibrations follow:
a 1.25x gain on the jaw output and a 0.05 deadband on Cartesian motion.

```bash
.venv/bin/python scripts/train_arm.py --history 16 --episodes 20 --updates 4000 --bc-lr 0.0003 --steps 0 --output out/rl/arm_history_padded
.venv/bin/python scripts/train_arm.py --history 16 --motion-deadband 0.05 --resume out/rl/arm_history_padded/imitation.zip --calibrate-jaw 1.25 --output out/rl/arm_padded_calibrated
.venv/bin/python scripts/run_arm.py --history 16 --motion-deadband 0.05 --checkpoint out/rl/arm_padded_calibrated/policy.zip --episodes 20 --seed 5000 --output out/rl/arm_padded_calibrated/validation
```

The shipped checkpoint is `checkpoints/flybrain_arm_padded_calibrated.zip`.
It is imitation only, zero PPO steps, with both calibrations. On its own
fixed-base test seeds it scored 4 of 10. A sibling candidate trained the same
way scored 18 of 20 on fresh held-out seeds; the two held-out sets differ and
are not a paired comparison. In the continuous demo it picked and placed the
blue cube. It runs there **without** the 20-of-20 validation gate that
`orchestrate.py` enforces, and the demo says so.

Checkpoints carry the gripper, station, control version, horizon, and
physics hashes. A mismatch is refused, not silently run.
`tests/test_checkpoints.py` verifies the shipped weights and graph match.

The older parallel-jaw experiment (20 of 20, with PPO) is historical. It used
a different gripper and is not evidence for the current policy. See
[docs/arm_rl.md](docs/arm_rl.md) and [docs/original_arm.md](docs/original_arm.md).

## Verification

The tests use `unittest`. `pytest` is not installed.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

120 tests run and pass. One is a declared expected failure: the bare
gripper, without pads, cannot grasp.

Standalone checks, all headless:

```bash
.venv/bin/python scripts/evaluate.py --push 300     # balance: 20 s, max lean 3 deg, drift 2 cm
.venv/bin/python scripts/plan_path.py               # 9 routes planned within limits
.venv/bin/python scripts/navigate.py                # 9 routes driven; arrival, falls, contacts
.venv/bin/python scripts/check_navigation.py        # 20 unit checks on grid, planner, navigator
.venv/bin/python scripts/validate_ik.py             # IK round-trip, then every object reached
.venv/bin/python scripts/build_room.py              # rebuild the room, print the clearance map
.venv/bin/python scripts/check_grasp.py             # 3 of 6 lifted with the default pads; see Known issues
.venv/bin/python scripts/check_slam_inputs.py       # lidar, odometry, projection, recording: all OK
.venv/bin/python scripts/check_arm_clearance.py     # arm-to-chassis clearance; exits 1 today, see Known issues
```

Live viewers (`mjpython`):

```bash
.venv/bin/mjpython scripts/balance.py                          # watch it balance
.venv/bin/mjpython scripts/room.py                             # W/S/A/D to drive, 1/2/3 to park at a table
.venv/bin/mjpython scripts/navigate.py --route cubes-pick --view
```

## Folder layout

```
models/bracketbot/          The original URDF and 50 meshes. Never edited.
models/bracketbot.xml       The robot in MuJoCo format. Made by build_mjcf.py.
models/room.xml             The room alone. models/room_scene.xml adds the robot.
models/balancer.xml         A toy two-wheeler for quick controller checks.

checkpoints/                ACT weights and configs, the fly-brain arm policy, graph_512.npz.
demo/                       Recordings: tour.mov, ACT.mov, pick_place.gif, fly_brain_point_goal_pilot.gif.
docs/                       arm_rl.md, brain_demo.md, original_arm.md, pick_place.md, results/*.json.

scripts/demo.py             One prompt, one simulation. The main entry point.
scripts/agent.py            The cubes skills agent, Fable or sweep.
scripts/orchestrate.py      Deterministic recognition-to-tool dispatch.
scripts/live_demo.py        Older two-window demo.
scripts/build_mjcf.py       URDF to MuJoCo, with the fixes.
scripts/build_room.py       Writes the room, checks clearance and reach.
scripts/evaluate.py         Headless balance test, optional shove.
scripts/balance.py          Balance viewer.  scripts/room.py: drive around.
scripts/plan_path.py        Plans and draws the nine routes.
scripts/navigate.py         Drives the nine routes in the sim.
scripts/check_navigation.py Unit checks for grid, planner, navigator.
scripts/validate_ik.py      IK checks.
scripts/check_grasp.py      Grasp probe on every object.
scripts/check_arm_clearance.py  Arm-to-chassis clearance. Broken; see Known issues.
scripts/check_slam_inputs.py    Local lidar/odometry checks. Broken; see Known issues.
scripts/record_slam_inputs.py   Records wheel, IMU, odometry and scans to an .npz.
scripts/check_ros_mapping.py    The ROS 2 SLAM acceptance gate; runs inside Docker.
scripts/teleop.py           Keyboard teleop, records demonstrations.
scripts/collect_demos.py    Scripted demonstrations.
scripts/render_demos.py     Renders cameras for recorded demonstrations.
scripts/audit_demos.py      Cuts bad demonstrations, writes a manifest.
scripts/train_act.py        Trains ACT.  scripts/rollout_act.py: runs it.
scripts/run_act.py          Describes or checks the shipped ACT checkpoint.
scripts/replay_demo.py      Plays recorded episodes back.
scripts/prepare_connectome.py   Downloads FlyWire tables, builds the graph.
scripts/train_connectome.py     Fly-brain navigation pilot.
scripts/evaluate_navigation.py  Scores navigation checkpoints.
scripts/record_brain_demo.py    Records the point-goal pilot to a GIF.
scripts/benchmark_connectome.py Full-graph forward-pass timing.
scripts/train_arm.py        Fly-brain arm policy: teacher, BC, PPO, calibration.
scripts/run_arm.py          Runs an arm checkpoint; viewer, record, validation.
scripts/pick_place.py       Scripted fixed-base pick-and-place baseline.
scripts/view_cameras.py     Shows the head and wrist cameras.

src/rlbot/robot.py          Load the robot, read its state, step the sim.
src/rlbot/control.py        PD balance and drive controllers.
src/rlbot/room.py           What is on each table and where.
src/rlbot/sensing.py        Lidar and wheel odometry.
src/rlbot/sensors.py        Lidar used by the ROS bridge and the recorder.
src/rlbot/odometry.py       Wheel and gyro odometry integration.
src/rlbot/navmap.py         Occupancy grids, inflation, footprint checks.
src/rlbot/planner.py        A*, splines, speed schedules.
src/rlbot/navigate.py       The phase-at-a-time navigator.
src/rlbot/navigation.py     Gymnasium env for the fly-brain navigation pilot.
src/rlbot/live.py           The one-model continuous simulation: drive, park, pads.
src/rlbot/live_arm.py       Runs the fly-brain arm policy on the live sim.
src/rlbot/arm.py            Arm IK.
src/rlbot/grasp.py          Grasp motions, the station keeper, the Rig.
src/rlbot/gripper_pads.py   Fitted pads for the fly-brain gripper.
src/rlbot/skills.py         The six skills the cubes agent calls.
src/rlbot/orchestration.py  Recognition validation and dispatch.
src/rlbot/act.py            ACT model, data loader, checkpoints.
src/rlbot/arm_env.py        Fly-brain arm env.
src/rlbot/connectome.py     ConnectomeFeatures.
src/rlbot/teleop.py         Jog controller and demonstration recorder.
src/rlbot/filming.py        Records a run to mp4.
src/rlbot/hybrid_ik.py      ctypes binding for the sponsors' IK library (Linux arm64).

ros2_ws/src/rlbot_bridge/   ROS 2 Jazzy package: simulation bridge, save_map, navigate.
docker/, Dockerfile         The rlbot:jazzy image.
```

## Known issues and limits

Scripts that look worse than they are:

- `scripts/check_arm_clearance.py` exits 1. The swing up to `cube_l` passes
  10 mm from the mast cover, on its 10 mm warn threshold. Every other
  waypoint clears by 15 mm or more. It also reports that about 30% of random
  poses inside the joint limits penetrate the chassis: the planner keeps the
  arm out, the workspace itself does not.
- `scripts/check_grasp.py` exits 1 at 3 of 6. It uses one default pad set.
  The ball is unpickable by design. `cube_l` and `pick_cube` fail here but
  succeed in the live demo, which swaps pads per table.

Limits:

- The robot's mass and motor limits are guesses. Weigh the real robot, then
  rerun `build_mjcf.py --total-mass` and retune the gains.
- The arms have no damping or friction.
- Navigation in the demo reads the simulator's pose, not SLAM.
- The physical lidar mount and hardware calibration are unvalidated.
- Low scans miss tabletop overhangs. The grid unions in known table tops.
- Nav2 was considered and dropped. Nothing here uses a Nav2 controller.
- The fly-brain arm policy has no 20-of-20 validation report. The demo runs
  it anyway and says so.
- ACT misses the shipped ball layout. The failure is the carry, not the
  reach.
- Generated checkpoints and replays under `out/` are ignored by git. Only
  `checkpoints/` and `demo/` ship.
- `rlbot:jazzy` copies the repo at build time. Rebuild the image after
  changing `models/`, `src/`, or `ros2_ws/`.

Next steps:

- Run all nine routes through the ROS `navigate` node on the SLAM pose
  automatically.
- Sense tabletop overhangs and place them in the map frame.
- Dock on perceived table edges instead of known poses.
- Replace the operator-supplied recognitions with a real detector.
- Get ACT to hold the ball through the carry. A better end effector is the
  likely answer.
- Fine-tune the fly-brain arm policy with PPO without hiding a scripted
  fallback.
