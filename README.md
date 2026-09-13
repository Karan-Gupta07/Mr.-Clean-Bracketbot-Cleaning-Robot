# RL-BOT

A robot that cleans up a room, built and tested in simulation first.

The robot is the **BracketBot**. It is a two-wheeled robot that balances on its own. It has a tall mast and two arms with grippers. We run it inside **MuJoCo**, a physics simulator, so we can test ideas fast and safely before touching real hardware.

## The goal

The project has three steps. Each step builds on the one before it.

1. **Put the BracketBot in the sim.** Load the robot model, make it stand up, and make it balance. This is the base for everything else.
2. **Drive around a room with SLAM.** SLAM stands for "Simultaneous Localization and Mapping". The robot builds a map of the room while it figures out where it is on that map. Then it can drive from one spot to another without bumping into things.
3. **Pick up and put down objects with task-specific controllers.** The current demo routes colored cubes to Fable, the red ball to ACT (Action Chunking with Transformers, awaiting integration), and the blue-cube/rectangle task to Flybrain. A deterministic selector replaces the proposed VLM orchestrator.

Put together: the robot maps the room, drives to an object, picks it up, drives to where it belongs, and puts it down.

## Where we are now

| Step | Status |
| --- | --- |
| 1. BracketBot in sim | Done. The robot loads, stands, and balances. It recovers from a shove. |
| 2. SLAM navigation | ROS 2 Jazzy / SLAM Toolbox mapping, map saving and localization restart pass in Ubuntu Docker. Custom curved navigation passes all nine sim routes on the true pose, and a ROS node drives the same navigator on the SLAM pose in Docker (two routes hand-tested). Nav2 is not implemented. |
| 3. Manipulation | Scripted cube transfers work with padded original grippers. The latest original-gripper Flybrain checkpoint succeeds on 0/20 held-out starts and is blocked from dispatch; ACT integration awaits its controller and checkpoint. |


### What works today

- **The BracketBot model.** It was converted from a URDF file into MuJoCo format by `scripts/build_mjcf.py`. The wheels spin, the robot can stand on the floor, and the mass numbers are fixed. See "How the robot model was fixed" below.
- **Balancing.** A hand-tuned PD controller keeps the robot upright for as long as you like. It survives a 300 N shove.
- **A room to work in.** A 6 x 4.5 m room with four walls, a pillar and a divider to map, and three tables: a red ball and crate, four colored cubes and crate, and a blue cube with a rectangular destination. It is written twice - `models/room.xml` is the environment on its own, with no robot in it at all, and `models/room_scene.xml` is the same room with the robot added.
- **A room the robot actually fits in.** By default, `scripts/build_room.py` regenerates the room files, then checks clearance and arm reach. The clearance check measures the robot's own footprint - 42 cm across, 1.61 m tall, read off its collision boxes. 16.2 of the 27 m2 of floor is standable, all of it reachable from the middle, and each table's docking pose leaves 14.6 cm of daylight. It prints the map and exits with a nonzero status if the checks fail; the room files have already been written.
- **Driving to a table.** A* searches a grid inflated for the robot's footprint, then the robot follows checked cubic curves with bounded motion profiles. All nine nominal routes from the start to each table and between table docks arrived within 10 cm and 5 degrees, without falling or touching furniture. The local check uses the known-room grid and the sim's true pose. In Docker, `ros2 run rlbot_bridge navigate` runs the same navigator on a saved SLAM map and the localised pose; two routes have been hand-tested that way.

  The four phases - exit, turn, curve, dock - are played open loop from a schedule worked out in advance, not steered step by step. Between phases the robot stops, waits for a steady pose, and replans from wherever it actually ended up; a stale pose, or drifting more than 15 cm off the schedule, stops it the same way. The design asked for 10 cm there; the extra 5 cm is simulation tuning, because a balancing robot writes phantom distance into its pose every time it catches itself. Waiting for a steady pose means holding zero for a braking dwell and then seeing 0.5 s of poses within 5 mm and half a degree of each other before it replans. If that window never arrives the wait re-arms after 8 s and tries again; only 60 replans on one route, or 30 s with no pose at all, ends it. Within about 20 cm of the goal - twice the arrival tolerance - the four phases give way to a guarded turn in place, each intermediate angle checked as a padded rectangle, so the last stretch is not a detour back out to an exact docking anchor and in again. The schedules cap speed at 0.15 m/s, yaw rate at 0.3 rad/s and linear acceleration at 0.1 m/s², with angular acceleration on curves best effort and the DriveController's ramp as a backstop. No controller limits were raised to make any of this work. These are nominal simulation results only: two routes finished with little of their time budget to spare, and noisy poses and hardware have not been tested.

  The check also watches the bridge's 2-degree scan-tilt gate, and it never tripped: the chassis spent 0.00 s past 2 degrees on all nine routes, so driving did not starve SLAM of scans in sim. That is only one of the three things the bridge asks of a scan, though. It also throws one away when the gyro's roll/pitch rate passes 0.2 rad/s, or when fewer than half the rays come back, and the local check measures neither - so a clean sweep here does not fully predict the ROS behaviour.
- **Arm control.** Inverse kinematics (IK) moves each 7-joint arm to a target pose. The arms and mast now have collision shapes, so they cannot pass through each other.
- **A grasp test.** `scripts/check_grasp.py` tries to pick up every object in the room. It approaches from above, closes the fingers, lifts, and checks the object came along. The supplied blades need contact pads to hold anything; `--bare` runs them without and lifts nothing.
- **Driving it around.** `scripts/room.py` opens a window with the robot in the room and lets you drive it with the keyboard while the balancer runs. The lidar scan is drawn live.
- **The sensors SLAM needs.** A 72-beam planar lidar and wheel odometry, in `src/rlbot/sensing.py`. There is also a head camera and a camera on each wrist.
- **A small toy balancer.** `models/balancer.xml` is a simple two-wheeled robot. It loads in a second and is a quick way to catch controller bugs without loading the full robot.

## Deterministic demo routing

The demo uses explicit recognition rules, not a VLM orchestrator:

| Recognized task | Station | Controller |
| --- | --- | --- |
| Colored cubes and a box | `cubes` | Fable tool-calling agent |
| Red ball and a box | `ball` | **ACT**, pending its code/checkpoint |
| Blue cube and a rectangle | `pick` | Flybrain learned arm policy |

The `pick` station replaces crockery, not the red ball. Source URDF and mesh
assets remain unchanged. See [original-gripper scope and training](docs/original_arm.md).

`src/rlbot/orchestration.py` accepts timestamped recognitions, rejects stale,
ambiguous or mismatched inputs, and invokes only the selected registered tool.
`classify_scene` distinguishes supplied shape/color/container observations.
These are recognition inputs, **not a camera detector**. The CLI labels its
input as user-provided; it does not pretend to recognize camera images.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-agent.txt
.venv\Scripts\python.exe scripts/orchestrate.py --station cubes --recognized colored_cubes
.venv\Scripts\python.exe scripts/orchestrate.py --station cubes --recognized colored_cubes --execute --planner sweep
.venv\Scripts\python.exe scripts/orchestrate.py --spot -2.25 0.90 --recognized blue_cube_rectangle --execute
```

Without `--execute`, this only reports the chosen tool. Execution first checks
controller availability, runs navigation, and proceeds only after successful
arrival. Manipulation uses a **separate fixed-base simulation** at the docking
pose; this is not yet a continuous balancing-to-manipulation handoff. Navigation
uses simulator truth here, not ROS localization. `--planner sweep` explicitly
tests Fable's skills without an API model; default `fable` needs a locally set
`ANTHROPIC_API_KEY`. ACT refuses execution until its real implementation is
registered; it never falls back to VLA or to a script. Flybrain refuses dispatch
without compatible model metadata and a passing, checkpoint-bound validation.

## Setup

### Fly-connectivity controller and demo

A separate RL prototype now drives the BracketBot toward point goals using a
512-neuron graph built from measured FlyWire connections. It includes imitation
and PPO training, an MLP comparison, and a synchronized robot/neuron replay.
The trained graph reached 17 of 20 separate evaluation goals with no falls;
the MLP reached 20 of 20. This is a simulator-state pilot, with PD balance, not
SLAM or a biological brain simulation. The full graph is prepared but untrained.

See [setup, demo commands, and measured results](docs/brain_demo.md). On the
development machine, open `out/demo/index.html` for the generated interactive
demo. Generated data/checkpoints stay in ignored `out/` and must be regenerated
on a fresh checkout. The linked guide includes the tested Windows commands.

### Pick-and-place simulation variant

The arm now has a contact-based cube pick-and-place demo with a **fixed base**.
It passes 20 tested starts across four cube sizes on the supplied gripper with
contact pads, and 25 on the older parallel-jaw replacement. This baseline uses
scripted IK and servos.
See [commands, results, and technical details](docs/pick_place.md).

```powershell
.venv\Scripts\python.exe scripts/pick_place.py --record --open
```

Models now use the **supplied hooked gripper with a contact pad on each blade**.
The CAD blades are untouched and are still what you see; the pads are what the
simulator collides, because MuJoCo collides each blade as its convex hull - 3.2x
the mesh's real volume - which fills the hook in solid and lifts nothing. With
pads the scripted baseline passes 20/20 across four cube sizes; with bare blades
it passes 0/20. `--gripper urdf` runs the bare blades and `--gripper parallel`
the older sliding-jaw replacement.

A separate [continuous arm RL experiment](docs/arm_rl.md) trains a new FlyWire
graph policy with demonstrations and PPO. Its four outputs select XYZ motion and
gripper opening; inference has no scripted phase controller. It uses the fixed
base and parallel jaws with simulated pad friction increased to 3.0.

```powershell
.venv\Scripts\python.exe scripts/train_arm.py --gripper parallel --steps 0
.venv\Scripts\python.exe scripts/train_arm.py --gripper parallel --output out/rl/arm_release --resume out/rl/arm_friction3/imitation.zip --demonstrations out/rl/arm_friction3/demonstrations.npz --correct-release --updates 750
.venv\Scripts\python.exe scripts/run_arm.py --gripper parallel --record --open
```

The last command writes a standalone replay to `out/arm_rl_demo/index.html`: the
robot views beside an orbitable cloud of the 512 mapped neurons, a circuit view
that lays their cell types out by synaptic distance from the policy's inputs, and
an Explore panel for searching any FlyWire root ID, reading its transmitter
prediction and connections, and following its activity through the episode.

The saved arm policy passed 20/20 tested starts. On a paired set of ten starts,
the pre-PPO checkpoint passed 2/10 and PPO plus demonstration rehearsal passed
10/10. These are small position variations of one cube with a fixed destination
offset; see [raw results](docs/results/arm_rl.json). These are historical
parallel-jaw results, not evidence for the current original-gripper policy.
Reproduce that environment at commit `2db0bea`; current runners deliberately
reject its unstamped checkpoints. Use `docs/original_arm.md` for current training.

### Base simulation setup

You need macOS (Apple Silicon is fine) and Python 3.10 or newer.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

```bash
# Headless balance test. Does the robot stay up? Does it drift?
.venv/bin/python scripts/evaluate.py

# Same test, but shove the robot with 300 N halfway through.
.venv/bin/python scripts/evaluate.py --push 300

# Live viewer. Watch the robot balance in real time.
.venv/bin/mjpython scripts/balance.py

# Drive the robot around the room. W/S/A/D to drive, 1/2/3 to park at a table.
.venv/bin/mjpython scripts/room.py

# Try to pick up every object in the room.
.venv/bin/python scripts/check_grasp.py

# Same, but with the robot balancing on its wheels instead of bolted to the floor.
.venv/bin/python scripts/check_grasp.py --balance

# Rebuild the robot model from the URDF, or rebuild the room. The room build also
# prints the clearance map and checks the arms can reach every object.
# All outputs are committed, so this is optional.
.venv/bin/python scripts/build_mjcf.py
.venv/bin/python scripts/build_room.py

# Check the arm IK: recover random reachable poses cold, then reach every object in the room.
.venv/bin/python scripts/validate_ik.py

# Check how close the arm gets to the robot's own mast and base along the grasp paths.
.venv/bin/python scripts/check_arm_clearance.py

# Plan the nine routes between the start pose and the three tables, draw them, check them.
.venv/bin/python scripts/plan_path.py

# Drive those routes in the sim on the balancer and check the robot arrives.
.venv/bin/python scripts/navigate.py
.venv/bin/mjpython scripts/navigate.py --route cubes-ware --view

# Unit checks for the grid, planner and navigator.
.venv/bin/python scripts/check_navigation.py

# Drive one arm by hand and record pick-and-place demonstrations for the VLA.
.venv/bin/mjpython scripts/teleop.py --cube m

# Put the camera frames back onto recorded demonstrations, and preview them.
.venv/bin/python scripts/render_demos.py out/demos/cubes --preview

# Run the sponsors' arm IK library (Linux arm64 only, so inside a container on a Mac).
docker run --rm --platform linux/arm64 -v "$PWD":/w -w /w python:3.12-slim \
    python3 src/rlbot/hybrid_ik.py /path/to/libhybrid_ik_lib.so models/bracketbot/chopped_urdf_v2.urdf right_eef
```

Use `mjpython`, not `python`, for anything that opens a window. On macOS the window must be made on the main thread, and `mjpython` takes care of that. Plain `python` will fail with `RuntimeError: Caught an unknown exception!`.

## Local SLAM inputs (no ROS required)

The local recorder and checks run independently of ROS. They produce sensor/odometry data, not maps. No additional packages or graphics context are needed for these commands. The ROS mapping entry point is described below.

```bash
.venv/bin/python scripts/check_slam_inputs.py
.venv/bin/python scripts/record_slam_inputs.py --output out/slam_inputs.npz
.venv/bin/python scripts/record_slam_inputs.py --push 300 --output out/slam_inputs_push.npz
```

The recorder balances the robot in the room, collecting wheel/IMU samples and odometry at the 500 Hz physics rate, and instantaneous 360-ray LiDAR scans at 10 Hz. Use `--seconds`, `--scan-hz`, `--beams`, and `--range-max` to adjust capture. It runs without a viewer, holds samples in memory until saving, and refuses to overwrite existing output.

The NumPy archive can be opened with `np.load(path, allow_pickle=False)`. It contains:

- `time`, `wheel_angles` (left/right, unwrapped radians), `imu_gyro` (rad/s), and `imu_accel` (m/s²).
- `odom_pose` (x/y/yaw), `odom_twist` (forward speed/yaw rate), and `odom_roll_pitch`. The pose tracks the wheel-axle midpoint projected onto the ground, relative to its initial odom frame. Body axes are +x forward, +y left, +z up; distances are metres and angles radians.
- `scan_time`, `scan_angles`, and `scan_ranges`, plus range limits, rate, frame names, wheel calibration, and static LiDAR/IMU mounting transforms relative to the model's `root` body. Scan times are synchronized to physics samples; all beams in one scan share a timestamp.
- `truth_pose`, the simulator's world-frame axle projection, strictly for evaluation. Replaying the estimator requires only timestamps, wheel angles, gyro readings, and calibration, not this reference.

**Limits that matter:**

- The existing LiDAR site is inside the mast. Ray casting masks only the robot's rigid mounting assembly, without changing the asset or physics. Moving robot links still occlude: their returns and too-close measurements are `NaN`, not free space. Out-of-range/no-return readings are `+inf`. `Lidar(..., mask_mount=False)` exposes mounting occlusion for diagnostics. The physical mount still needs validation.
- Raw rays follow chassis roll/pitch and retain real floor hits. The ROS bridge separately projects real returns into a fixed `lidar_planar` frame using gyro-estimated tilt and mounting geometry. It rejects tilt above 2 degrees, returns outside a 0.12-0.52 m height band, and scans with fewer than half the beams usable. It does not fill missing rays. This conservative projection passes local geometry checks and is exercised by the Ubuntu/Docker SLAM integration test; wider poses and hardware still need validation.
- Odometry adds gyro pitch rate to relative wheel rotation and blends wheel/gyro yaw increments. It assumes an upright start unless initial tilt is supplied, does not consume simulator orientation, and does not integrate accelerometer readings. Wheel calibration, gyro bias, signs, and yaw blend are configurable; drift, wheel slip, and gyro tilt drift remain. This is not a covariance-estimating filter.
- Local checks cover analytic scans/odometry, projection, timestamps, recording replay, simulated balancing, forward/reverse commands, a command watchdog, and a driven loop. Turning required fixing the yaw feedback sign and making both balance state readers measure pitch independently of heading. The separate ROS integration check below passes inside the Linux Docker runtime on this Mac.

## ROS 2 Jazzy mapping in Docker

This follows the original stack: Ubuntu 24.04, ROS 2 Jazzy, and SLAM Toolbox. The colcon package lives in `ros2_ws/src/rlbot_bridge/`. SLAM Toolbox owns scan matching, loop closure, `/map`, and `map -> odom`; it does not receive the room's known geometry or simulator truth. The Dockerfile builds the package and installs the matching Python dependencies. The cameras and robot assets are unchanged.

On this Mac, Docker runs in the dedicated `colima-rlbot` context. Build the image from the repository root:

```bash
colima start rlbot
DOCKER_CONTEXT=colima-rlbot docker-buildx build --load -t rlbot:jazzy .
```

On a Linux Docker host or Docker Desktop, use `docker build -t rlbot:jazzy .` and omit `--context colima-rlbot` from the commands below. Do not copy a macOS virtual environment into Linux; the image creates its own Python 3.12 environment.

Start mapping:

```bash
docker --context colima-rlbot run --rm -it --name rlbot-mapping -v "$PWD/out:/artifacts" rlbot:jazzy
```

From another terminal, publish a slow forward command. Stop the publisher with Ctrl+C; the 0.5-second command watchdog ramps the robot back to zero-speed balance. Use `'{angular: {z: 0.2}}'` instead to turn. These are manual mapping commands, not autonomous obstacle avoidance.

```bash
docker --context colima-rlbot exec -it rlbot-mapping /opt/rlbot/docker/entrypoint.sh ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.1}}'
```

After stopping travel, save the occupancy map and serialized SLAM state. Choose a new output directory each time; existing data is not overwritten:

```bash
docker --context colima-rlbot exec rlbot-mapping /opt/rlbot/docker/entrypoint.sh ros2 run rlbot_bridge save_map /artifacts/room_1
```

This writes `out/room_1/map.yaml`, `map.pgm`, `map.posegraph`, and `map.data`. Stop the mapping container, then restart in localization mode using the saved map stem (no extension):

```bash
docker --context colima-rlbot run --rm -it --name rlbot-localize -v "$PWD/out:/artifacts" rlbot:jazzy ros2 launch rlbot_bridge mapping.launch.py mode:=localization map_file:=/artifacts/room_1/map
```

The initial map-pose guess defaults to x/y/yaw = 0. Launch arguments `x:=... y:=... yaw:=...` change that localization guess, not the simulator spawn; the bridge currently starts at the room's `start` keyframe. `mode:=mapping` with `map_file:=...` resumes mapping instead of localization.

With the stack in localization mode, drive to a table on the SLAM pose. The `navigate` node reads the saved map, unions in the known table tops the lidar cannot see, looks up `map -> base_footprint`, and publishes `/cmd_vel` until it arrives within 10 cm and 5 degrees:

```bash
docker --context colima-rlbot exec -it rlbot-localize /opt/rlbot/docker/entrypoint.sh ros2 run rlbot_bridge navigate --ros-args -p use_sim_time:=true -p map_yaml:=/artifacts/room_1/map.yaml -p to:=cubes
```

`to` takes a table name (`ball`, `cubes`, `ware`) or `"x y yaw"`. It is the same navigator `scripts/navigate.py` runs locally on the sim's true pose.

Run the real integration check in a fresh output directory:

```bash
docker --context colima-rlbot run --rm -v "$PWD/out:/artifacts" rlbot:jazzy python scripts/check_ros_mapping.py --output /artifacts/mapping_check_1
```

It launches actual ROS nodes in a separate ROS domain, checks a nonempty SLAM map, drives a loop, saves all four map files, restarts the stack in localization mode, and checks pose error against a separately published simulator reference. Logs, maps and a machine-readable `result.json` remain in `out/mapping_check_1/`. This test, not a successful image build or the local sensor checks, is the end-to-end mapping acceptance gate.

The bridge integrates wheel/gyro odometry and controls physics at 500 Hz; ROS odometry, IMU, joint states, TF and clock are published at 50 Hz, and scans at 10 Hz. `/scan_raw` contains the original measurements; `/scan` contains tilt-gated projected measurements in `lidar_planar`; `/scan_valid` reports acceptance. Invalid projected bins are sent below `range_min`, not as free space. The TF chain is `map -> odom -> base_footprint -> base_link -> imu/lidar`, with a fixed `base_footprint -> lidar_planar` projection frame. Odometry covariance defaults are configurable conservative placeholders, not a calibrated filter. Ground-truth publication is off unless explicitly enabled for testing.

The physical LiDAR mount and hardware calibration remain unvalidated. Low scans miss tabletop overhangs, so this does not make the robot ready for autonomous navigation. No Nav2 planner or object CV has been added. To release the VM resources when finished, run `colima stop rlbot`.

## Folder layout

```
models/bracketbot/          The original URDF and its 50 mesh files. Never edited by hand.
models/bracketbot.xml       The robot in MuJoCo format. Made by build_mjcf.py.
models/bracketbot_scene.xml Floor, lights, and start poses. Use this to load just the robot.
models/room.xml             The room on its own: walls, pillar, divider, tables, objects. No robot.
models/room_scene.xml       The same room with the robot in it. Both made by build_room.py.
models/balancer.xml         The small toy balancer.

scripts/build_mjcf.py       Turns the URDF into MuJoCo format and fixes what is broken.
scripts/build_room.py       Writes the room and checks the arm can reach every object.
scripts/evaluate.py         Headless balance test with an optional shove.
scripts/balance.py          Live viewer.
scripts/check_grasp.py      Tries to pick up each object in the room.
scripts/validate_ik.py      Proves the arm IK: round-trip on random poses, then every object in the room.
scripts/check_arm_clearance.py  Measures arm-to-chassis clearance along the grasp paths (the sim filters self-collision).
scripts/plan_path.py        Plans, draws and checks the nine routes without running physics.
scripts/navigate.py         Drives the nine routes in the sim and checks arrival, falls and furniture contacts.
scripts/check_navigation.py Unit checks for the grid, planner and navigator.

scripts/room.py             Live viewer you can drive the robot around the room in.
scripts/agent.py            Tidy a table, driven by Claude Fable 5.1 or a fixed policy.
scripts/teleop.py           Keyboard teleop of one arm, recording demonstrations for the VLA.
scripts/render_demos.py     Renders the cameras for recorded demonstrations, after the fact.

src/rlbot/robot.py          Load the robot, read its state, step the sim.
src/rlbot/control.py        The PD balance controller.
src/rlbot/navmap.py         Known-room and saved PGM/YAML grids, obstacle overlays, inflation and footprint checks.
src/rlbot/planner.py        A* routes, cubic curves, guarded turns and bounded speed/yaw-rate schedules.
src/rlbot/navigate.py       Runs open-loop phases, holds zero to settle and replans from the actual pose.
src/rlbot/arm.py            Arm inverse kinematics and the grasp sequence.
src/rlbot/room.py           What is in the room and where. Shared by the builder and the grasp test.
src/rlbot/hybrid_ik.py      ctypes binding for the sponsors' libhybrid_ik_lib.so, set up the way their daemon uses it.
src/rlbot/sensing.py        The lidar and the wheel odometry.
src/rlbot/grasp.py          The motions a pick is made of, shared by the harness and the agent.
src/rlbot/skills.py         The robot as an agent sees it: typed skills, symbolic scene.
src/rlbot/filming.py        Records a run to an mp4.
src/rlbot/teleop.py         The jog controller and the demonstration recorder behind teleop.py.
```

## How the robot model was fixed

The URDF we got is a shape export from Onshape, not a physics model. Six things had to be fixed before MuJoCo could simulate it. All fixes live in `scripts/build_mjcf.py`, so the original URDF stays untouched and every fix can be reviewed in one place.

1. **The wheels could not spin.** They were welded on. The build script gives them real hinge joints.
2. **Nothing could touch the floor.** There were no collision shapes at all. Each wheel now has one.
3. **The mass numbers were wrong.** The whole robot weighed 0.29 kg in the file. Mass is now computed from the mesh volumes and scaled to a total.
4. **One frame high up the tree was rotated.** Anything that moves a part has to account for it, or the part lands sideways.
5. **Every joint had the same fake strength limit.** 10 N cannot hold the arm carriage up, so the arms slid down the rail. Limits are now sized from the real gravity load.
6. **The second gripper finger was getting its own motor.** It should only follow the first finger. The extra motor was removed.
7. **The fingers could not hold anything.** MuJoCo treats a mesh as its convex hull, and each finger is a hooked claw with a hollow inside. Hulled, the two of them fill the gap solid: a 55 mm cube placed dead centre between fingers 139 mm apart was already touching both of them, and closing shot it across the room. Each blade now gets a flat pad fitted to its real inner face instead, traced off the mesh slice by slice.

## Numbers

| | |
| --- | --- |
| Joints | 26 = 6 for the floating base + 2 wheels + 18 in the arms and grippers |
| Motors | 2 wheel motors (±8 N·m) + 16 arm servos |
| Sensors | Gyro, accelerometer, and orientation on the `imu` site. Wheel speeds. A 72-beam lidar and a head camera, cast and rendered from the `lidar` site and `head_cam`. |
| Gripper | Holds objects 40 to 60 mm across. See "What the gripper can actually hold". |
| Mass | 12.0 kg, center of mass 0.63 m up. **This is a placeholder.** |
| Balance gains | `kp_pitch=80, kd_pitch=15, kp_speed=0.010` |

From a 3° lean, the robot settles in about 2 seconds and stays up. It recovers from a 300 N shove with 4.6° of lean, and from 450 N with 13.9°. A 600 N shove knocks it over.

## What the gripper can actually hold

The fingertips part by 195 mm, which is not the same thing as being able to hold
something 195 mm wide. The fingers are hooks on pivots, not jaws on rails, and
they swing as they open: past about half travel the two gripping faces turn
outward and stop facing each other, so there is nothing to squeeze between. Open
them wide and the object is not really between anything.

Closing the hand on test blocks of different sizes gives the real answer:

| Object width | Result |
| --- | --- |
| Under 40 mm | The pads reach the table before they reach the object. |
| 40 to 60 mm | Held, reliably. |
| Over 60 mm | The faces are splayed too far apart to grip. |
| Any smooth ball | Not held. Flat rigid pads have nothing to bite on and it squirts out. |

Everything on the three tables is sized to that 40-60 mm window, which is why the
crockery is a small bowl and a mug rather than a dinner plate: a plate is 26 mm
tall, and the pads hang 22 mm below the middle of the jaw, so closing on a plate
means closing on the table.

The ball is the exception, left in deliberately. It is the object the current
hand cannot pick up, and it is worth keeping as the thing a better end effector -
or a policy that learns to trap it against something - has to beat.

## Driving it with an agent

`scripts/agent.py` parks the robot at one table and lets a planner tidy it by
calling robot skills. The base is welded at the docking pose, so the wheels never
turn - this is manipulation only, and navigation is a separate problem.

```bash
# No API key needed. A fixed policy - pick each object, put it in the crate -
# driving exactly the same skills. Start here.
.venv/bin/python scripts/agent.py --table cubes --planner sweep

# The same job, decided move by move by Claude Fable 5.1.
export ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/python scripts/agent.py --table ware --planner fable

# Record an mp4 of either.
.venv/bin/python scripts/agent.py --table ware --planner sweep --video out/ware.mp4
```

### The skill API

Six tools, and the shape of them follows what the published work on LLM-driven
manipulation actually found, rather than what is intuitive:

| Tool | Notes |
| --- | --- |
| `look` | The whole scene, symbolically: where each object is, what each hand holds. No coordinates. |
| `pick(object, arm?)` | `arm` is a hint. The robot chooses the hand. |
| `place(into?)` | Into the crate by default. |
| `home` | Arms back at rest. |
| `give_up(object, why)` | Declaring something impossible is a first-class action. |
| `finished(summary)` | Done. |

Four decisions worth calling out, each of which came from a measured failure:

- **Object names are an enum.** An unknown name is refused at the tool boundary
  with the list of real ones. Confidently asking for an object that is not there
  is the largest hallucination class in embodied agents, and corrective feedback
  does not reliably fix it.
- **The code picks the arm, not the model.** Published bimanual planners that let
  the LLM assign arms score near zero where the same model feeding a
  deterministic assigner scores near the ceiling. Here the choice comes from
  which hand is free and which can actually plan the reach.
- **Retries live inside `pick`.** It works through wrist angles and both hands
  itself. Models re-sequence and re-target well; they do not invent new
  low-level motion strategies.
- **The harness owns the loop rules**, not the prompt: a per-object failure cap,
  a repeated-call detector, and a step budget.

### What it gets today

Sweeping all three tables with no model in the loop:

| Table | Picked | Into the crate |
| --- | --- | --- |
| ball | 0 of 1 | 0 |
| cubes | 4 of 4 | 1 |
| tableware | 2 of 2 | 1 |

Six of the seven objects can be picked up. Most of them are then lost on the way
to the crate: the grasp survives a straight lift and about half the carries. The
ball cannot be picked up at all - flat rigid pads have nothing to bite on a
sphere. This is a gripper problem, not an agent problem, and the `sweep` planner
exists precisely so the two can be told apart.

## Collecting demonstrations

The VLA needs examples of the task being done. `scripts/teleop.py` parks the
robot at the cubes table, base welded, and puts one arm under the keyboard:

```
W / S      jaws forward / back (toward the table)    SPACE   close / open the hand
A / D      jaws left / right                         1-4     which cube the task is about
R / F      jaws up / down                            X       swap arms
Q / E      wrist counter-clockwise / clockwise       H       home: reset the scene
[ / ]      finer / coarser steps                     ENTER   start / stop recording
C          cancel the recording                      P       print where things are
```

The operator commands where the *jaws* go, not joints. Each press moves the
target 1 cm or 5 degrees, the hand always pointing down, and IK continues from
the servos' current command. A press that would need the arm to swing into a
different configuration is refused. The command chases the target at 0.15 m/s,
or 0.05 m/s with something in the hand, so holding a key down gives a smooth
move at that speed rather than a burst of steps - which matters, because a cube
in this hand is held by two pads and friction and a stepped carry shakes it
out. Closing the hand is the same stall-detected squeeze the grasp harness
uses. While the hand is open and at rest, the controller measures how far the
servos sag below their command (2 to 5 mm, and a cube leaves 6 mm either side
of the pads) and trims the command to cancel it.

Press ENTER, do the task, press ENTER again. The episode is scored - is the
cube inside the crate and out of the hand - and written to `out/demos/cubes/`
as one `.npz`: at 20 Hz, the full `qpos`, `qvel` and `ctrl`, both arms' joints
and gripper commands, the jaw pose, the jaw target and wrist yaw, whether the
hand is closed and what it holds, and every object's pose; plus the task text,
the key presses, and the success flag. Camera frames are not recorded. They are
a function of `qpos`, so `scripts/render_demos.py` renders them afterwards from
the head and both wrist cameras at whatever size the model wants.

`--jitter 0.02` scatters the cubes by up to 2 cm on each reset, for variety.
`--balance` runs the same thing on the wheels with the station keeper.

Driven by a script rather than a hand, the same controller picks and crates
every cube on the table with either arm, at key-repeat rate and at tap rate.
The two outer cubes sit at the edge of the wrist's range: the last centimetre
across to them gets refused at a wrist yaw of 90 degrees, and turning the wrist
gets it back.

## Plan for the next steps

**Step 2, SLAM navigation**

- Run all nine routes through the ROS node on the SLAM pose automatically, the way `scripts/check_ros_mapping.py` runs mapping; only two routes have been hand-tested in Docker so far.
- Account for tabletop overhangs that the low LiDAR scan cannot see. The loader already accepts extra obstacle boxes; sensing those overhangs and placing them in the map frame still need work.
- Validate the physical LiDAR mount and the 2-degree scan-tilt gate during autonomous routes, including pose loss and noisy localization, so driving does not starve SLAM of scans.
- Use perceived table edges for fine docking within the arms' reach, rather than relying on the known room's docking poses.

- Hook up a SLAM library to the lidar and odometry that are already there, so the robot can build a map of the room and know where it is.
- Add a path planner so the robot can drive to a target spot while it keeps its balance.
- Add a "dock at a table" move so the robot ends up in a good spot for the arms to reach.

**Step 3, VLA pick and place**

- Get more than 5 of 10 objects to lift with the scripted grasp. The four near misses (a 42 mm cube, a 54 mm cube, the bowl and the mug) all get picked up and then slip during the lift.
- Use the wrist and head cameras that are already on the robot.
- Collect demo data in the sim: the robot picks up an object and puts it somewhere.
- Fine-tune a VLA model on that data so it can follow text commands like "pick up the cup".
- Join it all together: map the room, drive to the object, pick it up, drive to the drop spot, and put it down.

## Things to know

- **The robot's weight and motor limits are guesses.** The URDF does not say what the robot weighs. We will need to weigh the real robot and check the motor specs before trusting any force or torque numbers from the sim. Then re-run `build_mjcf.py --total-mass` and re-tune the gains.
- **The arms have no damping or friction.** The URDF does not give any, and none was made up.
- **Turning was tipping the robot over.** The yaw part of the balance controller had its error the wrong way round, which is positive feedback. Standing still it looked fine, because the error was near zero; ask for a turn above about 0.4 rad/s and it put the robot on the floor. Fixed, but it is a reminder that the controller has only ever been tested standing still - it will need real work before it can drive to a table on its own.
- **The grasp test bolts the robot to the floor by default.** That way a failed grasp is the grasp's fault, not the balancer's. Use `--balance` to run it the honest way, on the wheels.
- **Nav2 was considered and dropped, not overlooked.** Nothing here uses a Nav2 controller, so running Nav2 only to plan would mean standing up a second Docker stack for one A* call.
- **If open loop plus replanning ever stops arriving, the answer is a different design.** `scripts/navigate.py` prints a replan count and a worst-distance-off-schedule for every route; those two columns are the evidence. If they climb and the robot stops reaching 10 cm - on noisier poses, or on hardware - what it needs is a cross-track tracking controller, not a tweak to this one.
