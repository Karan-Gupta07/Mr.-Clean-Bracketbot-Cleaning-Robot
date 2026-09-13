# RL-BOT

A robot that cleans up a room, built and tested in simulation first.

The robot is the **BracketBot**. It is a two-wheeled robot that balances on its own. It has a tall mast and two arms with grippers. We run it inside **MuJoCo**, a physics simulator, so we can test ideas fast and safely before touching real hardware.

## The goal

The project has three steps. Each step builds on the one before it.

1. **Put the BracketBot in the sim.** Load the robot model, make it stand up, and make it balance. This is the base for everything else.
2. **Drive around a room with SLAM.** SLAM stands for "Simultaneous Localization and Mapping". The robot builds a map of the room while it figures out where it is on that map. Then it can drive from one spot to another without bumping into things.
3. **Pick up and put down objects with a VLA policy.** VLA stands for "Vision-Language-Action". It is a model that looks at camera images, reads a short text command like "put the cup on the table", and outputs arm motions. We use it to do the pick and place tasks that make up "cleaning up".

Put together: the robot maps the room, drives to an object, picks it up, drives to where it belongs, and puts it down.

## Where we are now

| Step | Status |
| --- | --- |
| 1. BracketBot in sim | Done. The robot loads, stands, and balances. It recovers from a shove. |
| 2. SLAM navigation | ROS 2 Jazzy / SLAM Toolbox mapping, map saving and localization restart pass in Ubuntu Docker. Custom curved navigation passes all nine sim routes on the true pose, and a ROS node drives the same navigator on the SLAM pose in Docker (two routes hand-tested). Nav2 is not implemented. |
| 3. VLA pick and place | Started. The arms have inverse kinematics and a grasp test. Nothing lifts reliably yet - the gripper model is the blocker, see below. No VLA model yet. |

### What works today

- **The BracketBot model.** It was converted from a URDF file into MuJoCo format by `scripts/build_mjcf.py`. The wheels spin, the robot can stand on the floor, and the mass numbers are fixed. See "How the robot model was fixed" below.
- **Balancing.** A hand-tuned PD controller keeps the robot upright for as long as you like. It survives a 300 N shove.
- **A room to work in.** A 6 x 4.5 m room with four walls, a pillar and a divider to map, and three tables: one with a ball and a crate, one with four cubes and a crate, one with a bowl, a mug and a crate. It is written twice - `models/room.xml` is the environment on its own, with no robot in it at all, and `models/room_scene.xml` is the same room with the robot added.
- **A room the robot actually fits in.** By default, `scripts/build_room.py` regenerates the room files, then checks clearance and arm reach. The clearance check measures the robot's own footprint - 42 cm across, 1.61 m tall, read off its collision boxes. 16.2 of the 27 m2 of floor is standable, all of it reachable from the middle, and each table's docking pose leaves 14.6 cm of daylight. It prints the map and exits with a nonzero status if the checks fail; the room files have already been written.
- **Driving to a table.** A* searches a grid inflated for the robot's footprint, then the robot follows checked cubic curves with bounded motion profiles. All nine nominal routes from the start to each table and between table docks arrived within 10 cm and 5 degrees, without falling or touching furniture. The local check uses the known-room grid and the sim's true pose. In Docker, `ros2 run rlbot_bridge navigate` runs the same navigator on a saved SLAM map and the localised pose; two routes have been hand-tested that way.

  The four phases - exit, turn, curve, dock - are played open loop from a schedule worked out in advance, not steered step by step. Between phases the robot stops, waits for a steady pose, and replans from wherever it actually ended up; a stale pose, or drifting more than 15 cm off the schedule, stops it the same way. The design asked for 10 cm there; the extra 5 cm is simulation tuning, because a balancing robot writes phantom distance into its pose every time it catches itself. Waiting for a steady pose means holding zero for a braking dwell and then seeing 0.5 s of poses within 5 mm and half a degree of each other before it replans. If that window never arrives the wait re-arms after 8 s and tries again; only 60 replans on one route, or 30 s with no pose at all, ends it. Within about 20 cm of the goal - twice the arrival tolerance - the four phases give way to a guarded turn in place, each intermediate angle checked as a padded rectangle, so the last stretch is not a detour back out to an exact docking anchor and in again. The schedules cap speed at 0.15 m/s, yaw rate at 0.3 rad/s and linear acceleration at 0.1 m/s², with angular acceleration on curves best effort and the DriveController's ramp as a backstop. No controller limits were raised to make any of this work. These are nominal simulation results only: two routes finished with little of their time budget to spare, and noisy poses and hardware have not been tested.

  The check also watches the bridge's 2-degree scan-tilt gate, and it never tripped: the chassis spent 0.00 s past 2 degrees on all nine routes, so driving did not starve SLAM of scans in sim. That is only one of the three things the bridge asks of a scan, though. It also throws one away when the gyro's roll/pitch rate passes 0.2 rad/s, or when fewer than half the rays come back, and the local check measures neither - so a clean sweep here does not fully predict the ROS behaviour.
- **Arm control.** Inverse kinematics (IK) moves each 7-joint arm to a target pose. The arms and mast now have collision shapes, so they cannot pass through each other.
- **A grasp test.** `scripts/check_grasp.py` tries to pick up every object in the room. It approaches from above, closes the fingers, lifts, and checks the object came along. Nothing currently survives the lift - see "What the room is built around".
- **A small toy balancer.** `models/balancer.xml` is a simple two-wheeled robot. It loads in a second and is a quick way to catch controller bugs without loading the full robot.

## Setup

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
scripts/build_room.py       Writes the room, checks the robot fits in it, and checks the arms can reach every object.
scripts/evaluate.py         Headless balance test with an optional shove.
scripts/balance.py          Live viewer.
scripts/check_grasp.py      Tries to pick up each object in the room.
scripts/validate_ik.py      Proves the arm IK: round-trip on random poses, then every object in the room.
scripts/check_arm_clearance.py  Measures arm-to-chassis clearance along the grasp paths (the sim filters self-collision).
scripts/plan_path.py        Plans, draws and checks the nine routes without running physics.
scripts/navigate.py         Drives the nine routes in the sim and checks arrival, falls and furniture contacts.
scripts/check_navigation.py Unit checks for the grid, planner and navigator.

src/rlbot/robot.py          Load the robot, read its state, step the sim.
src/rlbot/control.py        The PD balance controller.
src/rlbot/navmap.py         Known-room and saved PGM/YAML grids, obstacle overlays, inflation and footprint checks.
src/rlbot/planner.py        A* routes, cubic curves, guarded turns and bounded speed/yaw-rate schedules.
src/rlbot/navigate.py       Runs open-loop phases, holds zero to settle and replans from the actual pose.
src/rlbot/arm.py            Arm inverse kinematics and the grasp sequence.
src/rlbot/room.py           What is in the room and where. Shared by the builder and the grasp test.
src/rlbot/hybrid_ik.py      ctypes binding for the sponsors' libhybrid_ik_lib.so, set up the way their daemon uses it.
```

## How the robot model was fixed

The URDF we got is a shape export from Onshape, not a physics model. Six things had to be fixed before MuJoCo could simulate it. All fixes live in `scripts/build_mjcf.py`, so the original URDF stays untouched and every fix can be reviewed in one place.

1. **The wheels could not spin.** They were welded on. The build script gives them real hinge joints.
2. **Nothing could touch the floor.** There were no collision shapes at all. Each wheel now has one.
3. **The mass numbers were wrong.** The whole robot weighed 0.29 kg in the file. Mass is now computed from the mesh volumes and scaled to a total.
4. **One frame high up the tree was rotated.** Anything that moves a part has to account for it, or the part lands sideways.
5. **Every joint had the same fake strength limit.** 10 N cannot hold the arm carriage up, so the arms slid down the rail. Limits are now sized from the real gravity load.
6. **The second gripper finger was getting its own motor.** It should only follow the first finger. The extra motor was removed.

## Numbers

| | |
| --- | --- |
| Joints | 26 = 6 for the floating base + 2 wheels + 18 in the arms and grippers |
| Motors | 2 wheel motors (±8 N·m) + 16 arm servos |
| Sensors | Gyro, accelerometer, and orientation on the `imu` site. Wheel speeds. |
| Mass | 12.0 kg, center of mass 0.63 m up. **This is a placeholder.** |
| Balance gains | `kp_pitch=80, kd_pitch=15, kp_speed=0.010` |

From a 3° lean, the robot settles in about 2 seconds and stays up. It recovers from a 300 N shove with 4.6° of lean, and from 450 N with 13.9°. A 600 N shove knocks it over.

## What the room is built around

Every dimension in the room answers to a measurement taken off the robot, so the
layout cannot quietly drift away from what the robot can do.

| | |
| --- | --- |
| Room | 6.0 x 4.5 m, walls 2.5 m. A pillar and a divider stub, so a map of it is not just a rectangle. |
| Tables | Three, 1.0 x 0.6 m, tops at 0.70 m - the hands hang at 0.715 m. |
| Docking | 0.24 m from the mast axis to the table edge. The robot is 0.19 m deep, so it parks with 14.6 cm to spare. |
| Objects | 0.09 m in from the near edge, which puts them 0.33 m from the mast axis against a top-down reach that runs out around 0.36 m. |
| Object size | 40 to 60 mm across the grasp axis. |

The object sizes come from closing the real gripper on test blocks rather than
from reading the fingertips: the tips part by 195 mm, but the fingers are hooks
on pivots and they splay as they open, so past about half travel the two
gripping faces stop facing each other. Hence a small bowl and a mug on the
crockery table rather than a dinner plate - a plate is 26 mm tall, and the
finger pads hang 22 mm below the middle of the jaw, so closing on a plate means
closing on the table.

The ball is the one object deliberately left outside that window. A smooth
sphere is the thing this gripper cannot pick up, and it is worth keeping in the
room as the case a better end effector has to beat.

**The gripper model is the blocker, not the room.** These sizes are what the
hand can hold once its fingers are modelled properly. As the model stands,
MuJoCo collides each finger mesh as its convex hull, and the fingers are hooked
claws with hollow insides - hulled, the two of them fill the jaw solid, so an
object placed dead centre between them is already touching both and closing
shoots it out. Until `scripts/build_mjcf.py` gives each blade a pad fitted to
its real inner face, `check_grasp.py` reports nothing lifting, whatever is on
the tables.

## Plan for the next steps

**Step 2, SLAM navigation**

- Run all nine routes through the ROS node on the SLAM pose automatically, the way `scripts/check_ros_mapping.py` runs mapping; only two routes have been hand-tested in Docker so far.
- Account for tabletop overhangs that the low LiDAR scan cannot see. The loader already accepts extra obstacle boxes; sensing those overhangs and placing them in the map frame still need work.
- Validate the physical LiDAR mount and the 2-degree scan-tilt gate during autonomous routes, including pose loss and noisy localization, so driving does not starve SLAM of scans.
- Use perceived table edges for fine docking within the arms' reach, rather than relying on the known room's docking poses.

**Step 3, VLA pick and place**

- Get more than 1 of 10 objects to lift with the scripted grasp.
- Add a wrist camera on each arm and a head camera.
- Collect demo data in the sim: the robot picks up an object and puts it somewhere.
- Fine-tune a VLA model on that data so it can follow text commands like "pick up the cup".
- Join it all together: map the room, drive to the object, pick it up, drive to the drop spot, and put it down.

## Things to know

- **The robot's weight and motor limits are guesses.** The URDF does not say what the robot weighs. We will need to weigh the real robot and check the motor specs before trusting any force or torque numbers from the sim. Then re-run `build_mjcf.py --total-mass` and re-tune the gains.
- **The arms have no damping or friction.** The URDF does not give any, and none was made up.
- **The grasp test bolts the robot to the floor by default.** That way a failed grasp is the grasp's fault, not the balancer's. Use `--balance` to run it the honest way, on the wheels.
- **Nav2 was considered and dropped, not overlooked.** Nothing here uses a Nav2 controller, so running Nav2 only to plan would mean standing up a second Docker stack for one A* call.
- **If open loop plus replanning ever stops arriving, the answer is a different design.** `scripts/navigate.py` prints a replan count and a worst-distance-off-schedule for every route; those two columns are the evidence. If they climb and the robot stops reaching 10 cm - on noisier poses, or on hardware - what it needs is a cross-track tracking controller, not a tweak to this one.
