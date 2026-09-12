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
| 2. SLAM navigation | Started. There is a room to map, a working lidar, wheel odometry, and a viewer you can drive the robot around in. No SLAM code yet. |
| 3. VLA pick and place | Started. The arms have inverse kinematics and a grasp test. 5 of 10 objects lift so far. No VLA model yet. |

### What works today

- **The BracketBot model.** It was converted from a URDF file into MuJoCo format by `scripts/build_mjcf.py`. The wheels spin, the robot can stand on the floor, and the mass numbers are fixed. See "How the robot model was fixed" below.
- **Balancing.** A hand-tuned PD controller keeps the robot upright for as long as you like. It survives a 300 N shove.
- **A room to work in.** A 6 x 4.5 m room with four walls, a pillar and a divider to map, and three tables: one with a ball and a crate, one with four cubes and a crate, one with a bowl, a mug and a crate. It is written twice: `models/room.xml` is the environment on its own, with no robot in it at all, and `models/room_scene.xml` is the same room with the robot added.
- **A room the robot actually fits in.** `scripts/build_room.py` measures the robot's own footprint - 42 cm across, 1.7 m tall - and checks the room against it before writing anything: 16.2 of the 27 m2 of floor is standable, all of it reachable from the middle, and each table's docking pose leaves 14.6 cm of daylight. It prints the map and refuses to write a room that fails.
- **Driving it around.** `scripts/room.py` opens a window with the robot in the room and lets you drive it with the keyboard while the balancer runs. The lidar scan is drawn live.
- **The sensors SLAM needs.** A 72-beam planar lidar and wheel odometry, in `src/rlbot/sensing.py`. There is also a head camera and a camera on each wrist.
- **Arm control.** Inverse kinematics (IK) moves each 7-joint arm to a target pose. The arms and mast now have collision shapes, so they cannot pass through each other.
- **A grasp test.** `scripts/check_grasp.py` tries to pick up every object in the room. It approaches from above, closes the fingers until they load up, lifts, and checks the object came along. 5 of 10 currently do.
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

# Run the sponsors' arm IK library (Linux arm64 only, so inside a container on a Mac).
docker run --rm --platform linux/arm64 -v "$PWD":/w -w /w python:3.12-slim \
    python3 src/rlbot/hybrid_ik.py /path/to/libhybrid_ik_lib.so models/bracketbot/chopped_urdf_v2.urdf right_eef
```

Use `mjpython`, not `python`, for anything that opens a window. On macOS the window must be made on the main thread, and `mjpython` takes care of that. Plain `python` will fail with `RuntimeError: Caught an unknown exception!`.

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
scripts/room.py             Live viewer you can drive the robot around the room in.

src/rlbot/robot.py          Load the robot, read its state, step the sim.
src/rlbot/control.py        The PD balance controller.
src/rlbot/arm.py            Arm inverse kinematics and the grasp sequence.
src/rlbot/room.py           What is in the room and where. Shared by the builder and the grasp test.
src/rlbot/hybrid_ik.py      ctypes binding for the sponsors' libhybrid_ik_lib.so, set up the way their daemon uses it.
src/rlbot/sensing.py        The lidar and the wheel odometry.
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

## Plan for the next steps

**Step 2, SLAM navigation**

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
