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
| 2. SLAM navigation | Started. There is a room to map, with walls, a pillar and a divider, and a lidar mount on the robot. No SLAM code yet. |
| 3. VLA pick and place | Started. The arms have inverse kinematics and a grasp test, and the supplied gripper now lifts and places cubes once its blades are given contact pads. No VLA model yet. |

### What works today

- **The BracketBot model.** It was converted from a URDF file into MuJoCo format by `scripts/build_mjcf.py`. The wheels spin, the robot can stand on the floor, and the mass numbers are fixed. See "How the robot model was fixed" below.
- **Balancing.** A hand-tuned PD controller keeps the robot upright for as long as you like. It survives a 300 N shove.
- **A room to work in.** A 6 x 4.5 m room with four walls, a pillar and a divider to map, and three tables: one with a ball and a crate, one with four cubes and a crate, one with a bowl, a mug and a crate. It is written twice - `models/room.xml` is the environment on its own, with no robot in it at all, and `models/room_scene.xml` is the same room with the robot added.
- **A room the robot actually fits in.** `scripts/build_room.py` measures the robot's own footprint - 42 cm across, 1.7 m tall, read off its collision boxes - and checks the room against it before writing anything. 16.2 of the 27 m2 of floor is standable, all of it reachable from the middle, and each table's docking pose leaves 14.6 cm of daylight. It prints the map and refuses to write a room that fails.
- **Arm control.** Inverse kinematics (IK) moves each 7-joint arm to a target pose. The arms and mast now have collision shapes, so they cannot pass through each other.
- **A grasp test.** `scripts/check_grasp.py` tries to pick up every object in the room. It approaches from above, closes the fingers, lifts, and checks the object came along. The supplied blades need contact pads to hold anything; `--bare` runs them without and lifts nothing.
- **A small toy balancer.** `models/balancer.xml` is a simple two-wheeled robot. It loads in a second and is a quick way to catch controller bugs without loading the full robot.

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
offset; see [raw results](docs/results/arm_rl.json).

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
scripts/build_room.py       Writes the room, checks the robot fits in it, and checks the arms can reach every object.
scripts/evaluate.py         Headless balance test with an optional shove.
scripts/balance.py          Live viewer.
scripts/check_grasp.py      Tries to pick up each object in the room.
scripts/validate_ik.py      Proves the arm IK: round-trip on random poses, then every object in the room.
scripts/check_arm_clearance.py  Measures arm-to-chassis clearance along the grasp paths (the sim filters self-collision).

src/rlbot/robot.py          Load the robot, read its state, step the sim.
src/rlbot/control.py        The PD balance controller.
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

- Add a lidar or depth camera to the lidar mount on the robot, and scan the room this PR adds.
- Hook up a SLAM library so the robot can build a map of the room and know where it is.
- Add a path planner so the robot can drive to a target spot while it keeps its balance.
- Add a "dock at a table" move so the robot ends up in a good spot for the arms to reach.

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
