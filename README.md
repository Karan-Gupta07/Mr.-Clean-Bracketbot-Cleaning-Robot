# Cleanup

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
| 1. BracketBot in sim | In progress. A simple two-wheeled balancer works on `main`. The full BracketBot model is in pull request #1. |
| 2. SLAM navigation | Not started. |
| 3. VLA pick and place | Not started. |

### What works today

- A small two-wheeled robot model (`models/balancer.xml`) that balances using a hand-tuned PD controller.
- A headless test that checks if the robot stays up for 20 seconds and how far it drifts.
- A live viewer so you can watch the robot balance.

### What is in pull request #1

- The real BracketBot model, converted from its URDF file into MuJoCo format.
- Wheels that actually spin, collision shapes so the robot can stand on the floor, and fixed mass numbers.
- A balance controller tuned for the bigger, heavier robot.

## Setup

You need macOS (Apple Silicon is fine) and Python 3.10 or newer.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

```bash
# Headless test. Does the robot stay up? Does it drift?
.venv/bin/python scripts/evaluate.py

# Live viewer. Watch the robot balance in real time.
.venv/bin/mjpython scripts/balance.py
```

Use `mjpython`, not `python`, for anything that opens a window. On macOS the window must be made on the main thread, and `mjpython` takes care of that. Plain `python` will fail with `RuntimeError: Caught an unknown exception!`.

## Folder layout

```
models/       Robot and scene files for MuJoCo (XML format)
scripts/      Things you run: evaluate.py (test), balance.py (viewer)
src/rlbot/    The Python code: load the robot, read its state, step the sim, control it
```

## Plan for the next steps

**Step 2, SLAM navigation**

- Build a room scene in MuJoCo with walls, furniture, and objects on the floor.
- Add a camera and a depth sensor (or a lidar) to the robot model.
- Hook up a SLAM library so the robot can build a map and know where it is.
- Add a path planner so the robot can drive to a target spot while it keeps its balance.

**Step 3, VLA pick and place**

- Add a wrist camera on each arm and a head camera.
- Collect demo data in the sim: the robot picks up an object and puts it somewhere.
- Fine-tune a VLA model on that data so it can follow text commands like "pick up the cup".
- Join it all together: map the room, drive to the object, pick it up, drive to the drop spot, and put it down.

## Things to know

- The BracketBot's real weight and motor limits are not known yet. The sim uses placeholder numbers. We will need to weigh the robot and check the motor specs before trusting any force or torque numbers from the sim.
- The simple balancer model is kept around on purpose. It loads in a second and is a quick way to catch bugs in the controller without loading the full robot.
