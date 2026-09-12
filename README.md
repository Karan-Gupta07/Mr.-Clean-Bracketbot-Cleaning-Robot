# RL-BOT

MuJoCo simulation of the BracketBot: a two-wheeled self-balancing robot with a
1.6 m mast and two 7-DOF arms. A hand-tuned PD controller is the current
baseline — it exists to prove the model is physically sound and to give a
learned policy something to beat.

## Setup

macOS/Apple Silicon, Python 3.10+ (the MuJoCo wheels no longer cover 3.9):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python scripts/evaluate.py                # headless: stays up? drifts?
.venv/bin/python scripts/evaluate.py --push 300     # 300 N shove at the halfway mark
.venv/bin/mjpython scripts/balance.py               # interactive viewer, real time
.venv/bin/python scripts/build_mjcf.py              # rebuild the MJCF from the URDF
```

`mjpython`, not `python`, for anything with a window: on macOS the window must be
created on the process main thread. `launch_passive` routes it there;
`mujoco.viewer.launch()` and `python -m mujoco.viewer` do not, and fail with
`RuntimeError: Caught an unknown exception!`.

## Layout

```
models/bracketbot/          vendored source: chopped_urdf_v2.urdf + 50 STL meshes
models/bracketbot.xml       generated MJCF - build output, committed
models/bracketbot_scene.xml floor, lighting, keyframes; include this one
models/balancer.xml         first-principles toy balancer, kept as a fast sanity model
scripts/build_mjcf.py       URDF -> MJCF, applying the repairs below
scripts/evaluate.py         headless rollout: survival, max lean, drift, push test
scripts/balance.py          viewer
src/rlbot/robot.py          model loading, state extraction, one stepping call
src/rlbot/control.py        cascaded PD: wheel speed -> pitch reference -> wheel torque
```

## The URDF, and what had to be repaired

`chopped_urdf_v2` is a geometry export (Onshape, 54 links / 53 joints, every
joint axis snapped to `(0,0,1)`). It is not a physics model, and three things
had to be fixed before MuJoCo could simulate it. All three are applied in
`scripts/build_mjcf.py`, so the vendored URDF stays byte-identical to the source
and every repair is reviewable:

1. **No wheel joints.** Both wheels are welded into a chain of fixed joints
   (`head_cover -> right_wheel_tire -> left_wheel_tire`), so the robot is one
   rigid lump — the 18 movable joints are all arm and gripper. The build lifts
   the four wheel geoms into two new bodies hinged about the axle, measured off
   the meshes: r = 0.0846 m, width 0.045 m, axle at y = ±0.1611 m, z = 0.0846 m
   (tyres exactly on z = 0, matching the URDF's root convention).
2. **No collision geometry.** 50 visuals, 0 collisions. Meshes are marked
   non-colliding (`group=2`) and a collision cylinder is added per wheel — the
   only part of the robot that touches the ground.
3. **Broken inertials.** The export totals **0.29 kg**, 95% of it in one link
   (`mirror30__mirror30`), with inertias down at 1e-9. These are discarded;
   mass is recomputed from mesh volume at a uniform density, scaled to hit
   `TOTAL_MASS`.

4. **A rotated frame 14 bodies up.** The whole robot hangs off
   `base_plate__base_plate`, which carries a −90° rotation about x
   (`quat = 0.7071 −0.7071 0 0`); every other body in that chain is identity.
   Anything that re-parents a geom has to compose the full chain to the world or
   it lands with y and z swapped — `world_pose()` in the build script.
5. **`effort=10` on every joint is boilerplate**, not a spec: the URDF gives
   every joint `effort=10` and `velocity=10` alike. 10 N cannot hold the 17.2 N
   mast carriage, so the arms slide down the rail on the first step. Limits are
   sized at 2.5× the worst-case gravity load instead (43 N for the carriages,
   17 N·m at the shoulder), keeping the URDF number as a floor. Note this has to
   be raised in **two** places — the actuator's `forcerange` *and* the joint's
   `actuatorfrcrange`, which the URDF importer also sets from `effort` and which
   clamps `qfrc_actuator` independently. Raise only the first and the servo asks
   for 43 N, receives 10, and the arms still fall.
6. **The mimic follower must not get a servo.** MuJoCo's URDF parser already
   converts `<mimic>` into a joint equality constraint, so adding one duplicates
   it — but nothing stops you putting a position servo on the follower, where it
   fights the constraint. Commanded fully open, the gripper reached −0.10 rad
   instead of +1.0. Followers are left unactuated; 18 arm joints, 16 servos.

### Verified

Full audit in `scripts/build_mjcf.py`'s output and the checks below:

- Joint anchors sit at real hardware locations (shoulder 1.29 m, wrist 0.86 m),
  and world-frame axes differ per joint — the link frames really do carry the
  rotations the URDF README describes.
- Left and right arms mirror to **0.00000 m**.
- Every arm joint reaches both of its limits under its servo; both grippers
  track their mimic follower to within 0.001 rad.
- 60 s of balancing: no MuJoCo warnings, no divergence, all state finite.
- Sensors read true: gyro ≈ 0 at rest, accelerometer +9.81 m/s² on z upright,
  framequat ≈ identity.
- Mass is no longer pathological — heaviest link is 25.9% of the total, down
  from 95%, and no body is under 0.1 g.

### Known gaps

- **No self-collision.** Only the two wheel cylinders and the floor collide; all
  50 meshes are visual-only. The arms will pass straight through the mast and
  through each other. Fine for balancing, wrong for manipulation — add collision
  primitives to the arm links before training anything that reaches.
- **No joint damping, armature or friction on the arms** — the URDF specifies
  none, and none has been invented. Wheels carry a small amount, set by the build.
- Four now-empty bodies (`*_wheel_tire`, `*_wheel_cap`) remain in the tree after
  their geoms were lifted into the hinged wheel bodies. Harmless, but clutter.
- `root` is massless and carries the freejoint; its subtree holds the 12 kg, so
  the dynamics are correct, but it trips naive "every body has inertia" checks.

> **`TOTAL_MASS = 12.0` kg is a placeholder.** Nothing in the URDF says what
> this robot weighs. The resulting CoM sits 0.63 m up. Wheel torque limits
> (±8 N·m) are likewise a guess. Weigh the robot and measure the motors before
> trusting any dynamics number out of this model.

## Model

| | |
| --- | --- |
| DOF | 26 = 6 root + 2 wheels + 18 arm/gripper |
| Actuators | 2 wheel torque motors (±8 N·m) + 18 position servos, limits sized from gravity load |
| Sensors | gyro, accelerometer, framequat on the `imu` site; wheel velocities |
| Mass / CoM | 12.0 kg / 0.63 m (placeholder, see above) |
| Keyframes | `home` (upright), `tipped` (3° forward) |

## Baseline

`Gains.for_bracketbot()` = `kp_pitch=80, kd_pitch=15, kp_speed=0.010`. The robot
presents m·g·h = 74 N·m/rad of destabilising torque, so `kp_pitch` has to sit
just above that; the gains come from a 64-point sweep scored on drift plus
residual wheel speed.

From the `tipped` keyframe: settles in ~2 s, holds 20 s, max lean 3.00°, drift
2 cm, peak wheel torque 4.2 N·m. A 50 ms shove of 300 N costs 4.6° of lean and
450 N costs 13.9°, both recovered; 600 N puts it on the floor.

The toy model (`--robot toy`) still balances in 20 s with 8 mm of drift — it runs
in a second and catches controller regressions without loading 50 meshes.

## Next

Gym-style env wrapper, domain randomisation over mass/friction/latency, then a
policy trained against the same rollout the PD baseline is scored on.
