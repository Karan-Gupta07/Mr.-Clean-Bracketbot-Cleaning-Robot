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

Also handled: the URDF `<mimic>` tags on the second gripper joint of each hand
have no MJCF equivalent and become `<equality joint>` constraints, so a gripper
stays one DOF.

> **`TOTAL_MASS = 12.0` kg is a placeholder.** Nothing in the URDF says what
> this robot weighs. The resulting CoM sits 0.63 m up. Wheel torque limits
> (±8 N·m) are likewise a guess. Weigh the robot and measure the motors before
> trusting any dynamics number out of this model.

## Model

| | |
| --- | --- |
| DOF | 26 = 6 root + 2 wheels + 18 arm/gripper |
| Actuators | 2 wheel torque motors (±8 N·m) + 18 position servos (±10 N·m, URDF effort) |
| Sensors | gyro, accelerometer, framequat on the `imu` site; wheel velocities |
| Mass / CoM | 12.0 kg / 0.63 m (placeholder, see above) |
| Keyframes | `home` (upright), `tipped` (3° forward) |

## Baseline

`Gains.for_bracketbot()` = `kp_pitch=80, kd_pitch=15, kp_speed=0.010`. The robot
presents m·g·h = 74 N·m/rad of destabilising torque, so `kp_pitch` has to sit
just above that; the gains come from a 64-point sweep scored on drift plus
residual wheel speed.

From the `tipped` keyframe: settles in ~2 s, holds 20 s, max lean 3.00°, drift
7 cm, peak wheel torque 4.2 N·m. Recovers from a 50 ms shove up to ~300 N
(15 N·s); falls at 600 N.

The toy model (`--robot toy`) still balances in 20 s with 8 mm of drift — it runs
in a second and catches controller regressions without loading 50 meshes.

## Next

Gym-style env wrapper, domain randomisation over mass/friction/latency, then a
policy trained against the same rollout the PD baseline is scored on.
