# RL-BOT

Two-wheeled self-balancing robot in MuJoCo. A hand-tuned PD controller is the
current baseline; it exists to prove the model is physically sound and to give a
learned policy something to beat.

## Setup

macOS/Apple Silicon, Python 3.10+ (the wheels no longer cover 3.9):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python scripts/evaluate.py     # headless: does it stay up, and does it drift?
.venv/bin/mjpython scripts/balance.py    # interactive viewer, real time
```

`mjpython`, not `python`, for anything with a window: on macOS the window must be
created on the process main thread. `launch_passive` routes it there;
`mujoco.viewer.launch()` and `python -m mujoco.viewer` do not, and fail with
`RuntimeError: Caught an unknown exception!`.

## Layout

```
models/balancer.xml     MJCF: chassis + two hinge wheels, gyro/accel/framequat, torque motors
src/rlbot/robot.py      model loading, state extraction, one stepping call
src/rlbot/control.py    cascaded PD: wheel speed -> pitch reference -> wheel torque
scripts/evaluate.py     20 s rollout, reports survival / max lean / drift
scripts/balance.py      viewer
```

## Model

Mass 1.1 kg, CoM 0.123 m up, wheels r=0.05 m at ±0.075 m. Starts from the
`tipped` keyframe (5° forward) so every run has something to recover from.
Torque control, `ctrlrange` ±1 N·m per wheel.

State the controller sees: pitch and pitch rate, yaw rate, mean wheel speed,
forward speed, chassis height.

## Baseline

`Gains(kp_pitch=14, kd_pitch=0.5, kp_speed=0.010, kp_yaw=0.05)` — from a 48-point
sweep scored on final drift plus residual wheel speed.

From 5° tipped: settles in ~2 s, 20 s upright, max lean 5.15°, final drift 8 mm,
wheels stationary.

## Next

Gym-style env wrapper, domain randomisation over mass/friction/latency, then a
policy trained against the same `evaluate.py` rollout the PD baseline is scored on.
