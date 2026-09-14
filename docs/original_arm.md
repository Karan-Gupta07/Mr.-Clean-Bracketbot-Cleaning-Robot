# Original-gripper pick-and-place station

The default arm task uses the supplied CAD finger meshes. It keeps the original
hinge joints, mimic coupling, joint ranges and actuator force limits. The source
`models/bracketbot/chopped_urdf_v2.urdf` and the robot mesh assets are unchanged.
The robot stays fixed at the docking pose during this manipulation task.

Every `out/...` path below is generated locally. Git ignores these paths, so they
are absent on a fresh clone.

## The gripper

The bare exported grippers failed all three initial grasp trials. The working
training setup therefore adds the allowed matching-color contact pads to the
original fingers.

| Pad property | Value |
| --- | --- |
| Face | 26 × 33 mm |
| Thickness | 6 mm |
| Friction coefficient | 5.0 |

The friction value comes from simulation, not from hardware measurement. Pad
collision replaces the oversized convex-hull collision of the hooked blades. The
CAD meshes stay visible. This is an explicit contact approximation. It is not a
claim that the entire original URDF collision model works unchanged. The default
demo uses no sliding-jaw replacement.

## The station

The former crockery station at `(-2.25, 0.90)` is now `table_pick`. It holds a
48 mm, 70 g blue cube named `pick_cube`. A rectangular destination marker sits
17 cm along the table. The red-ball/crate station and the Fable colored-cube
station remain. The room builder generates `models/room.xml` and
`models/room_scene.xml`. Use the `dock_pick` keyframe in the full scene.

| Model | Contact-solver impedance ratio |
| --- | --- |
| Shared room, navigation | 10 |
| Fixed-base manipulation | 200 |

Applying 200 to the shared room regressed the start-to-ball route. Separate
settings fix it. Fable uses its measured compliant-pad variant. Flybrain uses its
own fitted pads. Both preserve the original finger meshes, hinge joints and
gripper limits. Pad installation replaces the previous variant. It does not stack
contacts.

## Learned control

The fly-connectivity policy selects four continuous values at 20 Hz:
table-relative XYZ motion and jaw opening. No phase controller runs during policy
inference. World poses and velocities transform to the task frame. Station
rotation therefore does not change the meaning of the policy inputs.

| Jaw command bound | Value |
| --- | --- |
| Action range | about 0.314–0.516 rad |
| Ramp limit per policy step | 0.015 rad |
| Ramp limit per second | 0.3 rad/s |

The low end is closed on this cube. The high end is the approach opening. These
are controller command bounds. The original URDF joint ranges and force limits are
unchanged. Commanding the pincers fully shut would drive the blades through the
cube. The restricted command range is calibrated for this cube. It is not a
general grasp controller.

Training uses fresh demonstrations generated with this model. It adds optional
DAgger corrections, collected only during training. It then runs PPO with
demonstration rehearsal. Inference does not use the teacher. The episode limit is
480 steps: the stronger grip carries through to intentional release, and the
400-step limit truncated withdrawal. Three teacher trials now complete in 428
steps each.

### Earlier failures

A regression test identified hidden controller state. Changing the rate-limited
jaw command did not change the policy observation. Control version 4 adds that
command, the jaw angular velocity and the previous jaw action. The observation is
then 23 values. The replay reads the observation count from the actual recording.

| Run | Result |
| --- | --- |
| `out/rl/arm_original`, interrupted | 0/10 learned-policy successes |
| `out/rl/arm_integrated`, 4,000 imitation updates, three DAgger rounds, 8,192 PPO steps | 0/10 |
| `out/rl/arm_observable`, control version 4 | 0/10 evaluation starts, 0/20 held-out starts (seeds 3000–3019) |

No checkpoint above is a working demo. The failed artifacts are preserved. The
missing-input fix is verified, but this is not a working learned grasp policy.
Increasing the jaw-output gain in that single-frame experiment did not recover a
grasp, so it was not adopted. The history-based experiment below has a different
result. Measured integration results are in
[results/integration.json](results/integration.json).

### History and output calibration

The teacher's private wait counter gave nearly identical single-frame inputs
opposing open/close labels. `--history 16` supplies 368 values from 16 causal
observations. Padding resets at each episode start. No teacher phase or future
state enters the observation. Control version 5 records the history length.
Version 4 and its single-frame checkpoints remain the default for backward
compatibility.

| Policy | Development starts | Held-out starts |
| --- | --- | --- |
| History only, 4,000 imitation updates on 17 successful demonstrations | 0/5 | not run |
| Plus a recorded 1.25x learned jaw-output gain | 9/10 | 17/20 (seeds 4000–4019) |
| Plus a 0.05 Cartesian command deadband | 9/10 | 18/20 (seeds 5000–5019) |

The gain scales the actor's jaw-output weights. Command bounds, rate limits, URDF
and physics are unchanged. It does not invoke a teacher. The deadband suppresses
outputs below 0.2 mm per control step. It does not choose task phases. Control
version 6 records the deadband. Replay data includes both raw outputs and applied
commands.

Seeds 5011 and 5012 still failed release and placement. The two held-out sets
differ, so they are not a paired ablation. Both calibrated candidates are
imitation-only, with zero PPO steps. They are not successful PPO runs. They remain
blocked by the unchanged 20/20 dispatch requirement. No default checkpoint was
replaced.

The saved candidates are `out/rl/arm_history_calibrated/policy.zip` and
`out/rl/arm_history_deadband/policy.zip`. Each holds a checkpoint-bound report in
its `validation/` directory. A diagnostic recording of the first candidate was
made locally under `out/arm_history_demo/`. Its one successful episode is
separate from its 17/20 held-out score.

### The shipped checkpoint

The repository ships `checkpoints/flybrain_arm_padded_calibrated.zip`. It is
byte-identical to `out/rl/arm_padded_calibrated/policy.zip`. That run's
`report.json` records the recipe: resumed from
`out/rl/arm_history_padded/imitation.zip`, jaw gain 1.25, deadband 0.05, and
`ppo_steps: 0`. It scored 4 of 10 successes on its own evaluation seeds
2000–2009. The 17/20 and 18/20 candidates above were not shipped.

## Commands

```bash
# Train the original hinged grippers with matching-color pads at the new station.
.venv/bin/python scripts/train_arm.py --gripper padded --station pick --output out/rl/arm_observable --episodes 20 --updates 2500 --steps 8192

# Validate a trained policy over 20 held-out episodes.
.venv/bin/python scripts/run_arm.py --episodes 20 --seed 3000 --output out/rl/arm_observable/validation

# Test the bare exported gripper separately.
.venv/bin/python scripts/train_arm.py --teacher-check --gripper urdf --station pick
```

Reproduce the history training and the explicit calibration in new directories:

```bash
.venv/bin/python scripts/train_arm.py --history 16 --episodes 20 --updates 4000 --bc-lr 0.0003 --steps 0 --output out/rl/arm_history_reproduction
.venv/bin/python scripts/train_arm.py --history 16 --motion-deadband 0.05 --resume out/rl/arm_history_reproduction/imitation.zip --calibrate-jaw 1.25 --output out/rl/arm_calibrated_reproduction
.venv/bin/python scripts/run_arm.py --history 16 --motion-deadband 0.05 --checkpoint out/rl/arm_calibrated_reproduction/policy.zip --episodes 20 --seed 5000 --output out/rl/arm_calibrated_reproduction/validation
```

Run the shipped policy. Use `mjpython` for `--view`, because it opens a MuJoCo
window:

```bash
.venv/bin/mjpython scripts/run_arm.py --view --history 16 --motion-deadband 0.05 --checkpoint checkpoints/flybrain_arm_padded_calibrated.zip

# Record a GIF of the policy and open it.
.venv/bin/python scripts/run_arm.py --record --open --history 16 --motion-deadband 0.05 --checkpoint checkpoints/flybrain_arm_padded_calibrated.zip
```

Reusing a published evaluation seed set is a regression check. It is not a new
held-out result after further tuning. Release robustness and optional PPO
fine-tuning remain outstanding. Neither may hide a scripted fallback.

## Dispatch requirements

Checkpoints and demonstration datasets must match the gripper, station, control
version, horizon, physics fingerprint, and generated robot and room hashes. A
mismatch is rejected. This blocks stale original-gripper checkpoints and the
archived parallel-jaw policies. They never run silently against a different
environment. The orchestrator also requires a matching 20-episode validation
report with every trial successful. `run_arm.py` remains available for
failed-policy diagnostics. It exits nonzero when any requested episode fails.

The recording's `episode.json` holds source neuron IDs and annotations, actual
policy activations, input values and action outputs. Its `report.json` reads
the gripper description and RL results from the checkpoint. It does not copy
them from the old parallel-jaw experiment. The historical parallel-jaw results
in [arm_rl.md](arm_rl.md) are a separate experiment.
