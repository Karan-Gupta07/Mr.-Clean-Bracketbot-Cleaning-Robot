# Original-gripper pick-and-place station

The default arm task now uses the supplied CAD finger meshes, original hinge
joints, mimic coupling, joint ranges, and actuator force limits. The source
`models/bracketbot/chopped_urdf_v2.urdf` and robot mesh assets are unchanged.
The robot is still fixed at the docking pose during this manipulation task.

The bare exported grippers failed all three initial grasp trials. The working
training setup therefore uses the allowed matching-color contact pads on the
original fingers: 26 × 33 mm faces, 6 mm thick. The pad friction coefficient is
5.0, selected in simulation, not measured on hardware. Pad collision replaces
the oversized convex-hull collision of the hooked blades; the CAD meshes remain
visible. This is an explicit contact approximation, not a claim that the entire
original URDF collision model works unchanged. No sliding-jaw replacement is
used by the default demo.

The former crockery station at `(-2.25, 0.90)` is now `table_pick`, with a
48 mm, 70 g blue cube named `pick_cube` and a rectangular destination marker
17 cm along the table. The red-ball/crate station and the Fable colored-cube
station remain. The room builder generates both `models/room.xml` and
`models/room_scene.xml`; use the `dock_pick` keyframe in the full scene.

Navigation keeps its original contact-solver impedance ratio of 10. Fixed-base
manipulation selects 200 in its own model. Applying 200 to the shared room
regressed the start-to-ball route; keeping the settings separate fixes it.
Fable uses its measured compliant-pad variant; Flybrain uses its own fitted
pads. Both preserve the original finger meshes, hinge joints and gripper limits.
Pad installation replaces the previous variant rather than stacking contacts.

## Learned control

The fly-connectivity policy still selects four continuous values at 20 Hz:
table-relative XYZ motion and jaw opening. There is no phase controller during
policy inference. World poses and velocities are transformed to the task frame,
so the station's rotation does not change the meanings of the policy inputs.

For these pincer jaws, the action range maps to approximately 0.314–0.516 rad:
closed on this cube through approach opening. The requested angle ramps by at
most 0.015 rad per policy step, or 0.3 rad/s. These are controller command bounds;
the original URDF joint ranges and force limits are unchanged. Fully commanding
the pincers shut would make the blades cross through the cube. The restricted
command range is calibrated for this cube and is not a general grasp controller.

Training uses fresh demonstrations generated with this model, optional DAgger
corrections collected only during training, and PPO with demonstration rehearsal.
Inference does not use the teacher. The episode limit is 480 steps: the stronger
grip carries through to intentional release, and the 400-step limit truncated
withdrawal. Three teacher trials now complete in 428 steps each.

The interrupted `out/rl/arm_original` experiment finished with **0/10** learned
policy successes. The combined-model `out/rl/arm_integrated` run also finished
at **0/10**, despite 4,000 imitation updates, three DAgger rounds and 8,192 PPO
steps. Neither checkpoint is a working demo.

A regression test identified hidden controller state: changing the rate-limited
jaw command did not change the policy observation. Version 4 adds that command,
jaw angular velocity and the previous jaw action, making 23 observation values.
New training goes to `out/rl/arm_observable`; the failed artifacts are preserved.
The replay reads the observation count from the actual recording. This version
also failed: **0/10** evaluation starts and **0/20** held-out starts (3000–3019).
The missing-input fix is verified, but it is not a working learned grasp policy.
Increasing the jaw-output gain in that single-frame experiment did not recover
a grasp and was not adopted. The later history-based experiment below has a
different result. Measured integration results are in `docs/results/integration.json`.

### History and output calibration

The teacher's private wait counter gave nearly identical single-frame inputs
opposing open/close labels. Optional `--history 16` supplies 368 values from
16 causal observations, with episode-reset padding and no teacher phase or
future state. Control version 5 records that history length; version 4 and its
single-frame checkpoints remain the default for backward compatibility.

After 4,000 imitation updates on 17 successful demonstrations, history alone
still passed **0/5** development starts. A recorded **1.25x learned jaw-output
gain** then passed **9/10** development starts and **17/20** fresh held-out starts
(4000–4019). The gain scales the actor's jaw-output weights; command bounds,
rate limits, URDF and physics are unchanged. It does not invoke a teacher.

An additional Cartesian command deadband of **0.05** passed **9/10** development
starts and **18/20** new held-out starts (5000–5019). It suppresses outputs below
0.2 mm per control step, without choosing task phases. Control version 6 records
the deadband, and replay data includes both raw outputs and applied commands.
Seeds 5011 and 5012 still failed release/placement. The two held-out sets differ;
they are not a paired ablation. Both calibrated candidates are **imitation-only**
(zero PPO steps), not successful PPO runs. They remain blocked by the unchanged
20/20 dispatch requirement, and no default checkpoint was replaced.

The saved candidates are `out/rl/arm_history_calibrated/policy.zip` and
`out/rl/arm_history_deadband/policy.zip`, with checkpoint-bound reports in each
`validation/` directory. A diagnostic replay of the first candidate is in
`out/arm_history_demo/index.html`; its successful displayed episode is separate
from its 17/20 held-out score. All these artifacts remain local and ignored.

To reproduce the history training and explicit calibration in new directories:

```powershell
.venv\Scripts\python.exe scripts/train_arm.py --history 16 --episodes 20 --updates 4000 --bc-lr 0.0003 --steps 0 --output out/rl/arm_history_reproduction
.venv\Scripts\python.exe scripts/train_arm.py --history 16 --motion-deadband 0.05 --resume out/rl/arm_history_reproduction/imitation.zip --calibrate-jaw 1.25 --output out/rl/arm_calibrated_reproduction
.venv\Scripts\python.exe scripts/run_arm.py --history 16 --motion-deadband 0.05 --checkpoint out/rl/arm_calibrated_reproduction/policy.zip --episodes 20 --seed 5000 --output out/rl/arm_calibrated_reproduction/validation
```

Reusing a published evaluation seed set is a regression check, not a new
held-out result after further tuning. Further release robustness and optional
PPO fine-tuning remain outstanding; neither may hide a scripted fallback.

Checkpoints and demonstration datasets must match the gripper, station, control
version, horizon, physics fingerprint, and generated robot/room hashes. This
rejects both stale original-gripper checkpoints and the archived parallel-jaw
policies instead of silently running them against a different environment.
The orchestrator additionally requires a matching 20-episode validation report
with all trials successful; `run_arm.py` remains available for failed-policy
diagnostics and exits nonzero when any requested episode fails.

```powershell
# Train the original hinged grippers with matching-color pads at the new station.
.venv\Scripts\python.exe scripts/train_arm.py --gripper padded --station pick --output out/rl/arm_observable --episodes 20 --updates 2500 --steps 8192

# Run the saved policy in a live viewer, or make the neuron-explorer replay.
.venv\Scripts\python.exe scripts/run_arm.py --view
.venv\Scripts\python.exe scripts/run_arm.py --record --open
.venv\Scripts\python.exe scripts/run_arm.py --episodes 20 --seed 3000 --output out/rl/arm_observable/validation

# Test the bare exported gripper separately.
.venv\Scripts\python.exe scripts/train_arm.py --teacher-check --gripper urdf --station pick
```

Generated checkpoints and replays remain local in ignored `out/`. The neuron
explorer shows source neuron IDs and annotations, actual policy activations,
input values, and action outputs. Its gripper description and RL results are
read from the recording rather than copied from the old parallel-jaw experiment.
The historical parallel-jaw results in `arm_rl.md` are a separate experiment.
