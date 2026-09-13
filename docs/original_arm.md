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
Increasing the jaw-output gain in diagnostic trials did not recover a grasp and
was not adopted. Dispatch remains blocked. Further work must repair the learned
grip transitions rather than reuse the scripted teacher as a hidden fallback.
Measured integration results are in `docs/results/integration.json`.

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
