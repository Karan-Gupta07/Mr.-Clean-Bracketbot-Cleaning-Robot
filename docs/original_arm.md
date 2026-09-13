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

The former ball station at `(2.25, -1.10)` is now `table_pick`, with a 48 mm,
70 g cube named `pick_cube` and a clear destination 17 cm along the table.
The other cube and crockery stations remain. The room builder generated both
`models/room.xml` and `models/room_scene.xml` and passed its clearance and
reachability checks. Use the `dock_pick` keyframe in the full scene.

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

Training uses fresh demonstrations generated with this model, followed by PPO
with demonstration rehearsal. The previous parallel-jaw checkpoints are not
compatible. New checkpoints store a gripper/control-version identifier and the
runner rejects a mismatched padded-gripper checkpoint.

```powershell
# Train the original hinged grippers with matching-color pads at the new station.
.venv\Scripts\python.exe scripts/train_arm.py --gripper padded --station pick --output out/rl/arm_original --episodes 20 --updates 2000 --steps 8192

# Run the saved policy in a live viewer, or make the neuron-explorer replay.
.venv\Scripts\python.exe scripts/run_arm.py --view
.venv\Scripts\python.exe scripts/run_arm.py --record --open

# Test the bare exported gripper separately.
.venv\Scripts\python.exe scripts/train_arm.py --teacher-check --gripper urdf --station pick
```

Generated checkpoints and replays remain local in ignored `out/`. The neuron
explorer shows source neuron IDs and annotations, actual policy activations,
input values, and action outputs. Its gripper description and RL results are
read from the recording rather than copied from the old parallel-jaw experiment.
The historical parallel-jaw results in `arm_rl.md` are a separate experiment.
