# Learned arm motion

The arm policy reads simulator state and chooses four continuous values: XYZ
motion and gripper opening. It uses the same measured 512-neuron FlyWire graph
as the navigation experiment, with new input/output adapters and separately
trained weights. The navigation checkpoint is not used to move the arm.

At execution there is no scripted phase sequence, waypoint teacher, or timer
selecting movements. The observation has 20 values: grip position relative to
the initial cube position (3), cube relative to grip (3), goal relative to cube
(3), cube linear velocity (3), jaw opening (1), finger contacts (1), commanded
position relative to actual grip (3), and previous XYZ action (3). Positions and
velocities are scaled by 10, jaw displacement by 20, and contacts by 1/2.

Every 50 ms, three outputs increment the commanded grip position by up to
4 mm per axis (80 mm/s). The fourth requests a per-jaw opening between 0 and
45 mm. Downward wrist orientation is fixed. Damped-least-squares IK converts
the commanded position to seven joint servo targets; MuJoCo executes 25 steps
at 2 ms each. These low-level controllers do not choose task phases.

The graph has fixed measured adjacency, trainable input encoding, per-neuron
gains and biases, channel mixing, and output heads. Four propagation rounds
produce artificial rate-like activity in four channels per neuron. The actor
and critic each have two 128-unit layers after graph features. There are no
biological spikes or neural state carried between policy calls.

## Training

`scripts/train_arm.py` contains a demonstration teacher used only for training
data. Behavioral cloning initializes the actor with action MSE. PPO subsequently
collects its own stochastic rollouts and optimizes the clipped policy objective
plus a value loss. Demonstration rehearsal follows each PPO rollout to reduce
forgetting. This is demonstration-assisted RL, not learning from scratch.

The dense reward is `10 * (potential_now - potential_previous) - 0.01`, plus
30 for successful placement. Potential combines negative grip-to-cube distance,
clipped lift height, and horizontal goal progress. Success additionally requires
at least an 8 cm lift, over 0.5 s of elevated two-finger contact, final horizontal
error below 2.5 cm, cube center within 8 mm of its resting height, speed below
2.5 cm/s, released fingers, a withdrawn hand, and 0.5 s of stable placement.
The horizon is 400 actions / 20 seconds. Fallen cubes terminate the episode.

The successful run used 20 demonstration episodes (5,260 transitions), 3,000
cloning updates, then 138 targeted release corrections collected on training
seeds 30–34 and 750 further cloning updates (Adam, learning rate 0.0005, batch
256). Corrections were repeated eight times in the training dataset. It then
used 8,192 PPO transitions. PPO uses learning
rate 0.00001, rollout length 512, batch 128, three epochs, value coefficient
0.01, no entropy bonus, and initial log standard deviation -3.5. Each rollout
is followed by 32 rehearsal updates at learning rate 0.0001. Training seed is 7.

```powershell
.venv\Scripts\python.exe scripts/train_arm.py --output out/rl/arm_friction3 --steps 0
.venv\Scripts\python.exe scripts/train_arm.py --output out/rl/arm_release --resume out/rl/arm_friction3/imitation.zip --demonstrations out/rl/arm_friction3/demonstrations.npz --correct-release --updates 750 --steps 8192
.venv\Scripts\python.exe scripts/run_arm.py --episodes 10 --seed 2000
.venv\Scripts\python.exe scripts/run_arm.py --view
.venv\Scripts\python.exe scripts/run_arm.py --record --open
```

Checkpoints, raw before/after evaluation, and demonstrations are saved in the
specified output folder. The recorded replay uses the loaded policy's actual
actions and graph activations. `scripts/run_arm.py` does not import the teacher.

The correction collector intervenes only while generating training examples:
when a previously lifted cube reaches the destination and tabletop, it supplies
open-jaw and retreat targets. That logic is absent from the environment's action
selection and from the saved-policy runner. The graph learns the resulting
commands, and must generate them independently at evaluation.

## Measured results

| Checkpoint | Seeds 3000–3009, full task success |
| --- | --- |
| Demonstrations and release corrections, before PPO | 2/10 |
| Same initialization, after PPO plus rehearsal | 10/10 |

The final policy also passed all ten starts on seeds 2000–2009, for **20/20**
across these two sets. Final placement error across them was 6.2–10.7 mm.
This is one training seed on a narrow task distribution, not a general grasping
benchmark. Because rehearsal accompanies PPO, this comparison does not isolate
the effect of PPO from additional imitation updates.

The saved checkpoint records 8,192 environment actions and 48 PPO training
epochs (16 rollouts, three epochs each). Trainable parameter change from its
pre-PPO checkpoint is L2 0.7981; graph-neuron gain change is L2 0.0734. Raw
metrics, checkpoint hashes, and per-episode results are in
[results/arm_rl.json](results/arm_rl.json). The first run without targeted
release examples achieved 0/10 successes after PPO; that result is retained in
[results/arm_rl_initial_failed.json](results/arm_rl_initial_failed.json).

The replay in `out/arm_rl_demo/index.html` is the final PPO checkpoint on seed
2000: successful placement in 10.3 s, 16.0 cm maximum lift, and 7.2 mm final
error. It displays measured neuron coordinates and actual policy activations.
The GIF is `out/arm_rl_demo/demo.gif`; generated checkpoints and replays are
local artifacts in ignored `out/`, not included in the Git repository.

## Simulation scope

This task uses one 48 mm cube, randomized by ±8 mm in XY, with a destination
17 cm to the right. The arm starts above the cube. The base is fixed and the
original hooked jaws are replaced by the parallel-jaw simulation variant.
For this continuous-control experiment pad sliding friction is **3.0**, versus
1.2 in the scripted baseline. This change prevented observed transfer slips in
the demonstration check; it is not a measured hardware coefficient.

The cube remains a free body. Only reset assigns its pose. There is no object
attachment, hidden grasp assist, or scripted recovery during policy execution.
Robot self-collision is still filtered in the supplied model. Camera perception,
base balancing during manipulation, varying destinations, other objects, and
hardware performance remain unvalidated.
