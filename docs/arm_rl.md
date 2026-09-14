# Learned arm motion

## Historical status

This page records the parallel-jaw arm experiment. Models now default to the
supplied hooked gripper with contact pads on its blades. Every result on this
page was trained and measured with the sliding-jaw substitution. Pass `--gripper
parallel` to reproduce any of it.

The saved checkpoint does not transfer to the other builds, because it learned a
different jaw travel and pad friction. Retrain before this policy drives the
supplied gripper. [pick_place.md](pick_place.md) explains what the pads do.

## What the policy does

The arm policy reads simulator state. It chooses four continuous values: XYZ
motion and jaw opening. It uses the same measured 512-neuron FlyWire graph as
the navigation experiment. The input and output adapters are new. The weights
are trained separately. The navigation checkpoint does not move the arm.

At execution no scripted phase sequence, waypoint teacher, or timer selects
movements.

The observation has 20 values.

| Value | Count |
| --- | --- |
| Grip position relative to the initial cube position | 3 |
| Cube relative to the grip | 3 |
| Goal relative to the cube | 3 |
| Cube linear velocity | 3 |
| Jaw opening | 1 |
| Finger contacts | 1 |
| Commanded position relative to the actual grip | 3 |
| Previous XYZ action | 3 |

Positions and velocities are scaled by 10, jaw displacement by 20, and contacts
by 1/2.

The policy acts every 50 ms. Three outputs increment the commanded grip position
by up to 4 mm per axis, which is 80 mm/s. The fourth output requests a per-jaw
opening between 0 and 45 mm. Downward wrist orientation is fixed.

Damped-least-squares IK converts the commanded position to seven joint servo
targets. MuJoCo then executes 25 steps at 2 ms each. These low-level controllers
do not choose task phases.

The graph has fixed measured adjacency. It also has a trainable input encoding,
per-neuron gains and biases, channel mixing, and output heads. Four propagation
rounds produce artificial rate-like activity in four channels per neuron. The
actor and the critic each have two 128-unit layers after the graph features.
There are no biological spikes. No neural state carries between policy calls.

## Training

`scripts/train_arm.py` contains a demonstration teacher. The teacher only makes
training data. Behavioral cloning initializes the actor with action MSE. PPO
then collects its own stochastic rollouts. It optimizes the clipped policy
objective plus a value loss.

Demonstration rehearsal follows each PPO rollout and reduces forgetting. This is
demonstration-assisted RL, not learning from scratch.

The dense reward is `10 * (potential_now - potential_previous) - 0.01`. A
successful placement adds 30. Potential combines negative grip-to-cube distance,
clipped lift height, and horizontal goal progress. The horizon is 400 actions,
or 20 seconds. A fallen cube terminates the episode.

Success requires all of the following:

- a lift of at least 8 cm
- over 0.5 s of elevated two-finger contact
- final horizontal error below 2.5 cm
- the cube center within 8 mm of its resting height
- speed below 2.5 cm/s
- released fingers
- a withdrawn hand
- 0.5 s of stable placement

The successful run used these settings.

| Stage | Setting |
| --- | --- |
| Demonstrations | 20 episodes, 5,260 transitions |
| Cloning | 3,000 updates |
| Release corrections | 138, collected on training seeds 30–34 |
| Correction cloning | 750 updates, Adam, learning rate 0.0005, batch 256 |
| Correction repeats in the dataset | 8 |
| PPO | 8,192 transitions, learning rate 0.00001, rollout 512, batch 128, three epochs |
| PPO extras | value coefficient 0.01, no entropy bonus, initial log standard deviation -3.5 |
| Rehearsal | 32 updates per rollout, learning rate 0.0001 |
| Training seed | 7 |

Use `mjpython`, not `python`, for any MuJoCo window on macOS.

```bash
.venv/bin/python scripts/train_arm.py --gripper parallel --output out/rl/arm_friction3 --steps 0
.venv/bin/python scripts/train_arm.py --gripper parallel --output out/rl/arm_release --resume out/rl/arm_friction3/imitation.zip --demonstrations out/rl/arm_friction3/demonstrations.npz --correct-release --updates 750 --steps 8192
.venv/bin/python scripts/run_arm.py --gripper parallel --episodes 10 --seed 2000
.venv/bin/mjpython scripts/run_arm.py --gripper parallel --view
.venv/bin/python scripts/run_arm.py --gripper parallel --record --open
```

Checkpoints, raw before/after evaluation, and demonstrations land in the output
folder. Everything under `out/` is generated locally and git-ignored. No
checkpoint or replay on this page ships in the repository. The recorded replay
uses the loaded policy's actual actions and graph activations.
`scripts/run_arm.py` does not import the teacher.

The correction collector intervenes only while it generates training examples.
When a previously lifted cube reaches the destination and the tabletop, it
supplies open-jaw and retreat targets. That logic is absent from the
environment's action selection and from the saved-policy runner. The graph
learns the resulting commands. It must generate them alone at evaluation.

## Measured results

| Checkpoint | Seeds 3000–3009, full task success |
| --- | --- |
| Demonstrations and release corrections, before PPO | 2/10 |
| Same initialization, after PPO plus rehearsal | 10/10 |

The final policy also passed all ten starts on seeds 2000–2009. That is
**20/20** across the two sets. Final placement error across them was 6.2–10.7
mm.

This is one training seed on a narrow task distribution. It is not a general
grasping benchmark. Rehearsal accompanies PPO, so this comparison does not
isolate the effect of PPO from the extra imitation updates.

The saved checkpoint records 8,192 environment actions and 48 PPO training
epochs. That is 16 rollouts of three epochs each. Trainable parameter change
from the pre-PPO checkpoint is L2 0.7981. Graph-neuron gain change is L2 0.0734.
Raw metrics, checkpoint hashes, and per-episode results are in
[results/arm_rl.json](results/arm_rl.json).

The first run used no targeted release examples. It achieved 0/10 successes
after PPO. That result stays in
[results/arm_rl_initial_failed.json](results/arm_rl_initial_failed.json).

The replay in `out/arm_rl_demo/index.html` runs the final PPO checkpoint on seed
2000. It places the cube in 10.3 s, with a 16.0 cm maximum lift and 7.2 mm final
error. It displays measured neuron coordinates and actual policy activations.
The GIF is `out/arm_rl_demo/demo.gif`.

## Interactive neuron explorer

The replay uses a FlyJack-inspired control-room layout. The robot views and an
episode log sit on the left. The controller sits in the middle. An Explore panel
sits on the right. It is a standalone HTML file. It works offline after
recording and loads no rendering library from a CDN.

**Neuron cloud.** Drag to orbit, scroll to zoom, or expand to fullscreen. Click a
neuron, or search by exact FlyWire root ID, source class, transmitter, neuropil
group or policy role. Root IDs export as strings to avoid JavaScript integer
rounding. The panel reports the transmitter prediction and its score, source
classifications, side, flow, and source group. It also reports the synaptic hop,
the incoming and outgoing counts within this graph, and the strongest normalized
model-weight connections. Selecting a neuron highlights incoming links in blue
and outgoing links in gold.

Colour by policy role, source region, transmitter or activation. Filter by super
class. Switch between measured annotation positions and a conceptual
policy-interface layout.

**Circuit view.** The same activity groups into cell types, laid out by distance
from the policy's inputs. Columns are the median synaptic hop of each type's
neurons. Breadth-first search computes that hop from the 64 afferent input
neurons over the measured directed edges. The view draws the 30 types with the
highest total activation across the episode. It joins them with the 45 strongest
type-to-type links, ranked by summed absolute model weight. The layout is fixed
for the whole episode, so only the glow moves.

Node fill tracks each type's mean activation at the current frame, scaled
against the strongest type's episode peak. Clicking a type selects it, lists its
neurons, and highlights it in the neuron cloud. Cell types use FlyWire's class
annotation where one exists. Otherwise they use a super-class code plus neuropil
group (`CB · AVLP`, `DN · GNG`). The Circuit tab discloses that rule, because
325 of the 512 neurons carry no class annotation. Hop columns describe the
measured wiring, not the four propagation rounds the model runs.

**Explanation.** Twenty-two neuroscience and machine-learning terms are hoverable
and keyboard-focusable for a definition. A five-step guided tour opens once per
browser and replays from the header. The Science tab carries the measured
results below. It states that this arm experiment has no matched
conventional-network control run. It also states that the navigation
experiment's control run showed no advantage from the fly wiring.

The episode log beside the robot marks grasp, lift, arrival, release and
confirmation times. All come from the recorded task state, not from a script.

Scrub the replay to inspect any frame. Each frame shows the activity trace,
robot motion, 20 policy observations, and four action outputs. The values share
one recorded action step. Neuron metadata, per-neuron synaptic hops, cell-type
assignments and all 8,688 graph edges download as JSON from How it works.

Limits on the displayed numbers:

- Activities are mean absolute values across four artificial channels, not Hz or
  spike counts.
- Source annotation points represent neurons, not their complete morphologies.
- Connection weights are signed, incoming-normalized model weights, not raw
  synapse counts.
- The 64 input and 64 output roles are engineered adapters.
- Source biological classifications appear separately, and missing source
  annotations stay marked as unannotated.

The interface credits [FlyJack](https://fanpu.io/games/flyjack/) as its design
reference.

Headless Microsoft Edge drove the generated replay over the DevTools protocol.
It asserted playback, scrubbing, exact root-ID search, and neighbour navigation.
It asserted the circuit layout, its per-frame glow, and cell-type selection
carrying into the neuron cloud. It also asserted group filtering, every colour
mode, the glossary, all five tabs, the first-visit tour, and image loading. It
found no horizontal overflow at 1600 px or 400 px, and no JavaScript errors. The
saved record is `out/arm_rl_demo/explorer_checks.json`.

## Simulation scope

This task uses one 48 mm cube. Its XY position is randomized by ±8 mm. The
destination is 17 cm to the right. The arm starts above the cube. The base is
fixed. The parallel-jaw simulation variant replaces the original hooked jaws.

For this continuous-control experiment pad sliding friction is **3.0**. The
scripted baseline uses 1.2. This change prevented observed transfer slips in the
demonstration check. It is not a measured hardware coefficient.

The cube remains a free body. Only reset assigns its pose. There is no object
attachment, no hidden grasp assist, and no scripted recovery during policy
execution. Robot self-collision is still filtered in the supplied model.

These remain unvalidated: camera perception, base balancing during manipulation,
varying destinations, other objects, and hardware performance.
