# Fly-connectivity robot prototype

The robot's state and goal enter a learned input adapter. Signals pass through a
graph built from measured fly neuron connections. A learned output adapter turns
the resulting activity into a desired speed and yaw rate. A PD controller keeps
the BracketBot upright. Imitation initializes the policy. PPO with demonstration
rehearsal then updates it in MuJoCo.

This is a connectivity experiment, not a biological brain simulation. The trained
pilot uses **512 mapped neurons and 8,688 directed connections** from FlyWire
FAFB v783. The complete 139,255-neuron / 2,700,513-edge graph is also prepared,
and it passed a forward/backward benchmark. It has **not** been trained. Edge
counts here are neuron pairs, not individual synapse counts.

FlyBody is a reference, not a dependency. Its repository supplies a simulated fly
body and artificial controllers, not the brain wiring used here. We keep the
BracketBot body. We take the connectivity from
[FlyWire](https://www.nature.com/articles/s41586-024-07558-y). Source URLs and
SHA-256 checksums are saved beside the generated graph.

## The replay page

`scripts/record_brain_demo.py` writes `out/demo/index.html`. Open it in a
browser.

```bash
open out/demo/index.html
```

Everything under `out/` is generated locally and ignored by git. A fresh clone
has none of it. Two replay page templates ship instead: `demo/index.html` for this
navigation demo, and `demo/arm_rl.html` for the arm policy. The recorder fills the
`__DEMO_DATA__` placeholder in `demo/index.html`.

The page pairs the room render, the head camera, and the recorded policy activity
at each simulation timestamp. Play and pause it. Scrub the timeline. Drag the
neuron graph to rotate it. Hover a neuron to read its source ID.

What the page shows:

- The page draws the 500 strongest edges. The policy uses all 8,688.
- The GIF draws 200 edges for clarity.
- Activity colors are mean absolute artificial activation on a fixed 0–1 scale.
- Coordinates are annotated points on source neurons. They are not reconstructed
  morphology or guaranteed soma positions.
- This is **recorded replay**, not a live neural recording.
- The head camera is display-only.
- Terminal-frame commands are predictions. Nothing applies them after the episode
  ends.

The recorder writes three more files. `out/demo/demo.gif` is a shareable
animation. `episode.json` holds timestamps, states, actions, activities, and a
checkpoint hash. `report.json` holds the final outcome, dataset provenance,
target, and seed.

## Rebuild from a fresh checkout

Run every command from the repository root. This pilot runs with Python 3.12, CPU
PyTorch 2.14.0, MuJoCo 3.13.0, Gymnasium 1.3.0, and Stable-Baselines3 2.9.0. No
CUDA GPU is required.

Create the environment.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-rl.txt
```

Download about 58 MB of public source tables, build the graph, train, and record
the demo.

```bash
.venv/bin/python scripts/prepare_connectome.py
.venv/bin/python scripts/train_connectome.py --policy connectome --output out/rl/connectome_final
.venv/bin/python scripts/train_connectome.py --policy mlp --output out/rl/mlp_final
.venv/bin/python scripts/evaluate_navigation.py out/rl/connectome_final/imitation.zip out/rl/connectome_final/policy.zip out/rl/mlp_final/imitation.zip out/rl/mlp_final/policy.zip
.venv/bin/python scripts/record_brain_demo.py
open out/demo/index.html
```

CPU wheel installation differs by platform. On Windows, replace
`.venv/bin/python` with `.venv\Scripts\python.exe`. On Linux without a display,
MuJoCo rendering needs an EGL or OSMesa setup.

To change the target, rerun the recorder.

```bash
.venv/bin/python scripts/record_brain_demo.py --goal 0.8 -0.4 --yaw 0 --seed 2026 --seconds 20 --output out/demo_other
open out/demo_other/index.html
```

The policy has no obstacle avoidance. Choose clear routes near the room center.
It is not trained to drive around the tables or the divider.

## Measured results

One training seed (7). 40 scripted teacher episodes give 4,279 transitions. Then
800 imitation updates. Then 8,192 PPO transitions, with 32 demonstration
rehearsal updates per rollout. Both architectures use the same task and the same
training settings. Their parameter counts differ.

A run succeeds when it meets every threshold below.

| Success condition | Threshold |
| --- | --- |
| Distance to goal | under 0.12 m |
| Speed | under 0.06 m/s |
| Yaw rate | under 0.2 rad/s |
| Hold time | 0.5 seconds |
| Episode limit | 15 seconds |

| Policy | Development seeds 1000–1019 | Separate evaluation seeds 2000–2019 | Falls in separate evaluation |
| --- | --- | --- | --- |
| Fly graph, imitation only | 20/20 | 20/20 | 0 |
| Fly graph, PPO + rehearsal | 17/20 | 17/20 | 0 |
| MLP, imitation only | 18/20 | 18/20 | 0 |
| MLP, PPO + rehearsal | 20/20 | 20/20 | 0 |

**The graph works as a robot controller. This run does not demonstrate a benefit
from fly wiring, or from RL over its imitation initialization.** The development
seeds were reused while adjusting PPO. The separate seeds were evaluated
afterwards, with no further training changes. Multiple independent training seeds
and topology ablations are still needed.

An earlier plain-PPO graph run deteriorated from 10/10 to 0/10 goals. A lower
value-loss weight and demonstration rehearsal limited that forgetting. Both
learned node gains and adapters change during training. Gain changes include PPO
and rehearsal updates, so they do not isolate an RL effect. Raw results, including
the failed experiment, are in [results/](results/).

The room demo uses a 20-second limit. The scored evaluations use 15 seconds. The
first 15-second attempt stopped 0.118 m from the goal. It did not satisfy the
full success hold before the timeout. Its report stays in
`results/room_demo_15s.json`. Read the generated demo report for the current
outcome. Do not assume that every rollout succeeds.

The final recorded run reached the target. It satisfied the stop condition at
15.3 seconds, 0.115 m from the goal. The demo uses the post-PPO graph checkpoint,
not the stronger imitation-only checkpoint.

The full graph benchmark measured about 1.2 seconds per batch-1 feature forward
pass on this CPU, during other local work. The control budget is 50 ms, so the
graph is far too slow. The benchmark uses random parameters. It is a feasibility
measurement, not a trained policy and not a GPU benchmark. Rebuild and profile
it.

```bash
.venv/bin/python scripts/prepare_connectome.py --neurons 0
.venv/bin/python scripts/benchmark_connectome.py
```

## What is implemented and what remains

- The edges are fixed and measured. Each node carries four artificial channels.
  Each action runs four propagation rounds. Training changes the input and output
  adapters, the per-node gains and biases, and the shared channel mixing. No
  hidden neural state persists between robot steps.
- Transmitter signs are simplified. GABA and GLUT are negative. All others are
  positive. This is an engineering assumption. A wiring diagram does not recover
  cellular dynamics or receptor effects.
- The navigation policy runs at 20 Hz. It issues bounded speed and turn requests
  under acceleration limits, above the existing 500 Hz balancer. Two control bugs
  were corrected: turning feedback had the wrong sign, and pitch measurement
  depended on yaw.
- Observations come from simulator pose and velocity. SLAM, camera inputs,
  obstacle avoidance, hardware watchdogs, arm control, and whole-room cleanup stay
  outside this pilot. The demo is not evidence of hardware readiness.
- Full-graph training needs throughput work, memory work, and suitable compute.
  Before scaling, compare multiple seeds and randomized-graph controls. Also
  address the observed drop after PPO. Then replace privileged state with
  estimated sensors, and train obstacle-aware tasks.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/evaluate.py --seconds 10 --push 300
```

`tests/test_navigation_connectome.py` holds the four regressions for this pilot:

- heading-independent pitch
- stable turn direction
- deterministic resets and a teacher goal
- differentiability and stateless batching of the measured graph

The push test stayed upright for 10 seconds, with 4.56 degrees maximum lean. The
HTML was rendered and inspected in Microsoft Edge.

## Attribution

The data stays attributed to the
[FlyWire release](https://zenodo.org/records/10676866). Source and derived data
are downloaded separately, not relicensed as project code. No FlyBody or FlyGM
implementation or checkpoint was copied. The requested X clip could not be
retrieved. This dashboard is therefore an original provisional layout, not a
verified reproduction of that video.
