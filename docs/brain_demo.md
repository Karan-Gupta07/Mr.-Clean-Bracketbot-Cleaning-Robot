# Fly-connectivity robot prototype

The robot's state and goal enter a learned input adapter. Signals pass through a
graph built from measured fly neuron connections. A learned output adapter turns
the resulting activity into desired speed and yaw rate. A PD controller keeps the
BracketBot upright. Imitation initializes the policy; PPO with demonstration
rehearsal then updates it in MuJoCo.

This implements a connectivity experiment, not a biological brain simulation.
The trained pilot uses **512 mapped neurons and 8,688 directed connections** from
FlyWire FAFB v783. The complete 139,255-neuron / 2,700,513-edge graph has also been
prepared and passed a forward/backward benchmark, but has **not** been trained.
Edge counts here are neuron pairs, not individual synapse counts.

FlyBody is a reference, not a dependency: its repository supplies a simulated fly
body and artificial controllers, not the brain wiring requested here. We retain
the BracketBot body and obtain actual connectivity from
[FlyWire](https://www.nature.com/articles/s41586-024-07558-y), with source URLs and
SHA-256 checksums saved beside the generated graph. See the
[research plan](fly_connectome_rl_plan.md) for the later milestones.

## Run the demo already generated on this machine

From the repository root in PowerShell:

```powershell
Start-Process .\out\demo\index.html
```

The standalone HTML pairs the room render, head camera, and recorded policy
activity at each simulation timestamp. Play/pause, scrub the timeline, drag the
neuron graph to rotate, and hover to inspect source neuron IDs. Only 500 strongest
edges are drawn; the policy uses all 8,688. The GIF draws 200 edges for clarity.
Activity colors represent mean absolute artificial activation on a fixed 0–1
scale. Coordinates are annotated points on source neurons, not reconstructed
morphology or guaranteed soma positions. This is **recorded replay**, not a live
neural recording; the head camera is display-only. Terminal-frame commands are
predictions and are not applied after the episode ends.

`out/demo/demo.gif` is a shareable animation. `episode.json` includes timestamps,
states, actions, activities, and a checkpoint hash. `report.json` records the final
outcome, dataset provenance, target, and seed. Generated datasets, checkpoints,
and videos are local under ignored `out/`; they are not part of the Git push.

## Rebuild from a fresh checkout

Tested here with Windows, Python 3.12, CPU PyTorch 2.14.0, MuJoCo 3.13.0,
Gymnasium 1.3.0, and Stable-Baselines3 2.9.0. No CUDA GPU is required for the pilot.
Run from the repository root. For a new environment:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements-rl.txt
```

Download approximately 58 MB of public source tables, build the graph, train, and
record the demo:

```powershell
.venv\Scripts\python.exe scripts/prepare_connectome.py
.venv\Scripts\python.exe scripts/train_connectome.py --policy connectome --output out/rl/connectome_final
.venv\Scripts\python.exe scripts/train_connectome.py --policy mlp --output out/rl/mlp_final
.venv\Scripts\python.exe scripts/evaluate_navigation.py out/rl/connectome_final/imitation.zip out/rl/connectome_final/policy.zip out/rl/mlp_final/imitation.zip out/rl/mlp_final/policy.zip
.venv\Scripts\python.exe scripts/record_brain_demo.py
Start-Process .\out\demo\index.html
```

On Linux/macOS, substitute `.venv/bin/python` and open the resulting HTML in a
browser. CPU wheel installation differs by platform; the exact Windows setup
above was the one exercised in this session. On Linux without a display, MuJoCo
rendering requires an available EGL or OSMesa setup.

To change the target, rerun the recorder, for example:

```powershell
.venv\Scripts\python.exe scripts/record_brain_demo.py --goal 0.8 -0.4 --yaw 0 --seed 2026 --seconds 20 --output out/demo_other
Start-Process .\out\demo_other\index.html
```

The policy has no obstacle avoidance. Choose clear routes near the room center;
it is not trained to navigate around tables or the divider.

## Measured results

One training seed (7), 40 scripted teacher episodes (4,279 transitions), 800
imitation updates, then 8,192 PPO transitions with 32 demonstration rehearsal
updates per rollout. Both architectures use the same task and training settings;
their parameter counts differ. Success requires distance under 0.12 m, speed
under 0.06 m/s, and yaw rate under 0.2 rad/s for 0.5 seconds, within 15 seconds.

| Policy | Development seeds 1000–1019 | Separate evaluation seeds 2000–2019 | Falls in separate evaluation |
| --- | --- | --- | --- |
| Fly graph, imitation only | 20/20 | 20/20 | 0 |
| Fly graph, PPO + rehearsal | 17/20 | 17/20 | 0 |
| MLP, imitation only | 18/20 | 18/20 | 0 |
| MLP, PPO + rehearsal | 20/20 | 20/20 | 0 |

**The graph works as a robot controller, but this run does not demonstrate a
benefit from fly wiring or from RL over its imitation initialization.** The
development seeds were reused while adjusting PPO; the separate seeds were
evaluated afterwards without further training changes. Multiple independent
training seeds and topology ablations are still needed.

An earlier plain-PPO graph run deteriorated from 10/10 to 0/10 goals. Reducing
the value-loss weight and retaining demonstration rehearsal limited that
forgetting. Both learned node gains and adapters change during training; gain
changes include PPO and rehearsal updates, so they do not isolate an RL effect.
Raw results, including the failed experiment, are in [results/](results/).

The displayed room demo uses a 20-second limit, separate from the 15-second
scored evaluations. The first 15-second attempt stopped 0.118 m from the goal but
did not satisfy the complete success hold before timeout; its report is retained
in `results/room_demo_15s.json`. Consult the generated demo report for the current
outcome rather than assuming every rollout succeeds.

The final recorded run reached the target and satisfied the stop condition at
15.3 seconds (0.115 m final distance). The demo uses the actual post-PPO graph
checkpoint, not the stronger imitation-only checkpoint.

The full graph benchmark measured about 1.2 seconds per batch-1 feature forward
pass on this CPU during other local work, well above the 50 ms control budget.
It is a feasibility measurement with random parameters, not a trained policy or
a GPU benchmark. To rebuild and profile it:

```powershell
.venv\Scripts\python.exe scripts/prepare_connectome.py --neurons 0
.venv\Scripts\python.exe scripts/benchmark_connectome.py
```

## What is implemented and what remains

- Fixed measured edges; four artificial channels per node; four propagation
  rounds per action. Trainable input/output adapters, per-node gains/biases, and
  shared channel mixing. No hidden neural state persists between robot steps.
- Simplified transmitter signs: GABA/GLUT negative, others positive. This is an
  engineering assumption; actual cellular dynamics and receptor effects are not
  recovered from a wiring diagram.
- A 20 Hz navigation policy issuing bounded speed/turn requests, with acceleration
  limits and the existing 500 Hz balancer. Two control bugs were corrected:
  turning feedback had the wrong sign, and pitch measurement depended on yaw.
- Simulator pose and velocity supply observations. SLAM, camera inputs, obstacle
  avoidance, hardware watchdogs, arm control, and whole-room cleanup remain
  outside this pilot. The demo is not evidence of hardware readiness.
- Full-graph training needs throughput/memory work and suitable compute. Before
  scaling, compare multiple seeds and randomized-graph controls and address the
  observed drop after PPO. Then replace privileged state with estimated sensors
  and train obstacle-aware tasks.

Verification commands:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts/evaluate.py --seconds 10 --push 300
```

The four regressions cover heading-independent pitch, stable turn direction,
deterministic resets and a teacher goal, and differentiability/stateless batching
of the measured graph. The push test remained upright for 10 seconds, with
4.56 degrees maximum lean. The HTML was rendered and inspected in Microsoft Edge.

Data remains attributed to the [FlyWire release](https://zenodo.org/records/10676866).
Source/derived data is downloaded separately rather than relicensed as project
code. No FlyBody or FlyGM implementation/checkpoint was copied. The requested X
clip could not be retrieved, so this dashboard is an original provisional layout,
not a verified reproduction of that video.
