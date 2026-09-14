# Demo assets

Standalone artefacts for showing the project without running it. Every HTML page
is self-contained — open it in a browser, no server and no build step.

| File | What it shows |
| --- | --- |
| [`index.html`](index.html) | **Neural Drive** — the 512-neuron FlyWire graph driving the balancing robot to point goals, with the neuron cloud animating alongside the robot. |
| [`pick_place.html`](pick_place.html) | **Pick and place** — the scripted contact-based baseline on the parallel-jaw gripper, fixed base. |
| [`arm_rl.html`](arm_rl.html) | **Neural control room** — the arm policy replayed beside an orbitable neuron cloud, a circuit view by synaptic distance, and an Explore panel for looking up any FlyWire root ID. |
| `ACT.mov` | Screen recording of ACT running on the red-ball table. 23 s, 2268x1528, H.264. |

These are captures and replays, not live simulations. To run the real thing:

```bash
.venv/bin/python scripts/demo.py --planner sweep "clean the cubes, then pick up the blue cube, then go to the ACT table"
.venv/bin/mjpython scripts/demo.py --view          # with a window
```

`index.html` and `arm_rl.html` were generated from runs whose checkpoints live in
ignored `out/`; regenerating them needs the training steps in
[`../docs/brain_demo.md`](../docs/brain_demo.md) and
[`../docs/arm_rl.md`](../docs/arm_rl.md).
