# Demo assets

| File | What it is |
| --- | --- |
| `ACT.mov` | Screen recording of ACT on the red-ball table. 23 s, 2268x1528, H.264. Open it in any video player. |
| `index.html` | Template for the fly-brain navigation replay. `scripts/record_brain_demo.py` fills it and writes `out/demo/index.html`. |
| `arm_rl.html` | Template for the arm-policy neuron explorer. `scripts/run_arm.py --record` fills it and writes `out/arm_rl_demo/index.html`. |
| `pick_place.html` | Template for the scripted pick-and-place replay. `scripts/pick_place.py --record` fills it and writes `out/pick_place/index.html`. |

The three HTML files are templates, not replays. Each holds a placeholder
(`__DEMO_DATA__`, `__ARM_DATA__`, `__PICK_PLACE_DATA__`) that the recorder
replaces with a run's data. Open a template directly and it shows an empty
page. The filled pages land in `out/`, which git ignores.

To run the live demo instead:

```bash
.venv/bin/python scripts/demo.py --planner sweep "clean the cubes, then pick up the blue cube, then go to the ACT table"
.venv/bin/mjpython scripts/demo.py --view          # with a window
```

The recorders need their checkpoints. `pick_place.py --record` runs from a
fresh clone. The other two need the training steps in
[`../docs/brain_demo.md`](../docs/brain_demo.md) and
[`../docs/arm_rl.md`](../docs/arm_rl.md).
