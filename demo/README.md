# Demo recordings

Every file here plays as-is. No install, no browser page, no `out/`.

| File | What it shows | Length |
| --- | --- | --- |
| `tour.mov` | The full demo, one continuous simulation: A* drives the balancing robot to the cubes table, the Fable skills agent crates all four cubes, it drives to the pick table, the fly-brain arm policy picks and places the blue cube, it drives to the ball table, ACT tries the ball and knocks it to the floor. Over-the-shoulder camera, 960x720, 30 fps. | 7 min 48 s |
| `ACT.mov` | Screen recording of ACT on the red-ball table. 2268x1528. | 23 s |
| `pick_place.gif` | The scripted contact-based pick-and-place baseline on the fixed base. 1 of 1 picked, placed, and released. 640x480. | 18 s |
| `fly_brain_point_goal_pilot.gif` | The separate point-goal pilot: the 512-neuron FlyWire graph as the driving policy, neuron activity beside the room view. Reaches the goal in 6.5 s. **Not** the navigator in `tour.mov`; that is A*. 1120x560. | 7 s |

How each was made:

```bash
.venv/bin/python scripts/demo.py --planner sweep --video demo/tour.mov "clean the cubes, then pick up the blue cube, then go to the ACT table"
.venv/bin/python scripts/pick_place.py --record --output out/pick_place            # writes demo.gif
.venv/bin/python scripts/train_connectome.py --policy connectome --graph checkpoints/graph_512.npz --output out/rl/connectome_demo
.venv/bin/python scripts/record_brain_demo.py --checkpoint out/rl/connectome_demo/policy.zip --graph checkpoints/graph_512.npz --output out/demo   # writes demo.gif
```

`ACT.mov` is a manual screen capture of `scripts/demo.py --view`.

There is no fixed-base recording of the fly-brain arm policy.
`scripts/run_arm.py --record` refuses the shipped checkpoint because the room
and robot XML changed after it was trained, and it checks their hashes. The
live demo runs the same checkpoint without that gate; `tour.mov` shows it
picking the blue cube.
