# Project verification

- On Windows, use `.venv\Scripts\python.exe` directly; shell activation is unnecessary.
- Regression suite: `.venv\Scripts\python.exe -m unittest discover -s tests -v`. The bare-gripper grasp test is an expected failure, not proof that the original collision hulls work.
- Navigation unit checks: `.venv\Scripts\python.exe scripts/check_navigation.py`.
- Navigation simulation: `.venv\Scripts\python.exe scripts/navigate.py` runs all nine routes using simulator truth, not ROS localization.
- Regenerate models with `scripts/build_mjcf.py`, then `scripts/build_room.py`. The latter writes both room XML files before checking clearance and reachability.
- Keep `models/bracketbot/chopped_urdf_v2.urdf` and the mesh assets unchanged. Matching-color contact pads are allowed; do not replace the default demo fingers or hinge joints.
- Keep navigation's room solver impedance ratio at 10. Manipulation selects 200 in its own model; setting 200 globally regresses navigation.
- The required stations are red ball/box (`ball`, ACT), colored cubes (`cubes`, Fable), and blue cube/rectangle (`pick`, Flybrain). Crockery is not in the default room.
- Demos, graph data, checkpoints and replay outputs live in ignored `out/`; they are not included in a Git push. Checkpoint/demonstration metadata must match the current environment.
- `scripts/orchestrate.py --planner sweep` explicitly validates Fable's skills without API access. Real Fable needs `requirements-agent.txt` and a locally configured `ANTHROPIC_API_KEY`; never put a key in a tracked file.
- Do not equate a supplied recognition label or simulator metadata with camera-based perception, or the separate fixed-base demo with a continuous navigation/manipulation handoff.
