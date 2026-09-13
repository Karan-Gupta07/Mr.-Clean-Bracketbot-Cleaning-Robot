"""Play recorded demonstrations back in the MuJoCo viewer.

    .venv/bin/mjpython scripts/replay_demo.py out/demos/ball/ep_20260912_221119.npz
    .venv/bin/mjpython scripts/replay_demo.py out/demos/ball --speed 2      # every episode there
    .venv/bin/mjpython scripts/replay_demo.py out/demos/ball --loop

mjpython, not python: the window has to be made on the main thread.

Nothing is simulated.  Each recorded row is posed by joint name and object
name, the way the renderer does it, and the window is synced at the recording
rate.  The camera sits over the robot's shoulder looking at the table.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlbot.skills import Robot                                       # noqa: E402
from rlbot.teleop import Poser, load_demo                            # noqa: E402


def episodes(paths) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.glob("ep_*.npz")
                          if not q.name.endswith("_frames.npz"))
        else:
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="episode files or directories")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true", help="start over at the end")
    args = ap.parse_args()

    todo = episodes(args.paths)
    meta, _ = load_demo(todo[0])
    robot = Robot(meta["table"].split("_")[1], balancing=meta["balancing"])
    model, data = robot.model, robot.data

    with mujoco.viewer.launch_passive(model, data) as viewer:
        rot = data.body("root").xmat.reshape(3, 3)
        viewer.cam.lookat[:] = (data.body("root").xpos
                                + rot[:, 0] * 0.45 + np.array([0, 0, 0.3]))
        viewer.cam.distance = 1.6
        viewer.cam.elevation = -25
        viewer.cam.azimuth = math.degrees(math.atan2(rot[1, 0], rot[0, 0])) + 150
        while viewer.is_running():
            for path in todo:
                meta, arrays = load_demo(path)
                pose = Poser(robot, meta)
                dt = 1 / (meta["control_hz"] * args.speed)
                print(f"{path.name}: {meta['task']}, {meta['arm']} arm, "
                      f"{meta['seconds']:.0f}s, {'success' if meta['success'] else 'failed'}",
                      flush=True)
                for i in range(len(arrays["qpos"])):
                    if not viewer.is_running():
                        return
                    tick = time.time()
                    pose(arrays["qpos"][i], arrays["obj_pos"][i],
                         arrays["obj_quat"][i])
                    viewer.sync()
                    slack = dt - (time.time() - tick)
                    if slack > 0:
                        time.sleep(slack)
                time.sleep(0.5)
            if not args.loop:
                print("done - close the window to finish", flush=True)
                while viewer.is_running():
                    time.sleep(0.1)
                return


if __name__ == "__main__":
    main()
