"""Render the cameras for recorded demonstrations, after the fact.

    .venv/bin/python scripts/render_demos.py out/demos/ball
    .venv/bin/python scripts/render_demos.py out/demos/ball/ep_*.npz --size 224 --preview

Camera frames are a pure function of the pose, so the recorders keep only the
pose and this puts the pictures back: every row of every episode is posed in
the room and shot from the head camera and both wrist cameras.  Rendering at
recording time would have tied the frame size and camera set to a policy that
did not exist yet, and costs more than the sim step itself.

Rows are posed by *name*, not by copying the recorded state vector back.  The
room gets rebuilt, and a rebuild can move objects about in the state vector or
take their joints away; an episode recorded before that would then pose the
wrong thing.  So the robot's joints go in by joint name and every object on the
table by body name, from the poses the recorder kept, and the state vector's
layout at the time of recording does not matter.

Writes `<episode>_frames.npz` next to each episode: one uint8 array per camera
of shape (rows, size, size, 3), plus the cameras' vertical fields of view.
`--preview` also drops a PNG contact sheet of the first, middle and last rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlbot.skills import Robot                                       # noqa: E402
from rlbot.teleop import Poser, load_demo                            # noqa: E402

CAMERAS = ("head_cam", "wrist_right_cam", "wrist_left_cam")


def episodes(paths) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.glob("ep_*.npz")
                          if not q.name.endswith("_frames.npz"))
        elif not p.name.endswith("_frames.npz"):
            out.append(p)
    return out


def render(robot: Robot, meta: dict, arrays: dict, size: int, cameras) -> dict:
    """One array per camera, rows posed one after another in the same model."""
    model = robot.model
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, size)
    model.vis.global_.offheight = max(model.vis.global_.offheight, size)
    renderer = mujoco.Renderer(model, size, size)
    pose = Poser(robot, meta)
    n = len(arrays["qpos"])
    frames = {cam: np.empty((n, size, size, 3), dtype=np.uint8) for cam in cameras}
    for i in range(n):
        pose(arrays["qpos"][i], arrays["obj_pos"][i], arrays["obj_quat"][i])
        for cam in cameras:
            renderer.update_scene(robot.data, camera=cam)
            frames[cam][i] = renderer.render()
    renderer.close()
    return frames


def contact_sheet(frames: dict, path: Path) -> None:
    """First, middle and last rows, one line per camera."""
    from PIL import Image        # in the venv; used only for this preview
    n = next(iter(frames.values())).shape[0]
    picks = sorted({0, n // 2, n - 1})
    rows = [np.concatenate([frames[cam][i] for i in picks], axis=1)
            for cam in frames]
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="episode files or directories")
    ap.add_argument("--size", type=int, default=224, help="pixels, square")
    ap.add_argument("--cameras", nargs="+", default=list(CAMERAS))
    ap.add_argument("--preview", action="store_true",
                    help="also write a PNG contact sheet per episode")
    ap.add_argument("--force", action="store_true",
                    help="re-render episodes that already have frames")
    args = ap.parse_args()

    robots: dict[tuple, Robot] = {}
    for path in episodes(args.paths):
        out = path.with_name(path.stem + "_frames.npz")
        if out.exists() and not args.force:
            print(f"{path.name}: frames exist, skipping", flush=True)
            continue
        meta, arrays = load_demo(path)
        key = (meta["table"].split("_")[1], meta["balancing"])
        if key not in robots:
            robots[key] = Robot(key[0], balancing=key[1])
        robot = robots[key]

        frames = render(robot, meta, arrays, args.size, args.cameras)
        fovy = {cam: float(robot.model.cam_fovy[robot.model.camera(cam).id])
                for cam in args.cameras}
        np.savez_compressed(out, fovy=np.array([fovy[c] for c in args.cameras]),
                            cameras=np.array(args.cameras), **frames)
        print(f"{path.name}: {len(arrays['qpos'])} rows x {len(frames)} cameras "
              f"at {args.size}px -> {out.name}", flush=True)
        if args.preview:
            contact_sheet(frames, path.with_name(path.stem + "_preview.png"))


if __name__ == "__main__":
    main()
