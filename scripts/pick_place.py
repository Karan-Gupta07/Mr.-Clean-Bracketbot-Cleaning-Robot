"""Run contact-based pick/place and optionally record or view the simulation."""
from __future__ import annotations
import argparse
import base64
import io
import json
from pathlib import Path
import sys
import time
import webbrowser

import mujoco
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from rlbot.manipulation import PickPlace, make_model


def jpeg(pixels):
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode()


def camera(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def record(task, output):
    frames, gif = [], []
    task.model.vis.map.znear = 0.003 / task.model.stat.extent
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    overview = camera([-.26, -1.60, 1.0], 1.7, -60, -25)
    detail = camera([-.28, -1.72, .79], .65, 100, -30)
    with mujoco.Renderer(task.model, height=480, width=640) as renderer:
        step = 0
        while True:
            if step % 50 == 0 or task.finished:
                images = []
                for cam in (overview, detail):
                    renderer.update_scene(task.data, camera=cam, scene_option=options)
                    scene = renderer.scene
                    # Destination outline is a visual marker, not physical support.
                    for axis in range(2):
                        for sign in (-1, 1):
                            pos = task.goal + [0, 0, .003]
                            pos[axis] += sign * .038
                            size = np.array([.038, .038, .001])
                            size[axis] = .001
                            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
                                               size, pos, np.eye(3).ravel(), np.array([.2,1,.6,1],dtype=np.float32))
                            scene.ngeom += 1
                    images.append(renderer.render().copy())
                frames.append({**task.report(), "overview": jpeg(images[0]), "detail": jpeg(images[1]),
                               "qpos": task.data.qpos.tolist()})
                gif.append(Image.fromarray(images[0]))
            if task.finished:
                break
            task.step()
            step += 1
    report = task.report()
    payload = json.dumps({"report": report, "frames": frames}, separators=(",", ":"), allow_nan=False)
    output.mkdir(parents=True, exist_ok=True)
    (output / "episode.json").write_text(payload)
    template = (REPO / "demo/pick_place.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(template.replace("__PICK_PLACE_DATA__", payload), encoding="utf-8")
    gif[0].save(output / "demo.gif", save_all=True, append_images=gif[1:], duration=100, loop=0)
    gif[-1].save(output / "preview.png")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--item", choices=["cube_s", "cube_m", "cube_l", "cube_xl"], default="cube_m")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--open", action="store_true", help="Open recorded HTML after completion")
    parser.add_argument("--view", action="store_true", help="Run a live MuJoCo viewer")
    parser.add_argument("--gripper", choices=["padded", "urdf", "parallel"], default="padded",
                        help="Gripper model: 'padded' is the supplied hooked gripper with contact pads; 'urdf' is that gripper untouched, which does not grasp; 'parallel' is the sliding-jaw substitution the recorded RL results used")
    parser.add_argument("--output", type=Path, default=Path("out/pick_place"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.open and not args.record:
        parser.error("--open requires --record")
    if args.record and args.view:
        parser.error("Choose either --record or --view")
    args.output.mkdir(parents=True, exist_ok=True)
    model = make_model(args.item, args.gripper)
    reports = []
    for seed in range(args.seed, args.seed + args.episodes):
        task = PickPlace(seed=seed, model=model, item_name=args.item, gripper=args.gripper)
        if args.record:
            report = record(task, args.output if args.episodes == 1 else args.output / str(seed))
        elif args.view:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(task.model, task.data) as viewer:
                viewer.cam.lookat[:] = [-.26, -1.60, 1.0]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.7, -60, -25
                viewer.opt.geomgroup[3] = 0
                start = time.monotonic()
                while viewer.is_running() and not task.finished:
                    task.step()
                    viewer.sync()
                    delay = task.data.time - (time.monotonic() - start)
                    if delay > 0:
                        time.sleep(delay)
            report = task.report()
        else:
            while not task.finished:
                task.step()
            report = task.report()
        reports.append(report)
        print(json.dumps(report), flush=True)
    result = {"episodes": len(reports), "successes": sum(r["success"] for r in reports), "runs": reports}
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n", newline="\n")
    if args.open and args.episodes == 1:
        webbrowser.open((args.output / "index.html").resolve().as_uri())
    print(f"{result['successes']}/{result['episodes']} picked, placed, and released", flush=True)
    return 0 if result["successes"] == result["episodes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
