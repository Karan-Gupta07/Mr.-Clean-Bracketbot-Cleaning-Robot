"""Inspect the room and its three cameras with physics held at a fixed pose.

Windows/Linux: python scripts/view_cameras.py
macOS: mjpython scripts/view_cameras.py
Optional comparison image (requires Pillow): --snapshot out/camera_views.png
Keys: 0 = overview, 1 = head, 2 = left wrist, 3 = right wrist.
With --pose approach: P = parked arms, A = approach pose (static IK preview).
With --near-clip 0.005: C toggles the original and 5 mm rendering cutoff.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import queue
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


REPO = Path(__file__).resolve().parents[1]
CAMERAS = ("head_cam", "wrist_left_cam", "wrist_right_cam")
sys.path.insert(0, str(REPO / "src"))


def prepare_approach(model, data, args):
    from rlbot.arm import ArmIK, FOLLOWER, GRIPPER, Gripper, down_quat
    from rlbot.room import GRASP_YAWS, TABLES, grasp_pose

    matches = [(table, item) for table in TABLES for item in table.items if item.name == args.item]
    if not matches:
        raise ValueError(f"Unknown item: {args.item}")
    table, item = matches[0]
    expected_key = f"dock_{table.name.removeprefix('table_')}"
    if args.keyframe != expected_key:
        raise ValueError(f"Use --keyframe {expected_key} for {args.item}")
    ik, hand = ArmIK(model, args.arm), Gripper(model, args.arm)
    parked = data.qpos.copy()
    opening = hand.opening_for(item.width)
    target = grasp_pose(item, table) + [0, 0, args.approach_height]
    best = None
    for yaw in GRASP_YAWS:
        data.qpos[:] = parked
        for joint in (GRIPPER[args.arm], FOLLOWER[args.arm]):
            data.qpos[model.jnt_qposadr[model.joint(joint).id]] = opening
        mujoco.mj_forward(model, data)
        quat = down_quat(table.dock[2] + yaw)
        solution = ik.solve(data, hand.site_target(target, quat, opening), quat)
        if solution.ok:
            best = solution
            break
    if best is None:
        data.qpos[:] = parked
        raise RuntimeError(f"No valid {args.arm} arm approach found for {args.item}")
    mujoco.mj_forward(model, data)
    print(f"Static IK preview: {args.arm} arm -> {args.item}, "
          f"{args.approach_height * 100:.0f} cm above grasp target; "
          f"position error {best.pos_err * 1000:.2f} mm, "
          f"orientation error {np.degrees(best.rot_err):.2f} deg. "
          "Motion path and balance are not evaluated.", flush=True)
    return data.qpos.copy()


def overview(model, data):
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = data.body("root").xpos + [0, 0, 0.8]
    camera.distance = 2.6
    camera.azimuth = float(np.degrees(np.arctan2(camera.lookat[1], camera.lookat[0])))
    camera.elevation = -25
    return camera


def save_snapshot(model, data, path, keyframe):
    from PIL import Image, ImageDraw

    width, height, label_height = 640, 480, 36
    sheet = Image.new("RGB", (width * 2, (height + label_height) * 2), "#18212c")
    draw = ImageDraw.Draw(sheet)
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0  # Hide duplicate collision geometry.
    views = [("Overview", overview(model, data))] + [(name, name) for name in CAMERAS]
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for index, (label, camera) in enumerate(views):
            renderer.update_scene(data, camera=camera, scene_option=options)
            frame = Image.fromarray(renderer.render().copy())
            x, y = (index % 2) * width, (index // 2) * (height + label_height)
            draw.text((x + 12, y + 10), f"{index}: {label} | {keyframe} (paused)", fill="white")
            sheet.paste(frame, (x, y + label_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    print(f"Saved {path.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyframe", default="dock_cubes")
    parser.add_argument("--pose", choices=("parked", "approach"), default="parked")
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--item", default="cube_m")
    parser.add_argument("--approach-height", type=float, default=0.12, help="Height above grasp target in meters")
    parser.add_argument("--near-clip", type=float, help="Override rendering cutoff in meters for this viewer only; C toggles it")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--headless", action="store_true", help="Save a snapshot without opening the viewer")
    args = parser.parse_args()
    if args.headless and args.snapshot is None:
        parser.error("--headless requires --snapshot")
    if args.approach_height <= 0:
        parser.error("--approach-height must be positive")
    if args.near_clip is not None and args.near_clip <= 0:
        parser.error("--near-clip must be positive")

    model = mujoco.MjModel.from_xml_path(str(REPO / "models" / "room_scene.xml"))
    original_near = float(model.vis.map.znear)
    adjusted_near = original_near if args.near_clip is None else args.near_clip / model.stat.extent
    model.vis.map.znear = adjusted_near
    near_overridden = args.near_clip is not None
    print(f"Near clipping: {model.vis.map.znear * model.stat.extent * 1000:.1f} mm "
          f"(scene default: {original_near * model.stat.extent * 1000:.1f} mm)", flush=True)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(args.keyframe).id)
    mujoco.mj_forward(model, data)
    parked = data.qpos.copy()
    approach = prepare_approach(model, data, args) if args.pose == "approach" else None
    label = args.keyframe
    if approach is not None:
        label += f" / {args.arm} -> {args.item} +{args.approach_height * 100:.0f}cm"
    if args.near_clip is not None:
        label += f" / clip {args.near_clip * 1000:g}mm"
    if args.snapshot:
        save_snapshot(model, data, args.snapshot, label)
    if args.headless:
        return

    keys = queue.SimpleQueue()
    with mujoco.viewer.launch_passive(model, data, key_callback=keys.put) as viewer:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera = overview(model, data)
            viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance = camera.distance
            viewer.cam.azimuth = camera.azimuth
            viewer.cam.elevation = camera.elevation
            viewer.opt.geomgroup[3] = 0
        print("Viewer ready. Physics paused. Keys: 0 overview; 1 head; 2 left wrist; 3 right wrist.", flush=True)
        if approach is not None:
            print("Press P for parked arms; A for the approach pose. Camera mounts are unchanged.", flush=True)
        if args.near_clip is not None:
            print("Press C to compare the original and overridden near clipping distance.", flush=True)
        while viewer.is_running():
            while not keys.empty():
                key = keys.get()
                with viewer.lock():
                    if key == ord("C") and args.near_clip is not None:
                        near_overridden = not near_overridden
                        model.vis.map.znear = adjusted_near if near_overridden else original_near
                        print(f"Near clipping: {model.vis.map.znear * model.stat.extent * 1000:.1f} mm", flush=True)
                    elif key in (ord("P"), ord("A")) and approach is not None:
                        data.qpos[:] = parked if key == ord("P") else approach
                        mujoco.mj_forward(model, data)
                        print("Pose: " + ("parked" if key == ord("P") else "approach"), flush=True)
                    elif key == ord("0"):
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    elif ord("1") <= key <= ord("3"):
                        name = CAMERAS[key - ord("1")]
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        viewer.cam.fixedcamid = model.camera(name).id
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
